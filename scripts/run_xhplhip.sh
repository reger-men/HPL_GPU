#!/usr/bin/env bash
# set -x #echo on

hpl_bin=./xhplhip
rocblas_dir=/opt/rocm-5.2.0/lib
blas_dir=/global/home/lulu/hyc/rocHPL-main/tpl/blis/lib

filename=./HPL.dat
p=-1
q=-1
export LD_LIBRARY_PATH=${rocblas_dir}:${blas_dir}:$LD_LIBRARY_PATH

oversubscribe=true

P=$(sed -n "11, 1p" ${filename} | awk '{print $1}')
Q=$(sed -n "12, 1p" ${filename} | awk '{print $1}')
np=$(($P*$Q))

set +u
if [[ -n ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]]; then
  globalRank=$OMPI_COMM_WORLD_RANK
  globalSize=$OMPI_COMM_WORLD_SIZE
  rank=$OMPI_COMM_WORLD_LOCAL_RANK
  size=$OMPI_COMM_WORLD_LOCAL_SIZE
elif [[ -n ${SLURM_LOCALID+x} ]]; then
  globalRank=$SLURM_PROCID
  globalSize=$SLURM_NTASKS
  rank=$SLURM_LOCALID
  size=$SLURM_TASKS_PER_NODE
  #Slurm can return a string like "2(x2),1". Get the first number
  size=$(echo $size | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/')
fi
set -u

#Determing node-local grid size
if [[ "$p" -lt 1 && "$q" -lt 1 ]]; then
  # no node-local grid was specified, pick defaults
  q=$(( (Q<=size) ? Q : size))

  if [[ $((size % q)) -gt 0 ]]; then
    echo "Invalid MPI grid parameters; Unable to form node-local grid; aborting";
    exit 1
  fi

  p=$(( size/q ))

elif [[ "$p" -lt 1 ]]; then
  #q was specified

  if [[ $((size % q)) -gt 0 ]]; then
    echo "Invalid MPI grid parameters; Unable to form node-local grid; aborting";
    exit 1
  fi

  p=$(( size/q ))

elif [[ "$q" -lt 1 ]]; then
  #p was specified

  if [[ $((size % p)) -gt 0 ]]; then
    echo "Invalid MPI grid parameters; Unable to form node-local grid; aborting";
    exit 1
  fi

  q=$(( size/p ))

else
  #Both p and q were specified
  if [[ $size -ne $((p*q)) ]]; then
    echo "Invalid MPI grid parameters; Unable to form node-local grid; aborting";
    exit 1
  fi
fi

# Check that the columns are evenly divided among nodes
if [[ $((P % p)) -gt 0 ]]; then
  echo "Invalid MPI grid parameters; Must have the same number of P rows on every node; aborting";
  exit 1
fi

# Check that the rows are evenly divided among nodes
if [[ $((Q % q)) -gt 0 ]]; then
  echo "Invalid MPI grid parameters; Must have the same number of Q columns on every node; aborting";
  exit 1
fi

# count the number of physical cores on node
num_cpu_cores=$(lscpu | grep "Core(s)" | awk '{print $4}')
num_cpu_sockets=$(lscpu | grep Socket | awk '{print $2}')
total_cpu_cores=$(($num_cpu_cores*$num_cpu_sockets))

# Ranks in different processes rows will take distinct chunks of cores
row_stride=$((total_cpu_cores/p))
col_stride=$((row_stride/q))

myp=$((rank%p))
myq=$((rank/p))

#Although ranks are column-major order, we select GPUs in row-major order on node
mygpu=$((myq+myp*q))

# Try to detect special Bard-peak core mapping
if [[ -n ${HPL_PLATFORM+x} ]]; then
  platform=$HPL_PLATFORM
else
  platform=$(cat /sys/class/dmi/id/product_name)
fi

if [[ "$platform" == "BardPeak" || "$platform" == "HPE_CRAY_EX235A" ]]; then
  # Special core mapping for BardPeak

  # Debug
  # if [[ $globalRank == 0 ]]; then
  #   echo "BardPeak platform detected"
  # fi

  # Sanity check
  if [[ $size -gt 8 ]]; then
    echo "Unsupported number of ranks on BardPeak platform; aborting";
    exit 1
  fi

  # GCD0 cores="48-55"
  # GCD1 cores="56-63"
  # GCD2 cores="16-23"
  # GCD3 cores="24-31"
  # GCD4 cores="0-7"
  # GCD5 cores="8-15"
  # GCD6 cores="32-39"
  # GCD7 cores="40-47"

  root_cores=(48 56 16 24 0 8 32 40)
  root_core=${root_cores[mygpu]}

  # First omp place is the root core
  omp_places="{$root_core}"

  # First assign the CCD
  for i in $(seq $((root_core+1)) $((root_core+8-1)))
  do
    omp_places+=",{$i}"
  done
  omp_num_threads=8

  places="{$root_core-$((root_core+7))}"

  # Loop through unassigned CCDs
  for c in $(seq $((mygpu+size)) $size 7)
  do
    iroot_core=${root_cores[c]}
    for i in $(seq $((iroot_core)) $((iroot_core+8-1)))
    do
      omp_places+=",{$i}"
    done
    omp_num_threads=$((omp_num_threads+8))
    places+=",{$iroot_core-$((iroot_core+7))}"
  done

  if [[ "${oversubscribe}" == true ]]; then
    # Add cores from different columns, without their root cores
    for j in $(seq 0  $((q-1)))
    do
      if [[ "$j" == "$myq" ]]; then
        continue
      fi
      for jj in $(seq 0 $size 7)
      do
        q_gpu=$((jj+j+myp*q))
        q_core=$((root_cores[q_gpu]))
        offset=$(( (q_gpu>=size) ? 0 : 1))
        for i in $(seq $((q_core+offset)) $((q_core+8-1)))
        do
          omp_places+=",{$i}"
        done
        omp_num_threads=$((omp_num_threads+8-offset))
        places+=",{$((q_core+offset))-$((q_core+7))}"
      done
    done
  fi

else
  # Default core mapping
  root_core=$((myp*row_stride + myq*col_stride))

  omp_num_threads=${col_stride}
  # First omp place is the root core
  omp_places="{$root_core}"

  # Make contiuguous chunk of cores (to maximize L1/L2 locality)
  for i in $(seq $((root_core+1)) $((root_core+col_stride-1)))
  do
    omp_places+=",{$i}"
  done

  if [[ $col_stride -gt 1 ]]; then
    places="{$root_core-$((root_core+col_stride-1))}"
  else
    places="{$root_core}"
  fi

  if [[ "${oversubscribe}" == true ]]; then
    # Add cores from different columns, without their root cores
    for j in $(seq 0 $((q-1)))
    do
      if [[ "$j" == "$myq" ]]; then
        continue
      fi
      q_core=$((myp*row_stride + j*col_stride))
      for i in $(seq $((q_core+1)) $((q_core+col_stride-1)))
      do
        omp_places+=",{$i}"
      done
      omp_num_threads=$((omp_num_threads+col_stride-1))

      if [[ $col_stride -gt 2 ]]; then
        places+=",{$((q_core+1))-$((q_core+col_stride-1))}"
      elif [[ $col_stride -gt 1 ]]; then
        places+=",{$((q_core+1))}"
      fi

    done
  fi
fi
# Export OpenMP config
export OMP_NUM_THREADS=${omp_num_threads}
export OMP_PLACES=${omp_places}
export OMP_PROC_BIND=true
if [[ $globalRank -lt $size ]]; then
  echo "Node Binding: Process $rank [(p,q)=($myp,$myq)] CPU Cores: $omp_num_threads - $places"
fi
#run
${hpl_bin} 
