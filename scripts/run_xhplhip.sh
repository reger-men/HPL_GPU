#!/usr/bin/env bash
# set -x #echo on

hpl_bin=./xhplhip
rocblas_dir=/opt/rocm/rocblas/lib
blas_dir=$HOME/OpenBLAS

filename=./HPL.dat

export LD_LIBRARY_PATH=${rocblas_dir}:${blas_dir}:$LD_LIBRARY_PATH

oversubscribe=true

P=$(sed -n "11, 1p" ${filename} | awk '{print $1}')
Q=$(sed -n "12, 1p" ${filename} | awk '{print $1}')
np=$(($P*$Q))

# Get local process numbering
set +u
if [[ -n ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]]; then
  rank=$OMPI_COMM_WORLD_LOCAL_RANK
  size=$OMPI_COMM_WORLD_LOCAL_SIZE
elif [[ -n ${SLURM_LOCALID+x} ]]; then
  rank=$SLURM_LOCALID
  size=$SLURM_TASKS_PER_NODE
  #Slurm can return a string like "2(x2),1". Get the first number
  size=$(echo $size | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/')
fi
set -u

# count the number of physical cores on node
num_cpu_cores=$(lscpu | grep "Core(s)" | awk '{print $4}')
num_cpu_sockets=$(lscpu | grep Socket | awk '{print $2}')
total_cpu_cores=$(($num_cpu_cores*$num_cpu_sockets))

# We assume a row-major process mapping to nodes
columns_per_node=$(( Q < size ? Q : size ))

# Check that the columns are evenly divided among nodes
if [[ $((Q % columns_per_node)) -gt 0 ]]; then
  echo "Invalid MPI grid parameters; Must have the same number of Q columns on every node; aborting";
  exit 1
fi

# Check that the rows are evenly divided among nodes
if [[ $((size % columns_per_node)) -gt 0 ]]; then
  echo "Invalid MPI grid parameters; Must have the same number of P rows on every node; aborting";
  exit 1
fi

rows_per_node=$(( size/columns_per_node ))

# Ranks in different processes rows will take distinct chunks of cores
row_stride=$((total_cpu_cores/rows_per_node))
col_stride=$((row_stride/columns_per_node))

myP=$((rank/columns_per_node))
myQ=$((rank%columns_per_node))

# Default core mapping

root_core=$((myP*row_stride + myQ*col_stride))

omp_num_threads=${col_stride}
# First omp place is the root core
omp_places="{$root_core}"

# Make contiuguous chunk of cores (to maximize L1/L2 locality)
for i in $(seq $((root_core+1)) $((root_core+col_stride-1)))
do
  omp_places+=",{$i}"
done

if [[ "${oversubscribe}" == true ]]; then
    # Add cores from different columns, without their root cores
    for q in $(seq 0 $((columns_per_node-1)))
    do
      if [[ "$q" == "$myQ" ]]; then
        continue
      fi
      q_core=$((myP*row_stride + q*col_stride))
      for i in $(seq $((q_core+1)) $((q_core+col_stride-1)))
      do
        omp_places+=",{$i}"
      done
      omp_num_threads=$((omp_num_threads+col_stride-1))
    done
fi

# Export OpenMP config
export OMP_NUM_THREADS=${omp_num_threads}
export OMP_PLACES=${omp_places}
export OMP_PROC_BIND=true

#run
${hpl_bin} 
