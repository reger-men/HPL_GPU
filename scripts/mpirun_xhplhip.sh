#!/usr/bin/env bash
# set -x #echo on

hpl_bin=./xhplhip
mpi_dir=$HOME/ompi
mpi_bin=${mpi_dir}/bin/mpiexec
mpi_lib=${mpi_dir}/lib
hpl_runscript=./run_xhplhip.sh

filename=./HPL.dat

P=$(sed -n "11, 1p" ${filename} | awk '{print $1}')
Q=$(sed -n "12, 1p" ${filename} | awk '{print $1}')
np=$(($P*$Q))

# count the number of physical cores
num_cpu_cores=$(lscpu | grep "Core(s)" | awk '{print $4}')
num_cpu_sockets=$(lscpu | grep Socket | awk '{print $2}')
total_cpu_cores=$(($num_cpu_cores*$num_cpu_sockets))

export LD_LIBRARY_PATH=${mpi_lib}:$LD_LIBRARY_PATH
#Default MPI options
mpi_args="--map-by node:PE=${total_cpu_cores} --bind-to core:overload-allowed --mca pml ucx --mca btl ^vader,tcp,openib,uct ${mpi_args}"

${mpi_bin} --allow-run-as-root -np ${np} ${mpi_args} ${hpl_runscript} 
grep --color "e+" HPL.out
