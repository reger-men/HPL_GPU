#!/bin/bash

MPI_DIR=/home/lulu/hpl-lib/ompi/bin
num_cpu_cores=4
num_process=1

# FOR OMP
export OMP_NUM_THREADS=${num_cpu_cores}
export LD_LIBRARY_PATH=openblas:$LD_LIBRARY_PATH

# ./xhpl
HSA_ENABLE_SDMA=1 ${MPI_DIR}/mpirun --allow-run-as-root -np ${num_process} --map-by node:PE=${num_cpu_cores} --bind-to core:overload-allowed --report-bindings -x UCX_TLS=sm,rocm ./xhplhip
grep --color "e+" HPL.out


