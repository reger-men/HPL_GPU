#!/bin/bash

MPI_DIR=$HOME/hpl-lib/ompi/bin
num_cpu_cores=16
num_process=4

# FOR OMP
export OMP_NUM_THREADS=${num_cpu_cores}
export LD_LIBRARY_PATH=openblas:$LD_LIBRARY_PATH

# ./xhpl
HSA_ENABLE_SDMA=1 ${MPI_DIR}/mpirun --allow-run-as-root -np ${num_process} --map-by node:PE=${num_cpu_cores} --bind-to core:overload-allowed --report-bindings ./xhplhip
grep --color "e+" HPL.out
