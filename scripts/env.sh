export MPI_DIR=/home/lulu/hpl-lib/ompi
export BLAS_DIR=/home/lulu/hpl-lib/openblas

export PATH=$PATH:$MPI_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib:$BLAS_HOME/lib

export C_INCLUDE_PATH=$MPI_HOME/include
export CPLUS_INCLUDE_PATH=$MPI_HOME/include
