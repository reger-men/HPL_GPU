export MPI_HOME=/opt/mpi/ompi
export BLAS_HOME=/home/hpl/hpl/local/blis/install

export PATH=$PATH:$MPI_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib:$BLAS_HOME/lib

export C_INCLUDE_PATH=$MPI_HOME/include
export CPLUS_INCLUDE_PATH=$MPI_HOME/include
