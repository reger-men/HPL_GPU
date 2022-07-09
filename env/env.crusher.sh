module reset

module load craype-accel-amd-gfx90a
#module load PrgEnv-cray
module load PrgEnv-amd #maybe?...
module load amd/5.1.0
module load rocm/5.1.0
#module load cray-mpich/8.1.15
module load cray-mpich/8.1.16
module load openblas/0.3.17-omp

module load htop

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

#module load rocm/5.1.0
#module load craype-x86-trento
#module load libfabric
#module load craype-network-ofi
#module load perftools-base
#module load cce
#module load craype
#module load cray-dsmml
#module load cray-mpich
#module load cray-libsci
#module load craype-accel-amd-gfx90a

#export MPICH_GPU_SUPPORT_ENABLED=1

#export LD_LIBRARY_PATH=${ROCM_PATH}/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=${CRAY_MPICH_DIR}/lib:${LD_LIBRARY_PATH}

# workaround for missing libomp.so
#export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib:$LD_LIBRARY_PATH

# workaround for linking ibamdhip64.so.4 by Cray netwokr. Just a softlink in directory to redirect to libamdhip64.so.5
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/lib
