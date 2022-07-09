module reset

module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load amd/5.1.0
module load rocm/5.1.0
module load cray-mpich/8.1.16
module load openblas/0.3.17-omp

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
