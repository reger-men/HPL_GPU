# modules
module reset

module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load amd/5.2.0
module load rocm/5.2.0
module load cray-mpich/8.1.17
module load openblas/0.3.17-omp

#
# env
#
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
# enable GPU aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
# to work around the OFI registration cache issue for > 8 nodes
export FI_MR_CACHE_MAX_COUNT=0
#export MPICH_SMP_SINGLE_COPY_MODE=NONE # does not work
export MPICH_RANK_REORDER_DISPLAY=1
