module reset

module load rocm/5.3.0-10584

# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/global/software/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/openblas-0.3.20-qbm5uv3ntjerkx4jzrprmelytviwoq2e/lib:/global/software/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/openmpi-4.1.4-3z7jsddbvczl4duixalzrtap3q5nuvjk/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/global/home/lulu/blis/lib:/global/home/lulu/ompi/lib:/global/software_internal/rocm/rocm-5.3.0-10584/lib"
# export MPICH_GPU_SUPPORT_ENABLED=1
