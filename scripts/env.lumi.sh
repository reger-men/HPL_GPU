# Any other commands must follow the #SBATCH directives
module load LUMI/22.06 partition/G
#module load rocm/5.1.4

module use /project/project_462000075/paklui/modulefiles
#module load rocm/5.3.0-10584
#module load rocm/5.3.0-10619
module load rocm/5.3.0-10670
#module load openblas/0.3.17-omp
#module load cce/14.0.2
#module load cray-libsci/22.08.1.1
module load cray-mpich/8.1.18
#module load craype/2.7.17

#
# env
#
# enable GPU aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
# to work around the OFI registration cache issue for > 8 nodes
#export FI_MR_CACHE_MAX_COUNT=0
export MPICH_RANK_REORDER_DISPLAY=1
