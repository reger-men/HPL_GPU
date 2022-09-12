#!/bin/bash
#SBATCH -v
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -A VEN114
#SBATCH -J xhplhip
#SBATCH --gpu-bind=closest
#SBATCH --job-name=hpl_gpu   # Job name
#SBATCH --output=hpl.o%j # Name of stdout output file
#SBATCH --error=hpl.e%j  # Name of stderr error file
#SBATCH --partition=gpu    # Partition (queue) name
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0-01:00:00      # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000075  # Project for billing
#SBATCH --exclusive

source ../env/env.lumi.sh

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

NP=$SLURM_NPROCS
NODES=$SLURM_NNODES
DATE=$(date +%y%m%d-%H%M%S)
LOG=log.hpl-gpu-${NP}np-${HOSTNAME}-${DATE}.txt

cp config/HPL_${NP}GPU.dat HPL.dat

EXE="./xhplhip -p 4 -q 2"
CMD=""
CMD+="srun "
CMD+="-v "
CMD+="-n $NP "
CMD+="-N $NODES "
# CMD+="-A VEN114 "
CMD+="--gpu-bind=closest "
CMD+="--ntasks-per-node=8 "
CMD+="--gpus-per-node=8 "
CMD+="--exclusive "
CMD+="-c 8 "
CMD+="-o $LOG -e $LOG "
#CMD+="${HOME}/mpich_bind.sh "
CMD+="$EXE"

#export MPICH_SMP_SINGLE_COPY_MODE=NONE # does not work
export FI_MR_CACHE_MAX_COUNT=0
export MPICH_RANK_REORDER_DISPLAY=1

echo $CMD >> $LOG
echo $CMD 2>&1 | tee -a $LOG
     $CMD 2>&1 | tee -a $LOG
cat HPL.dat 2>&1 | tee -a $LOG
cat HPL.out 2>&1 | tee -a $LOG