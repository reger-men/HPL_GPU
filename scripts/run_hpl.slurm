#!/bin/bash
#SBATCH -v
#SBATCH -N 1
##SBATCH -n 8
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -A VEN114
#SBATCH -J xhplhip
#SBATCH --gpu-bind=closest
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

source ../env/env.crusher-rocm510.sh

NP=$SLURM_NPROCS
NODES=$SLURM_NNODES
DATE=$(date +%y%m%d-%H%M%S)
LOG=log.hpl-gpu-${NP}np-${HOSTNAME}-${DATE}.txt

cp config/HPL_${NP}GPU.dat HPL.dat

EXE="../build/xhplhip"
CMD=""
CMD+="srun "
CMD+="-v "
CMD+="-n $NP "
CMD+="-N $NODES "
CMD+="-A VEN114 "
CMD+="--gpu-bind=closest "
CMD+="--ntasks-per-node=8 "
CMD+="--gpus-per-node=8 "
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
