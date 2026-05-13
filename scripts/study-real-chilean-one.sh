#!/bin/bash

#SBATCH --job-name=eduranker_main_real_chilean_sim                             
#SBATCH --nodes=1                  
#SBATCH --cpus-per-task=96          
#SBATCH --mem=16GB                     
#SBATCH --time=40:10:00             
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --output=/scratch/rm6609/MatchingInferenceEngine/experimental_output/mass-sim-logs/job_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

SEED=40
K=6
M=15
MAX_ITER=20
MAX_ITER_OPT=15
N_JOBS=96
LR=0.05
PROFILE_TIMING=1
SAVE_PARAMS=1
SAVE_BEST_SAMPLE=1
MAX_P=10
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

PROFILE_ARG=""
if [[ "$PROFILE_TIMING" -eq 1 ]]; then
    PROFILE_ARG="--profile_timing"
fi

SAVE_PARAMS=""
if [[ "$SAVE_PARAMS" -eq 1 ]]; then
    SAVE_PARAMS="--save_params"
fi

SAVE_BEST_SAMPLE=""
if [[ "$SAVE_BEST_SAMPLE" -eq 1 ]]; then
    SAVE_BEST_SAMPLE="--save_best_sample"
fi

echo "========================================"
echo "Job Start: $TIMESTAMP | Seed: $SEED"
echo "Profile timing: $PROFILE_TIMING"
echo "========================================"

OVERLAY="/scratch/rm6609/research/overlay-persistent-manual-2.ext3"

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "
    conda activate research
    export HOME=/ext3/conda_home
    cd /scratch/rm6609/MatchingInferenceEngine
    python3 src/chilean_experiment_driver.py --seed $SEED --K $K --M $M --lr $LR --max_iter $MAX_ITER --max_iter_opt $MAX_ITER_OPT  --n_jobs $N_JOBS --max_p $MAX_P $PROFILE_ARG $SAVE_PARAMS $SAVE_BEST_SAMPLE
"

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S')"
