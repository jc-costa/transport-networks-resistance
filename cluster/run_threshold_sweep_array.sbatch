#!/bin/bash
# E1 of the revision plan: percolation rate only, 50 points (5 boxes x 10 values of N a^2),
# 2000 draws each. Box 20 takes ~20-30 min per point; the smaller boxes a few minutes.
# mkdir -p logs before the first sbatch.
#SBATCH --job-name=threshold_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-49%16

set -eo pipefail
set -x
source ~/envs/piano_sweep/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ~/ColabProject/cluster
python run_threshold_sweep_task.py "$SLURM_ARRAY_TASK_ID"
