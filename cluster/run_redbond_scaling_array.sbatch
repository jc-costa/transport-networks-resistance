#!/bin/bash
# E2 of the revision plan: red bonds at the threshold (N a^2 = 5.40, 5.64, 5.90) in boxes of
# 5, 6.67, 10, 14.3 and 20 stick lengths; one task per point (15). The box-20 points draw
# ~2500 sticks, ~10 s per draw in identify_intersections; the task stops itself at 5.5 h.
# mkdir -p logs before the first sbatch.
#SBATCH --job-name=redbond_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-14%15

set -eo pipefail
set -x
source ~/envs/piano_sweep/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd ~/ColabProject/cluster
python run_redbond_scaling_task.py "$SLURM_ARRAY_TASK_ID"
