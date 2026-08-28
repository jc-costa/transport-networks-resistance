#!/bin/bash
# Parallel version of inverse_N_a_from_R_eff.ipynb's Section 6 validation:
# 28 candidates (7 R_eff targets x 4 (N,a) candidates each), one SLURM array
# task per candidate, each drawing its own 30 percolating networks
# independently -- they don't depend on each other, so this is the
# "embarrassingly parallel" case the earlier sequential single-job version
# didn't take advantage of.
#
# --array=0-27%10: 28 tasks total, but AT MOST 10 running at the same time,
# even if the cluster is otherwise idle -- a deliberate courtesy cap so this
# doesn't grab a big slice of shared capacity all at once. Each task is
# small (CPU/RAM-wise), so 10 concurrent is a light footprint on a cluster
# with ~400 compute-node cores.
#
# REQUIRES prepare_validation_plan.sbatch to have already completed
# successfully (validation_plan.json must exist) -- every task reads that
# same frozen plan by index.
#
#SBATCH --job-name=inverse_val_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-27%10

set -eo pipefail
set -x

module load Python/3.10.8-GCCcore-12.2.0
source ~/envs/piano_sweep/bin/activate

cd ~/ColabProject/cluster
python run_validation_task.py "$SLURM_ARRAY_TASK_ID"
