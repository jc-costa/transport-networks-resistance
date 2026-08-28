#!/bin/bash
# Parallel version of test_150_random_networks.ipynb's Section 4 batch run:
# 1500 cases split into 20 chunks of 75, one SLURM array task per chunk --
# each task independently recomputes the same deterministic 1500-config list
# (see run_random_networks_task.py) and only processes its own slice.
#
# --array=0-19%4: the "simple" QOS caps concurrent jobs per user at 4
# regardless of what's requested (observed directly on the earlier array run:
# SLURM showed QOSMaxJobsPerUserLimit when asked for %10) -- %4 just makes
# that explicit instead of relying on the QOS to silently throttle a higher
# number. Each task is small (2 CPUs, 4GB), so this stays a light footprint
# on the cluster either way.
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as
# every other script here).
#
#SBATCH --job-name=random_nets_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-19%4

set -eo pipefail
set -x

module load Python/3.10.8-GCCcore-12.2.0
source ~/envs/piano_sweep/bin/activate

export MPLBACKEND=Agg

cd ~/ColabProject/cluster
python run_random_networks_task.py "$SLURM_ARRAY_TASK_ID"
