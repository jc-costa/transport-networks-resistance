#!/bin/bash
# Runs ONCE, before the array job: pools experiments/exp_*/ data, fits the
# model, and freezes the 28-candidate validation plan to validation_plan.json
# so every array task reads the exact same plan (no risk of drift if
# experiments/ changes between when tasks happen to start).
# No percolation networks drawn here -- just JSON I/O + a numpy least-squares
# fit, so this finishes in seconds even though it's still a real SLURM job.
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as
# the other scripts here -- SLURM resolves --output/--error before this
# script's first line runs).
#
#SBATCH --job-name=prep_val_plan
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=short-simple
#SBATCH --qos=simple

set -eo pipefail
set -x

module load Python/3.10.8-GCCcore-12.2.0
source ~/envs/piano_sweep/bin/activate

cd ~/ColabProject/cluster
python run_validation_task.py --prepare
