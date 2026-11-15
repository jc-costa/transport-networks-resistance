#!/bin/bash
# E2b (PREREGISTRATION_JUNCTION_STRATEGIES_MORE.md): more readings (up to 2187 attempts), tips landing on chosen
# sites against a wide landing, and the mean and median of the readings, on the SAME 80 geometries as the
# strategies run. All geometries in ONE job, 16 at a time. Resumable.
#   sbatch run_junction_strategies_more_parallel.sbatch
# Copy run_junction_strategies_more_task.py, run_junction_strategies_task.py and run_junction_pairwise_task.py
# into ~/ColabProject/cluster first and run `python -W ignore run_junction_strategies_more_task.py --validate`
# on the login node. mkdir -p logs first.
#SBATCH --job-name=jun_st2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=short-simple
#SBATCH --qos=simple

set -eo pipefail
set -x
source ~/envs/piano_sweep/bin/activate
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd ~/ColabProject/cluster
python -W ignore run_junction_strategies_more_task.py --parallel ${SLURM_CPUS_PER_TASK}
