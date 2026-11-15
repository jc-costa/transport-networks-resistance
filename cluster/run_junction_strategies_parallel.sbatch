#!/bin/bash
# E2, measurement strategies (PREREGISTRATION_JUNCTION_STRATEGIES.md): one movable tip, two movable tips and a
# four-point probe on two-dimensional stick films with an unknown junction resistance. 80 off-grid geometries,
# 30 networks each, 4 junction ratios per network; all geometries in ONE job, 16 at a time. Resumable.
#   sbatch run_junction_strategies_parallel.sbatch
# Copy run_junction_strategies_task.py (and run_junction_pairwise_task.py) into ~/ColabProject/cluster first and
# run `python -W ignore run_junction_strategies_task.py --validate` on the login node. mkdir -p logs first.
#SBATCH --job-name=jun_st
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
python -W ignore run_junction_strategies_task.py --parallel ${SLURM_CPUS_PER_TASK}
