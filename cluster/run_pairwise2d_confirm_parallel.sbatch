#!/bin/bash
# Confirmatory 2D pairwise test (paper item e; see PREREGISTRATION.md).
# 21 candidates x 60 fresh networks, one job on short-simple with 16 workers
# (one candidate per worker; with 21 candidates more cores would mostly idle).
# The costliest candidate takes ~8-10 min on one core, so the job should end in
# about 20-30 min. Resumable: finished candidates are appended to
# pairwise2d_confirm_results/partial.jsonl and skipped on resubmission.
# Memory: E7 used 96 GB for 48 workers (~2 GB each); 16 workers need ~32 GB.
#SBATCH --job-name=pw2d_confirm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
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
python run_pairwise2d_confirm_task.py --parallel ${SLURM_CPUS_PER_TASK}
