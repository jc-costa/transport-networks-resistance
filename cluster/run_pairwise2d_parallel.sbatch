#!/bin/bash
# E7 of the revision plan: pairwise resistances in 2D (22 validation candidates regenerated and
# checked against validation_results, plus the 101 cells of the pooled 2D sweeps, 30 networks each).
# One 48-core job on the complex QoS. Tested locally: the largest cell takes ~4 min on one core,
# the whole plan ~1 core-day at most, so well under the time limit below.
#SBATCH --job-name=pairwise2d
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=short-complex
#SBATCH --qos=complex

# Resumable: each finished candidate/cell is appended to pairwise2d_results/partial.jsonl, and a
# resubmitted job skips the units already there. Needs pairwise2d_plan.json next to the script.
set -eo pipefail
set -x
source ~/envs/piano_sweep/bin/activate
export MPLBACKEND=Agg
# one BLAS thread per pool worker (48 workers on 48 cores)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd ~/ColabProject/cluster
python run_pairwise2d_task.py --parallel ${SLURM_CPUS_PER_TASK}
