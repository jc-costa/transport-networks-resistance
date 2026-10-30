#!/bin/bash
# E4b of the revision plan: flat sticks at uniform height, same (N, a, d) grid plus d = 0.02, 24 realizations each.
# One 48-core job on the complex QoS, like run_onion_grid_parallel.sbatch.
#SBATCH --job-name=onion_flat_par
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=short-complex
#SBATCH --qos=complex

# Resumable: each finished combination is appended to <results>/partial.jsonl, and a
# resubmitted job skips the combinations already there.
set -eo pipefail
set -x
source ~/envs/piano_sweep/bin/activate
export MPLBACKEND=Agg
# one BLAS thread per pool worker (48 workers on 48 cores)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd ~/ColabProject/cluster
python run_flat_sticks_task.py --parallel ${SLURM_CPUS_PER_TASK}
