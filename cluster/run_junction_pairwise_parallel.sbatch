#!/bin/bash
# Junction resistance + pairwise resistances (paper item A), all cells of one grid
# in ONE job, 16 at a time. Choose the grid with GRID=pilot (default) or GRID=full:
#   sbatch run_junction_pairwise_parallel.sbatch
#   sbatch --export=ALL,GRID=confirm run_junction_pairwise_parallel.sbatch
# GRID=dense is the pre-registered dense grid (PREREGISTRATION_JUNCTION_DENSE.md): up to 100 cells,
# 100 networks per cell, 9 conditions; ~2.6x the work of GRID=confirm, so expect 4-6 h on 16 CPUs.
#   sbatch --export=ALL,GRID=dense run_junction_pairwise_parallel.sbatch
# GRID=transfer is the transfer run (PREREGISTRATION_JUNCTION_TRANSFER.md): 150 new geometries, 40 networks
# each, 12 conditions; about 0.8x the work of GRID=dense, so expect 4-6 h on 16 CPUs.
#   sbatch --export=ALL,GRID=transfer run_junction_pairwise_parallel.sbatch
# GRID=confirm is the pre-registered run (PREREGISTRATION_JUNCTION.md): 60 networks per
# cell, 16 conditions. Measured locally (10 processes, multithreaded BLAS) the 30-network,
# 6-condition grid took ~5 min; confirm is ~5x heavier per cell (2x networks, 16/6 conditions),
# and one thread per worker is slower, so expect 1-3 h. Resumable: finished cells are skipped.
# Copy run_junction_pairwise_task.py into ~/ColabProject/cluster first; run its
# --validate on the login node before submitting. mkdir -p logs before the first sbatch.
#SBATCH --job-name=jun_pw
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
python -W ignore run_junction_pairwise_task.py --grid "${GRID:-pilot}" --parallel ${SLURM_CPUS_PER_TASK}
