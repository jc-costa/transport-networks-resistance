#!/bin/bash
# Large-film sweep (paper item d): all 70 points in ONE job, 16 at a time
# (Apuana runs at most a few jobs at once, so one multi-core job is better than
# a 70-task array). Films 5 to 80 stick lengths across, N a^2 from 4 to 20,
# 50 percolating networks per point, sparse solver; see run_large_film_task.py.
#
# Sizing (measured locally): the largest point (Lambda = 80, N a^2 = 20,
# ~130000 sticks, ~650000 nodes) peaks at 3.1 GB and takes ~4 s per network;
# 16 workers stay under ~50 GB even if all were at that point. Points far below
# their threshold stop after 600 failed draws in a row. Expected wall time: a few
# hours. Resumable: finished points are skipped on resubmission.
#
# Copy sparse_network.py and run_large_film_task.py into ~/ColabProject/cluster
# first. mkdir -p logs before the first sbatch.
#SBATCH --job-name=large_film
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
python run_large_film_task.py --parallel ${SLURM_CPUS_PER_TASK}
