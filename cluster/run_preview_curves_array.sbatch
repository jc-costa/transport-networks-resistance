#!/bin/bash
# Part A preview onion curves (notebooks/r_eff_ij_vs_N_a_d.ipynb Part A, the
# three-curve validation figure in chapters/resultados/resultados.tex) at
# N_TRIALS=50, spread over an array. See run_preview_curves_task.py's docstring
# for why the preview needed recomputing and why its seeding differs from
# run_onion_grid_task.py's.
#
# --array=0-24%20: 150 work units (3 combos x 50 realizations) dealt round-robin
# across 25 tasks, so the expensive (45, 0.2, 0.10) units spread evenly instead
# of piling onto one task. The %20 throttle follows the same reasoning as
# run_onion_grid_array.sbatch: it is a self-imposed ceiling above what qos=simple
# actually grants (4 concurrent, as of 2026-08), so it costs nothing now and
# picks up any future increase for free.
#
# --time=02:00:00: the whole job is about one core-hour, dominated by the
# (45, 0.2, 0.10) draws (roughly 1 draw in 3000 percolates, so ~3000 draws per
# realization at ~20 ms each). Two units per task on average, but the draw count
# per unit is geometrically distributed with a long tail, so the budget is set
# well above the ~5 minute expectation.
#
# --cpus-per-task=4 / --mem=4G: smaller than the grid jobs on purpose. These are
# 45-70 stick networks, so the pseudoinverse is tiny and the runtime is almost
# entirely single-threaded geometry draws; the extra cores the grid asks for
# would sit idle here.
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as
# every other script here).
#SBATCH --job-name=preview_curves
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-24%20

set -eo pipefail
set -x

# No `module load`: see run_onion_grid_array.sbatch for the full history. The venv
# must be the one built from /usr/bin/python3, not from the login node's
# /opt/miniforge3, or this dies on a compute node with `python: command not found`.
source ~/envs/piano_sweep/bin/activate

export MPLBACKEND=Agg

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd ~/ColabProject/cluster
python run_preview_curves_task.py "$SLURM_ARRAY_TASK_ID"
