#!/bin/bash
# Parallel version of the R_eff,ij onion-layer curve grid over (N, a, d):
# same 420 (N, a, d) combinations, same 20-chunk split, as
# run_effective_resistance_d_array.sbatch -- each task independently
# recomputes the same deterministic 420-combo list (see
# run_onion_grid_task.py) and only processes its own slice.
#
# --partition=short-simple / --qos=simple / --array=...%4: the short-complex /
# qos=complex probe (see run_effective_resistance_d_array.sbatch's history)
# was tested directly on this job on 2026-08-18 (jobs 12587, 12591) -- both
# times squeue showed exactly 1 task RUNNING and the other 19 stuck PENDING
# with REASON=QOSMaxJobsPerUserLimit, i.e. "complex" caps this user at 1
# concurrent job, worse than "simple"'s already-confirmed 4. Reverted back to
# the known-good simple/%4 config; don't re-attempt complex without new
# evidence the cap has changed.
#
# --time=01:30:00: each combo here does everything
# run_effective_resistance_d_task.py's combo does (same dense O(n^3) solve
# for the scalar R_eff) PLUS one Laplacian pseudoinverse per accepted network
# (utils/laplacian_utils.py's effective_resistance_from_reference, called via
# multimeter.run_onion_protocol_pinv) -- roughly another O(n^3) op on top, so
# budgeted at ~2x that script's 45-minute margin. Revisit after the first
# real chunk finishes (see the local smoke-test timing from
# `python run_onion_grid_task.py 0`).
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as
# every other script here).
#
#SBATCH --job-name=onion_grid_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=01:30:00
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
python run_onion_grid_task.py "$SLURM_ARRAY_TASK_ID"
