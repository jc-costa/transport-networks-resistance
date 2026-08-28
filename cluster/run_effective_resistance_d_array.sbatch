#!/bin/bash
# Parallel version of effective_resistance_vs_d.ipynb's Part C extended grid:
# 420 (N, a, d) combinations split into 20 chunks of ~21, one SLURM array
# task per chunk -- each task independently recomputes the same
# deterministic 420-combo list (see run_effective_resistance_d_task.py) and
# only processes its own slice. Same pattern as run_random_networks_array.sbatch.
#
# --partition=short-simple / --qos=simple / --array=...%4: the short-complex /
# qos=complex probe was tested directly on the sibling onion-grid array job
# (run_onion_grid_array.sbatch, jobs 12587 and 12591 on 2026-08-18) -- both
# times squeue showed exactly 1 task RUNNING and the other 19 stuck PENDING
# with REASON=QOSMaxJobsPerUserLimit, i.e. "complex" caps this user at 1
# concurrent job, worse than "simple"'s already-confirmed 4 (observed via the
# same REASON code on an earlier run of this exact script). Reverted back to
# the known-good simple/%4 config here too; don't re-attempt complex without
# new evidence the cap has changed.
#
# --time=00:45:00: combos are NOT evenly costly -- near-percolation-threshold
# combos (small N) can need hundreds of retries even though each retry is
# cheap, while large-N/large-a/thin-d combos are individually slow (dense
# solve). Contiguous chunking (same as run_random_networks_task.py) doesn't
# load-balance for this, so some chunks land unlucky; 45 min gives real
# margin over the ~20 min single-chunk cost observed in local timing.
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as
# every other script here).
#
#SBATCH --job-name=reff_d_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:45:00
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
python run_effective_resistance_d_task.py "$SLURM_ARRAY_TASK_ID"
