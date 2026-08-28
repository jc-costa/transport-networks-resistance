#!/bin/bash
# Boosted-N_TRIALS rerun of the strongest-signal (N, a, d) groups from the onion-curve grid:
# 69 combos (top 5 ANOVA-strongest matched-R_eff groups, see run_onion_grid_boost_task.py's
# --prepare mode / cluster/onion_grid_boost_plan.json) split into 20 chunks of ~4, one SLURM
# array task per chunk. IMPORTANT: run `python run_onion_grid_boost_task.py --prepare` locally
# once BEFORE submitting this (writes onion_grid_boost_plan.json) -- every task reads that same
# frozen plan.
#
# --partition=short-simple / --qos=simple / --array=...%4: same config
# run_onion_grid_array.sbatch settled on after testing showed qos=complex caps this user at 1
# concurrent job (worse than simple's 4) -- reused directly here rather than re-probing.
#
# --time=02:00:00: each combo here does the same per-network work as
# run_onion_grid_task.py's combos (scalar solve + one Laplacian pseudoinverse) but at
# N_TRIALS_BOOST=50 instead of 6 (~8.3x as many percolating draws needed per combo), while each
# task only gets ~4 combos instead of up to 21 -- net cost per task is somewhat higher than that
# script's 90-minute budget. Not empirically timed at the worst-case corner (N=150) yet; revisit
# after the first real chunk finishes, same as that script's own note.
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as every other script
# here).
#
#SBATCH --job-name=onion_boost_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
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
python run_onion_grid_boost_task.py "$SLURM_ARRAY_TASK_ID"
