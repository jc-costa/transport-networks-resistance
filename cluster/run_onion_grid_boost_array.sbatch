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
# --array=0-19%20: the throttle is deliberately set ABOVE the concurrency the QoS grants.
# The %N suffix is only a self-imposed ceiling; the QoS enforces the real one, holding
# the surplus tasks as PENDING with REASON=QOSMaxJobsPerUserLimit (exactly what was
# observed in August, when qos=simple granted 4 and qos=complex granted 1). Asking for
# 20 therefore behaves identically to asking for 4 while that cap stands, and picks up
# the benefit for free if it is ever raised. slurmdbd was down on the login node when
# this was written, so the cap could not be queried directly.
# --cpus-per-task=8 / --mem=8G: the short-simple nodes carry 64+ cores and ~126 GB each
# (9 of them), so a task asking for 8 cores is still tiny. The extra cores go to the
# dense solves and the Laplacian pseudoinverse through the BLAS thread settings below;
# the percolation retries that dominate the median combo stay single-threaded, so the
# gain shows up on the large-N tail rather than across the board.
#SBATCH --job-name=onion_boost_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-19%20

set -eo pipefail
set -x

# No `module load` here on purpose: the Python/3.10.8-GCCcore-12.2.0 module these jobs
# used until 2026-08 no longer exists on Apuana (`module spider Python` finds nothing
# at all), and its removal took the ~/envs/piano_sweep venv built against
# /opt/easybuild/software/Python/3.10.8-GCCcore-12.2.0 with it.
#
# The venv is now built from /usr/bin/python3 (3.10.12), NOT from the login node's
# conda at /opt/miniforge3. That distinction matters: /opt is node-local here, so a
# venv rooted in /opt/miniforge3 activates fine on the login node and then dies on a
# compute node with `python: command not found`, because bash skips the dangling
# symlink (job 13754). /usr/bin/python3 exists on every node (verified on
# cluster-node1, job 13755). Re-run smoke_test.sbatch after any change to this setup.
source ~/envs/piano_sweep/bin/activate

export MPLBACKEND=Agg

# Keep the BLAS thread pool inside the CPU allocation. numpy sizes its pool from the
# node's core count, not from the cgroup, so without this a --cpus-per-task=2 task
# spawns one thread per core on the node and they fight over 2 CPUs, which is slower
# than running single-threaded. Raise --cpus-per-task to give the dense solves more
# threads; these follow it automatically.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd ~/ColabProject/cluster
python run_onion_grid_boost_task.py "$SLURM_ARRAY_TASK_ID"
