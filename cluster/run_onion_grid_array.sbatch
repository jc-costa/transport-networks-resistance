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
#
# Measured 2026-08-31 on the post-fix rerun: with --cpus-per-task=8 and the array
# throttle raised, the whole 420-combo scalar grid finished in ~7 minutes of wall
# clock and the onion grid in ~30, not the ~20 min PER CHUNK estimated from the
# August timing above. The old figure came from a single local chunk on 2 cores.
#SBATCH --job-name=onion_grid_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
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
python run_onion_grid_task.py "$SLURM_ARRAY_TASK_ID"
