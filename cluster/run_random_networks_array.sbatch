#!/bin/bash
# Parallel version of test_150_random_networks.ipynb's Section 4 batch run:
# 1500 cases split into 20 chunks of 75, one SLURM array task per chunk --
# each task independently recomputes the same deterministic 1500-config list
# (see run_random_networks_task.py) and only processes its own slice.
#
# --array=0-19%4: the "simple" QOS caps concurrent jobs per user at 4
# regardless of what's requested (observed directly on the earlier array run:
# SLURM showed QOSMaxJobsPerUserLimit when asked for %10) -- %4 just makes
# that explicit instead of relying on the QOS to silently throttle a higher
# number. Each task is small (2 CPUs, 4GB), so this stays a light footprint
# on the cluster either way.
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
#SBATCH --job-name=random_nets_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:20:00
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
python run_random_networks_task.py "$SLURM_ARRAY_TASK_ID"
