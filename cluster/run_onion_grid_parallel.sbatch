#!/bin/bash
# ONE job on the complex QoS (48 CPUs) running the whole onion (N, a, d) grid
# through run_grid_parallel.py's process pool, instead of the 20-task array on
# the simple QoS, which is capped at 4 concurrent single-threaded tasks. Writes
# the same 20 chunk files. See cluster/README.md, "Running a grid in one job".
#SBATCH --job-name=onion_par
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --partition=short-complex
#SBATCH --qos=complex

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
# BLAS threads are pinned to 1 inside run_grid_parallel.py: the pool is the parallelism.

cd ~/ColabProject/cluster
python run_grid_parallel.py --task onion --workers ${SLURM_CPUS_PER_TASK}
