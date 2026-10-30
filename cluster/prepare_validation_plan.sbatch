#!/bin/bash
# Runs ONCE, before the array job: pools experiments/exp_*/ data, fits the
# model, and freezes the 28-candidate validation plan to validation_plan.json
# so every array task reads the exact same plan (no risk of drift if
# experiments/ changes between when tasks happen to start).
# No percolation networks drawn here -- just JSON I/O + a numpy least-squares
# fit, so this finishes in seconds even though it's still a real SLURM job.
#
# mkdir -p logs in this folder before the first `sbatch` call (same rule as
# the other scripts here -- SLURM resolves --output/--error before this
# script's first line runs).
#
# Sizing matches the grid arrays reworked on 2026-08-31: the short-simple nodes carry 64+
# cores and ~126 GB each (9 of them), so 8 cores per task is still tiny, and the array
# throttle is set ABOVE what the QoS grants because the QoS enforces the real cap anyway
# (surplus tasks simply wait with REASON=QOSMaxJobsPerUserLimit). The partition allows 2
# days, so the time limits carry real margin instead of risking a killed task.
#SBATCH --job-name=prep_val_plan
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --partition=short-simple
#SBATCH --qos=simple

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
# node's core count, not from the cgroup, so without this a task spawns one thread per
# core on the node and they fight over the few CPUs it was actually given, which is
# slower than running single-threaded.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd ~/ColabProject/cluster
python run_validation_task.py --prepare
