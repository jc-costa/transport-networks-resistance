#!/bin/bash
# Parallel version of inverse_N_a_from_R_eff.ipynb's Section 6 validation:
# 28 candidates (7 R_eff targets x 4 (N,a) candidates each), one SLURM array
# task per candidate, each drawing its own 30 percolating networks
# independently -- they don't depend on each other, so this is the
# "embarrassingly parallel" case the earlier sequential single-job version
# didn't take advantage of.
#
# --array=0-27%10: 28 tasks total, but AT MOST 10 running at the same time,
# even if the cluster is otherwise idle -- a deliberate courtesy cap so this
# doesn't grab a big slice of shared capacity all at once. Each task is
# small (CPU/RAM-wise), so 10 concurrent is a light footprint on a cluster
# with ~400 compute-node cores.
#
# REQUIRES prepare_validation_plan.sbatch to have already completed
# successfully (validation_plan.json must exist) -- every task reads that
# same frozen plan by index.
#
# Sizing matches the grid arrays reworked on 2026-08-31: the short-simple nodes carry 64+
# cores and ~126 GB each (9 of them), so 8 cores per task is still tiny, and the array
# throttle is set ABOVE what the QoS grants because the QoS enforces the real cap anyway
# (surplus tasks simply wait with REASON=QOSMaxJobsPerUserLimit). The partition allows 2
# days, so the time limits carry real margin instead of risking a killed task.
#SBATCH --job-name=inverse_val_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-27%28

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
python run_validation_task.py "$SLURM_ARRAY_TASK_ID"
