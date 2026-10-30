#!/bin/bash
# Electrical backbone (red bonds, cut vertices, dead fraction) on the networks
# of the near-threshold sweep, one SLURM array task per (a, N a^2) point (33
# points; see run_backbone_task.py --plan). Regenerates the critical-sweep
# networks from their seeds and checks each R_eff against critical_sweep_results/.
#
# Sizing: same as run_critical_sweep_array.sbatch. The extra work per network is
# one Cholesky inverse of the grounded Laplacian (well under a minute at 5000
# nodes, ~0.3 s at 2700) and two n x n matrices, so the regeneration of the
# networks, not the measurement, sets the walltime. 16 GB and 6 h per task is
# generous for both regimes.
#
# mkdir -p logs before the first sbatch (SLURM resolves --output/--error before
# the script runs).
#SBATCH --job-name=backbone_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --array=0-32%33

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
python run_backbone_task.py "$SLURM_ARRAY_TASK_ID"
