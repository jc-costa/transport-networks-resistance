#!/bin/bash
# Runs effective_resistance_vs_d.ipynb end-to-end (Parts A/B/C) headlessly on
# a SLURM compute node (Apuana, CIn/UFPE), in place, via nbclient -- same
# pattern as run_effective_resistance_sweep.sbatch (the 2D N,a sweep), just
# pointed at the 3D film-thickness notebook.
#
# Part C (the extended N x a x d grid) is the part worth scaling up here --
# widen N_values_ext/a_values_ext/d_values_ext and/or raise n_trials_ext
# directly in the notebook before submitting, since it checkpoints after
# every (N, a, d) combination (extended_sweep_CHECKPOINT.json) and can
# safely be re-run or resubmitted without losing prior progress.
#
# CPU-only workload (dense numpy linear solve), no GPU used -- short-simple
# is the right partition (short-complex would reserve a GPU for nothing).
#
# IMPORTANT: SLURM resolves --output/--error BEFORE this script's first line
# runs, so the "logs/" directory below must already exist at submission
# time -- create it once with `mkdir -p logs` in this same folder before the
# first `sbatch` call, or the job fails instantly with no log written
# anywhere (nothing to grep for the error, and it vanishes from `squeue`
# right away since there's nothing to show).
#
#SBATCH --job-name=reff_d_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=short-simple
#SBATCH --qos=simple

set -eo pipefail   # no -u: sourced rc files often reference unset vars, that's not our bug to fail on
set -x             # trace every command to stderr, so a silent failure still leaves a trail

# Login node and compute nodes have DIFFERENT Lmod module registries on
# Apuana: the login node only has miniforge3 (conda); compute nodes only
# have Python/*-GCCcore-* (confirmed via an interactive `srun` session on
# cluster-node9) -- no miniforge3 there at all. So this job uses the
# compute-node-side Python module + a plain venv under $HOME (shared across
# all nodes), not conda.
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

export MPLBACKEND=Agg   # headless node: never try to open an interactive plot window

# NOT $(dirname "$0") -- SLURM copies the submitted script into a per-job
# spool dir (/tmp/slurmd/job<id>/) before running it, so $0 doesn't point at
# ~/ColabProject/cluster at runtime. Use the fixed project path instead.
cd ~/ColabProject/notebooks

python -c "
import nbformat
from nbclient import NotebookClient

nb = nbformat.read('effective_resistance_vs_d.ipynb', as_version=4)
client = NotebookClient(nb, timeout=14400, kernel_name='python3',
                         resources={'metadata': {'path': '.'}})
client.execute()
nbformat.write(nb, 'effective_resistance_vs_d.ipynb')
print('Done -- notebook re-saved with fresh outputs.')
"
