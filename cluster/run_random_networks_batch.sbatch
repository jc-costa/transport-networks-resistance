#!/bin/bash
# Runs test_150_random_networks.ipynb (now scaled to 1500 cases) end-to-end
# headlessly on a SLURM compute node (Apuana, CIn/UFPE), in place, via
# nbclient. Same working recipe as the other .sbatch files in this folder
# (see run_effective_resistance_sweep.sbatch's history for why): compute
# nodes only have Python/*-GCCcore-* modules (no miniforge3), so this uses
# the module-provided Python + the venv at ~/envs/piano_sweep, not conda.
#
# CPU-only workload (dense numpy linear solve per percolating case), no GPU
# used -- short-simple is the right partition. Per-case cost is much lighter
# than the R_eff sweep (N capped at 90, a capped at 0.80 here, vs up to 150/1.1
# there), so this is a single sequential job, not a job array. Estimated
# ~15-25min typical (only ~38% of cases percolate and reach the actual
# impedance solve, same rate as the original 150-case run; the rest exit
# early and are cheap), up to ~1-1.5h in an unlucky worst case if many
# solved cases cluster near N=90/a=0.80 -- --time below has margin for that.
#
# IMPORTANT: SLURM resolves --output/--error BEFORE this script's first line
# runs, so the "logs/" directory below must already exist at submission
# time -- `mkdir -p logs` in this same folder before the first `sbatch` call.
#
#SBATCH --job-name=random_nets
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

export MPLBACKEND=Agg   # headless node: never try to open an interactive plot window

cd ~/ColabProject/notebooks

python -c "
import nbformat
from nbclient import NotebookClient

nb = nbformat.read('test_150_random_networks.ipynb', as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name='python3',
                         resources={'metadata': {'path': '.'}})
client.execute()
nbformat.write(nb, 'test_150_random_networks.ipynb')
print('Done -- notebook re-saved with fresh outputs.')
"
