#!/bin/bash
# Runs inverse_N_a_from_R_eff.ipynb end-to-end headlessly on a SLURM compute
# node (Apuana, CIn/UFPE), in place, via nbclient. Same working recipe as
# run_effective_resistance_sweep.sbatch (see that file's history for why):
# compute nodes only have Python/*-GCCcore-* modules (no miniforge3), so this
# uses the module-provided Python + the venv at ~/envs/piano_sweep, not conda.
#
# CPU-only workload (dense numpy linear solve), no GPU used -- short-simple
# is the right partition (short-complex would reserve a GPU for nothing).
#
# IMPORTANT: SLURM resolves --output/--error BEFORE this script's first line
# runs, so the "logs/" directory below must already exist at submission
# time -- `mkdir -p logs` in this same folder before the first `sbatch` call,
# or the job fails instantly with no log written anywhere.
#
#SBATCH --job-name=inverse_val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=short-simple
#SBATCH --qos=simple

set -eo pipefail   # no -u: sourced rc files often reference unset vars, that's not our bug to fail on
set -x             # trace every command to stderr, so a silent failure still leaves a trail

# Login node and compute nodes have DIFFERENT Lmod module registries on
# Apuana: the login node only has miniforge3 (conda); compute nodes only
# have Python/*-GCCcore-* -- no miniforge3 there at all. So this uses the
# compute-node-side Python module + the plain venv under $HOME (shared
# across all nodes, already set up for run_effective_resistance_sweep.sbatch).
module load Python/3.10.8-GCCcore-12.2.0
source ~/envs/piano_sweep/bin/activate

export MPLBACKEND=Agg   # headless node: never try to open an interactive plot window

# NOT $(dirname "$0") -- SLURM copies the submitted script into a per-job
# spool dir (/tmp/slurmd/job<id>/) before running it, so $0 doesn't point at
# ~/ColabProject/cluster at runtime. Use the fixed project path instead.
cd ~/ColabProject/notebooks

python -c "
import nbformat
from nbclient import NotebookClient

nb = nbformat.read('inverse_N_a_from_R_eff.ipynb', as_version=4)
client = NotebookClient(nb, timeout=7200, kernel_name='python3',
                         resources={'metadata': {'path': '.'}})
client.execute()
nbformat.write(nb, 'inverse_N_a_from_R_eff.ipynb')
print('Done -- notebook re-saved with fresh outputs.')
"
