#!/bin/bash
# One-task pre-flight for the grid arrays: proves, from an actual compute node,
# that the environment the arrays depend on is really there.
#
# Exists because `srun --pty`-style interactive steps fail on Apuana with
# "Task launch ... Communication connection failure", so the only way to see a
# compute node's view of the filesystem is to submit a batch job and read its log.
#
# Checks, in order: the venv activates (it lives in /opt/miniforge3-derived paths,
# which may or may not be shared with the compute nodes), numpy/scipy/matplotlib
# import, utils/ is importable, and the merge-geometry fix is actually present in
# the code the job would run.
#
#   sbatch smoke_test.sbatch && sleep 30 && cat logs/smoke_test_*.out
#
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --partition=short-simple
#SBATCH --qos=simple

set -eo pipefail
set -x

source ~/envs/piano_sweep/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd ~/ColabProject/cluster

python - <<'PYEOF'
import sys, os, inspect
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(".")), "utils"))
sys.path.insert(0, "../utils")

print("hostname     :", os.uname().nodename)
print("python       :", sys.executable)
print("cpus         :", os.environ.get("SLURM_CPUS_PER_TASK"))

import numpy, scipy, matplotlib
print("numpy        :", numpy.__version__)
print("scipy        :", scipy.__version__)
print("matplotlib   :", matplotlib.__version__)

from pipeline_utils import distance_dependent_impedance as f
fixed = "use_merged_geometry" in inspect.signature(f).parameters
print("merge fix    :", "FIXED" if fixed else "STILL OLD CODE")

from pipeline_utils_3d import generate_percolating_network_3d
from multimeter import run_onion_protocol_pinv
print("utils imports: OK")

if not fixed:
    raise SystemExit("the compute node is running the OLD utils/: rsync it before submitting")
print("\nPRE-FLIGHT PASSED")
PYEOF
