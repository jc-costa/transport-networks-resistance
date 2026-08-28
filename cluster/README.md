# `cluster/`: large sweeps on SLURM

The large parameter grids in this project are too expensive to run in a notebook. The
impedance solve is a dense `O(n_nodes³)` operation and the node count grows sharply with
`N`, with `a`, and (in 3D) with *shrinking* film thickness `d`. These jobs ran on the
**Apuana** cluster at CIn/UFPE.

**Their output is committed.** The `*_results/` directories hold the raw JSON chunks each
job produced, so the analysis notebooks reload them instead of recomputing. Nothing here
has to be re-run to reproduce a figure.

---

## How the jobs are structured

Every sweep follows the same pattern:

1. A `run_*_task.py` script takes an array-task index, computes only its own slice of the
   grid, and writes `<results_dir>/chunk_NN.json`.
2. A `run_*_array.sbatch` file submits the whole set as a SLURM array job.
3. The corresponding notebook checks for a complete set of chunk files. If they are all
   present it loads them; otherwise it computes serially in-notebook, checkpointing after
   every combination so an interrupt does not lose the sweep.

Because the grid slicing and the seed derivation are identical across scripts, results are
independent of how the work was split across tasks.

```bash
cd cluster
mkdir -p logs
sbatch run_onion_grid_array.sbatch
```

**On the QOS setting.** The jobs specify `--qos=simple` with `--array=0-19%4`, running
four tasks concurrently. The `complex` QOS was tried and is worse here: it capped
concurrency at one.

---

## The jobs

| Job | Task script | Output | Feeds |
|---|---|---|---|
| `run_effective_resistance_d_array.sbatch` | `run_effective_resistance_d_task.py` | `effective_resistance_d_results/` (20 chunks) | `effective_resistance_vs_d.ipynb` Part C |
| `run_onion_grid_array.sbatch` | `run_onion_grid_task.py` | `onion_grid_results/` (20 chunks) | `r_eff_ij_vs_N_a_d.ipynb` Parts B and C |
| `run_onion_grid_boost_array.sbatch` | `run_onion_grid_boost_task.py` | `onion_grid_boost_results/` (18 chunks) | `r_eff_ij_vs_N_a_d.ipynb` Part D |
| `run_validation_array.sbatch` | `run_validation_task.py` | `validation_results/` (31 files) | `inverse_N_a_from_R_eff.ipynb` section 6 |
| `run_random_networks_array.sbatch` | `run_random_networks_task.py` | (batch statistics) | `test_150_random_networks.ipynb` section 4 |

All array jobs request 2 CPUs and 4 GB on the `short-simple` partition. Walltimes range
from 20 minutes for the random-network batch to 2 hours for the boosted onion grid.

### Whole-notebook jobs

Two scripts re-execute an entire notebook end to end through `nbclient` as a single job,
rather than slicing a grid. Simpler to launch, with no chunk files to assemble, but
serial.

| Script | Runs |
|---|---|
| `run_effective_resistance_sweep.sbatch` | `effective_resistance_vs_N_a.ipynb` |
| `run_effective_resistance_d_sweep.sbatch` | `effective_resistance_vs_d.ipynb`, Parts A, B, and C |
| `run_inverse_validation.sbatch` | `inverse_N_a_from_R_eff.ipynb` |

### Plan files

`run_onion_grid_boost_task.py` and `run_validation_task.py` need their work enumerated
before submission, since the set of combinations depends on the analysis of a previous
run. `onion_grid_boost_plan.json` and `validation_plan.json` are those enumerations, and
are committed so the exact submitted workload is on record.

```bash
python run_onion_grid_boost_task.py --prepare   # regenerates the plan, run before sbatch
```

---

## Paired grids

`run_onion_grid_task.py` reproduces the grid definition, the seed formula, and the
network-draw order of `run_effective_resistance_d_task.py` exactly. The consequence is
that a given `(N, a, d)` combination yields the *same network* in both datasets, so the
onion curves and the scalar resistances are paired one to one by construction rather than
matched after the fact. The whole matched-group analysis in
[`../docs/RESULTS.md`](../docs/RESULTS.md) depends on this.

The boosted rerun extends the same property. It reuses the same `stable_seed()` values, so
its first 6 realizations per combination are byte-identical to the original 6-realization
data (verified directly, maximum absolute difference 0.0). It is a strict extension of the
same draws rather than an independent resample.

This pairing only holds for data produced **after** the seed fix described in the main
[README](../README.md). The scalar grid in `experiments/exp_829/` predates it.

---

## A note on the results directories

The `*_results/` directories hold real data synced back from the cluster, not scratch
output. Running a task script locally overwrites the chunk it owns. Check what is there
before running anything that writes into these directories.
