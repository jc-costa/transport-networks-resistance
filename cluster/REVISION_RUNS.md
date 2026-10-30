# Apuana runs for the thesis revision (1 Oct 2026)

New files only; nothing existing was edited. Sync this folder to `~/ColabProject` on Apuana, then:

```bash
cd ~/ColabProject/cluster && mkdir -p logs
sbatch run_redbond_scaling_array.sbatch     # E2  15 tasks, <= 6 h each (box 20 is the slow one)
sbatch run_threshold_sweep_array.sbatch     # E1  50 tasks, minutes each (box 20: ~30 min)
sbatch run_onion_inplane_parallel.sbatch    # E4a one 48-core job (same cost as the onion grid)
sbatch run_onion_flat_parallel.sbatch       # E4b one 48-core job (same order as the onion grid)
```

The two `complex`-QoS jobs can't run at the same time (one 48-core job per user): submit E4a first,
and E4b with `--dependency=afterany:<E4a job id>`. E1 and E2 run on the `simple` QoS alongside.

| Job | What it answers (revision plan) | Writes | Analysis after syncing back |
|---|---|---|---|
| E2 `run_redbond_scaling_task.py` | H3d: red bonds at N a^2 = 5.4/5.64/5.9 in boxes 5-20; does the count grow as L^{3/4}? Also the A7 bound on every network | `redbond_results/` | `python redbond_scaling_analysis.py` |
| E1 `run_threshold_sweep_task.py` | M5: half-percolation density per box at steps of 0.2, with intervals | `threshold_results/` | `python threshold_sweep_analysis.py` |
| E4a `run_onion_inplane_task.py` | H2: does the d signal survive shells measured in the plane? (checks every network against onion_grid_results first) | `onion_inplane_results/` | `python onion_tolerance_sweep.py --onion onion_inplane_results`; `python curve_inversion.py --results onion_inplane_results`; `python ../revision_analyses/a4_permutation.py --onion onion_inplane_results` |
| E4b `run_flat_sticks_task.py` | H2: does the curve recover d when sticks lie flat and are not cut? d = 0.02 is the thin limit | `onion_flat_results/` | as E4a with `onion_flat_results` (use it as `--scalar` too) |

Already run locally (no cluster needed): `run_random_networks_fixed_task.py` (robustness batch with the
crossing loop over all sticks) -> `random_networks_results_fixed/`; A1, A2, A4, A6, A7 (mesh) in
`../revision_analyses/`.

Quick local checks before submitting (each takes seconds):
```bash
python run_redbond_scaling_task.py --plan
REDBOND_OUT=/tmp/rb python run_redbond_scaling_task.py 1 --max-trials 3
THRESHOLD_OUT=/tmp/th python run_threshold_sweep_task.py 4 --draws 40
INPLANE_OUT=/tmp/ip python run_onion_inplane_task.py 5 --limit 1 --trials 2
FLAT_OUT=/tmp/fl python run_flat_sticks_task.py 3 --limit 1 --trials 2
```

## E7 (2 Oct 2026): pairwise resistances in 2D

New files: `run_pairwise2d_task.py`, `run_pairwise2d_parallel.sbatch`, `pairwise2d_plan.json`
(built locally with `python run_pairwise2d_task.py --prepare`; it carries the stored validation
R_eff values and the 101 cell list, so the cluster needs nothing else), `pairwise2d_analysis.py`.

```bash
cd ~/ColabProject/cluster && mkdir -p logs && sbatch run_pairwise2d_parallel.sbatch
```
Quick local check (seconds): `python run_pairwise2d_task.py --units A21,B_N20_a0.5 --trials 2 --out /tmp/pw`

Part A stops with an error if any regenerated validation network does not reproduce its stored
terminal R_eff to 1e-9. Analysis after syncing `pairwise2d_results/` back:
`python pairwise2d_analysis.py` (criteria and every analysis choice are in its docstring, fixed
before the run).
