# `paper/`: material of the junction-resistance paper

Code, pre-registrations, summaries and manifest for the paper *Estimating Junction
Resistance Ratios in Two-Dimensional Random Stick Films Using Simulated Probe
Measurements*. The paper builds on the dissertation pipeline in this repository (`utils/`,
`cluster/`) and adds the runs listed below, which are not in the dissertation.

## Where things are

| What | Where |
|---|---|
| Run scripts and SLURM files of the new runs | `cluster/run_junction_pairwise_*`, `cluster/run_junction_strategies_*` |
| Pre-registrations | `cluster/PREREGISTRATION*.md` |
| Registered analyses | `cluster/junction_pairwise_*_analysis.py`, `cluster/junction_strategies_analysis.py`, `cluster/junction_strategies_more_analysis.py` |
| Post hoc analyses (labeled as such in the paper) | `cluster/junction_pairwise_review_checks.py`, `junction_pairwise_prior_scenario.py`, `junction_pairwise_transfer_estimands.py`, `junction_strategies_decomposition.py`, `junction_strategies_narrowing.py`, `junction_strategies_contact.py`, `junction_strategies_contact_robust.py`, `junction_strategies_signal.py`, `junction_strategies_more_medians.py` |
| Frozen estimator of the transfer run | `cluster/freeze_junction_estimator.py`, `paper/figures/data/junction_estimator_frozen.json` |
| Summaries behind every figure and table | `paper/figures/data/` |
| Figure scripts (Figs. 1 to 4, supplementary figures, TOC graphic) | `paper/figures/scripts/` |
| Stored records of the runs, with sizes and SHA-256 checksums | `paper/DATA-MANIFEST.md`, `paper/DATA-MANIFEST-files.txt` |

## The runs

| Run | Script (grid) | Pre-registration | Analysis |
|---|---|---|---|
| Junction confirmation, 2160 networks at 36 geometries | `run_junction_pairwise_task.py --grid confirm` | `PREREGISTRATION_JUNCTION.md`, with Addendum 1 | `junction_pairwise_confirm_analysis.py` |
| Dense grid, 9900 networks at 99 geometries | `--grid dense` | `PREREGISTRATION_JUNCTION_DENSE.md` | `junction_pairwise_dense_analysis.py` |
| Transfer, 6000 networks at 150 new geometries | `--grid transfer` | `PREREGISTRATION_JUNCTION_TRANSFER.md` | `junction_pairwise_transfer_analysis.py` |
| Measurement strategies, 2400 networks at 80 new geometries | `run_junction_strategies_task.py` | `PREREGISTRATION_JUNCTION_STRATEGIES.md` | `junction_strategies_analysis.py` |
| Follow-up on the same films (more readings, chosen sites, the median) | `run_junction_strategies_more_task.py` | `PREREGISTRATION_JUNCTION_STRATEGIES_MORE.md` | `junction_strategies_more_analysis.py` |
| Large-system sweep and two-dimensional pairwise repeat | `run_large_film_task.py`, `run_pairwise2d_confirm_task.py` | `PREREGISTRATION.md` | `analyze_large_film.py`, `pairwise2d_confirm_analysis.py` |

Every network is seeded from its parameters (`stable_seed`), so each run can be rebuilt
from the scripts alone.

## Timestamps of the pre-registrations

Each pre-registration was committed before its run in the repository where the paper was
written, which is not public. The files here are copies; the dates of those commits are:

| File | Commit | Date |
|---|---|---|
| `PREREGISTRATION_JUNCTION.md` | `e6c567f` | 2026-11-12 |
| Addendum 1 of the same file (the fault of the first run and its fix) | `0efb27e` | 2026-11-13 |
| `PREREGISTRATION_JUNCTION_DENSE.md` | `02c12fb` | 2026-11-14 |
| `PREREGISTRATION_JUNCTION_TRANSFER.md` | `bd45c1f` | 2026-11-14 |
| `PREREGISTRATION_JUNCTION_STRATEGIES.md` | `4212266` | 2026-11-14 |
| `PREREGISTRATION_JUNCTION_STRATEGIES_MORE.md` | `261f961` | 2026-11-15 |

The SLURM files are as run, except that the e-mail notification lines were removed.

## Stored records

The records of the runs (JSON lines, one row per geometry, about 343 MB in all) are not in
this repository. They are in the data deposit named in the paper, and
`paper/DATA-MANIFEST.md` gives the checksum of each file. To rerun an analysis or a figure
script that reads them, place each `*_results/` directory of the deposit under `cluster/`.

## Reproducing a figure

```bash
cd paper/figures/scripts
python fig1_model.py        # writes paper/figures/fig1_model.pdf
python fig4_strategies.py   # needs cluster/junction_strategies_results/partial.jsonl
```

The scripts read the pipeline from the root of this repository; set the `COLAB`
environment variable to point elsewhere.
