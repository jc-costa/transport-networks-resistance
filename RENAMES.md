# Renames: onion curve to radial profile

The dissertation and the paper call the curve of the mean pairwise resistance from a reference node, against distance, the **radial profile**. Earlier versions of this repository called it the *onion curve*. Every name and every key that carried the old word was renamed, in the files and in their contents, by one scripted rule, so that code and stored results stay consistent.

**Nothing is lost.** The state before the renaming is the git tag `dissertation-2026-11-pre-rename`, and the commit that the paper cites (`0afc0f5`) keeps the old names. To see the old layout, run `git checkout dissertation-2026-11-pre-rename`.

## Rules

| old | new |
|---|---|
| `onion curve`, `onion curves`, `onion-curve` (prose) | `radial profile`, `radial profiles`, `radial-profile` |
| `onion layer`, `onion layers`, `onion-layer` (the shells) | `radial-profile shell`, `radial-profile shells`, `radial-profile` |
| `onion protocol`, `onion grid` (prose) | `radial-profile protocol`, `radial-profile grid` |
| `--onion` (command-line flag) | `--radial-profile` |
| `onion` in a file, directory, function, variable or JSON key | `radial_profile` |
| `ONION` | `RADIAL_PROFILE` |

Stored result files keep every value. Only keys and strings that contained the word were changed, and each JSON file was checked to equal the original once the same rule was applied to its keys and strings. The random seeds never depended on the word, so regenerated networks are identical. The historical snapshot `cluster/backups/20260831_111505/` and the files `cluster/PREREGISTRATION_*.md`, which are dated records, were left as they were.

## Directories

| old | new | files |
|---|---|---|
| `cluster/onion_flat_results/` | `cluster/radial_profile_flat_results/` | 21 |
| `cluster/onion_grid_boost_results/` | `cluster/radial_profile_grid_boost_results/` | 18 |
| `cluster/onion_grid_results/` | `cluster/radial_profile_grid_results/` | 22 |
| `cluster/onion_grid_results_reject/` | `cluster/radial_profile_grid_results_reject/` | 22 |
| `cluster/onion_inplane_results/` | `cluster/radial_profile_inplane_results/` | 21 |

## Other files

| old | new |
|---|---|
| `cluster/onion_grid_boost_plan.json` | `cluster/radial_profile_grid_boost_plan.json` |
| `cluster/onion_tolerance_sweep.py` | `cluster/radial_profile_tolerance_sweep.py` |
| `cluster/run_onion_flat_parallel.sbatch` | `cluster/run_radial_profile_flat_parallel.sbatch` |
| `cluster/run_onion_grid_array.sbatch` | `cluster/run_radial_profile_grid_array.sbatch` |
| `cluster/run_onion_grid_boost_array.sbatch` | `cluster/run_radial_profile_grid_boost_array.sbatch` |
| `cluster/run_onion_grid_boost_task.py` | `cluster/run_radial_profile_grid_boost_task.py` |
| `cluster/run_onion_grid_parallel.sbatch` | `cluster/run_radial_profile_grid_parallel.sbatch` |
| `cluster/run_onion_grid_task.py` | `cluster/run_radial_profile_grid_task.py` |
| `cluster/run_onion_inplane_parallel.sbatch` | `cluster/run_radial_profile_inplane_parallel.sbatch` |
| `cluster/run_onion_inplane_task.py` | `cluster/run_radial_profile_inplane_task.py` |
| `experiments/exp_833/partA_preview_onion_curves.png` | `experiments/exp_833/partA_preview_radial_profile_curves.png` |
| `experiments/exp_834/partA_preview_onion_curves.png` | `experiments/exp_834/partA_preview_radial_profile_curves.png` |
| `experiments/exp_835/partA_preview_onion_curves.png` | `experiments/exp_835/partA_preview_radial_profile_curves.png` |
| `experiments/exp_836/partA_preview_onion_curves.png` | `experiments/exp_836/partA_preview_radial_profile_curves.png` |
| `experiments/exp_837/partA_preview_onion_curves.png` | `experiments/exp_837/partA_preview_radial_profile_curves.png` |
| `experiments/exp_838/partA_preview_onion_curves.png` | `experiments/exp_838/partA_preview_radial_profile_curves.png` |
| `experiments/exp_839/partA_preview_onion_curves.png` | `experiments/exp_839/partA_preview_radial_profile_curves.png` |
| `experiments/exp_840/partA_preview_onion_curves.png` | `experiments/exp_840/partA_preview_radial_profile_curves.png` |
| `experiments/exp_852/partA_preview_onion_curves.png` | `experiments/exp_852/partA_preview_radial_profile_curves.png` |
| `experiments/exp_858/partA_preview_onion_curves.png` | `experiments/exp_858/partA_preview_radial_profile_curves.png` |
| `experiments/exp_874/partA_preview_onion_curves.png` | `experiments/exp_874/partA_preview_radial_profile_curves.png` |
| `experiments/exp_877/partA_preview_onion_curves.png` | `experiments/exp_877/partA_preview_radial_profile_curves.png` |
| `figures/partA_preview_onion_curves_errorbars.png` | `figures/partA_preview_radial_profile_curves_errorbars.png` |
| `revision_analyses/a4_result_onion_flat_results.json` | `revision_analyses/a4_result_radial_profile_flat_results.json` |
| `revision_analyses/a4_result_onion_inplane_results.json` | `revision_analyses/a4_result_radial_profile_inplane_results.json` |
