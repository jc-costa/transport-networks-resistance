# Data manifest

Stored records of the runs reported in the paper and its supplement. Each is a JSON-lines file with one row per geometry, regenerable from the seeds (`stable_seed`) with the scripts in `cluster/`. SHA-256 checksums are of the files as analysed. The run scripts and the analyses are in `cluster/`; the summaries behind every figure and table are in `figures/data/`.

| file | content | pre-registration and analysis | size (MB) | SHA-256 |
|---|---|---|---|---|
| `junction_pairwise_confirm_results/partial.jsonl` | Junction confirmation, repeat (job 30331), 2160 networks, 36 geometries | PREREGISTRATION_JUNCTION.md; junction_pairwise_confirm_analysis.py | 29.5 | `8fb05decb2b2ac5b261ac126721f87881001df1c5b042e11e7a5a98bb3b460ac` |
| `junction_pairwise_confirm_results_v1_defective/partial.jsonl` | Junction confirmation, first run with the fault (job 30206), kept for the disclosure | Addendum 1 of PREREGISTRATION_JUNCTION.md | 29.5 | `6e5b609b3b66cd1db8d77c9abebbac4fec5fbf6f03d553da5e9735f265ef41f9` |
| `junction_pairwise_dense_results/partial.jsonl` | Dense grid (job 30359), 9900 networks, 99 geometries | PREREGISTRATION_JUNCTION_DENSE.md; junction_pairwise_dense_analysis.py | 76.9 | `375b28a3d2ae6c611faef97024db9828c1d3830635c9b35cf74e47474c6bec2b` |
| `junction_pairwise_transfer_results/partial.jsonl` | Transfer run (job 30377), 6000 networks, 150 geometries | PREREGISTRATION_JUNCTION_TRANSFER.md; junction_pairwise_transfer_analysis.py | 63.5 | `5c272392e754535a9776d2bb41fe6f8caa7806be0033bfb1bccdedae58154522` |
| `junction_strategies_results/partial.jsonl` | Strategies run (job 30616), 2400 networks, 80 geometries | PREREGISTRATION_JUNCTION_STRATEGIES.md; junction_strategies_analysis.py | 73.2 | `133ca88621d68bc8fe2155731a951819f23f05ed4b7ff2ee9c1125afb7dc4da4` |
| `junction_strategies_more_results/partial.jsonl` | Follow-up run (job 30780), more readings, chosen sites, mean and median, same 2400 networks | PREREGISTRATION_JUNCTION_STRATEGIES_MORE.md; junction_strategies_more_analysis.py | 65.2 | `a4f373dc1c9e4e8c003f71e787be6722b02eb550c85765c3d218c5091b31ae07` |
| `large_film_results/` (70 files) | Large-system sweep (job 17598), 70 points, 3254 networks | PREREGISTRATION.md (two-dimensional repeat) | 0.7 | per-file checksums in `DATA-MANIFEST-files.txt` |
| `pairwise2d_confirm_results/` (1 files) | Two-dimensional pairwise repeat (job 17599), 1260 networks | PREREGISTRATION.md (two-dimensional repeat) | 4.5 | per-file checksums in `DATA-MANIFEST-files.txt` |

Total about 343 MB. The results of the dissertation's earlier experiments are in its own repository (`utils/`, `cluster/`, `experiments/`).

## Reproducing

1. `python run_junction_strategies_task.py --validate` (and the same flag of `run_junction_pairwise_task.py`): the readings and the junction graph reproduce direct solves and the dissertation's pairwise code to 1e-12.
2. Rerun a run on the cluster with the sbatch of the same name (`run_junction_pairwise_parallel.sbatch`, `GRID=confirm|dense|transfer`; `run_junction_strategies_parallel.sbatch`). Seeds are fixed, so the networks are identical.
3. Run the pre-registered analysis of that run on its `partial.jsonl` (see the table).
4. Figures: `figures/scripts/` (they read `figures/data/` and the stored records through the `COLAB` environment variable).

Git commits that fix the pre-registrations: `e6c567f` (confirmation), `0efb27e` (its addendum), `02c12fb` (dense grid), `bd45c1f` (transfer), `4212266` (strategies), `261f961` (follow-up of the strategies run).
