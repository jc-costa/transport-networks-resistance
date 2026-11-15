# Pre-registration: transfer of the junction estimator to new films (paper A, run E1)

This file is written and committed **before** `run_junction_pairwise_task.py --grid transfer` is run. The git commit that adds it is the timestamp. Nothing below may change after the run. Any extra analysis is reported as exploratory.

## Why this run

The dense-grid regression (`PREREGISTRATION_JUNCTION_DENSE.md`) measured the junction ratio at nine fixed levels, in geometries on a grid, with whole geometries held out of training. A pre-submission review points out that a held-out geometry still meets junction ratios that appeared in training, and that the error intervals say nothing about new ratio values, a wrong wire resistance, or other junction rules. This run tests the frozen estimator on films that share none of those with the training set.

## Background (exploratory, seen before this file was written)

On the dense-grid records, with a whole level of the junction ratio also held out, R² of log10 rho was 0.929 with root-mean-square error 0.306 (median error +0.44 decades at rho = 0.03 and −0.48 at rho = 100). With a scale error of 30 % in the wire resistance, common to the terminal resistance and the profile, the RMSE was 0.299 (R² 0.932). The mean level of the profile with the terminal resistance gave R² 0.931 against 0.962 for the full profile (`figures/data/junction_review_checks.json`). **The margins below were set after seeing these values, and above them**, because new geometries and continuous ratios should be at least as hard.

## The estimators are frozen

`freeze_junction_estimator.py` trained three estimators of log10 rho on the dense-grid records and `figures/data/junction_estimator_frozen.json` holds their coefficients: **S** (log10 of the terminal resistance), **S+level** (log10 S and the mean of the nine log10 shell values of the all-pairs profile P) and **S+P** (log10 S and the nine values). Each is a quadratic ridge regression (penalty 0.001) on standardized features. The training set is 9652 networks in 99 geometries with a complete all-pairs profile, uniform junctions, rho > 0 (`junction_pairwise_dense_results/partial.jsonl`, SHA-256 `375b28a3d2ae6c611faef97024db9828c1d3830635c9b35cf74e47474c6bec2b`). The half-width of the 90 % interval of each estimator is the 90th percentile of its absolute out-of-fold error with whole geometries held out: S 1.149, S+level 0.481, S+P 0.378 (log10 units). The coefficients are not refitted on the new data.

## Data

- **Geometries.** 150 new geometries, none on the dense grid: the stick length a is log-uniform in [0.2, 1], the density Na² is uniform in [5, 16], N = round(Na²/a²); seeds from `stable_seed(N, round(a, 3), 'junction_pairwise_transfer')`. 40 new percolating networks per geometry, 6000 in all. Networks above 6500 nodes are skipped and counted.
- **Junction ratios.** For each network, four values of rho drawn log-uniformly and continuously in [0.03, 100], by a generator seeded from the network, and **the same four values for each of three rules**: equal junctions; log-normal junctions (log-standard deviation 1.5, arithmetic mean rho R₀ l); the bimodal rule (a stick is metallic with probability 1/3, a mixed junction is 100 times the base value). Twelve conditions per network.
- **Features.** The terminal resistance S (strips shorted) and the all-pairs profile P (nine shells), computed in the largest connected component, exactly as in `run_junction_pairwise_task.py` at the commit of this file. The script must pass `--validate` first.

## Analysis (decides the result)

`junction_pairwise_transfer_analysis.py`, unchanged. A network enters if its all-pairs profile has all nine shells filled; the numbers that enter and do not, by rule and by stick length, are reported. A pairwise value that is not positive and finite stops the analysis. Intervals: 2000 bootstrap resamples of the networks within each geometry, applied to the fixed predictions, so they do not include the choice of the training set.

The **primary population** is the rows with equal junctions and rho in [0.03, 100]; the **physical range** is rho in [1, 30], which contains the silver-nanowire values of the paper after Section IV-A is checked against Coleman et al.

## Hypotheses

- **T1.** On the primary population, the RMSE of log10 rho for S+P has an upper bound of its 95 % interval at most **0.35**.
- **T2.** On the primary population, the profile adds to the mean level: the lower bound of the interval of RMSE(S+level) − RMSE(S+P) exceeds **0.03**.
- **T3.** The 90 % interval of S+P has a coverage on the primary population between **0.80 and 0.95**.
- **T4.** As T1, with rho restricted to the physical range [1, 30].
- **T5.** With a scale error, one random factor per network with log-standard deviation **0.30** applied to S and every shell value of the test films only, the RMSE of S+P on the primary population has an upper bound of its interval at most **0.40**.

**Descriptive, not tested.** RMSE, median and 90th-percentile factor error, bias, interval coverage and the slope of the prediction on the true log10 rho, for S, S+level and S+P, in the primary population and for each variable rule against the **base value** rho, so the bias of an estimator trained on equal junctions on log-normal and bimodal junctions is shown (the arithmetic mean of the bimodal junction is 45 times the base value, the geometric 7.7 times); the same by stick length and density; the flow of attempted, percolating and usable networks by stick length.

## What would change the paper's claim

If T1 or T4 fails, the paper says the estimator does not transfer to new films and gives the error that was found, and drops the claim of recovery in that range. If T2 fails, the paper says the shape of the profile adds nothing measurable to its mean level for new films. If T3 fails, it reports the miscalibration. If T5 fails, it states how accurately the wire resistance must be known. The variable-rule biases are reported whatever they show.
