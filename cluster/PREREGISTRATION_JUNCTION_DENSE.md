# Pre-registration: dense grid for the junction ratio, the stick length and the density (paper item A, part 2)

This file is written and committed **before** `run_junction_pairwise_task.py --grid dense` is run. The git commit that adds it is the timestamp. Nothing below may change after the run. Any extra analysis is reported as exploratory.

## Background (exploratory, seen before this file was written)

The confirmation run (`PREREGISTRATION_JUNCTION.md`, with its Addendum 1) used 36 cells with six values of a, six of N a^2 and uniform, log-normal and bimodal junctions. Its registered tests passed. An exploratory regression on those data (quadratic ridge regression of log10 rho, log10 a and log10 N a^2 on log10 S with the nine values of P, 5 folds) gave, with whole cells held out of training, R^2 = 0.91 for log10 rho, 0.25 for log10 a and 0.68 for log10 N a^2, against 0.46, 0 and 0.19 for log10 S alone (`figures/data/junction_regime_exploratory_v2.json`). The same code was run on the confirmation data, after being written, to test it; its output is the exploratory result above, and **the margins below were set after seeing it**, below the exploratory values. The results on the confirmation data are limited by two things this run removes: only six distinct values of a and of N a^2, which makes the R^2 of those two a measure of interpolation between few levels, and no junction ratio below 0.3.

## Data

- **Cells.** a in {0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00} and N a^2 in {5, 5.75, 6.5, 7.25, 8, 9, 10, 12, 14, 16} (N = round(N a^2 / a^2); cells that round to the same N are one cell, which gives 99 cells).
- **Networks.** 100 fresh percolating networks per cell, seeds `stable_seed(N, round(a, 3), 'junction_pairwise_dense')`. No network of an earlier run is reused. Networks above 6500 nodes are skipped and counted.
- **Conditions per network (the same network in all of them).** Uniform junctions, rho in {0, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}.
- **Features.** S (strips shorted), the onion curves C1, C4, C16 and the all-pairs profile P, from the largest connected component of the junction graph, as in `run_junction_pairwise_task.py` at commit `0efb27e` and after (grid code only). The script must pass `--validate` before the run.

## Analysis (decides the result)

`junction_pairwise_dense_analysis.py`, unchanged.

**Part 1, regression with whole cells held out.**
1. Targets: log10 rho (samples with rho > 0), log10 a and log10 N a^2 (all samples).
2. Feature sets: log10 S alone, and log10 S together with the nine values of C1, of C4, of C16 or of P.
3. Quadratic ridge regression (`junction_pairwise_regime_analysis.cv_predict`, penalty 1e-3), 5 folds, each fold a group of whole cells chosen by a seeded permutation (seed 0).
4. Score: R^2 = 1 − MSE / variance of the target, over the held-out samples.
5. Intervals: 2000 bootstrap resamples of the networks within each cell, applied to the out-of-fold predictions (95 % percentile interval). As in the confirmation run, they are conditional on the cell split and the predictions, and do not reflect the sensitivity to that design.

**Part 2, Cramér–Rao bound.** The mean of the log features is a cubic polynomial in (log10 a, log10 N a^2, log10 rho) fitted to the group means (cell, rho > 0); the covariance is the pooled within-group covariance of the networks plus an isotropic measurement noise of relative size 0, 5 % or 10 %. The bound is the inverse of F = J' Sigma^-1 J at each group, reported as a standard deviation in log10 units, as the median over groups. A feature set with singular F has an infinite bound.

## Hypotheses

- **D1.** S + P recovers rho: the lower bound of the 95 % interval of R^2 for log10 rho is above **0.80**.
- **D2.** S + P recovers part of a: the lower bound of the interval of R^2(S + P) − R^2(S) for log10 a is above **0.15**.
- **D3.** S + P recovers part of N a^2: the lower bound of the interval of R^2(S + P) − R^2(S) for log10 N a^2 is above **0.40**.
- **D4.** Four anchors are almost as good as all pairs for rho: the lower bound of the interval of R^2(S + C4) − R^2(S + P) for log10 rho is above **−0.05**.

**Descriptive, not tested.** C1 and C16 against P; the predicted log10 rho by true rho (calibration), which shows the lowest rho the data tell from zero; the RMSE of log10 rho by regime of a and N a^2; the Cramér–Rao bounds by feature set and measurement noise; and how well the polynomial mean model fits (its R^2 is reported, and a bound is not claimed where it fits poorly).

## What would change the paper's claim

If D1 fails, the statement that the pairwise resistances recover the junction ratio is dropped or limited to the regimes where it holds. If D2 or D3 fails, the paper says that the stick length or the density is not recovered beyond the terminal resistance. D4 decides whether the paper recommends four anchors or all pairs. The Cramér–Rao bounds are reported whatever they show.
