# Pre-registration: junction resistance and the pairwise resistances (paper item A)

This file is written and committed **before** `run_junction_pairwise_task.py --grid confirm` is run. The git commit that adds it is the timestamp. Nothing below may change after the run. Any extra analysis is reported as exploratory.

## Background (exploratory, not confirmatory)

A local pilot (33 cells, 30 networks per cell, tag `junction_pairwise`, uniform junctions, rho = 0, 0.3, 1, 3, 10, 30) was analysed after seeing the data, with a classifier that did not keep the copies of one network in the same fold. Within narrow bins of the terminal resistance S, adding the all-pairs profile P to S identified the junction ratio rho at +0.37 above the majority rate (S alone: −0.10) and did not identify the stick length a (+0.04). Independent log-normal noise of 10 % on every measured value left the rho result at +0.35. This run tests those findings on fresh networks and adds the variable-junction rules, which the pilot did not cover. The margins below were set after the pilot and are deliberately well inside its effects.

## Data

- **Cells.** a in {0.2, 0.3, 0.4, 0.6, 0.8, 1.0} and N a^2 in {5, 6.5, 8, 10, 12, 16} (N = round(N a^2 / a^2)); cells whose networks have an empty distance shell are dropped by the analysis, as in the pilot.
- **Networks.** 60 fresh percolating networks per cell, seeds `stable_seed(N, round(a, 3), 'junction_pairwise_confirm')`. No network of the pilot or of the thesis is reused. Networks above 6500 nodes are skipped and counted.
- **Conditions per network (the same network in all of them).**
  - uniform junctions, rho in {0, 0.3, 1, 3, 10, 30};
  - lognormal junctions, log-sd 1.5, arithmetic mean rho R0 a, rho in {0.3, 1, 3, 10, 30};
  - bimodal junctions (a stick is metallic with probability 1/3; mixed junction = 100 times the same-kind one), rho in {0.3, 1, 3, 10, 30}.
- **Features.** S (strips shorted), the onion curves C1, C4, C16 and the all-pairs profile P, as in `run_junction_pairwise_task.py`, validated by `--validate` (rho = 0 must reproduce the thesis, S to 1e-9 and P to 1e-6).

## Analysis (decides the result)

`junction_pairwise_confirm_analysis.py`, unchanged.

1. Samples are (cell, network, condition). Features are log10 of S, and of S together with the nine values of P.
2. Samples are grouped into bins of 0.10 decades of log10 S. In a bin, classes with at least 8 samples are kept, and the bin is used if two classes remain. The classes are the values of rho (task rho) or of a (task a).
3. A shrunk-covariance linear discriminant (shrinkage 0.1) is cross-validated in 5 folds. All conditions of one network fall in the same fold, folds are assigned to networks by a seeded permutation (seed 0).
4. The score of a feature set is the accuracy minus the majority-class rate, averaged over bins weighted by the number of samples.
5. Confidence intervals: 2000 bootstrap resamples of the networks within each cell (95 % percentile interval).

## Hypotheses

- **H1 (uniform junctions, task rho).** S + P identifies rho better than S alone. Confirmed if the lower bound of the 95 % interval of (score of S + P) − (score of S) exceeds **0.20**.
- **H2 (uniform junctions, task a).** S + P does not identify a. Confirmed if the upper bound of the interval of the score of S + P, task a, is below **0.10**.
- **H3 (variable junctions, task rho within each rule).** For lognormal and for bimodal junctions separately, with the classes the mean ratio rho, the lower bound of the interval of (S + P) − S exceeds **0.10**. Each rule is reported on its own; a rule that fails is reported as not confirmed.
- **H4 (descriptive, not tested).** The difference (S + P) − S for rho with 10 % independent noise on S and on every shell value, mean of 5 draws.

C1, C4 and C16 are reported descriptively and not tested.

## What would change the paper's claim

If H1 fails, the junction result is dropped from the paper. If H2 fails, the claim that a is not identified is dropped. If H3 fails for a rule, the paper states the result for the uniform rule only and says the rule did not confirm.

---

## Addendum 1: a defect in the first run, and the rerun

Written after the first run and after its analysis were seen, and committed before the rerun.

**What happened.** The first run (commit `e6c567f`, job results later moved to `junction_pairwise_confirm_results_v1_defective`) was analysed as registered and gave H1 +0.236 [+0.221, +0.252], H2 +0.010, H3 +0.227 (lognormal) and +0.238 (bimodal), all meeting the criteria. While computing an exploratory regime analysis, 185 values of P turned out negative or of order 10^12 to 10^16, so their logarithm was NaN.

**Cause.** The pairwise values come from the junction graph without the terminals merged. In some networks that graph falls into two or more connected components, because strip nodes are joined to the rest only through the electrode. Its Laplacian is then singular and the inverse returns noise. The thesis's E7 avoids this by keeping the nodes reachable from the anchor; the new script did not. The validation (`--validate`) used connected networks and could not show it. The analysis passed the NaN features to the discriminant without a check, and one NaN in a training fold sets the mean of its class to NaN.

**Extent.** In the uniform-junction data, 6 of 1377 networks have a negative pairwise value. Disconnected networks whose garbage values are positive are not detected by this check, so the number is a lower bound.

**What was seen before the rerun.** Removing those 6 networks and rerunning the registered analysis gives an H1 difference of +0.577 instead of +0.236. This was seen before the rerun is started. H1 meets its criterion either way, and no decision below depends on it.

**Fix.** `run_junction_pairwise_task.py` now computes the pairwise values inside the largest connected component of the junction graph (anchors, shells and pairs are taken from it, the terminal resistance S is unchanged), and records the size of that component. `junction_pairwise_analysis.py` now stops with an error if any pairwise value is non-positive or not finite. Checks: with rho = 0 the script still reproduces the thesis's E7 (S to 8e-13, P to 3e-12); on a connected network it reproduces the first run to 1e-13; on a network that was defective the values are now positive.

**Rerun.** Same grid, same tag `junction_pairwise_confirm`, same seeds and number of networks, so the geometry of every network is identical and only the pairwise features of the disconnected networks, and the anchors drawn for them, differ. The hypotheses, margins, bins, folds, shrinkage and bootstrap are unchanged, and `junction_pairwise_confirm_analysis.py` is unchanged. The rerun is the confirmatory result. The first run's outcome is reported as it was, with this defect, in the supplement.
