# Pre-registration: confirmatory 2D pairwise test (paper item e)

This file was written and committed **before** `run_pairwise2d_confirm_task.py` was run. The git commit that adds it is the timestamp. Nothing below may change after the run. Any extra analysis is reported as exploratory.

## Background

The thesis (Results, "More anchors and all pairs in two dimensions"; experiment E7) found the following on 630 networks of 21 candidates:

- **Pre-set criterion.** C₁₆ and P cleared the pre-set criterion (accuracy above chance ≥ 0.20) by less than its standard error (0.205 and 0.201, SE ≈ 0.02).
- **Comparison with R alone (post hoc).** C₁₆ and P beat the terminal resistance S alone, with exact McNemar p = 0.0027 and 0.0011. Their accuracy gain over S was about 0.07. This comparison was chosen after the results were seen.

This run tests that post-hoc finding on fresh data.

## Data

- **Candidates.** The same 21 candidates of `validation_plan.json`, Part A of E7 (A19 excluded).
- **Networks.** 60 fresh percolating networks per candidate, with seeds `stable_seed(N, round(a,3), 'pairwise2d_confirm')`. None of these networks was used in the thesis.
- **Features.** Computed by `run_pairwise2d_task.network_features`, unchanged.
- **Usable networks.** If a candidate gives up before 60 networks, its networks are still used if there are at least 10. If not, the candidate is excluded and the exclusion is reported.

## Hypotheses

- **H1.** Adding C₁₆ to S raises the share of networks assigned to their own candidate.
- **H2.** Adding P to S raises it.

C₁ and C₄ are reported descriptively. They are not tested.

## Primary analysis (decides the result)

1. **Split.** For each target, make one stratified 5-fold split of its networks with seed 0. Fold k receives the networks at positions k, k+5, … of a seeded permutation within each candidate. This is exactly the split of `pairwise2d_robustness.py`.
2. **Classifier.** For each information set (S, S+C₁₆, S+P), train the LDA of `pairwise2d_analysis.lda_fit` (shrinkage γ = 0.1, standardized log10 features, empty shells filled as in `pairwise2d_analysis.fill_curve`) on four folds and predict the fifth.
3. **Pooling.** Pool the networks of all targets. For each of S+C₁₆ and S+P, count b (networks correct with the set but wrong with S) and c (the reverse).
4. **Test.** Run an exact two-sided McNemar test on (b, c).
5. **Decision.** Use Holm's correction over the two tests at family-wise α = 0.05:
   - H_i is **confirmed** if its Holm-adjusted p < 0.05 **and** b > c;
   - **refuted in direction** if the adjusted p < 0.05 and b < c;
   - **not confirmed** otherwise.

## Secondary analyses (reported in full, do not change the verdict)

- **Effect size.** Two measures:
  - the difference in accuracy above chance (mean over targets) between each set and S, under the thesis's pre-set procedure, `pairwise2d_analysis.classify(F, y, 1)` (stratified 5-fold, 10 repeats, seed 0);
  - on the primary split, the paired difference in the share of correct networks, (b − c)/n, with a 95 % interval from 2000 bootstrap resamples of networks within candidates (seed 0).
- **The thesis's pre-set criterion.** Accuracy above chance ≥ 0.20, for S+C₁, S+C₄, S+C₁₆ and S+P.
- **Per-target accuracies.**

## Reporting

- The paper will report the verdict of the primary analysis whatever it is, together with the effect sizes.
- If H1 and H2 are not confirmed, the paper's sentence on the 2D gain becomes "not confirmed on fresh networks", and the thesis's post-hoc result stays labeled as post hoc.
