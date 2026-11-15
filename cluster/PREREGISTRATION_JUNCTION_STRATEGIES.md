# Pre-registration: measurement strategies on two-dimensional stick films (paper A, run E2)

This file is written and committed **before** `run_junction_strategies_task.py` is run on the cluster. The git commit that adds it is the timestamp. Nothing below may change after the run. Any extra analysis is reported as exploratory.

## The question (the advisor's)

A thin film of sticks has a wire resistance per length R₀, a stick length (a) and a number of sticks (N) that are known only roughly, within a margin; the junction resistance is completely unknown. **What do different resistance measurements teach: how well do they estimate the junction resistance, and how far do they narrow the margins of the parameters that were known roughly?** The strategies compared are: the terminal resistance alone; a fixed electrode with **one** movable tip; **two** movable tips; a **four-point probe**. The scope is two-dimensional sheets with equal junctions. The study is a theoretical investigation by computational experiments of measurement protocols; it makes no comparison with experimental data.

## Background (exploratory, seen before this file was written)

On the dense-grid records, with R₀, a and N each known within ±10 % and the junction ratio ρ unknown (log-uniform from 0.03 to 100), the RMSE of log₁₀ ρ with whole geometries held out was 0.47 from the terminal resistance alone, 0.25 adding the mean level of the all-pairs profile, and 0.18 with all pairs. By regime of the true ρ, the terminal resistance gave 0.76 for ρ < 0.1 and 0.36 for ρ > 10, against 0.31 and 0.12 with all pairs (`cluster/junction_pairwise_prior_scenario.py`). **The margins below were set after seeing these values.** The code of this run was tested on a small local sample (14 geometries × 3 networks); its numbers were not used for any choice.

## Data

- **Geometries.** 80 geometries off any grid: a log-uniform in [0.2, 1], N a² uniform in [5, 16], N = round(N a²/a²); seed 1 of `units()`; film seeds from `stable_seed(N, round(a, 3), 'junction_strategies')`. 30 new percolating networks per geometry (2400); networks above 6500 nodes are skipped and counted.
- **Films.** For each network, four junction ratios ρ drawn log-uniformly and continuously in [0.03, 100], equal junctions: 9600 films.
- **Prior errors.** For each film, three independent errors, stored as u_c, u_a, u_N uniform in [−1, 1]; a relative error is margin × u. The error c of R₀ multiplies every resistance of the film; the nominal a and N are the true values times (1 + margin × u). The main margin is **10 %**.
- **Readings** (`run_junction_strategies_task.py`, validated against direct solves of the circuit to 10⁻¹², and its `--validate` must pass before the run), on the graph with the two strips tied to their electrodes:
  - *one*: the resistance between the source strip and a tip, tips at positions across the gap;
  - *two*: the resistance between two tips at set separations (0.1 to 0.9, or 0.2, 0.5, 0.8);
  - *four*: current in at the outer tips of four collinear, equally spaced tips and the voltage between the inner tips (spacings 0.03 to 0.27, or 0.06, 0.15, 0.24); the contact resistance does not enter.
- **Tip landing.** A tip lands on the nearest point of a retained stick within a footprint f of its target, after a Gaussian position error σ_pos, and is attached to the nearer of the two crossings that bound that segment inside the film region; otherwise it **misses** and the attempt counts. Landing configurations: base (f = 0.02, σ_pos = 0.01), f = 0.01, f = 0.04, and σ_pos = 0.03.
- **Designs.** m positions × r readings: 3×1, 3×3, 9×1, 9×3, 9×9, 9×27, that is 3, 9, 9, 27, 81 and 243 attempts.
- **Added in the analysis** to the stored noise-free readings: the tip contact resistance of κ wire resistances R₀ l per contact (one contact for *one*, two for *two*, none for *four*), main κ = **1**, swept over 0, 1, 10; multiplicative noise of **5 %** per reading (a shell value of r readings has noise 5 %/√r) and on the terminal resistance; a shell with no successful reading is imputed by linear interpolation in the logarithm of the other shells.
- **Reference.** The terminal resistance alone, and the all-pairs profile P (nine shells) as the upper reference with all information.

## Analysis (decides the result)

`junction_strategies_analysis.py`, unchanged. For each strategy and design the user has the terminal resistance (with its noise and the error of R₀), the readings, and the nominal a and N. A quadratic ridge regression (penalty 10⁻³, `junction_pairwise_review_checks.fit_predict`) estimates log₁₀ ρ and the logarithm of each of the three errors (R₀, a, N), with **whole geometries held out** (5 folds, seed 0). Scores: RMSE of log₁₀ ρ, the median and 90th-percentile factor error, the success rate of the attempts, the RMSE by regime of the true ρ (wire-limited ρ < 0.1, intermediate 0.1 to 10, junction-limited ρ > 10), and **narrowing**, the RMSE of a parameter error after the measurement divided by its prior spread (below 1 the measurement narrows the margin). Intervals: 2000 bootstrap resamples of the networks within each geometry, on the fixed predictions, so they do not include the choice of design or of tip parameters.

## Hypotheses (main case: margin 10 %, κ = 1, base landing)

- **H1** (per strategy: one, two, four). At 27 attempts (9×3) the strategy lowers the RMSE of log₁₀ ρ below that of the terminal resistance alone by more than **0.05**: the lower bound of the interval of RMSE(terminal) − RMSE(strategy) exceeds 0.05.
- **H2.** With resistive contacts (κ = 10) at 81 attempts (9×9), the four-point probe has a lower RMSE of log₁₀ ρ than two tips: the lower bound of RMSE(two) − RMSE(four) exceeds 0.
- **H3.** With the all-pairs reference, the readings narrow the margin of R₀ where the junction is small and not where it dominates: the ratio of the RMSE of the R₀ error to its prior spread has an upper bound at most **0.90** for ρ < 0.1 and a lower bound at least **0.95** for ρ > 10. H3 is confirmed only if both hold.

**Descriptive, not tested.** The full grid of strategies and designs; the smallest design that reaches an RMSE of 0.35 (the error of the transfer run) for each strategy; the success rate of the attempts; the sweeps of the landing configuration, κ (0, 1, 10) and the margin (5 %, 10 %, 20 %, 50 %), one factor at a time, at 27 and 81 attempts; the narrowing of a and N; the regime breakdown.

## Modeling limits stated in advance

Tips are point contacts attached to a crossing; no current spreading under the tip, no tip on an electrode, no probe finite size beyond the footprint; equal junctions; two dimensions. The tip parameters are assumptions swept rather than measured. The success rate of a strategy is a property of this landing model, in particular of the footprint, and is reported as such.

## What would change the paper's claim

If H1 fails for a strategy, the paper says that strategy does not improve the estimate of the junction ratio at that budget under these tip parameters. If H2 fails, the paper does not claim an advantage of the four-point probe. If H3 fails, the paper does not claim a regime dependence of the narrowing of the wire resistance. Whatever the outcome, the table of strategies, budgets and success rates is reported.
