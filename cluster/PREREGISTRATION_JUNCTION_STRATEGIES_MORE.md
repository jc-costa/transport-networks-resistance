# Pre-registration: three follow-ups of the strategies run (paper A, run E2b)

This file is written and committed **before** `run_junction_strategies_more_task.py` is run on the cluster. The git commit that adds it is the timestamp. Nothing below may change after the run. Any extra analysis is reported as exploratory.

## Why this run

The strategies run (`PREREGISTRATION_JUNCTION_STRATEGIES.md`) found that a few tip readings capture little of what the all-pairs profile contains, and none of its five hypotheses held. Three questions follow, and the paper states them as untested: (1) does the error keep falling with many more readings; (2) does choosing where the tips land, from an image of the film, help; (3) does a better summary of the readings of a position help. This run answers them on the **same films**.

## Background (exploratory, seen before this file was written)

In the strategies run, two tips at 243 attempts reached an RMSE of log₁₀ ρ of 0.329 (contact of one wire resistance per tip), the terminal resistance alone 0.423, all pairs 0.145. A post hoc analysis (`junction_strategies_contact.py`) with the contact relative to the film's resistance gave 0.234 to 0.252 for a contact of 0.1 to 0.5 times the film's. **The margins below were set after seeing these values.** The code of this run was tested on a small local sample (14 geometries × 3 networks), and its numbers were not used for any choice.

## Data

- **The same films as the strategies run**: the same 80 geometries (seed 1 of `units()`), the same networks (seeds `stable_seed(N, round(a, 3), 'junction_strategies')`), the same four junction ratios per network, equal junctions. The analysis joins the two files by geometry, network and film index and **stops if a junction ratio or a terminal resistance differs** (`load_join`).
- **New readings**, from a new random stream (tag `junction_strategies_more`), for one tip, two tips and the four-point probe, with the readings and the tip model of the strategies run:
  1. *More readings*, random landing in the base configuration (footprint 0.02, position error 0.01): 9 positions × 9, 27, 81 and 243 readings, that is **81, 243, 729 and 2187** attempts.
  2. *Chosen sites*: landing with a wider footprint of **0.05** (position error 0.01), either anywhere on a wire (**wide**, the control) or only on wire points whose local wire length is at least the median of the film (**chosen**). The local wire length of a point is the number of wire sample points (0.004 apart) within 0.05 of it; it is what an image of the film shows and does not depend on the junction resistance or on the currents. Designs 9×9, 9×27, 9×81.
  3. *Other estimators*: for every position, both the **mean** and the **median** of the successful readings are stored.
- **Added in the analysis**, as in the strategies run: the error of R₀ (uniform ±10 %, stored in the strategies run), the tip contact resistance (**one wire resistance R₀ l per tip**, none for the four-point probe), 5 % reading noise, the nominal a and N with ±10 % errors.

## Analysis (decides the result)

`junction_strategies_more_analysis.py`, unchanged. A quadratic ridge regression (penalty 10⁻³) estimates log₁₀ ρ from the terminal resistance, the shell values of the protocol and the nominal a and N, with **whole geometries held out** (5 folds, seed 0), on the films whose all-pairs profile has no empty shell. Score: the RMSE of log₁₀ ρ. Intervals: 2000 bootstrap resamples of the networks within each geometry, on the fixed predictions. The references are those of the strategies run on the same films: the terminal resistance alone and the all-pairs profile.

## Hypotheses (two tips, main case: margin 10 %, contact of one wire resistance, base landing)

- **F1 (more readings).** The error keeps falling: the lower bound of the interval of RMSE(243 attempts) − RMSE(2187 attempts), random landing, exceeds **0.02**.
- **F2 (chosen sites).** At 243 attempts, with the same footprint of 0.05, choosing the sites lowers the error: the lower bound of RMSE(wide) − RMSE(chosen) exceeds **0.01**.
- **F3 (estimator).** At 243 attempts, the median of the readings of a position lowers the error against the mean: the lower bound of RMSE(mean) − RMSE(median) exceeds **0.01**.

**Descriptive, not tested.** The same comparisons for one tip and the four-point probe; the RMSE against the number of attempts for each protocol (81 to 2187) and the share of the gap between the terminal resistance and all pairs that it closes; the share of attempts that land with wide and with chosen landing (choosing lowers it: the well-connected points are fewer); mean against median at every design; the contact resistance relative to the film's (0.1, 0.5 and 1 times its terminal resistance) at 243 and 2187 attempts.

## Modeling limits stated in advance

Those of the strategies run: point-contact tips attached to a crossing, no current spreading, equal junctions, two dimensions, tip parameters assumed. Specific to this run: the chosen sites use only the geometry of the wires (an image), not the current; a protocol that knew the currents, or that chose sites by another criterion, could do better or worse; 2187 attempts is the largest budget; the estimator is the same quadratic regression on shell values, with the mean or the median.

## What would change the paper's claim

If F1 fails, the paper says the error saturates before 2187 attempts and gives the level. If F2 fails, the paper says that choosing sites by local wire length does not help under this model, which does not exclude other criteria. If F3 fails, it says that the median does not improve on the mean. Whatever the outcome, the tables of the three comparisons are reported, and the statement of the paper that its conclusions are about these protocols and estimators stays.
