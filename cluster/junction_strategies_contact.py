"""
EXPLORATORY (post hoc, E2b): the tip contact resistance relative to the resistance of the FILM, not of a wire.
In the pre-registered analysis a contact is kappa wire resistances (R0 l); in dense films that is comparable to
or larger than the film's terminal resistance. Here a contact is kappa_rel times the terminal resistance S of the
film (one contact for one tip, two for two tips, none for four points). Everything else as in
junction_strategies_analysis.py (margin 10 %, base landing, 5 % noise, whole geometries held out).
Usage: python junction_strategies_contact.py <partial.jsonl>
"""
import sys
import numpy as np
import junction_strategies_analysis as sa

films = [f for f in sa.load(sys.argv[1]) if all(v is not None for v in f["P"])]
D = sa.Data(films)
MARGIN = 0.10


def rmse_rel(kind, dz, kappa_rel, seed=0):
    rng = np.random.default_rng(seed)
    c = 1 + MARGIN * D.uc
    Sobs = c * D.S * np.exp(sa.NOISE * rng.standard_normal(D.n))
    prior = np.column_stack([np.log10(D.a * (1 + MARGIN * D.ua)), np.log10(D.N * (1 + MARGIN * D.uN))])
    mean, cnt = D.reading(kind, "base", dz)
    tip = {"one": 1.0, "two": 2.0, "four": 0.0}[kind]
    val = c[:, None] * mean + tip * kappa_rel * (c * D.S)[:, None]
    val = val * np.exp(sa.NOISE / np.sqrt(np.maximum(cnt, 1)) * rng.standard_normal(mean.shape))
    lv = np.where(cnt > 0, np.log10(np.where(val > 0, val, np.nan)), np.nan)
    X = np.hstack([np.log10(Sobs)[:, None], sa.impute(lv, np.log10(Sobs)), prior])
    p = D.cv(X, D.y)
    return float(np.sqrt(np.mean((p - D.y) ** 2)))


ref = sa.evaluate(D, "terminal", MARGIN)["rmse"]; ap = sa.evaluate(D, "allpairs", MARGIN)["rmse"]
print(f"RMSE of log10 rho against the contact resistance of one tip, in units of the terminal resistance S of the film")
print(f"(terminal resistance alone {ref:.3f}; all pairs {ap:.3f})")
print(f"{'contact / S':>11s}  {'one 9x9':>8s} {'one 9x27':>9s}  {'two 9x9':>8s} {'two 9x27':>9s}")
for k in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
    print(f"{k:11.1f}  {rmse_rel('one', '9x9', k):8.3f} {rmse_rel('one', '9x27', k):9.3f}  {rmse_rel('two', '9x9', k):8.3f} {rmse_rel('two', '9x27', k):9.3f}")
