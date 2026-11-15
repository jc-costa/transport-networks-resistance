"""
Post hoc (second revision, item 4): the scale of the tip contact resistance.

In the registered analysis a tip contact is kappa wire resistances on the NOMINAL scale: the reading is
c * (network part) + n_tips * kappa * a, where c is the error of R0 of the film. The error of R0 therefore multiplies
the network part of a reading and not its contact term. Here the contact follows the physical wire of each film
instead, c * (network part + n_tips * kappa * a), with the same means, counts, noise and folds.

Usage: python junction_strategies_contact_scale.py <E2 partial.jsonl> [--out summary.json]
"""
import sys, json
import numpy as np
import junction_strategies_analysis as sa

films = [f for f in sa.load(sys.argv[1]) if all(v is not None for v in f["P"])]
D = sa.Data(films)


def rmse_scale(kind, design, margin, physical):
    rng = np.random.default_rng(sa.SEED)
    c = 1 + margin * D.uc
    Sobs = c * D.S * np.exp(sa.NOISE * rng.standard_normal(D.n))
    prior = np.column_stack([np.log10(D.a * (1 + margin * D.ua)), np.log10(D.N * (1 + margin * D.uN))])
    mean, cnt = D.reading(kind, sa.CFG, design)
    tip = {"one": 1.0, "two": 2.0, "four": 0.0}[kind]
    contact = tip * sa.KAPPA * D.a[:, None]
    val = c[:, None] * (mean + contact) if physical else c[:, None] * mean + contact
    val = val * np.exp(sa.NOISE / np.sqrt(np.maximum(cnt, 1)) * rng.standard_normal(mean.shape))
    lv = np.where(cnt > 0, np.log10(np.where(val > 0, val, np.nan)), np.nan)
    X = np.hstack([np.log10(Sobs)[:, None], sa.impute(lv, np.log10(Sobs)), prior])
    e = D.cv(X, D.y) - D.y
    return float(np.sqrt(np.mean(e ** 2)))


out = {}
print("RMSE of log10 rho, contact of one wire resistance per tip: nominal scale (registered) / physical wire of the film")
for kind in ("one", "two"):
    for design in ("9x3", "9x27"):
        for margin in (0.10, 0.50):
            a, b = rmse_scale(kind, design, margin, False), rmse_scale(kind, design, margin, True)
            out[f"{kind}|{sa.ATTEMPTS[design]}|{margin:g}"] = dict(nominal=a, physical=b)
            print(f"  {kind:4s} {sa.ATTEMPTS[design]:4d} attempts, margin {margin:.0%}: {a:.3f} / {b:.3f}")
if "--out" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
