"""E7, descriptive checks written AFTER the results of pairwise2d_analysis.py were seen (they do not
change its pre-set criteria or verdicts). Part A, k = 1 only, the 21 candidates of the analysis:
  1. CV-seed sensitivity: the pre-set statistic (mean over targets of accuracy - chance) for seeds 0-9,
     each the full stratified 5-fold x 10 procedure.
  2. Sampling uncertainty: binomial SE of that statistic, sqrt(sum_t p_t(1-p_t)/n_t) / n_targets,
     from one stratified 5-fold split per target (seed 0).
  3. Gain over S itself: on the same split, per network, correct under S vs under S+X; exact
     two-sided McNemar test on the discordant networks pooled over targets."""
import sys
from math import comb
import numpy as np
sys.argv = sys.argv[:1]
import pairwise2d_analysis as pa

rows, nets, _ = pa.load("pairwise2d_results")
A = [n for n in nets if n["part"] == "A" and n["unit"] != "A19"]
targets = sorted({n["R_target"] for n in A})
subs = {t: [n for n in A if n["R_target"] == t] for t in targets}

print("1. CV-seed sensitivity (seed 0 is the pre-set run)")
res = {i: [] for i in pa.INFO}
for seed in range(10):
    for i in pa.INFO:
        ex = []
        for t in targets:
            y = np.array([n["unit"] for n in subs[t]])
            ex.append(pa.classify(pa.matrix(subs[t], i), y, 1, seed=seed) - 1 / len(set(y)))
        res[i].append(np.mean(ex))
for i in pa.INFO:
    v = np.array(res[i]); d = v - np.array(res["S"])
    print(f"   {i:6s} seed0 {v[0]:+.3f}  mean {v.mean():+.3f}  sd {v.std(ddof=1):.3f}  range [{v.min():+.3f}, {v.max():+.3f}]"
          f"  seeds >= 0.20: {(v >= 0.2).sum()}/10  minus S {d.mean():+.3f}")

print("2-3. one split per target (seed 0)")
corr = {i: [] for i in pa.INFO}; se2 = {i: 0.0 for i in pa.INFO}
for t in targets:
    sub = subs[t]; y = np.array([n["unit"] for n in sub])
    rng = np.random.default_rng(0); fold = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); fold[idx] = np.arange(len(idx)) % 5
    for i in pa.INFO:
        F = pa.matrix(sub, i); ok = np.zeros(len(y), bool)
        for f in range(5):
            te = fold == f; ok[te] = pa.lda_fit(F[~te], y[~te])(F[te]) == y[te]
        corr[i].append(ok); p = ok.mean(); se2[i] += p * (1 - p) / len(y)
S = np.concatenate(corr["S"])
for i in pa.INFO:
    line = f"   {i:6s} binomial SE ~ {np.sqrt(se2[i]) / len(targets):.3f}"
    if i != "S":
        X = np.concatenate(corr[i]); b = int((~S & X).sum()); c = int((S & ~X).sum()); n = b + c
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
        line += f"   vs S on {len(S)} networks: gained {b}, lost {c}, McNemar p = {p:.2g}"
    print(line)
