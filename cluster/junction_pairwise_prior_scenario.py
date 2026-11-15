"""
EXPLORATORY (post hoc): the advisor's scenario. The wire resistance per length R0, the stick length a and
the number of sticks N are known only within a margin (a relative error drawn uniformly in +-margin, one
for each, independent), and the junction resistance is completely unknown (rho log-uniform, 0.03 to 100).
What does each amount of measurement say about log10 rho? Dense-grid records, whole cells held out.

The error in R0 multiplies the terminal resistance and every shell value (all resistances scale with R0);
the errors in a and N enter as nominal values given to the estimator as features.

Usage: python junction_pairwise_prior_scenario.py <dense partial.jsonl>
"""
import sys
import numpy as np
import junction_pairwise_review_checks as rc

rows, _ = rc.load_all(sys.argv[1])
rows = [r for r in rows if r["rho"] > 0 and rc.complete(r["P"])]
y = np.log10([r["rho"] for r in rows]); rho = np.array([r["rho"] for r in rows])
fm = rc.folds_for([r["cell"] for r in rows]); fold = np.array([fm[r["cell"]] for r in rows])
S = np.log10([r["S"] for r in rows])[:, None]; P = np.log10(np.array([r["P"] for r in rows]))
a = np.array([r["a"] for r in rows]); N = np.array([r["x"] for r in rows]) / a ** 2


def cv(X):
    p = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        p[te] = rc.fit_predict(X[tr], y[tr], X[te])
    return p


def scenario(margin, seed=0):
    rng = np.random.default_rng(seed)
    U = lambda: np.log10(rng.uniform(1 - margin, 1 + margin, len(rows)))[:, None] if margin else np.zeros((len(rows), 1))
    c, ua, uN = U(), U(), U()
    prior = np.hstack([np.log10(a)[:, None] + ua, np.log10(N)[:, None] + uN])
    Sx, Px = S + c, P + c
    return {"R only": cv(np.hstack([Sx, prior])), "R + level": cv(np.hstack([Sx, Px.mean(1, keepdims=True), prior])),
            "R + all pairs": cv(np.hstack([Sx, Px, prior]))}


print("RMSE of log10 rho against the margin of R0, a and N (prior spread of log10 rho: %.3f)" % y.std())
print("  margin |  " + "  ".join(f"{k:>13s}" for k in ("R only", "R + level", "R + all pairs")))
for m in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
    pp = scenario(m)
    print(f"  {m:6.0%} |  " + "  ".join(f"{np.sqrt(np.mean((p - y) ** 2)):13.3f}" for p in pp.values()))
pp = scenario(0.10)
print("\nMargin 10 %: RMSE of log10 rho by regime of the true rho")
print(f"  {'regime':26s} {'rows':>6s}  " + "  ".join(f"{k:>13s}" for k in pp))
for lab, lo, hi in (("wire-limited, rho < 0.1", 0, 0.1), ("0.1 to 1", 0.1, 1), ("1 to 10", 1, 10), ("junction-limited, rho > 10", 10, 1e9)):
    mk = (rho >= lo) & (rho < hi)
    print(f"  {lab:26s} {mk.sum():6d}  " + "  ".join(f"{np.sqrt(np.mean((p[mk] - y[mk]) ** 2)):13.3f}" for p in pp.values()))
