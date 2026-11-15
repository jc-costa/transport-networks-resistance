"""
EXPLORATORY analysis (not pre-registered): where are the junction ratio rho and the
stick length a recoverable from S and the pairwise profile, as a function of the
regime (a, N a^2)?

The pre-registered test (junction_pairwise_confirm_analysis.py) averaged the
accuracy of a classifier over bins of S and found that it varies from below chance
to 0.8 between bins. Here rho is estimated as a continuous variable instead:

  target     log10 rho, on the samples with rho > 0 and uniform junctions
  features   log10 S, or log10 S with the nine log10 P values; a quadratic ridge
             regression on standardized features
  folds      5, all samples of one network in the same fold (seeded permutation)
  report     root-mean-square error of log10 rho by regime, for S alone and S + P;
             the spread of the true log10 rho is the error of guessing the mean

The same is done for log10 a, and for log10 (N a^2), over all samples.

Usage: python junction_pairwise_regime_analysis.py <partial.jsonl> [--out regime.json]
"""
import sys, json
import numpy as np
import junction_pairwise_analysis as ja

FOLDS, SEED, LAM = 5, 0, 1e-3


def design(X):
    """Quadratic features of standardized columns."""
    Z = (X - X.mean(0)) / X.std(0)
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1])]
    cols += [Z[:, i] * Z[:, j] for i in range(Z.shape[1]) for j in range(i, Z.shape[1])]
    return np.column_stack(cols)


def cv_predict(X, y, fold):
    pred = np.zeros(len(y))
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        A, B = design_like(Ztr), design_like(Zte)
        pen = LAM * np.eye(A.shape[1]); pen[0, 0] = 0
        w = np.linalg.solve(A.T @ A + len(A) * pen, A.T @ y[tr])
        pred[te] = B @ w
    return pred


def design_like(Z):
    d = Z.shape[1]
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(d)]
    cols += [Z[:, i] * Z[:, j] for i in range(d) for j in range(i, d)]
    return np.column_stack(cols)


def rmse_by(group, resid):
    return {str(g): float(np.sqrt(np.mean(resid[group == g] ** 2))) for g in np.unique(group)}


def main():
    rows = [r for r in ja.load(sys.argv[1]) if r["variant"] == "uniform"]
    groups = sorted({(r["cell"], r["net"]) for r in rows})
    order = np.random.default_rng(SEED).permutation(len(groups))
    fo = {g: int(order[i] % FOLDS) for i, g in enumerate(groups)}
    S = np.log10([r["S"] for r in rows])[:, None]
    P = np.log10(np.array([r["P"] for r in rows]))
    fold = np.array([fo[(r["cell"], r["net"])] for r in rows])
    a = np.array([r["a"] for r in rows]); x = np.array([r["x"] for r in rows]); rho = np.array([r["rho"] for r in rows])
    nominal = np.array([5.0, 6.5, 8.0, 10.0, 12.0, 16.0])
    xbin = nominal[np.abs(x[:, None] - nominal[None, :]).argmin(1)]      # N = round(Na2 / a^2) moves Na2 a little
    report = {}
    targets = {
        "log10 rho (rho>0)": (np.log10(np.where(rho > 0, rho, 1.0)), rho > 0),
        "log10 a": (np.log10(a), np.ones(len(rows), bool)),
        "log10 Na2": (np.log10(x), np.ones(len(rows), bool)),
    }
    for name, (y, m) in targets.items():
        sd = float(y[m].std())
        out = {"sd_of_target": sd, "n": int(m.sum())}
        for label, X in (("S", S), ("S+P", np.hstack([S, P]))):
            pred = cv_predict(X[m], y[m], fold[m])
            res = pred - y[m]
            out[label] = {"rmse": float(np.sqrt(np.mean(res ** 2))),
                          "by_a": rmse_by(a[m], res), "by_Na2": rmse_by(xbin[m], res)}
        report[name] = out
        print(f"\n{name}: n {out['n']}, sd of the target {sd:.3f}")
        print(f"   RMSE  S alone {out['S']['rmse']:.3f}   S + P {out['S+P']['rmse']:.3f}")
        for key, title in (("by_a", "a"), ("by_Na2", "N a^2")):
            ks = sorted(out["S"][key], key=float)
            print(f"   by {title:5s}: " + "  ".join(f"{k}: {out['S'][key][k]:.2f}/{out['S+P'][key][k]:.2f}" for k in ks))
    print("\n(by-regime entries are RMSE for S alone / S + P)")
    if "--out" in sys.argv:
        json.dump(report, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
