"""
Freezes the estimators used by the transfer run (PREREGISTRATION_JUNCTION_TRANSFER.md).

Trained on the dense-grid records (cluster/junction_pairwise_dense_results), uniform junctions, rho > 0,
every network with a complete all-pairs profile. Three estimators of log10 rho, all quadratic ridge
regressions (penalty 1e-3) on standardized features:
  S        log10 of the terminal resistance
  S+level  log10 S and the mean of the nine log10 shell values of the all-pairs profile P
  S+P      log10 S and the nine log10 shell values of P
The half-width of the 90 % interval of each is the 90th percentile of the absolute out-of-fold error
with whole cells held out (the registered five folds, seed 0).

Writes figures/data/junction_estimator_frozen.json. Run once, before the transfer run, and committed.
Usage: python freeze_junction_estimator.py <dense partial.jsonl> <out.json>
"""
import sys, json, hashlib
import numpy as np
import junction_pairwise_review_checks as rc
import junction_pairwise_regime_analysis as rg

SETS = ("S", "S+level", "S+P")


def main():
    path, out = sys.argv[1], sys.argv[2]
    rows, _ = rc.load_all(path)
    rows = [r for r in rows if r["rho"] > 0 and rc.complete(r["P"])]
    y = np.log10([r["rho"] for r in rows])
    fmap = rc.folds_for([r["cell"] for r in rows]); fold = np.array([fmap[r["cell"]] for r in rows])
    est = {}
    for name in SETS:
        X = rc.features(rows, name)
        mu, sd = X.mean(0), X.std(0)
        A = rg.design_like((X - mu) / sd)
        pen = rc.rg.LAM * np.eye(A.shape[1]); pen[0, 0] = 0
        w = np.linalg.solve(A.T @ A + len(A) * pen, A.T @ y)
        p = rc.cv(rows, name, y, fold)
        est[name] = {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(),
                     "q90": float(np.percentile(np.abs(p - y), 90)),
                     "cv_rmse": float(np.sqrt(np.mean((p - y) ** 2))), "cv_r2": float(rc.r2(y, p))}
        print(f"{name:8s} cv R2 {est[name]['cv_r2']:.3f}  cv RMSE {est[name]['cv_rmse']:.3f}  90% half-width {est[name]['q90']:.3f}")
    meta = {"training_file": path.split("/")[-2] + "/" + path.split("/")[-1],
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "rows": len(rows), "networks": len({(r["cell"], r["net"]) for r in rows}),
            "cells": len({r["cell"] for r in rows}), "ridge_penalty": rc.rg.LAM, "folds_seed": rc.SEED}
    json.dump({"meta": meta, "estimators": est}, open(out, "w"), indent=1)
    print("wrote", out, meta["networks"], "networks,", meta["cells"], "cells")


if __name__ == "__main__":
    main()
