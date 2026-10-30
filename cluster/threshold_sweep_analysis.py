"""Analysis of run_threshold_sweep_task.py: per box, a logistic fit of the percolation rate
against N a^2 gives the half-percolation density p_1/2 and the window width; 95% intervals by
a parametric bootstrap of the binomial counts."""
import os, json, glob
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "threshold_results", "point_*.json")))]
rng = np.random.default_rng(56)
def fit(x, k, n):
    def nll(th):
        p = 1 / (1 + np.exp(-(x - th[0]) / th[1])); p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
    return minimize(nll, [5.6, 0.3], method="Nelder-Mead").x
for a in sorted({p["a"] for p in pts}, reverse=True):
    sel = sorted([p for p in pts if p["a"] == a], key=lambda p: p["x"])
    x = np.array([p["x"] for p in sel]); k = np.array([p["percolating"] for p in sel]); n = np.array([p["draws"] for p in sel])
    c, w = fit(x, k, n)
    boots = np.array([fit(x, rng.binomial(n, np.clip(k / n, 1e-6, 1 - 1e-6)), n) for _ in range(500)])
    lo, hi = np.percentile(boots[:, 0], [2.5, 97.5])
    print(f"box {1/a:5.2f}: p_1/2 = {c:.3f} [{lo:.3f}, {hi:.3f}]  width {w:.3f}  "
          f"({'contains' if lo <= 5.637 <= hi else 'excludes'} 5.64)   rates: " +
          " ".join(f"{xx:.1f}:{kk/nn:.2f}" for xx, kk, nn in zip(x, k, n)))
