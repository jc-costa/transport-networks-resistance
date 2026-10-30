"""Table robustness and the Sec. 4.1.2 statistics, stored batch vs corrected rerun."""
import json, glob, os
import numpy as np
from scipy import stats
CL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster")
def load(d): return [r for f in sorted(glob.glob(os.path.join(CL, d, "chunk_*.json"))) for r in json.load(open(f))]
for d in ("random_networks_results", "random_networks_results_fixed"):
    rows = load(d); P = [r for r in rows if r["has_percolation"] and np.isfinite(r["effective_resistance"])]
    R = np.array([r["effective_resistance"] for r in P]); N = np.array([r["N"] for r in P]); a = np.array([r["a"] for r in P])
    sing = sum(1 for r in rows if r["is_singular"]); pr = sum(1 for r in rows if r["needed_pruning"])
    lo, hi = N <= 25, N > 70
    rN = stats.spearmanr(N, R); ra = stats.spearmanr(a, R)
    print(f"{d}: total {len(rows)}, percolating {len(P)} ({len(P)/len(rows):.1%}), needed pruning {pr} ({pr/len(rows):.1%}), singular {sing}")
    print(f"   R min/median/max {R.min():.1f} / {np.median(R):.1f} / {R.max():.1f}")
    print(f"   median R: N<=25 {np.median(R[lo]):.0f} (n={lo.sum()}, mean a {a[lo].mean():.2f}); N>70 {np.median(R[hi]):.0f} (n={hi.sum()}, mean a {a[hi].mean():.2f})")
    print(f"   Spearman N: rho={rN.correlation:.2f} p={rN.pvalue:.0e}; a: rho={ra.correlation:.2f} p={ra.pvalue:.0e}")
