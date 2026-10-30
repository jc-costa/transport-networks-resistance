"""A10 (advisor questions 1-2): on the 3D slab grid, what does each piece of information add?
Ridge, 10-fold grouped by combination, one realization per sample, as in curve_inversion.py:
scalar alone; + onion curve; + a known; + d known; + a and d known."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster")); sys.argv = sys.argv[:1]
import curve_inversion as ci
rows = ci.load_rows(os.path.join(os.path.dirname(ci.__file__), "onion_grid_results"))
X, Y, G, S = ci.dataset_realizations(rows)
logN, loga, logd = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
fit = lambda Xtr, Ytr: ci.ridge_fit(Xtr, Ytr, 1.0)
def run(name, feats, targets, tn):
    rmse, r2, _ = ci.cross_validate(feats, targets, G, fit, 10)
    print(f"  {name:34s} " + "  ".join(f"{t}: R2 {r:5.2f} factor {10**e:4.2f}" for t, r, e in zip(tn, r2, rmse)))
print(f"{len(Y)} realizations, {len(set(G.tolist()))} combinations")
run("scalar -> N, a, d", S, Y, "Nad")
run("curve+scalar -> N, a, d", np.hstack([X, S]), Y, "Nad")
run("scalar + a known -> N, d", np.hstack([S, loga]), np.hstack([logN, logd]), "Nd")
run("curve+scalar + a known -> N, d", np.hstack([X, S, loga]), np.hstack([logN, logd]), "Nd")
run("scalar + d known -> N, a", np.hstack([S, logd]), np.hstack([logN, loga]), "Na")
run("curve+scalar + d known -> N, a", np.hstack([X, S, logd]), np.hstack([logN, loga]), "Na")
run("scalar + a, d known -> N", np.hstack([S, loga, logd]), logN, "N")
run("curve+scalar + a, d known -> N", np.hstack([X, S, loga, logd]), logN, "N")
