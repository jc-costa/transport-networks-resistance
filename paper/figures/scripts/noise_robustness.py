"""New analysis (not in the thesis): how measurement noise degrades the thickness
signal of the onion curve. Re-analysis of stored curves only, no simulation.

Each realization's nine shell means and its terminal resistance are multiplied
by independent log-normal factors exp(sigma * z), z ~ N(0, 1), before the
regression; sigma is the relative noise of one measured value (0.05 = 5 %).
Noise is drawn per realization, so averaging k realizations averages it too,
as repeated measurements would. The regression, grouped cross-validation and
k-averaging are those of k_curves_controls.py (and of curve_inversion.py).

Model variants: the slab grid with 3D shells (119 cells, the curve of Fig. 4a)
and the flat-stick control with in-plane shells (118 cells).

Usage: python noise_robustness.py <ColabProject dir> <out.json>
"""
import os, sys, json
import numpy as np

COLAB, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(COLAB, "cluster"))
import curve_inversion as ci  # noqa: E402

LAYERS = ci.LAYERS


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r["n"] > 0 and r["onion_layer_curves"]]


def build(rows, key, exclude_d=()):
    """As in k_curves_controls.py: one sample per realization, cells whose 24
    realizations all have complete curves."""
    X, Y, G, S = [], [], [], []
    for r in rows:
        ok = all(all(cv.get(str(l)) is not None for l in LAYERS) for cv in r[key])
        if r["d"] in exclude_d or r["n"] < 24 or not ok:
            continue
        for i, cv in enumerate(r[key]):
            X.append([cv[str(l)] for l in LAYERS]); S.append(r["R_eff_values"][i])
            Y.append([r["N"], r["a"], r["d"]]); G.append(f'{r["N"]}|{r["a"]}|{r["d"]}')
    return (np.log10(np.array(X)), np.log10(np.array(Y)), np.array(G), np.log10(np.array(S))[:, None])


SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
KS = (1, 24)
LN10 = np.log(10.0)


def scores(X, Y, G, S, sigma, seed=0, folds=10, lam=1.0, draws=10):
    """R^2 for log10 d at k = 1 and 24, with noise added in log10 units."""
    counts = {g: int((G == g).sum()) for g in set(G.tolist())}
    keys = sorted(g for g, c in counts.items() if c >= 24)
    rng = np.random.default_rng(seed)
    F = np.hstack([X, S])
    out = {}
    for fname, cols in (("R", slice(9, 10)), ("curve+R", slice(0, 10))):
        out[fname] = {}
        for k in KS:
            r2s = []
            for _ in range(draws):
                Fn = F + sigma / LN10 * rng.standard_normal(F.shape)   # fresh noise per draw
                Xs, Ys, Gs = [], [], []
                for key in keys:
                    idx = np.where(G == key)[0]
                    for _r in range(max(1, len(idx) // k)):
                        sel = rng.choice(idx, k, replace=False)
                        Xs.append(Fn[sel][:, cols].mean(0)); Ys.append(Y[idx[0]]); Gs.append(key)
                _, r2, _ = ci.cross_validate(np.array(Xs), np.array(Ys), np.array(Gs),
                                             lambda a, b: ci.ridge_fit(a, b, lam), folds)
                r2s.append(r2[2])
            out[fname][str(k)] = float(np.mean(r2s))
    return out, len(keys)


res = {"sigmas": SIGMAS, "ks": KS}
grid = ci.load_rows(os.path.join(COLAB, "cluster", "onion_grid_results"))
flat = load_jsonl(os.path.join(COLAB, "cluster", "onion_flat_results", "partial.jsonl"))
data = {"slab_3d": ci.dataset_realizations(grid),
        "flat_inplane": build(flat, "per_realization_curves", exclude_d=(0.02,))}
for name, (X, Y, G, S) in data.items():
    res[name] = {}
    for sg in SIGMAS:
        r, n = scores(X, Y, G, S, sg)
        res[name][str(sg)] = r
        print(f"{name:13s} cells={n} sigma={sg:4.2f}: curve+R k=1 {r['curve+R']['1']:.2f} k=24 {r['curve+R']['24']:.2f}"
              f" | R alone k=1 {r['R']['1']:.2f} k=24 {r['R']['24']:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
