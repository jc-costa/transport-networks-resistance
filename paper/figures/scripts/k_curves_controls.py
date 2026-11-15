"""Out-of-sample R^2 for log10 d against the number k of realizations averaged,
for the slab with 3D shells, the slab with in-plane shells, and the flat-stick
control. Re-analysis of stored results only (no new simulation).

It reuses the regression and the averaging procedure of
ColabProject/cluster/curve_inversion.py (imported, not modified) and reads the
in-plane and flat-stick records from their partial.jsonl files.

Thesis values it must reproduce (Results, "What the curve determines"):
  slab, 3D shells, 118 cells complete under both definitions: 0.15 (k=1), 0.77 (k=24)
  slab, in-plane shells, same 118 cells:                       0.21 (k=1), 0.83 (k=24)
  flat sticks, in-plane shells, 118 cells (d = 0.02 excluded):                     0.56 (k=1), 0.93 (k=24)
  terminal resistance alone: about 0 in all three

Usage: python k_curves_controls.py <ColabProject dir> <out.json>
"""
import os, sys, json
import numpy as np

COLAB = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, os.path.join(COLAB, "cluster"))
import curve_inversion as ci  # noqa: E402

KS = (1, 2, 3, 6, 12, 24)
LAYERS = ci.LAYERS


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r["n"] > 0 and r["onion_layer_curves"]]


def complete(curves):
    return all(all(cv.get(str(l)) is not None for l in LAYERS) for cv in curves)


def build(rows, key, require_keys, exclude_d=()):
    """One sample per realization, cells whose 24 realizations are complete
    under every curve definition in require_keys."""
    X, Y, G, S = [], [], [], []
    for r in rows:
        if r["d"] in exclude_d or r["n"] < 24 or not all(complete(r[k]) for k in require_keys):
            continue
        for i, cv in enumerate(r[key]):
            X.append([cv[str(l)] for l in LAYERS]); S.append(r["R_eff_values"][i])
            Y.append([r["N"], r["a"], r["d"]]); G.append(f'{r["N"]}|{r["a"]}|{r["d"]}')
    return (np.log10(np.array(X)), np.log10(np.array(Y)), np.array(G),
            np.log10(np.array(S))[:, None])


def averaging(X, Y, G, S, folds=10, lam=1.0):
    """Copy of the averaging loop of curve_inversion.main, returning R^2 for d."""
    counts = {g: int((G == g).sum()) for g in set(G.tolist())}
    keys = sorted(g for g, c in counts.items() if c >= max(KS))
    rng = np.random.default_rng(0)
    out = {"n_cells": len(keys)}
    for fname, F in (("scalar", S), ("curve+scalar", np.hstack([X, S]))):
        out[fname] = {}
        for k in KS:
            r2s = []
            for _ in range(10):
                Xs, Ys, Gs = [], [], []
                for key in keys:
                    idx = np.where(G == key)[0]
                    for _r in range(max(1, len(idx) // k)):
                        sel = rng.choice(idx, k, replace=False)
                        Xs.append(F[sel].mean(0)); Ys.append(Y[idx[0]]); Gs.append(key)
                _, r2, _ = ci.cross_validate(np.array(Xs), np.array(Ys), np.array(Gs),
                                             lambda a, b: ci.ridge_fit(a, b, lam), folds)
                r2s.append(r2)
            out[fname][str(k)] = np.mean(r2s, 0).tolist()  # (N, a, d)
    return out


res = {"ks": KS}
inplane = load_jsonl(os.path.join(COLAB, "cluster", "onion_inplane_results", "partial.jsonl"))
both = ("per_realization_curves", "per_realization_curves_3d")
res["slab_3d"] = averaging(*build(inplane, "per_realization_curves_3d", both))
res["slab_inplane"] = averaging(*build(inplane, "per_realization_curves", both))
flat = load_jsonl(os.path.join(COLAB, "cluster", "onion_flat_results", "partial.jsonl"))
# the seven thicknesses of the slab grid; d = 0.02 was added to the flat runs only
res["flat_inplane"] = averaging(*build(flat, "per_realization_curves", ("per_realization_curves",), exclude_d=(0.02,)))
# the thesis curve: full slab grid with 3D shells (119 cells)
grid = ci.load_rows(os.path.join(COLAB, "cluster", "onion_grid_results"))
res["slab_grid_3d"] = averaging(*ci.dataset_realizations(grid))

for name in ("slab_grid_3d", "slab_3d", "slab_inplane", "flat_inplane"):
    r = res[name]
    print(f'{name:14s} cells={r["n_cells"]:3d}  curve+R d: k=1 {r["curve+scalar"]["1"][2]:.2f}'
          f'  k=24 {r["curve+scalar"]["24"][2]:.2f}   R only d: k=1 {r["scalar"]["1"][2]:.2f}'
          f'  k=24 {r["scalar"]["24"][2]:.2f}')
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
