"""A4: permutation null for the per-layer ANOVA of the matched-R_eff groups (revision plan, M15).
Reuses the grouping and normalization of cluster/onion_tolerance_sweep.py. Under the null, the
triple labels of whole realizations are exchangeable within a group; each permutation recomputes
every per-layer F test of every group and records the fraction with p < 0.05."""
import sys, os, json
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "cluster")
sys.path.insert(0, CL)
import onion_tolerance_sweep as ots

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--onion", default="onion_grid_results"); _ap.add_argument("--scalar", default=None)
_args = _ap.parse_args()
onion = ots.load_rows(os.path.join(CL, _args.onion))
scalar = ots.load_rows(os.path.join(CL, _args.scalar or ("effective_resistance_d_results" if _args.onion == "onion_grid_results" else _args.onion)))
paired = [k for k in onion if k in scalar and onion[k]["onion_layer_curves"] and scalar[k]["mean_ohm"] is not None]
paired.sort(key=lambda k: scalar[k]["mean_ohm"])

def group_data(groups):
    out = []
    for group in groups:
        shared = sorted(set.intersection(*[set(float(d) for d in onion[k]["onion_layer_curves"]) for k in group]))
        if len(shared) < 2: continue
        X, lab = [], []
        for k in group:
            c = ots.aligned_curves(onion[k], shared)
            if c.shape[0] >= 3:
                X.append(c / c[:, -1:]); lab.append(np.full(c.shape[0], len(X) - 1))
        if len(X) < 2: continue
        out.append((np.vstack(X)[:, :-1], np.concatenate(lab)))
    return out

def pvals_batch(X, labels):            # labels: (P, n) -> p-values (P, L)
    P, n = labels.shape; K = labels.max() + 1; L = X.shape[1]
    G = np.zeros((P, n, K)); G[np.arange(P)[:, None], np.arange(n)[None, :], labels] = 1.0
    cnt = G.sum(1)                                   # (P, K)
    sums = np.einsum('pnk,nl->pkl', G, X)
    grand = X.mean(0)
    ssb = np.einsum('pk,pkl->pl', cnt, (sums / cnt[:, :, None] - grand) ** 2)
    sst = ((X - grand) ** 2).sum(0)
    ssw = sst[None, :] - ssb
    F = (ssb / (K - 1)) / (ssw / (n - K))
    return stats.f.sf(F, K - 1, n - K)

def run(tol, n_perm, rng, kw=False):
    data = group_data(ots.build_groups(paired, scalar, tol))
    obs = np.concatenate([pvals_batch(X, lab[None, :])[0] for X, lab in data])
    frac_obs = np.mean(obs < 0.05); n_tests = len(obs)
    fracs = np.zeros(n_perm); done = 0; B = 250
    while done < n_perm:
        b = min(B, n_perm - done); sig = 0
        for X, lab in data:
            perm = np.array([rng.permutation(lab) for _ in range(b)])
            sig = sig + (pvals_batch(X, perm) < 0.05).sum(1)
        fracs[done:done + b] = sig / n_tests; done += b
    p_glob = (1 + np.sum(fracs >= frac_obs)) / (1 + n_perm)
    res = dict(tol=tol, n_groups=len(data), n_tests=n_tests, frac_obs=frac_obs, null_mean=fracs.mean(),
               null_q99=np.percentile(fracs, 99), null_max=fracs.max(), p_global=p_glob,
               bonf=int(np.sum(obs < 0.05 / n_tests)))
    # Benjamini-Hochberg at q = 0.05
    ps = np.sort(obs); thr = ps[ps <= 0.05 * np.arange(1, n_tests + 1) / n_tests]
    res["bh"] = int(np.sum(obs <= thr.max())) if thr.size else 0
    if kw:
        kwp = []
        for X, lab in data:
            for li in range(X.shape[1]):
                kwp.append(stats.kruskal(*[np.log10(X[lab == g, li]) for g in np.unique(lab)]).pvalue)
        kwp = np.array(kwp); res["kw_frac"] = float(np.mean(kwp < 0.05)); res["kw_bonf"] = int(np.sum(kwp < 0.05 / len(kwp)))
    return res

rng = np.random.default_rng(20261001)
out = []
for tol, P in [(0.15, 10000)] + [(t, 2000) for t in ots.TOLERANCES if t != 0.15]:
    r = run(tol, P, rng, kw=(tol == 0.15)); out.append(r)
    print(f"tol {tol:5.3f}: groups {r['n_groups']:2d} tests {r['n_tests']:3d}  observed {100*r['frac_obs']:5.1f}%  "
          f"null mean {100*r['null_mean']:4.1f}% q99 {100*r['null_q99']:4.1f}% max {100*r['null_max']:4.1f}%  "
          f"p_global {r['p_global']:.1e} ({'10k' if P==10000 else '2k'} perms)  Bonf {r['bonf']}  BH {r['bh']}"
          + (f"  | Kruskal-Wallis on log10: {100*r['kw_frac']:.1f}% sig, {r['kw_bonf']} Bonf" if 'kw_frac' in r else ''), flush=True)
json.dump([{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()} for r in out], open(f"a4_result_{_args.onion}.json", "w"), indent=1)
