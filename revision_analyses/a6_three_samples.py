"""A6: the three samples of (N, a, d) = (150, 0.2, 0.10) quoted in Sec. 4.8 (revision plan, M16)."""
import json, glob, os
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); CL = os.path.join(HERE, "..", "cluster")
key = (150, 0.2, 0.1)
def match(N, a, d): return (N, round(a, 3), round(d, 3)) == key
pv = []
for f in sorted(glob.glob(os.path.join(CL, "preview_curves_results", "chunk_*.json"))):
    for r in json.load(open(f))["results"]:
        if match(r["N"], r["a"], r["d"]) and r.get("percolated"): pv.append(r["R_eff"])
dc = json.load(open(os.path.join(HERE, "..", "figures", "partA_constant_density_control.json")))
ci = [i for i, c in enumerate(dc["combos"]) if match(*c)][0]
dcv = [r["R_eff"] for r in dc["realizations"] if r["combo_index"] == ci and r.get("percolated")]
grid = []
for f in sorted(glob.glob(os.path.join(CL, "effective_resistance_d_results", "chunk_*.json"))):
    for r in json.load(open(f)):
        if match(r["N"], r["a"], r["d"]): grid = r["R_eff_values"]
S = {"preview (50)": np.array(pv), "density control (250)": np.array(dcv), "scalar grid (24)": np.array(grid)}
rng = np.random.default_rng(7)
for k, v in S.items():
    lv = np.log10(v); bm = [np.mean(rng.choice(v, len(v))) for _ in range(5000)]
    print(f"{k:22s} n={len(v):3d} mean={v.mean():7.1f}±{v.std(ddof=1)/np.sqrt(len(v)):4.1f} "
          f"[boot95 {np.percentile(bm,2.5):6.1f},{np.percentile(bm,97.5):6.1f}] median={np.median(v):7.1f} "
          f"GM={10**lv.mean():7.1f} log10 mean={lv.mean():.4f}±{lv.std(ddof=1)/np.sqrt(len(v)):.4f} skew={stats.skew(v):.2f} max/median={v.max()/np.median(v):.1f}")
names = list(S)
for i in range(3):
    for j in range(i + 1, 3):
        a_, b_ = S[names[i]], S[names[j]]
        la, lb = np.log10(a_), np.log10(b_)
        z = (la.mean() - lb.mean()) / np.sqrt(la.var(ddof=1)/len(la) + lb.var(ddof=1)/len(lb))
        print(f"{names[i]} vs {names[j]}: log-mean diff = {z:+.2f} SE; Mann-Whitney p={stats.mannwhitneyu(a_, b_).pvalue:.3f}; KS p={stats.ks_2samp(a_, b_).pvalue:.3f}")
