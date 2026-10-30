"""Analysis of run_redbond_scaling_task.py: slope of log <N_red> against log L at each N a^2,
with a bootstrap over realizations, plus the A7 bound summary."""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
pts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "redbond_results", "point_*.json")))]
rng = np.random.default_rng(34)
print(f"{'Na2':>5} {'box':>6} {'n':>4} {'rate':>6} {'<N_red>':>8} {'redR':>6} {'mism':>5} {'min bound ratio':>15}")
for p in pts:
    nets = p["networks"]
    if not nets: continue
    print(f"{p['x_target']:5.2f} {1/p['a']:6.2f} {p['n']:4d} {p['perc_rate']:6.3f} {np.mean([m['n_red_curr'] for m in nets]):8.2f} "
          f"{np.mean([m['red_R_fraction'] for m in nets]):6.3f} {sum(m['red_mismatch'] for m in nets):5d} {min(m['affine_bound_ratio'] for m in nets):15.3f}")
for x in sorted({p["x_target"] for p in pts}):
    sel = [p for p in pts if p["x_target"] == x and p["networks"]]
    if len(sel) < 2: continue
    logL = np.log([1 / p["a"] for p in sel]); counts = [np.array([m["n_red_curr"] for m in p["networks"]], float) for p in sel]
    slope = np.polyfit(logL, np.log([c.mean() for c in counts]), 1)[0]
    boots = [np.polyfit(logL, np.log([rng.choice(c, len(c)).mean() for c in counts]), 1)[0] for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"N a^2 = {x:.2f}: slope of log<N_red> vs log L = {slope:.3f}  95% [{lo:.3f}, {hi:.3f}]  "
          f"({'contains' if lo <= 0.75 <= hi else 'excludes'} 3/4; {'contains' if lo <= 1 <= hi else 'excludes'} 1)")
allr = [m["affine_bound_ratio"] for p in pts for m in p["networks"]]
if allr:
    print(f"A7 bound: {sum(r < 1 - 1e-9 for r in allr)} violations in {len(allr)} networks; ratio median {np.median(allr):.3f}")
