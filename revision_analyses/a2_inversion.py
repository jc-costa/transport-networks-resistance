"""A2: re-score the out-of-sample inversion (revision plan, block M14)."""
import json, glob, os
import numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster", "validation_results")
rows = []
for f in sorted(glob.glob(os.path.join(D, "candidate_*.json"))):
    c = json.load(open(f)); v = np.array(c["R_eff_values"], float); T = c["R_target"]
    gm = np.exp(np.mean(np.log(v)))
    rows.append(dict(T=T, N=c["N"], a=c["a"], n=len(v), x=c["N"] * c["a"]**2,
                     e_am=np.mean(v) / T - 1, e_gm=gm / T - 1, e_med=np.median(v) / T - 1,
                     fac=np.exp(np.median(np.abs(np.log(v / T)))), cv=np.std(v, ddof=1) / np.mean(v),
                     se=np.std(v, ddof=1) / np.sqrt(len(v)) / T))
print(f"{'target':>6} {'a':>4} {'N':>4} {'Na2':>5} {'n':>3} {'AM err':>7} {'GM err':>7} {'med err':>7} {'1-net factor':>12} {'CV':>5} {'SE/T':>5}")
for r in rows:
    print(f"{r['T']:6.0f} {r['a']:4.1f} {r['N']:4d} {r['x']:5.1f} {r['n']:3d} {100*r['e_am']:+6.1f}% {100*r['e_gm']:+6.1f}% {100*r['e_med']:+6.1f}% {r['fac']:12.2f} {r['cv']:5.2f} {100*r['se']:4.1f}%")
for key, lab in (("e_am", "arithmetic mean"), ("e_gm", "geometric mean"), ("e_med", "median")):
    e = np.array([abs(r[key]) for r in rows]); hi = np.array([abs(r[key]) for r in rows if r["x"] >= 10])
    print(f"{lab:16s}: median |err| {100*np.median(e):.1f}% (all 22), {100*np.median(hi):.1f}% (Na^2>=10, n={len(hi)}); within 30%: {np.sum(e<=0.3)}/22; mean signed {100*np.mean([r[key] for r in rows]):+.1f}%")
print("single network vs target, median factor over candidates:", round(float(np.median([r['fac'] for r in rows])), 2),
      "range", round(min(r['fac'] for r in rows), 2), "-", round(max(r['fac'] for r in rows), 2))
for a in (0.2, 0.4, 0.7, 1.0):
    e = [abs(r["e_gm"]) for r in rows if r["a"] == a]
    print(f"a={a}: mean |GM err| {100*np.mean(e):.1f}% over {len(e)}")
