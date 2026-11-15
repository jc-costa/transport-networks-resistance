"""Fig. 3 (single column): solution curves of the inverse problem. For each of
the seven target resistances, the line is every (a, N) at which the fitted law
R = 28254 N^-1.694 a^-2.459 (Ohm, R0 = 1000 Ohm per unit length) predicts the
target; filled markers are the 22 simulated candidates, open markers the six
infeasible ones; gray squares are the fitted cells. Resistances are labeled
in units of R0 L.

Inputs (read-only, ColabProject): cluster/validation_plan.json,
cluster/validation_results/candidate_*.json, the 2D sweep summaries, selected
as in revision_analyses/iso_resistance_figure.py.
"""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from style import COLAB, ONE_COL, INK2, R0, grid, save

CL = os.path.join(COLAB, "cluster")
sys.path.insert(0, CL); sys.path.insert(0, os.path.join(COLAB, "utils"))
import run_validation_task as rv  # noqa: E402

vp = json.load(open(os.path.join(CL, "validation_plan.json")))
Ap, p, q = vp["fit"]["Ap"], vp["fit"]["p_exp"], vp["fit"]["q_exp"]
plan, infeasible = vp["plan"], vp["infeasible"]
achieved = []
for i, c in enumerate(plan):
    r = json.load(open(os.path.join(CL, "validation_results", f"candidate_{i:02d}.json")))
    assert r["N"] == c["N"] and abs(r["a"] - c["a"]) < 1e-12
    achieved.append((r["achieved_mean_ohm"], r["n"]))
err = [abs(m / c["R_target"] - 1) for c, (m, _) in zip(plan, achieved)]
print(f"Ap={Ap:.0f} p={p:.3f} q={q:.3f}; {len(plan)} candidates, median |error| = {100 * np.median(err):.1f}%")

cells, seen = set(), set()
for path in sorted(glob.glob(os.path.join(rv.EXPERIMENTS_DIR, "exp_*", "effective_resistance_sweep_summary*.json"))):
    if not rv._is_post_fix(path):
        continue
    for rec in json.load(open(path)).get("sweep", []):
        vals = rec.get("R_eff_values") or []
        key = (int(rec["N"]), round(float(rec["a"]), 3))
        if vals and (key, tuple(vals)) not in seen:
            seen.add((key, tuple(vals))); cells.add(key)
assert len(cells) == vp["fit"]["n_points"] == 101

targets = sorted({c["R_target"] for c in plan} | {c["R_target"] for c in infeasible})
ramp = LinearSegmentedColormap.from_list("r", ["#8fb8ea", "#2a78d6", "#0d3b75"])
col = {t: ramp(i / (len(targets) - 1)) for i, t in enumerate(targets)}
N_of = lambda a, R: (R / (Ap * a ** q)) ** (1 / p)

fig, ax = plt.subplots(figsize=(ONE_COL, 2.5))
ag = np.geomspace(0.15, 1.3, 300)
ylo, yhi = 5, 1000
ax.fill_between(ag, np.sqrt(2 * 3000) / ag, yhi, color="#efeeea", lw=0, zorder=0)
ax.text(1.25, 800, "over node budget", ha="right", fontsize=6.5, color=INK2, va="center")
ax.fill_between(ag, ylo, 10, color="#efeeea", lw=0, zorder=0)
ax.text(0.17, 7.2, "fewer than 10 sticks", fontsize=6.5, color=INK2, va="center")
ax.plot(ag, 5.64 / ag ** 2, ls="-.", color=INK2, lw=0.7, zorder=1)
ca, cN = zip(*[(a, N) for N, a in cells])
ax.scatter(ca, cN, s=4, marker="s", color="#bdbcb6", lw=0, zorder=2)
ly = {t: np.log10(N_of(1.3, t)) for t in targets}
for lo, hi in zip(targets[::-1], targets[::-1][1:]):
    ly[hi] = max(ly[hi], ly[lo] + 0.085)
for t in targets:
    ax.plot(ag, N_of(ag, t), color=col[t], lw=1.0, zorder=3)
    ax.text(1.33, 10 ** ly[t], f"{t / R0:.3g}", color="#0b0b0b", va="center", ha="left", fontsize=6.5,
            clip_on=False)
ax.text(1.33, 10 ** (ly[targets[0]] + 0.17), r"$R_{\rm target}/(R_0L)$", fontsize=6.5, ha="left", clip_on=False)
for c in plan:
    ax.scatter([c["a"]], [c["N"]], s=14, color=col[c["R_target"]], edgecolor="white", lw=0.5, zorder=5)
for c in infeasible:
    ax.scatter([c["a"]], [c["N_curve"]], s=14, facecolor="white", edgecolor=col[c["R_target"]], lw=0.8, zorder=5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.15, 1.3); ax.set_ylim(ylo, yhi)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.7, 1.0]); ax.set_xticklabels(["0.2", "0.3", "0.4", "0.5", "0.7", "1"])
ax.set_yticks([10, 20, 50, 100, 200, 500, 1000]); ax.set_yticklabels(["10", "20", "50", "100", "200", "500", "1000"])
ax.minorticks_off()
ax.set_xlabel(r"stick length $a = l/L$")
ax.set_ylabel(r"number of sticks $N$")
grid(ax)
save(fig, "fig3_inverse.pdf")
