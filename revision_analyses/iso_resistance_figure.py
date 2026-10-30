"""Finding 3 (inverse problem): the iso-resistance curves of the general law in the (a, N) plane.

Existing data only, nothing is simulated:
  cluster/validation_plan.json   fit coefficients (R = Ap N^p a^q), the 22 feasible candidates
                                 and the 6 infeasible ones (with the N the curve asked for)
  cluster/validation_results/    achieved mean R_eff of each candidate (30 networks; 4 at
                                 890 ohm, a = 0.2)
  ../experiments/exp_*           the (N, a) cells of the pooled 2D sweeps the law was fitted on,
                                 selected and de-duplicated exactly as run_validation_task.pool_and_fit
For each target R_t the curve is N(a) = (R_t / (Ap a^q))^(1/p): every point on it is an answer of
the inverse problem with the same predicted R. Output: iso_resistance.png (thesis style)."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.join(HERE, "..", "cluster")
sys.path.insert(0, CL); sys.path.insert(0, os.path.join(HERE, "..", "utils"))
import run_validation_task as rv  # noqa: E402

vp = json.load(open(os.path.join(CL, "validation_plan.json")))
Ap, p, q = vp["fit"]["Ap"], vp["fit"]["p_exp"], vp["fit"]["q_exp"]
plan, infeasible = vp["plan"], vp["infeasible"]
achieved = []
for i, c in enumerate(plan):
    r = json.load(open(os.path.join(CL, "validation_results", f"candidate_{i:02d}.json")))
    assert r["N"] == c["N"] and abs(r["a"] - c["a"]) < 1e-12
    achieved.append((r["achieved_mean_ohm"], r["n"]))

cells, seen = set(), set()
for path in sorted(glob.glob(os.path.join(rv.EXPERIMENTS_DIR, "exp_*", "effective_resistance_sweep_summary*.json"))):
    if not rv._is_post_fix(path):
        continue
    for rec in json.load(open(path)).get("sweep", []):
        vals = rec.get("R_eff_values") or []
        key = (int(rec["N"]), round(float(rec["a"]), 3))
        if vals and (key, tuple(vals)) not in seen:
            seen.add((key, tuple(vals))); cells.add(key)
assert len(cells) == vp["fit"]["n_points"], (len(cells), vp["fit"]["n_points"])

targets = sorted({c["R_target"] for c in plan} | {c["R_target"] for c in infeasible})
cmap = plt.get_cmap("plasma")
col = {t: cmap(0.85 * i / (len(targets) - 1)) for i, t in enumerate(targets)}
N_of = lambda a, R: (R / (Ap * a ** q)) ** (1 / p)

plt.rcParams.update({"font.size": 10.5})
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ag = np.geomspace(0.15, 1.3, 300)
ylo, yhi = 5, 1000
ax.fill_between(ag, np.sqrt(2 * 3000) / ag, yhi, color="0.9", lw=0, zorder=0, label="over the 3000-node budget")
ax.fill_between(ag, ylo, 10, color="0.9", lw=0, hatch="///", edgecolor="0.75", zorder=0, label="fewer than 10 sticks")
ax.plot(ag, 5.64 / ag ** 2, ls="-.", color="0.35", lw=1, zorder=1, label=r"$Na^2 = 5.64$")
ca, cN = zip(*[(a, N) for N, a in cells])
ax.scatter(ca, cN, s=9, marker="s", color="0.6", lw=0, zorder=2, label="fitted sweep cells")
ly = {t: np.log10(N_of(1.3, t)) for t in targets}          # right-edge labels, nudged apart
for lo, hi in zip(targets[::-1], targets[::-1][1:]):        # from the highest R (lowest N) upward
    ly[hi] = max(ly[hi], ly[lo] + 0.06)
for t in targets:
    ax.plot(ag, N_of(ag, t), color=col[t], lw=1.6, zorder=3)
    ax.text(1.32, 10 ** ly[t], f"{t:g} $\\Omega$", color=col[t], va="center", ha="left", fontsize=8.5, clip_on=False)
for j, (c, (Rach, n)) in enumerate(zip(plan, achieved)):
    t = c["R_target"]; side = 1 if targets.index(t) % 2 else -1
    ax.scatter([c["a"]], [c["N"]], s=34, color=col[t], edgecolor="k", lw=0.6, zorder=5)
    lab = (f"{Rach:.0f}" if Rach >= 100 else f"{Rach:.3g}") + ("*" if n < 30 else "")
    ax.annotate(lab, (c["a"], c["N"]), xytext=(5 * side, 0), textcoords="offset points",
                ha="left" if side > 0 else "right", va="center", fontsize=7, zorder=6,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
for c in infeasible:
    ax.scatter([c["a"]], [c["N_curve"]], s=34, facecolor="white", edgecolor=col[c["R_target"]], lw=1.2, zorder=5)
ax.scatter([], [], s=34, color="0.5", edgecolor="k", lw=0.6, label="candidate")
ax.scatter([], [], s=34, facecolor="white", edgecolor="0.4", lw=1.2, label="infeasible candidate")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.15, 1.3); ax.set_ylim(ylo, yhi)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.7, 1.0]); ax.set_xticklabels(["0.2", "0.3", "0.4", "0.5", "0.7", "1.0"])
ax.minorticks_off()
ax.set_yticks([5, 10, 20, 50, 100, 200, 500, 1000]); ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500", "1000"])
ax.set_xlabel("stick length $a$")
ax.set_ylabel("number of sticks $N$")
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
fig.tight_layout()
out = os.path.join(HERE, "iso_resistance.png")
fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"{len(plan)} candidates, {len(infeasible)} infeasible, {len(cells)} cells -> {out}")
print("* = fewer than 30 networks:", [(c["N"], c["a"], n) for c, (_, n) in zip(plan, achieved) if n < 30])
