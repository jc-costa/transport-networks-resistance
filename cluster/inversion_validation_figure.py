"""Achieved mean resistance against target for the simulated inverse candidates (thesis Appendix C).

Reads cluster/validation_results/candidate_*.json, written by run_validation_task.py, and writes
inversion_validation_candidates.png next to them. Each point is one candidate (N, a); color and
marker give its stick length a. Same data and colors as the figure of
notebooks/inverse_N_a_from_R_eff.ipynb, with labels that name a instead of informal tags.
Usage: python inversion_validation_figure.py [out.png]"""
import os, sys, glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "validation_results", "candidate_*.json")))]
rows = [r for r in rows if r["achieved_mean_ohm"] is not None]
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "validation_results", "inversion_validation_candidates.png")

STYLE = {0.2: ("tab:red", "o", "$a = 0.2$ (many short sticks)"), 0.4: ("tab:blue", "s", "$a = 0.4$"),
         0.7: ("tab:orange", "D", "$a = 0.7$"), 1.0: ("tab:green", "^", "$a = 1.0$ (few long sticks)")}
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(5.6, 5.4))
for a, (col, mk, lab) in STYLE.items():
    pts = [(r["R_target"], r["achieved_mean_ohm"]) for r in rows if abs(r["a"] - a) < 1e-9]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, marker=mk, color=col, edgecolor="k", linewidth=0.5, label=lab, zorder=3)
vals = [r["R_target"] for r in rows] + [r["achieved_mean_ohm"] for r in rows]
lims = [min(vals) * 0.7, max(vals) * 1.3]
ax.plot(lims, lims, "k--", alpha=0.5, label="achieved = target")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"target $R_{\mathrm{eff}}$ ($\Omega$)")
ax.set_ylabel(r"achieved mean $R_{\mathrm{eff}}$ ($\Omega$)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(out, dpi=170, metadata={"Software": None}); plt.close(fig)
print(f"wrote {out} ({len(rows)} candidates)")
