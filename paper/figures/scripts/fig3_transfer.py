"""Fig. 3 (double column): what the full profile contains, and where.
(a) Transfer run (E2 of the plan, `transfer`): predicted against true log10 rho for 150 new geometries and
    continuous junction ratios, equal junctions, by the estimator frozen from the dense grid (the terminal
    resistance with the all-pairs profile). Dashed: unity; band: the 90 % interval of the estimator.
(b) Scenario of the advisor (R0, a and N known within 10 %, junction unknown), strategy run `strategies`:
    RMSE of log10 rho by regime of the true rho, for the terminal resistance alone and for the all-pairs profile.
Inputs: COLAB/cluster/junction_pairwise_transfer_results/partial.jsonl, data/junction_estimator_frozen.json,
        data/junction_strategies_summary.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from style import COLAB, TWO_COL, CAT, INK2, GRAY, DATADIR, grid, panel, save

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cluster"))
import junction_pairwise_transfer_analysis as ta        # noqa: E402
import junction_pairwise_review_checks as rc            # noqa: E402

frozen = json.load(open(os.path.join(DATADIR, "junction_estimator_frozen.json")))
rows = [r for r in ta.load(os.path.join(COLAB, "cluster", "junction_pairwise_transfer_results", "partial.jsonl"))
        if r["variant"] == "uniform" and rc.complete(r["P"])]
S = np.array([r["S"] for r in rows]); P = np.array([r["P"] for r in rows])
y = np.log10([r["rho"] for r in rows]); p = ta.predict(frozen, "S+P", S, P)
q90 = frozen["estimators"]["S+P"]["q90"]
cover = float(np.mean(np.abs(p - y) <= q90)); rmse = float(np.sqrt(np.mean((p - y) ** 2)))
print(f"{len(y)} rows; RMSE {rmse:.3f}; coverage {cover:.3f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(TWO_COL, 1.95), gridspec_kw=dict(wspace=0.30, width_ratios=[1.25, 1]))
hb = axa.hexbin(y, p, gridsize=34, extent=(np.log10(0.03), 2, np.log10(0.03) - 0.9, 2.9), bins="log", cmap="Blues", mincnt=1, lw=0.1)
lo, hi = np.log10(0.03), 2.0
axa.plot([lo, hi], [lo, hi], color=INK2, ls="--", lw=0.8, zorder=5)
for sgn in (-1, 1):
    axa.plot([lo, hi], [lo + sgn * q90, hi + sgn * q90], color=CAT[1], ls=":", lw=0.9, zorder=5)
axa.axvline(-1.0, color=GRAY, lw=0.6, ls=":"); axa.axvline(1.0, color=GRAY, lw=0.6, ls=":")
axa.text(-1.25, 2.55, "wire-limited", fontsize=6.3, color=INK2, ha="center"); axa.text(0.0, 2.55, "intermediate", fontsize=6.3, color=INK2, ha="center")
axa.text(1.5, 2.55, "junction-\nlimited", fontsize=6.3, color=INK2, ha="center", va="center")
axa.text(0.03, 0.04, f"RMSE {rmse:.2f}; 90 % band covers {100 * cover:.0f} %", transform=axa.transAxes, fontsize=6.5, color=INK2, va="bottom")
axa.set_xlim(lo - 0.05, hi + 0.05); axa.set_ylim(lo - 0.9, 2.9)
axa.set_xlabel(r"true $\log_{10}\rho$"); axa.set_ylabel(r"estimated $\log_{10}\rho$")
grid(axa); panel(axa, "a", x=-0.19)

sm = json.load(open(os.path.join(DATADIR, "junction_strategies_summary.json")))["main"]
labs = ["wire-limited\n" + r"$\rho<0.1$", "intermediate\n" + r"$0.1$–$10$", "junction-limited\n" + r"$\rho>10$"]
keys = ["wire-limited (rho < 0.1)", "intermediate (0.1 to 10)", "junction-limited (rho > 10)"]
x = np.arange(3); w = 0.36
t = [sm["terminal"]["by_regime"][k] for k in keys]; a_ = [sm["allpairs"]["by_regime"][k] for k in keys]
axb.bar(x - w / 2, t, w, color=GRAY, label="terminal resistance")
axb.bar(x + w / 2, a_, w, color=CAT[0], label="all pairs")
for xi, v in zip(x - w / 2, t):
    axb.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.3)
for xi, v in zip(x + w / 2, a_):
    axb.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.3)
axb.set_xticks(x); axb.set_xticklabels(labs, fontsize=6.6); axb.set_ylim(0, 0.8)
axb.set_ylabel(r"RMSE of $\log_{10}\rho$"); axb.legend(loc="upper right", fontsize=6.5, handlelength=1.0)
grid(axb, "y"); panel(axb, "b", x=-0.2)
save(fig, "fig3_transfer.pdf")
