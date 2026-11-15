"""Fig. 5 (double column): what the pairwise resistances recover when the junction
resistance is a third unknown (dense grid: 99 cells, 100 networks per cell, rho = 0 to 100).
(a) Out-of-sample R^2 of log10 rho, log10 a and log10 N a^2 predicted from the terminal
    resistance S alone and with C1, C4, C16 or P added; folds are groups of whole cells.
(b) Cramer-Rao bound on the same three parameters (median standard deviation in log10
    units over the cells and the junction ratios, one network per film), with no
    measurement noise (filled) and 10 % (open); S alone has an infinite bound.
Both from data/junction_dense_summary.json, written by cluster/junction_pairwise_dense_analysis.py
(pre-registered, PREREGISTRATION_JUNCTION_DENSE.md).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from style import TWO_COL, CAT, MARK, INK2, GRAY, DATADIR, grid, panel, save

d = json.load(open(os.path.join(DATADIR, "junction_dense_summary.json")))
FS = ["S", "S+C1", "S+C4", "S+C16", "S+P"]
LAB = [r"$R$", r"$+C_1$", r"$+C_4$", r"$+C_{16}$", r"$+P$"]
TG = [("log10 rho", r"$\rho$ (junction)", CAT[0], MARK[0]),
      ("log10 a", r"$a$ (stick length)", CAT[1], MARK[1]),
      ("log10 Na2", r"$Na^2$ (density)", CAT[2], MARK[2])]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(TWO_COL, 1.75), gridspec_kw=dict(wspace=0.30))

x = np.arange(len(FS))
for (key, lab, col, m), off in zip(TG, (-0.2, 0.0, 0.2)):
    r2 = [d["regression"][key][f]["r2"] for f in FS]
    lo = [d["regression"][key][f]["ci"][0] for f in FS]; hi = [d["regression"][key][f]["ci"][1] for f in FS]
    axa.errorbar(x + off, r2, yerr=[np.array(r2) - lo, np.array(hi) - r2], fmt=m, ms=3.6, color=col, mec="white",
                 mew=0.4, elinewidth=0.8, capsize=1.5, label=lab)
    axa.plot(x + off, r2, color=col, lw=0.6, alpha=0.5)
    print(key, [round(v, 3) for v in r2])
axa.axhline(0, color=INK2, lw=0.5)
axa.set_xticks(x); axa.set_xticklabels(LAB); axa.set_ylim(-0.05, 1.02)
axa.set_ylabel(r"held-out-cell $R^2$"); axa.set_xlabel("information used, always with $R$")
axa.legend(loc="center right", bbox_to_anchor=(1.0, 0.50), handlelength=1.2, labelspacing=0.2, fontsize=6.5)
grid(axa, "y"); panel(axa, "a", x=-0.17)

for (key, lab, col, m), off, j in zip(TG, (-0.2, 0.0, 0.2), (2, 0, 1)):
    for noise, filled in ((0.0, True), (0.1, False)):
        y = [d["cramer_rao"][f"{f}|{noise}"]["median_sd"][j] for f in FS]
        xs = [i + off + (0.045 if not filled else -0.045) for i, v in enumerate(y) if np.isfinite(v)]
        ys = [v for v in y if np.isfinite(v)]
        axb.plot(xs, ys, m, ms=3.6, color=col, mfc=col if filled else "white", mew=0.8)
    print(key, "CRB, no noise:", [round(d["cramer_rao"][f"{f}|0.0"]["median_sd"][j], 3) for f in FS[1:]])
axb.text(0, 0.62, r"$\infty$", ha="center", va="center", fontsize=9, color=INK2)
axb.set_xticks(x); axb.set_xticklabels(LAB); axb.set_xlim(-0.5, 4.5)
axb.set_yscale("log"); axb.set_ylim(0.08, 0.8)
axb.set_yticks([0.1, 0.2, 0.4, 0.8]); axb.set_yticklabels(["0.1", "0.2", "0.4", "0.8"]); axb.minorticks_off()
axb.set_ylabel("Cramér–Rao bound, $\\log_{10}$ units"); axb.set_xlabel("information used, always with $R$")
axb.legend(handles=[Line2D([], [], color=INK2, marker="o", ls="", ms=3.6, label="no measurement noise"),
                    Line2D([], [], color=INK2, marker="o", ls="", ms=3.6, mfc="white", label="10 % noise")],
           loc="upper right", fontsize=6.3, handletextpad=0.3)
grid(axb, "y"); panel(axb, "b", x=-0.22)
save(fig, "fig5_junction.pdf")
