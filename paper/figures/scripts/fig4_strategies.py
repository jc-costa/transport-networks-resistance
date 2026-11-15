"""Fig. 4 (double column): what practical protocols capture (run `strategies`, scenario of the advisor).
(a) RMSE of log10 rho against the number of attempted readings for one tip, two tips and a four-point probe
    (base landing, contact resistance of one wire resistance per tip, 5 % noise; 3 to 243 attempts from the
    strategies run, 729 and 2187 from E2b), with two tips at a contact of a tenth of the film's resistance
    (dashed orange) and the terminal resistance with the nominal a and N and the all-pairs profile as references.
(b) Success rate of an attempt (a tip lands on a wire) against the stick length, 9 x 9 readings.
Inputs: data/junction_strategies_summary.json, data/junction_strategies_more_summary.json,
        data/junction_strategies_more_medians.json, COLAB/cluster/junction_strategies_results/partial.jsonl.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from style import COLAB, TWO_COL, CAT, MARK, INK2, GRAY, DATADIR, grid, panel, save

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "cluster"))
import junction_strategies_analysis as sa              # noqa: E402

sm = json.load(open(os.path.join(DATADIR, "junction_strategies_summary.json")))["main"]
more = json.load(open(os.path.join(DATADIR, "junction_strategies_more_summary.json")))["random"]
med = json.load(open(os.path.join(DATADIR, "junction_strategies_more_medians.json")))["kappa_rel"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(TWO_COL, 1.95), gridspec_kw=dict(wspace=0.30))
K = [3, 9, 27, 81, 243]; DZ = ["3x3", "9x1", "9x3", "9x9", "9x27"]       # 3x1 and 9x1 share 3 and 9 attempts
DZ_ATT = {"3x1": 3, "3x3": 9, "9x1": 9, "9x3": 27, "9x9": 81, "9x27": 243}
names = {"one": "one tip", "two": "two tips", "four": "four points"}
for (kd, nm), col, m in zip(names.items(), CAT, MARK):
    xs = [DZ_ATT[d] for d in ("3x1", "9x1", "9x3", "9x9", "9x27")]
    ys = [sm[kd][d]["rmse"] for d in ("3x1", "9x1", "9x3", "9x9", "9x27")]
    xs += [729, 2187]; ys += [more[kd]["729"]["rmse"], more[kd]["2187"]["rmse"]]
    axa.plot(xs, ys, color=col, marker=m, ms=3.4, lw=1.0, mec="white", mew=0.4, label=nm)
axa.plot([243, 2187], [med["0.1|243"]["rmse"], med["0.1|2187"]["rmse"]], color=CAT[1], marker=MARK[1], ms=3.4, lw=1.0, ls="--", mfc="white", label="two tips, low contact")
axa.axhline(sm["terminal"]["rmse"], color=GRAY, ls="--", lw=0.9); axa.text(2187, sm["terminal"]["rmse"] + 0.006, "terminal + nominal", fontsize=8.4, color=INK2, ha="right")
axa.axhline(sm["allpairs"]["rmse"], color=INK2, ls="--", lw=0.9); axa.text(3.1, sm["allpairs"]["rmse"] + 0.006, "all pairs (ideal)", fontsize=8.4, color=INK2)
axa.axhline(0.35, color=GRAY, ls=":", lw=0.7)
axa.set_xscale("log"); axa.set_xticks([3, 27, 243, 2187]); axa.set_xticklabels(["3", "27", "243", "2187"]); axa.minorticks_off()
axa.set_ylim(0.1, 0.5); axa.set_xlabel("attempted readings"); axa.set_ylabel(r"RMSE of $\log_{10}\rho$")
axa.legend(loc="lower left", bbox_to_anchor=(0.0, 0.17), fontsize=8.4, handlelength=1.5, labelspacing=0.2)
grid(axa); panel(axa, "a", x=-0.17)

films = [f for f in sa.load(os.path.join(COLAB, "cluster", "junction_strategies_results", "partial.jsonl")) if all(v is not None for v in f["P"])]
D = sa.Data(films)
edges = np.array([0.2, 0.28, 0.4, 0.55, 0.75, 1.01]); mid = 0.5 * (edges[1:] + edges[:-1])
for (kd, nm), col, m in zip(names.items(), CAT, MARK):
    mean, cnt = D.reading(kd, "base", "9x9"); s = cnt.sum(1) / 81
    y = [s[(D.a >= lo) & (D.a < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
    axb.plot(mid, y, color=col, marker=m, ms=3.4, lw=1.0, mec="white", mew=0.4, label=nm)
axb.set_ylim(0, 0.7); axb.set_xlabel(r"stick length $a = l/L$"); axb.set_ylabel("share of attempts with a reading")
axb.legend(loc="upper right", fontsize=8.4, handlelength=1.5, labelspacing=0.2)
grid(axb); panel(axb, "b", x=-0.17)
save(fig, "fig4_strategies.pdf")
