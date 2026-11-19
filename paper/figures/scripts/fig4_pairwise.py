"""Fig. 4 (double column): what pairwise resistances add to the terminal one.
(a) Films of finite thickness: out-of-sample R^2 for log10 d against the
    number k of realizations averaged, for the radial profile with R (three model
    variants) and for R alone. From data/k_curves_controls.json, written by
    k_curves_controls.py, which reproduces the thesis values at k = 1 and 24.
(b) Two dimensions: accuracy above chance of assigning one network to its
    (N, a) among candidates of equal predicted resistance, for R alone and with
    C1, C4, C16 or P added; the dashed line is the threshold fixed before the
    run. Filled markers: the thesis run (630 networks); open markers: the
    pre-registered confirmatory run on 1260 fresh networks (Apuana job 17599,
    data/pairwise2d_confirm_summary.json), with its Holm-adjusted McNemar p-values
    for the two tested sets. From cluster/pairwise2d_results/analysis.json,
    robustness.txt and data/pairwise2d_confirm_summary.json.
"""
import os, re, json
import numpy as np
import matplotlib.pyplot as plt
from style import COLAB, TWO_COL, CAT, MARK, INK2, GRAY, DATADIR, grid, panel, save

fig, (axa, axb) = plt.subplots(1, 2, figsize=(TWO_COL, 1.75), gridspec_kw=dict(wspace=0.32))

# (a) thickness signal against k
kc = json.load(open(os.path.join(DATADIR, "k_curves_controls.json")))
ks = kc["ks"]
series = [("slab_grid_3d", "slab, 3D shells (119 cells)"),
          ("slab_inplane", "slab, in-plane shells (118)"),
          ("flat_inplane", "flat sticks, in-plane shells (118)")]
for (key, lab), col, m in zip(series, CAT, MARK):
    y = [kc[key]["curve+scalar"][str(k)][2] for k in ks]
    axa.plot(ks, y, color=col, marker=m, ms=3, lw=1.0, mec="white", mew=0.4, label=lab)
    print(f"{key}: curve+R R2(log d) k=1 {y[0]:.2f}, k=24 {y[-1]:.2f}")
y = [kc["slab_grid_3d"]["scalar"][str(k)][2] for k in ks]
axa.plot(ks, y, color=GRAY, ls="--", lw=1.0, marker="o", ms=2.5, mfc="white", label=r"$R$ alone (slab)")
axa.axhline(0, color=INK2, lw=0.5)
axa.set_xscale("log"); axa.set_xticks(ks); axa.set_xticklabels([str(k) for k in ks]); axa.minorticks_off()
axa.set_ylim(-0.08, 1.0)
axa.set_xlabel(r"realizations averaged, $k$")
axa.set_ylabel(r"out-of-sample $R^2$ for $\log_{10} d$")
axa.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), handlelength=1.8, labelspacing=0.2, fontsize=6.5)
grid(axa); panel(axa, "a", x=-0.17)

# (b) two-dimensional classification
RES = os.path.join(COLAB, "cluster", "pairwise2d_results")
an = json.load(open(os.path.join(RES, "analysis.json")))
rob = open(os.path.join(RES, "robustness.txt")).read()
INFO = ["S", "S+C1", "S+C4", "S+C16", "S+P"]
se = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(S\S*)\s+binomial SE ~ ([0-9.]+)", rob, re.M)}
mc = {m.group(1): m.group(4) for m in re.finditer(
    r"^\s+(S\S*)\s+binomial SE.*gained (\d+), lost (\d+), McNemar p = (\S+)", rob, re.M)}
ex = {i: an["part_A"]["summary"][i]["k=1"]["mean_excess_over_chance"] for i in INFO}
LAB = {"S": r"$R$", "S+C1": r"$+C_1$", "S+C4": r"$+C_4$", "S+C16": r"$+C_{16}$", "S+P": r"$+P$"}
shade = {"S": GRAY, "S+C1": "#8fb8ea", "S+C4": "#5b97df", "S+C16": "#2a78d6", "S+P": "#0d3b75"}
MK = {"S": "o", "S+C1": "s", "S+C4": "D", "S+C16": "^", "S+P": "v"}
conf = json.load(open(os.path.join(DATADIR, "pairwise2d_confirm_summary.json")))
cex = conf["secondary"]["excess_over_chance"]; cpr = conf["primary"]
SE_CONF = 0.014        # binomial SE on the confirmatory primary split (see claims-map)
axb.axhline(0.20, color=INK2, ls="--", lw=0.7)
axb.text(-0.45, 0.2025, "pre-set criterion", ha="left", va="bottom", fontsize=6.5, color=INK2)
for k, i in enumerate(INFO):
    axb.errorbar(k - 0.13, ex[i], yerr=se[i], fmt=MK[i], ms=4.2, color=shade[i], mec="white", mew=0.5,
                 ecolor=shade[i], elinewidth=0.9, capsize=2)
    axb.errorbar(k + 0.13, cex[i], yerr=SE_CONF, fmt=MK[i], ms=4.2, mfc="white", color=shade[i], mew=0.9,
                 ecolor=shade[i], elinewidth=0.9, capsize=2)
    if i in cpr:
        pv = cpr[i]["p_holm"]
        axb.text(k + 0.13, cex[i] + SE_CONF + 0.004, f"$p={pv:.2g}$".replace("e-0", r"\times10^{-") + ("}" if "e-0" in f"{pv:.2g}" else ""),
                 ha="center", va="bottom", fontsize=6.3, color=INK2)
    print(f"{i}: thesis {ex[i]:.3f} (SE {se[i]:.3f}); confirmatory {cex[i]:.3f}")
from matplotlib.lines import Line2D  # noqa: E402
axb.legend(handles=[Line2D([], [], color=INK2, marker="o", ls="", ms=4, mec="white", label="pairwise test (630 networks)"),
                    Line2D([], [], color=INK2, marker="o", ls="", ms=4, mfc="white", label="pre-registered repeat (1260)")],
           loc="lower right", fontsize=6.3, handletextpad=0.3)
axb.set_xticks(range(len(INFO))); axb.set_xticklabels([LAB[i] for i in INFO])
axb.set_xlim(-0.5, len(INFO) - 0.5); axb.set_ylim(0.08, 0.275)
axb.set_ylabel("accuracy above chance")
grid(axb, "y"); panel(axb, "b", x=-0.17)

save(fig, "fig4_pairwise.pdf")
