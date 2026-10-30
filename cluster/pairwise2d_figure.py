"""E7 figure and table, built only from pairwise2d_results/analysis.json (pre-set analysis) and
pairwise2d_results/robustness.txt (post-hoc checks). Writes
  pairwise2d_results/pairwise2d_classification_a.png, pairwise2d_classification_b.png
  pairwise2d_results/pairwise2d_table.tex
(a) mean over the 7 targets of (accuracy - chance), one network (k = 1), per information set, with the
    binomial standard error of robustness.txt and the pre-set threshold 0.20; the McNemar p-values
    against S (post hoc) are written above the points.
(b) accuracy per target, k = 1, with chance marked."""
import os, re, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "pairwise2d_results")
an = json.load(open(os.path.join(RES, "analysis.json")))
rob = open(os.path.join(RES, "robustness.txt")).read()
A, B, V = an["part_A"], an["part_B"], an["verdicts"]
INFO = ["S", "S+C1", "S+C4", "S+C16", "S+P"]
TEX = {"S": "$S$", "S+C1": "$S{+}C_1$", "S+C4": "$S{+}C_4$", "S+C16": "$S{+}C_{16}$", "S+P": "$S{+}P$"}
se = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s+(S\S*)\s+binomial SE ~ ([0-9.]+)", rob, re.M)}
mc = {m.group(1): (int(m.group(2)), int(m.group(3)), m.group(4))
      for m in re.finditer(r"^\s+(S\S*)\s+binomial SE.*gained (\d+), lost (\d+), McNemar p = (\S+)", rob, re.M)}
assert set(se) == set(INFO) and set(mc) == set(INFO[1:]), (se, mc)

cm = plt.get_cmap("plasma")
COL = {"S": "#7a7a7a", "S+C1": cm(0.0), "S+C4": cm(0.25), "S+C16": cm(0.5), "S+P": cm(0.72)}
MK = {"S": "o", "S+C1": "s", "S+C4": "D", "S+C16": "^", "S+P": "v"}
LAB = {"S": r"$R_{\mathrm{eff}}$", "S+C1": "$+C_1$", "S+C4": "$+C_4$", "S+C16": "$+C_{16}$", "S+P": "$+P$"}
ex = {i: A["summary"][i]["k=1"]["mean_excess_over_chance"] for i in INFO}
targets = sorted(A["targets"], key=float)

plt.rcParams.update({"font.size": 10.5})
# one file per panel; the panel titles go to the LaTeX subcaptions
fig1, ax1 = plt.subplots(figsize=(5.2, 4.3))
fig2, ax2 = plt.subplots(figsize=(5.2, 4.3))
x = np.arange(len(INFO))
ax1.axhline(0.20, color="0.25", ls="--", lw=1, zorder=1)
ax1.text(-0.42, 0.203, "pre-set threshold", ha="left", va="bottom", fontsize=8, color="0.25")
ax1.axhline(ex["S"], color="0.6", lw=1, zorder=1)
for k, i in enumerate(INFO):
    ax1.errorbar(k, ex[i], yerr=se[i], fmt=MK[i], ms=7, color=COL[i], mec="white", mew=1.0, ecolor=COL[i], elinewidth=1.4, capsize=3, zorder=3)
    if i in mc:
        ax1.text(k, ex[i] + se[i] + 0.004, f"$p$={mc[i][2]}", ha="center", va="bottom", fontsize=7.5, color="0.3")
ax1.set_xticks(x); ax1.set_xticklabels([LAB[i] for i in INFO])
ax1.set_xlim(-0.5, len(INFO) - 0.5); ax1.set_ylim(0.08, 0.26)
ax1.set_ylabel("accuracy $-$ chance, mean over targets")

off = np.linspace(-0.28, 0.28, len(INFO))
for j, t in enumerate(targets):
    ch = A["targets"][t]["chance"]
    ax2.plot([j - 0.38, j + 0.38], [ch, ch], color="0.25", lw=1.2, zorder=1)
for k, i in enumerate(INFO):
    ys = [A["targets"][t]["accuracy"][i]["k=1"] for t in targets]
    ax2.scatter(np.arange(len(targets)) + off[k], ys, marker=MK[i], s=34, color=COL[i], edgecolor="white", linewidth=0.8, zorder=3, label=LAB[i])
ax2.plot([], [], color="0.25", lw=1.2, label="chance")
ax2.set_xticks(np.arange(len(targets)))
ax2.set_xticklabels([f"{float(t):g}\n({len(A['targets'][t]['candidates'])})" for t in targets])
ax2.set_xlabel(r"target $R_{\mathrm{eff}}$ ($\Omega$) (number of candidates)")
ax2.set_ylabel("classification accuracy")
ax2.set_ylim(0.15, 0.85)
ax2.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
for ax in (ax1, ax2):
    ax.grid(axis="y", color="0.9", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
pngs = []
for fig, tag in ((fig1, "a"), (fig2, "b")):
    fig.tight_layout()
    pngs.append(os.path.join(RES, f"pairwise2d_classification_{tag}.png"))
    fig.savefig(pngs[-1], dpi=200, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- table
f3 = lambda v: f"{v:.3f}"
L = []
L.append(r"\begin{tabular}{rrr" + "r" * len(INFO) + "}")
L.append(r"\toprule")
L.append(r"Target ($\Omega$) & $C$ & Chance & " + " & ".join(TEX[i] for i in INFO) + r" \\")
L.append(r"\midrule")
for t in targets:
    r = A["targets"][t]
    L.append(f"{float(t):g} & {len(r['candidates'])} & {r['chance']:.2f} & " + " & ".join(f"{r['accuracy'][i]['k=1']:.2f}" for i in INFO) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{3}{l}{Mean accuracy $-$ chance} & " + " & ".join(f3(ex[i]) for i in INFO) + r" \\")
L.append(r"\multicolumn{3}{l}{Difference from $S$} & --- & " + " & ".join(f3(ex[i] - ex["S"]) for i in INFO[1:]) + r" \\")
L.append(r"\multicolumn{3}{l}{Gained\,/\,lost against $S$\textsuperscript{a}} & --- & " + " & ".join(f"{mc[i][0]}/{mc[i][1]}" for i in INFO[1:]) + r" \\")
L.append(r"\multicolumn{3}{l}{McNemar $p$\textsuperscript{a}} & --- & " + " & ".join(mc[i][2] for i in INFO[1:]) + r" \\")
L.append(r"\midrule")
r2 = {i: B["results"][i]["k=1"]["r2_loga"] for i in INFO}
L.append(r"\multicolumn{3}{l}{$R^2$ for $\log a$, " + f"{B['cells_fixed']}" + r" cells} & " + " & ".join(f3(r2[i]) for i in INFO) + r" \\")
L.append(r"\multicolumn{3}{l}{Gain in $R^2$ over $S$} & --- & " + " & ".join(f3(V[i]["B_delta_r2_loga"]) for i in INFO[1:]) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{3}{l}{Pre-set verdict} & --- & " + " & ".join(V[i]["verdict"] for i in INFO[1:]) + r" \\")
L.append(r"\bottomrule")
L.append(r"\end{tabular}")
tex = os.path.join(RES, "pairwise2d_table.tex")
open(tex, "w").write("\n".join(L) + "\n")
print(*pngs, sep="\n"); print(tex); print("\n".join(L))
