"""Node potentials of the three-node chain before, with and after loops (thesis Appendix B).

Solves the chain T1 of utils/test_networks.py (source, one internal node, drain) with the
pipeline's own solver, three times for each of two cases:
  * self-loops: one at the source, two at the internal node and one at the drain
    (case T1e of notebooks/test_self_loops.ipynb, loop nodes [1, 0, 0, 2]);
  * a dangling loop: two extra nodes joined to the internal node in a ring
    (utils/node_loop_test_runner.run_node_loop_test, host 0, two loop nodes, "linear").
Each panel is written as its own PNG; the thesis sets them as subfigures.
Usage: python loop_potential_figures.py [outdir]   (default: ../figures)"""
import os, sys
from io import StringIO
from contextlib import redirect_stdout
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
from test_networks import intersections_t1, graph_t1, seg_t1, src_t1, drn_t1, k_test  # noqa: E402
from self_loop_utils import run_self_loop_test  # noqa: E402
from node_loop_test_runner import run_node_loop_test  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
NAME = {src_t1: "source", 0: "internal", drn_t1: "drain"}
CMAP = plt.get_cmap("RdYlGn")


def panel(path, inters, seg, volts, ring=()):
    """Chain drawn left to right; node color and label give the potential V (unit drop)."""
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(0.18, 1.0); ax.axis("off")
    for x0, x1, col, lab in ((-0.08, 0.1, "tab:blue", "source strip"), (0.9, 1.08, "tab:red", "drain strip")):
        ax.axvspan(x0, x1, color=col, alpha=0.08, lw=0, zorder=0)  # the electrode strips, x < 0.1 and x > 0.9
        ax.text((x0 + x1) / 2, 0.19, lab, ha="center", va="bottom", fontsize=8, color="0.35")
    loops = {}
    for n1, n2 in seg.values():
        if n1 == n2:
            loops[n1] = loops.get(n1, 0) + 1
        else:
            ring_edge = n1 in ring or n2 in ring
            ax.plot([inters[n1][0], inters[n2][0]], [inters[n1][1], inters[n2][1]],
                    color="tab:orange" if ring_edge else "k", lw=2, zorder=2)
    for node, cnt in loops.items():
        x, y = inters[node]
        for j in range(cnt):  # nested arcs above the node, one per self-loop
            h = 0.16 + 0.07 * j
            ax.add_patch(Arc((x, y + h / 2), 0.09 + 0.03 * j, h, theta1=-60, theta2=240,
                             color="tab:orange", lw=1.8, zorder=1))
        ax.text(x, y + 0.18 + 0.07 * cnt, f"{cnt} self-loop" + ("s" if cnt > 1 else ""),
                fontsize=8, color="tab:orange", ha="center", va="bottom")
    for i, (x, y) in enumerate(inters):
        ax.scatter(x, y, s=300, color=CMAP(volts[i]), edgecolors="k", lw=1.2, zorder=5)
        lab = NAME.get(i, "loop node")
        below = i not in ring
        ax.text(x, y - 0.13 if below else y + 0.11, f"{lab}\n$V$ = {volts[i]:.3f}",
                ha="center", va="top" if below else "bottom", fontsize=8.5, linespacing=1.2)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(path, dpi=170, metadata={"Software": None}); plt.close(fig)


with redirect_stdout(StringIO()):
    S = run_self_loop_test(intersections_t1, graph_t1, seg_t1, src_t1, drn_t1, k_test, [1, 0, 0, 2], label="T1e")
    L, ring = run_node_loop_test(intersections_t1, graph_t1, seg_t1, src_t1, drn_t1, k_test,
                                 host_node=0, n_loop_nodes=2, topology="linear", label="dangling")
for case in ("baseline", "looped", "cleaned"):
    assert abs(S[case]["impedance"] - S["baseline"]["impedance"]) < 1e-6
    assert abs(L[case]["impedance"] - L["baseline"]["impedance"]) < 1e-6
assert np.allclose(S["looped"]["voltages"], S["baseline"]["voltages"])
pos = [[0.5, 0.5], [0.05, 0.5], [0.95, 0.5]]  # drawn positions of the chain (x as in test_networks)
for tag, case in (("before", "baseline"), ("with", "looped"), ("after", "cleaned")):
    panel(os.path.join(OUT, f"voltage_{tag}_selfloop.png"), pos, S[case]["seg"], S[case]["voltages"])
assert len(L["looped"]["intersections"]) == 5 and list(ring) == [3, 4]
pos_loop = pos + [[0.40, 0.76], [0.60, 0.76]]  # drawn positions of the two loop nodes, above the internal node
for tag, case, inters, rg in (("baseline", "baseline", pos, ()),
                             ("withloops", "looped", pos_loop, tuple(ring)),
                             ("afterremoval", "cleaned", pos, ())):
    panel(os.path.join(OUT, f"voltage_nodeloop_{tag}.png"), inters, L[case]["seg"], L[case]["voltages"], ring=rg)
print("R_eff:", S["baseline"]["impedance"], "potentials:", np.round(S["baseline"]["voltages"], 3).tolist(),
      "dangling-loop potentials:", np.round(L["looped"]["voltages"], 3).tolist())
print(f"wrote six panels to {os.path.normpath(OUT)}")
