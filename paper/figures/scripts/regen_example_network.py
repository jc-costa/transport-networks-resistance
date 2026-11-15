"""Rebuild the near-threshold example network of the thesis (N = 30, a = 0.4,
R_eff = 1092 Ohm, thesis Fig. "Two percolating networks") from its stored stick
geometry, and mark its red bonds and cut vertices with the resistance criteria
of ColabProject/cluster/run_backbone_task.py (imported, not modified).

No new simulation: the sticks are the stored ones; only the solve and the
backbone measurement are repeated. The script refuses to continue unless the
rebuilt R_eff matches the thesis value, and it checks that the three
definitions of a red bond agree, as the thesis does on 990 networks.

Usage: python regen_example_network.py <ColabProject dir> <out.json>
"""
import os, sys, io, json, glob, contextlib, warnings
import numpy as np

COLAB, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(COLAB, "utils"))
sys.path.insert(0, os.path.join(COLAB, "cluster"))
from pipeline_utils import run_full_pipeline, distance_dependent_impedance  # noqa: E402
from run_backbone_task import measure_backbone  # noqa: E402

R0 = 1000.0
THESIS_R = 1092.0   # rounded value printed in the thesis caption

best = None
for path in sorted(glob.glob(os.path.join(COLAB, "experiments", "exp_*", "network_N30_a0.4.json"))):
    net = json.load(open(path))
    p1 = [tuple(l["start_point"]) for l in net["lines"]]
    p2 = [tuple(l["end_point"]) for l in net["lines"]]
    ang = [l["angle_rad"] for l in net["lines"]]
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_full_pipeline(p1, p2, ang, net["N"], net["a"], k=R0, verbose=False)
    if not res.get("has_percolation"):
        continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rl = distance_dependent_impedance(res, R0)
    R = float(rl["impedance"])
    print(f"{os.path.relpath(path, COLAB)}: R_eff = {R:.2f} Ohm")
    if abs(R - THESIS_R) < 0.5 and best is None:
        best = (path, p1, p2, res, rl)

if best is None:
    sys.exit("no stored network reproduces R_eff = 1092 Ohm; stopping")
path, p1, p2, res, rl = best
m = measure_backbone(res, rl)
assert m["red_elec_vs_curr_mismatch"] == 0 and m["red_elec_vs_topo_mismatch"] == 0
assert m["cut_elec_vs_topo_mismatch"] == 0 and m["bridge_elec_vs_topo_mismatch"] == 0

# per-edge flags for the drawing, recomputed exactly as measure_backbone does
stn = res["merged_segment_to_nodes"]; s, d = res["merged_source_node"], res["merged_drain_node"]
edges = sorted(stn.items())
e_idx = np.array([e for e, _ in edges]); u = np.array([uv[0] for _, uv in edges]); v = np.array([uv[1] for _, uv in edges])
r_edge = np.asarray(rl["edge_resistances"], float)[e_idx]
I = np.abs(np.asarray(rl["currents"], float)[e_idx]); It = float(rl["I_total"])
red = I >= (1 - 1e-8) * It
alive = I > 1e-10 * It
coords = np.asarray(res["merged_intersections"], float)

# the merge parks each terminal at its strip centroid; for the drawing, carry the
# flags back to the pre-merge geometry, where every segment keeps its real position
from impedance_calculator import build_merge_node_mapping  # noqa: E402
pre_xy = np.asarray(res["pre_merge_intersections"], float)
mapping, _, _, _ = build_merge_node_mapping(len(pre_xy), res["pre_merge_source_nodes"],
                                            res["pre_merge_drain_nodes"])
flag = {frozenset((int(a), int(b))): (bool(r_), bool(al)) for a, b, r_, al in zip(u, v, red, alive)}
pre_edges = []
for _, (p, q) in sorted(res["pre_merge_segment_to_nodes"].items()):
    key = frozenset((mapping[p], mapping[q]))
    if len(key) == 1 or key not in flag:
        continue          # inside one terminal strip, or pruned away
    pre_edges.append([int(p), int(q), *flag[key]])

out = {
    "source_file": os.path.relpath(path, COLAB), "R_eff_ohm": float(rl["impedance"]),
    "R0_ohm_per_unit_length": R0,
    "sticks": [[list(a), list(b)] for a, b in zip(p1, p2)],
    "nodes": coords.tolist(), "source_node": int(s), "drain_node": int(d),
    "edges": [[int(a), int(b)] for a, b in zip(u, v)],
    "edge_red": red.tolist(), "edge_alive": alive.tolist(),
    "edge_current_fraction": (I / It).tolist(),
    "pre_merge_nodes": pre_xy.tolist(), "pre_merge_edges_red_alive": pre_edges,
    "n_red": m["n_red_elec"], "red_R_fraction": m["red_R_fraction"], "n_cut": m["n_cut_elec"],
}
json.dump(out, open(OUT, "w"))
print(f"using {out['source_file']}: R_eff = {out['R_eff_ohm']:.2f} Ohm, red bonds = {m['n_red_elec']}, "
      f"red share of R_eff = {m['red_R_fraction']:.3f}, cut vertices = {m['n_cut_elec']}, "
      f"criterion mismatches = 0")
print("wrote", OUT)
