"""
Electrical identification of the percolation backbone on the near-threshold
sweep: red bonds and cut vertices read off the effective-resistance matrix,
checked against the current field and against the graph topology.

Redraws exactly the networks of run_critical_sweep_task.py (same plan, same
stable_seed(N, a, 'critical'), same draw order) and refuses to continue unless
each regenerated R_eff reproduces the value stored in critical_sweep_results/.
On each network it computes the full pairwise effective-resistance matrix of
the pruned, merged graph and records

  * the current-carrying backbone: edges with |I_e| > 0. The degree-based
    pruning removes dangling chains but not dangling blobs (a loop hung on
    the backbone by a single edge has every node at degree >= 2), so the
    pruned graph is larger than the backbone; the dead fraction is recorded;
  * red bonds, three ways that must agree:
      - electrical: bridges by the cut-edge criterion R_eff,uv = r_uv
        (Bennett et al., restated for weighted edges) that lie on every
        source-drain path, R_eff,su + r_uv + R_eff,vd = R_eff,sd;
      - current:    edges carrying the whole terminal current, |I_e| = I_total
        (the percolation definition of a singly connected bond);
      - topological: bridges of the pruned graph (Tarjan) whose removal
        separates the source from the drain;
    bridges that hang a dead blob on the backbone are bridges but not red
    bonds, and are counted separately;
  * cut vertices of the source-drain backbone, two ways that must agree:
      - electrical: vertices b with R_eff,sb + R_eff,bd = R_eff,sd (tight
        triangle inequality), i.e. a one-to-many measurement from each
        terminal;
      - topological: articulation points whose removal separates s from d;
  * the bypass ratio R_eff,uv / r_uv of every edge (1 for a red bond, smaller
    the more the rest of the network shunts the edge), kept as a histogram;
  * the series resistance of the red bonds and its share of R_eff, the wire
    length they represent, and their positions along the transport axis.

One array task per (a, N a^2) point; each writes backbone_results/point_NN.json.

Usage:
  python run_backbone_task.py --plan       # print the 33 points and exit
  python run_backbone_task.py <task_id>    # run one point
  python run_backbone_task.py <task_id> --max-trials 3   # quick local check
"""
import sys
import os
import json
import time
import warnings

import numpy as np
import scipy.linalg

HERE = os.path.dirname(os.path.abspath(__file__))
UTILS_PATH = os.path.join(HERE, "..", "utils")
RESULTS_DIR = os.path.join(HERE, "backbone_results")
CRITICAL_DIR = os.path.join(HERE, "critical_sweep_results")

sys.path.insert(0, UTILS_PATH)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402
from run_critical_sweep_task import build_plan, R0, N_TRIALS, MAX_ATTEMPTS, DRAW_BUDGET  # noqa: E402

REL_TOL = 1e-8          # equality tolerance for R_eff,uv = r_uv and the tight triangle
ZERO_TOL = 1e-10        # |I_e| below this fraction of I_total is no current
REPRO_TOL = 1e-9        # relative agreement demanded with the stored R_eff
HIST_BINS = 100         # bypass-ratio histogram on [0, 1]


def pairwise_resistance(L, ground):
    """
    Full matrix R[u, v] = R_eff,uv of a connected weighted Laplacian L, from the
    inverse of L with one node grounded (a Cholesky solve rather than pinv: the
    grounded block is positive definite). R does not depend on the choice of
    ground.
    """
    n = L.shape[0]
    keep = np.array([i for i in range(n) if i != ground])
    Lg = L[np.ix_(keep, keep)]
    cf = scipy.linalg.cho_factor(Lg, lower=True, check_finite=False)
    Ginv = scipy.linalg.cho_solve(cf, np.eye(n - 1), check_finite=False)
    Gfull = np.zeros((n, n))
    Gfull[np.ix_(keep, keep)] = Ginv
    diag = np.diag(Gfull)
    return diag[:, None] + diag[None, :] - 2.0 * Gfull


def adjacency(n, u_arr, v_arr):
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(zip(u_arr.tolist(), v_arr.tolist())):
        adj[u].append((v, i))
        adj[v].append((u, i))
    return adj


def bridges_and_articulation_points(n, adj):
    """
    Tarjan's lowlink DFS, iterative. Returns (is_bridge over edges, is_ap over
    nodes). Parallel edges have already been collapsed by the merge, so an edge
    is skipped on the way back only by its own index.
    """
    disc = [-1] * n
    low = [0] * n
    n_edges = sum(len(a) for a in adj) // 2
    is_bridge = np.zeros(n_edges, dtype=bool)
    is_ap = np.zeros(n, dtype=bool)
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t; t += 1
        root_children = 0
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            v, pe, it = stack[-1]
            advanced = False
            for w, e in it:
                if e == pe:
                    continue
                if disc[w] == -1:
                    disc[w] = low[w] = t; t += 1
                    stack.append((w, e, iter(adj[w])))
                    advanced = True
                    break
                low[v] = min(low[v], disc[w])
            if advanced:
                continue
            stack.pop()
            if stack:
                p, _, _ = stack[-1]
                low[p] = min(low[p], low[v])
                if low[v] > disc[p]:
                    is_bridge[pe] = True
                if p == root:
                    root_children += 1
                elif low[v] >= disc[p]:
                    is_ap[p] = True
        if root_children > 1:
            is_ap[root] = True
    return is_bridge, is_ap


def connected(n, adj, s, d, skip_edge=-1, skip_node=-1):
    """Is there a path from s to d avoiding one edge index and/or one node?"""
    seen = np.zeros(n, dtype=bool)
    seen[s] = True
    stack = [s]
    while stack:
        v = stack.pop()
        if v == d:
            return True
        for w, e in adj[v]:
            if e == skip_edge or w == skip_node or seen[w]:
                continue
            seen[w] = True
            stack.append(w)
    return False


def measure_backbone(res, rl):
    """All backbone observables of one network; returns a JSON-ready dict."""
    stn = res["merged_segment_to_nodes"]
    s, d = res["merged_source_node"], res["merged_drain_node"]
    L = rl["L"]
    n = L.shape[0]
    r_edge = np.asarray(rl["edge_resistances"], dtype=float)
    lengths = np.asarray(rl["edge_lengths"], dtype=float)
    I_edge = np.asarray(rl["currents"], dtype=float)
    I_total = float(rl["I_total"])
    R_sd = float(rl["impedance"])
    coords = np.asarray(res["merged_intersections"], dtype=float)

    edges = sorted(stn.items())
    e_idx = np.array([e for e, _ in edges])
    u_arr = np.array([uv[0] for _, uv in edges])
    v_arr = np.array([uv[1] for _, uv in edges])
    mult = np.array([res["edge_conductance"].get(e, 1) for e, _ in edges])

    R = pairwise_resistance(L, ground=s)
    if abs(R[s, d] - R_sd) > REL_TOL * R_sd:
        raise RuntimeError(f"pairwise matrix disagrees with the terminal solve: "
                           f"{R[s, d]} vs {R_sd}")

    # --- current-carrying backbone ------------------------------------------
    I_abs = np.abs(I_edge[e_idx])
    alive = I_abs > ZERO_TOL * I_total
    alive_nodes = np.zeros(n, dtype=bool)
    alive_nodes[u_arr[alive]] = True
    alive_nodes[v_arr[alive]] = True

    # --- red bonds -------------------------------------------------------
    R_uv = R[u_arr, v_arr]
    bypass = R_uv / r_edge[e_idx]                      # in (0, 1]
    bridge_elec = bypass >= 1.0 - REL_TOL
    # a bridge separates s from d iff the path s-u-v-d (either way round) is tight
    through = np.minimum(np.abs(R[s, u_arr] + r_edge[e_idx] + R[v_arr, d] - R_sd),
                         np.abs(R[s, v_arr] + r_edge[e_idx] + R[u_arr, d] - R_sd))
    red_elec = bridge_elec & (through <= REL_TOL * R_sd)
    red_curr = I_abs >= (1.0 - REL_TOL) * I_total
    adj = adjacency(n, u_arr, v_arr)
    bridge_topo, ap_topo = bridges_and_articulation_points(n, adj)
    red_topo = np.zeros(len(edges), dtype=bool)
    for i in np.flatnonzero(bridge_topo):
        red_topo[i] = not connected(n, adj, s, d, skip_edge=int(i))

    red = red_elec
    red_R = float(np.sum(r_edge[e_idx][red]))
    red_len = float(np.sum(lengths[e_idx][red] * mult[red]))   # wire length, bundles counted per wire
    tot_len = float(np.sum(lengths[e_idx] * mult))
    alive_len = float(np.sum(lengths[e_idx][alive] * mult[alive]))
    red_mid_x = ((coords[u_arr[red], 0] + coords[v_arr[red], 0]) / 2.0).tolist()

    # --- cut vertices of the s-d backbone ----------------------------------
    tri = R[s, :] + R[:, d]                            # = R_sd exactly on a cut vertex
    cut_elec = np.abs(tri - R_sd) <= REL_TOL * R_sd
    cut_elec[[s, d]] = False
    cut_topo = np.zeros(n, dtype=bool)
    for b in np.flatnonzero(ap_topo):
        if b in (s, d):
            continue
        cut_topo[b] = not connected(n, adj, s, d, skip_node=int(b))

    # bridges can round to 1 + 1e-16; clip so they land in the last bin instead of
    # falling outside the range (the cluster run of 2026-09-17 predates this clip)
    hist, _ = np.histogram(np.minimum(bypass, 1.0), bins=HIST_BINS, range=(0.0, 1.0))
    non_red = bypass[~bridge_elec]

    return {
        "n_nodes": int(n), "n_edges": int(len(edges)), "R_eff": R_sd,
        "n_backbone_edges": int(alive.sum()), "n_backbone_nodes": int(alive_nodes.sum()),
        "dead_edge_fraction": float(1.0 - alive.mean()),
        "dead_node_fraction": float(1.0 - alive_nodes.mean()),
        "dead_length_fraction": (tot_len - alive_len) / tot_len if tot_len > 0 else 0.0,
        "n_bridges_elec": int(bridge_elec.sum()), "n_bridges_topo": int(bridge_topo.sum()),
        "bridge_elec_vs_topo_mismatch": int(np.sum(bridge_elec != bridge_topo)),
        "n_red_elec": int(red_elec.sum()), "n_red_curr": int(red_curr.sum()),
        "n_red_topo": int(red_topo.sum()),
        "red_elec_vs_curr_mismatch": int(np.sum(red_elec != red_curr)),
        "red_elec_vs_topo_mismatch": int(np.sum(red_elec != red_topo)),
        "n_red_bundles": int(np.sum(mult[red] > 1)),
        "red_R_series": red_R, "red_R_fraction": red_R / R_sd,
        "red_length": red_len, "total_length": tot_len, "backbone_length": alive_len,
        "red_length_fraction": red_len / tot_len if tot_len > 0 else 0.0,
        "red_backbone_length_fraction": red_len / alive_len if alive_len > 0 else 0.0,
        "red_mid_x": red_mid_x,
        "n_cut_elec": int(cut_elec.sum()), "n_cut_topo": int(cut_topo.sum()),
        "cut_elec_vs_topo_mismatch": int(np.sum(cut_elec != cut_topo)),
        "cut_x": coords[cut_elec, 0].tolist(),
        "bypass_hist": hist.tolist(),
        "bypass_max_non_bridge": float(non_red.max()) if non_red.size else None,
        "bypass_mean": float(bypass.mean()),
        "source_x": float(coords[s, 0]), "drain_x": float(coords[d, 0]),
    }


def run_point(task_id, max_trials=None):
    plan = build_plan()
    if task_id >= len(plan):
        print(f"task_id {task_id} out of range (plan has {len(plan)} points)")
        return
    pt = plan[task_id]
    N, a = pt["N"], pt["a"]
    n_trials = N_TRIALS if max_trials is None else min(max_trials, N_TRIALS)

    ref_path = os.path.join(CRITICAL_DIR, f"point_{task_id:02d}.json")
    with open(ref_path) as f:
        ref = json.load(f)
    if (ref["N"], ref["a"]) != (N, a):
        raise RuntimeError(f"{ref_path} is for (N={ref['N']}, a={ref['a']}), plan says ({N}, {a})")
    ref_R = ref["R_eff_values"]

    print(f"[task {task_id}] a={a}  N={N}  N a^2={pt['x']:.2f}  box={1/a:.1f} stick lengths across; "
          f"{n_trials} realizations, checked against {os.path.basename(ref_path)}", flush=True)

    np.random.seed(stable_seed(N, round(a, 3), "critical"))
    nets = []
    total_attempts = 0
    t_start = time.time()
    while len(nets) < n_trials and total_attempts < DRAW_BUDGET:
        res, attempts = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS)
        total_attempts += attempts
        if res is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rl = distance_dependent_impedance(res, R0)
        k = len(nets)
        R_eff = float(rl["impedance"])
        if k < len(ref_R) and abs(R_eff - ref_R[k]) > REPRO_TOL * abs(ref_R[k]):
            raise RuntimeError(f"realization {k} does not reproduce the stored network: "
                               f"R_eff={R_eff!r} vs stored {ref_R[k]!r}")
        t0 = time.time()
        m = measure_backbone(res, rl)
        m["realization"] = k
        m["seconds_measure"] = time.time() - t0
        nets.append(m)
        print(f"  {k+1:2d}/{n_trials}  R_eff={R_eff:9.2f}  nodes={m['n_nodes']:5d}  "
              f"dead={m['dead_edge_fraction']:.2f}  bridges={m['n_bridges_elec']:3d}  "
              f"red={m['n_red_elec']:3d} (curr {m['n_red_curr']:3d}, topo {m['n_red_topo']:3d})  "
              f"red R frac={m['red_R_fraction']:.3f}  cut={m['n_cut_elec']:3d}/{m['n_cut_topo']:3d}  "
              f"{m['seconds_measure']:5.1f}s  {time.time()-t_start:6.0f}s", flush=True)

    out = {
        "task_id": task_id, "a": a, "N": N, "x": pt["x"], "x_target": pt["x_target"],
        "R0": R0, "n": len(nets), "n_target": n_trials,
        "total_draws": total_attempts,
        "rel_tol": REL_TOL, "hist_bins": HIST_BINS,
        "networks": nets,
        "seconds": time.time() - t_start,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    mism = sum(m["bridge_elec_vs_topo_mismatch"] + m["red_elec_vs_curr_mismatch"]
               + m["red_elec_vs_topo_mismatch"] + m["cut_elec_vs_topo_mismatch"] for m in nets)
    print(f"[task {task_id}] done: {len(nets)} realizations, mean red bonds "
          f"{np.mean([m['n_red_elec'] for m in nets]):.2f}, mean red R fraction "
          f"{np.mean([m['red_R_fraction'] for m in nets]):.3f}, mean dead edge fraction "
          f"{np.mean([m['dead_edge_fraction'] for m in nets]):.3f}, criterion mismatches {mism}, "
          f"{out['seconds']:.0f}s -> {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    if args[0] == "--plan":
        for i, pt in enumerate(build_plan()):
            print(f"{i:2d}  a={pt['a']:.2f}  N={pt['N']:5d}  N a^2={pt['x']:6.2f}")
        sys.exit(0)
    max_trials = None
    if "--max-trials" in args:
        max_trials = int(args[args.index("--max-trials") + 1])
    run_point(int(args[0]), max_trials=max_trials)
