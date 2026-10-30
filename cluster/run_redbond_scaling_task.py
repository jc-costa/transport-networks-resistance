"""
E2 of the thesis revision plan: red bonds AT the percolation threshold, as a function
of the box size, to test the L^{1/nu} (1/nu = 3/4) growth that Sec. 4.12 could only
check below the threshold (N a^2 = 4 and 5) and in three boxes.

Plan: N a^2 in {5.40, 5.64, 5.90} x box 1/a in {5, 6.67, 10, 14.3, 20} stick lengths
(15 points, one SLURM array task each). Each task draws percolating networks under the
clipping rule (generate_percolating_network, the same pipeline as every 2D result) and,
on each, counts the red bonds two exact ways that must agree:
  * current:  edges carrying the whole terminal current, |I_e| >= (1 - 1e-8) I_total;
  * topology: bridges of the pruned graph (Tarjan) whose removal separates s from d.
No pairwise-resistance matrix is needed for the count, so the large boxes stay cheap.
It also records the red bonds' share of R_eff and of the wire length, and the A7 check:
the affine trial potential V(x) = (0.9 - x)/0.8 bounds R_eff from below (Dirichlet's
principle), so R_eff * P_aff >= 1 on every network.

Seeds: stable_seed(N, a, 'redbond'). Output: redbond_results/point_NN.json.
Usage:
  python run_redbond_scaling_task.py --plan
  python run_redbond_scaling_task.py <task_id> [--max-trials K]
Analysis: python redbond_scaling_analysis.py
"""
import sys, os, json, time, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
sys.path.insert(0, HERE)
from pipeline_utils import generate_percolating_network, distance_dependent_impedance, stable_seed  # noqa: E402
from run_backbone_task import adjacency, bridges_and_articulation_points, connected  # noqa: E402

RESULTS_DIR = os.environ.get("REDBOND_OUT", os.path.join(HERE, "redbond_results"))
R0 = 1000.0
N_TRIALS = 200            # percolating realizations wanted per point
MAX_ATTEMPTS = 500        # per generate_percolating_network call
DRAW_BUDGET = 6000        # total draws per point
TIME_BUDGET_S = 5.5 * 3600  # stop cleanly before a 6 h wall limit, keeping what was collected
REL_TOL = 1e-8
A_VALUES = [0.20, 0.15, 0.10, 0.07, 0.05]
X_VALUES = [5.40, 5.64, 5.90]


def build_plan():
    return [{"a": a, "x_target": x, "N": int(round(x / a ** 2)), "x": int(round(x / a ** 2)) * a ** 2}
            for x in X_VALUES for a in A_VALUES]


def _edge_list(stn):
    items = stn.items() if isinstance(stn, dict) else enumerate(stn)
    return [(e, tuple(uv)) for e, uv in items]


def affine_bound_ratio(res, R_eff):
    """R_eff * P_aff on the pre-merge geometry (Dirichlet lower bound: must be >= 1)."""
    pos = np.asarray(res["pre_merge_intersections"], dtype=float)
    S, D = set(res["pre_merge_source_nodes"]), set(res["pre_merge_drain_nodes"])
    V = np.array([1.0 if i in S else 0.0 if i in D else (0.9 - p[0]) / 0.8 for i, p in enumerate(pos)])
    P = 0.0
    for _, (u, v) in _edge_list(res["pre_merge_segment_to_nodes"]):
        L = float(np.linalg.norm(pos[u] - pos[v]))
        if L > 0:
            P += (V[u] - V[v]) ** 2 / (R0 * L)
    return R_eff * P


def measure(res, rl):
    stn = res["merged_segment_to_nodes"]
    s, d = res["merged_source_node"], res["merged_drain_node"]
    n = int(rl["L"].shape[0])
    r_edge = np.asarray(rl["edge_resistances"], float); lengths = np.asarray(rl["edge_lengths"], float)
    I_edge = np.asarray(rl["currents"], float); I_tot = float(rl["I_total"]); R_sd = float(rl["impedance"])
    edges = sorted(stn.items())
    e_idx = np.array([e for e, _ in edges]); u = np.array([uv[0] for _, uv in edges]); v = np.array([uv[1] for _, uv in edges])
    mult = np.array([res["edge_conductance"].get(e, 1) for e, _ in edges])
    red_curr = np.abs(I_edge[e_idx]) >= (1.0 - REL_TOL) * I_tot
    adj = adjacency(n, u, v)
    is_bridge, _ = bridges_and_articulation_points(n, adj)
    red_topo = np.zeros(len(edges), bool)
    for i in np.flatnonzero(is_bridge):
        red_topo[i] = not connected(n, adj, s, d, skip_edge=int(i))
    tot_len = float(np.sum(lengths[e_idx] * mult))
    return {"n_nodes": n, "n_edges": len(edges), "R_eff": R_sd,
            "n_red_curr": int(red_curr.sum()), "n_red_topo": int(red_topo.sum()),
            "red_mismatch": int(np.sum(red_curr != red_topo)), "n_bridges": int(is_bridge.sum()),
            "red_R_fraction": float(np.sum(r_edge[e_idx][red_curr]) / R_sd),
            "red_length_fraction": float(np.sum(lengths[e_idx][red_curr] * mult[red_curr]) / tot_len) if tot_len else 0.0,
            "affine_bound_ratio": affine_bound_ratio(res, R_sd)}


def run_point(task_id, max_trials=None):
    plan = build_plan(); pt = plan[task_id]; N, a = pt["N"], pt["a"]
    n_target = N_TRIALS if max_trials is None else max_trials
    print(f"[task {task_id}] a={a} N={N} N a^2={pt['x']:.3f} box={1/a:.2f} stick lengths", flush=True)
    np.random.seed(stable_seed(N, round(a, 3), "redbond"))
    nets, draws, t0 = [], 0, time.time()
    while len(nets) < n_target and draws < DRAW_BUDGET and time.time() - t0 < TIME_BUDGET_S:
        res, att = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS)
        draws += att
        if res is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rl = distance_dependent_impedance(res, R0)
        m = measure(res, rl); m["realization"] = len(nets); nets.append(m)
        print(f"  {len(nets):3d}/{n_target} R={m['R_eff']:9.2f} nodes={m['n_nodes']:5d} red={m['n_red_curr']:3d} "
              f"(topo {m['n_red_topo']:3d}) redR={m['red_R_fraction']:.3f} bound ratio={m['affine_bound_ratio']:.3f} "
              f"draws={draws:5d} {time.time()-t0:6.0f}s", flush=True)
    out = dict(pt, task_id=task_id, R0=R0, n=len(nets), n_target=n_target, total_draws=draws,
               perc_rate=len(nets) / draws if draws else 0.0, seconds=time.time() - t0, networks=nets)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json")
    json.dump(out, open(path, "w"), indent=1)
    mism = sum(m["red_mismatch"] for m in nets); viol = sum(m["affine_bound_ratio"] < 1 - 1e-9 for m in nets)
    print(f"[task {task_id}] {len(nets)} networks, mean red {np.mean([m['n_red_curr'] for m in nets]) if nets else float('nan'):.2f}, "
          f"current/topology mismatches {mism}, bound violations {viol} -> {path}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    if args[0] == "--plan":
        for i, p in enumerate(build_plan()):
            print(f"{i:2d}  a={p['a']:.2f}  box={1/p['a']:5.2f}  N={p['N']:5d}  N a^2={p['x']:.3f}")
        sys.exit(0)
    mt = int(args[args.index("--max-trials") + 1]) if "--max-trials" in args else None
    run_point(int(args[0]), max_trials=mt)
