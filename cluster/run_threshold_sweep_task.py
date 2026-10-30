"""
E1 of the thesis revision plan: where half of the draws percolate, in boxes of 5 to 20
stick lengths, at steps of 0.2 in N a^2 around the threshold (the near-threshold sweep has
steps of 1 there). Percolation only, no solve: sticks are drawn with the clipping rule
(point_generator.generate_points, the generator of every 2D result), crossings are found by a
vectorized segment test, and a draw percolates when a crossing in the source strip (x < 0.1)
and one in the drain strip (x > 0.9) lie on sticks of the same connected cluster, which is the
breadth-first criterion of run_full_pipeline.

The vectorized crossing test is checked, on the first CHECK_DRAWS draws of every task, against
intersection_utils.identify_intersections (same crossings) and against run_full_pipeline
(same percolation flag); the task stops with an error on any disagreement.

Seeds: stable_seed(N, a, 'threshold'). Output: threshold_results/point_NN.json.
Usage:
  python run_threshold_sweep_task.py --plan
  python run_threshold_sweep_task.py <task_id> [--draws K]
Analysis: python threshold_sweep_analysis.py
"""
import sys, os, json, time, io, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
from point_generator import generate_points  # noqa: E402
from intersection_utils import identify_intersections  # noqa: E402
from pipeline_utils import run_full_pipeline, stable_seed  # noqa: E402

RESULTS_DIR = os.environ.get("THRESHOLD_OUT", os.path.join(HERE, "threshold_results"))
A_VALUES = [0.20, 0.15, 0.10, 0.07, 0.05]
X_VALUES = [4.6, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.6, 7.0]
N_DRAWS = 2000
CHECK_DRAWS = 10
TIME_BUDGET_S = 5.5 * 3600


def build_plan():
    return [{"a": a, "x_target": x, "N": int(round(x / a ** 2)), "x": int(round(x / a ** 2)) * a ** 2}
            for a in A_VALUES for x in X_VALUES]


def crossings(p1, p2, eps=1e-9, block=256):
    P1, P2 = np.asarray(p1, float), np.asarray(p2, float); n = len(P1)
    R = P2 - P1; out_i, out_j, out_x = [], [], []
    for s in range(0, n, block):
        I = np.arange(s, min(s + block, n))
        A, r = P1[I][:, None, :], R[I][:, None, :]
        C, q = P1[None, :, :], R[None, :, :]
        den = r[..., 0] * q[..., 1] - r[..., 1] * q[..., 0]
        w = C - A
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[..., 0] * q[..., 1] - w[..., 1] * q[..., 0]) / den
            u = (w[..., 0] * r[..., 1] - w[..., 1] * r[..., 0]) / den
        ok = (np.abs(den) > 1e-14) & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
        ok &= np.arange(n)[None, :] > I[:, None]
        ii, jj = np.nonzero(ok)
        out_i.append(I[ii]); out_j.append(jj)
        out_x.append(P1[I[ii], 0] + t[ii, jj] * R[I[ii], 0])
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_x)


def percolates(n, ci, cj, cx):
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(ci.tolist(), cj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    src = {find(int(k)) for k in ci[cx < 0.1]} | {find(int(k)) for k in cj[cx < 0.1]}
    drn = {find(int(k)) for k in ci[cx > 0.9]} | {find(int(k)) for k in cj[cx > 0.9]}
    return bool(src & drn)


def run_point(task_id, n_draws=None):
    pt = build_plan()[task_id]; N, a = pt["N"], pt["a"]; n_draws = n_draws or N_DRAWS
    np.random.seed(stable_seed(N, round(a, 3), "threshold"))
    hits, done, t0, checked = 0, 0, time.time(), 0
    while done < n_draws and time.time() - t0 < TIME_BUDGET_S:
        p1, p2, ang = [], [], []
        generate_points(N, p1, p2, ang, a)
        ci, cj, cx = crossings(p1, p2)
        perc = percolates(len(p1), ci, cj, cx)
        if checked < CHECK_DRAWS:
            ref = identify_intersections(p1, p2, [], [], [], [], ang, a, len(p1))
            if len(ref[0]) != len(ci) or set(map(tuple, ref[3])) != set(zip(ci.tolist(), cj.tolist())):
                raise RuntimeError(f"crossing mismatch: routine {len(ref[0])}, vectorized {len(ci)}")
            with contextlib.redirect_stdout(io.StringIO()):
                full = run_full_pipeline(p1, p2, ang, N, a, k=1000.0, verbose=False)
            if bool(full.get("has_percolation") and "merged_segment_to_nodes" in full) != perc:
                raise RuntimeError("percolation flag differs from run_full_pipeline")
            checked += 1
        hits += perc; done += 1
    out = dict(pt, task_id=task_id, draws=done, percolating=hits, rate=hits / done if done else 0.0,
               checked_against_pipeline=checked, seconds=time.time() - t0)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS_DIR, f"point_{task_id:02d}.json"), "w"), indent=1)
    print(f"[task {task_id}] a={a} N={N} N a^2={pt['x']:.3f}: {hits}/{done} percolate ({out['rate']:.3f}), "
          f"{checked} draws checked against the pipeline, {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(2)
    if args[0] == "--plan":
        for i, p in enumerate(build_plan()):
            print(f"{i:2d}  a={p['a']:.2f}  box={1/p['a']:5.2f}  N={p['N']:5d}  N a^2={p['x']:.3f}")
        sys.exit(0)
    nd = int(args[args.index("--draws") + 1]) if "--draws" in args else None
    run_point(int(args[0]), n_draws=nd)
