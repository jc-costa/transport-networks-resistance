"""
E2b: three follow-ups of the strategies run (PREREGISTRATION_JUNCTION_STRATEGIES_MORE.md), on the SAME films.

The same 80 geometries, networks and junction ratios as run_junction_strategies_task.py (same seeds and the same
draw order, so the films are identical; the analysis joins the two files by geometry, network and film index and
checks the terminal resistance of each film). New readings, from a new random stream:

  1. MORE READINGS. Landing as in the base configuration (footprint 0.02, position error 0.01), designs of
     9 positions x 9, 27, 81 and 243 readings: 81, 243, 729 and 2187 attempts.
  2. CHOSEN SITES. Landing with a wider footprint of 0.05 (position error 0.01), either anywhere on a wire
     ("wide") or only on wire points whose local wire length is at least the median of the film ("chosen"):
     an operator who images the film moves the tip to the nearest well-connected wire. The wide landing is the
     control, so that the gain of choosing is not a gain of a larger footprint. Designs 9x9, 9x27, 9x81.
  3. OTHER ESTIMATORS. For every position the mean AND the median of the successful readings are stored.

Local wire length of a point: the number of wire sample points (spacing 0.004) within 0.05 of it. Nothing of this
depends on the junction resistance or on the currents: it is what an image of the film shows.

Output: <out>/partial.jsonl, one row per geometry; resumable.
Usage:
  python run_junction_strategies_more_task.py --validate
  python run_junction_strategies_more_task.py --parallel 16
  python run_junction_strategies_more_task.py --units S_N100_a0.4 --trials 1 --out /tmp/st2    # quick check
"""
import sys, os, json, time
if "--parallel" in sys.argv:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
import run_junction_pairwise_task as jp      # noqa: E402
import run_junction_strategies_task as st    # noqa: E402

R0 = jp.R0
TAG_FILMS = st.TAG                      # the films are those of the strategies run
TAG_NEW = "junction_strategies_more"    # the new readings
RANDOM_DESIGNS = {"9x9": 9, "9x27": 27, "9x81": 81, "9x243": 243}      # readings per position, 9 positions
WIDE_DESIGNS = {"9x9": 9, "9x27": 27, "9x81": 81}
MODES = {"random": dict(f=0.02, sig=0.01, chosen=False, designs=RANDOM_DESIGNS),
         "wide": dict(f=0.05, sig=0.01, chosen=False, designs=WIDE_DESIGNS),
         "chosen": dict(f=0.05, sig=0.01, chosen=True, designs=WIDE_DESIGNS)}
DENSITY_RADIUS = 0.05


class ChosenTips(st.Tips):
    """Tips with the landing restricted, when asked, to wire points of at least the median local wire length."""

    def __init__(self, net, Lp, m, s, rng):
        super().__init__(net, Lp, m, s, rng)
        dens = self.tree.query_ball_point(self.pts, DENSITY_RADIUS, return_length=True)
        keep = dens >= np.median(dens)
        self.tree_d = cKDTree(self.pts[keep]); self.seg_d = self.seg[keep]; self.tpar_d = self.tpar[keep]
        self.use_chosen = False

    def land(self, pt, f, sig):
        p = pt + self.rng.normal(0.0, sig, 2) if sig > 0 else pt
        if not (st.LO <= p[0] <= st.HI and 0.0 <= p[1] <= 1.0):
            return None
        tree, seg, tpar = (self.tree_d, self.seg_d, self.tpar_d) if self.use_chosen else (self.tree, self.seg, self.tpar)
        dist, i = tree.query(p)
        if dist > f:
            return None
        e, t = seg[i], tpar[i]
        first, second = (self.su[e], self.sv[e]) if t < 0.5 else (self.sv[e], self.su[e])
        for c in (first, second):
            if st.LO <= self.xy[c, 0] <= st.HI:
                return int(self.m[c])
        return None


def sample(tips, kind, r, f, sig):
    mean, med, cnt = [], [], []
    for k in range(9):
        vals = np.array([v for v in (tips.reading(kind, k, 9, f, sig) for _ in range(r)) if v is not None])
        cnt.append(int(len(vals)))
        mean.append(float(f"{vals.mean() / R0:.5g}") if len(vals) else None)
        med.append(float(f"{np.median(vals) / R0:.5g}") if len(vals) else None)
    return {"mean": mean, "median": med, "cnt": cnt}


def run_unit(u, trials=None):
    t0 = time.time()
    N, a = int(u["N"]), float(u["a"]); trials = trials or u["trials"]
    np.random.seed(jp.stable_seed(N, round(a, 3), TAG_FILMS))            # the networks of the strategies run
    nets, att_total, skipped = [], 0, 0
    while len(nets) < trials:
        res, att = jp.generate_percolating_network(N, a, k_uniform=R0, max_attempts=jp.MAX_ATTEMPTS, boundary=jp.BOUNDARY)
        att_total += att
        if res is None:
            break
        k = len(nets) + skipped
        net = jp.build_junction_network(res)
        if len(net["coords"]) > jp.MAX_NODES:
            skipped += 1
            if skipped > 3 * trials:
                break
            continue
        rr = np.random.default_rng(jp.stable_seed(N, round(a, 3), TAG_FILMS, k, "rho"))
        rhos = 10 ** rr.uniform(st.RHO_LOG[0], st.RHO_LOG[1], size=st.N_RHO)
        films = []
        for j, rho in enumerate(rhos):
            Rj = np.full(net["n_cross"], rho * R0 * a)
            Lp, m, s, d = st.merged_inverse(net, Rj)
            Rsd = float(Lp[s, s] + Lp[d, d] - 2 * Lp[s, d])
            rng = np.random.default_rng(jp.stable_seed(N, round(a, 3), TAG_NEW, k, "tips", j))
            tips = ChosenTips(net, Lp, m, s, rng)
            strat = {}
            for kind in ("one", "two", "four"):
                strat[kind] = {}
                for mode, cfg in MODES.items():
                    tips.use_chosen = cfg["chosen"]
                    strat[kind][mode] = {dz: sample(tips, kind, r, cfg["f"], cfg["sig"]) for dz, r in cfg["designs"].items()}
            films.append({"j": j, "rho": float(rho), "S": float(f"{Rsd / R0:.6g}"), "strat": strat})
        nets.append({"net": k, "n_nodes": int(len(net["coords"])), "films": films})
    return {"id": u["id"], "N": N, "a": a, "x": u["x"], "tag": TAG_NEW, "n": len(nets), "skipped_too_large": skipped,
            "attempts_total": att_total, "seconds": time.time() - t0, "networks": nets}


def _job(args):
    u, t = args
    return u["id"], run_unit(u, t)


def validate():
    """The films are those of the strategies run, and the chosen sites are well connected."""
    import run_junction_strategies_task as s2
    u = s2.units()[3]
    row = run_unit(u, 1)
    f = row["networks"][0]["films"][0]
    # the same network and junction ratio as the strategies run: recompute and compare the terminal resistance
    np.random.seed(jp.stable_seed(u["N"], round(u["a"], 3), TAG_FILMS))
    res, _ = jp.generate_percolating_network(u["N"], u["a"], k_uniform=R0, max_attempts=jp.MAX_ATTEMPTS, boundary=jp.BOUNDARY)
    net = jp.build_junction_network(res)
    Rj = np.full(net["n_cross"], f["rho"] * R0 * u["a"])
    ref = jp.solve_two_terminal(net, R0, Rj)["R_eff"] / R0
    print(f"film of {u['id']}: terminal resistance {f['S']:.6g} against the solver {ref:.6g}")
    assert abs(f["S"] - ref) < 1e-4 * ref
    # chosen sites have a higher local wire length than wide landing
    Lp, m, s, d = st.merged_inverse(net, Rj)
    tips = ChosenTips(net, Lp, m, s, np.random.default_rng(0))
    dens = tips.tree.query_ball_point(tips.pts, DENSITY_RADIUS, return_length=True)
    dd = tips.tree_d.query_ball_point(tips.tree_d.data, DENSITY_RADIUS, return_length=True)
    print(f"median local wire length: all points {np.median(dens):.0f}, chosen points {np.median(dd):.0f} (wire points, 0.004 apart)")
    assert np.median(dd) >= np.median(dens)
    cnts = {mode: sum(f["strat"]["one"][mode]["9x9"]["cnt"]) for mode in MODES}
    print("successful readings of 81 attempts, one tip:", cnts)
    print("validate: the films and the chosen sites behave as defined")


def main():
    args = sys.argv[1:]
    if "--validate" in args:
        validate(); return
    ul = st.units()
    if "--units" in args:
        want = args[args.index("--units") + 1].split(",")
        ul = [u for u in ul if u["id"] in want]
    trials = int(args[args.index("--trials") + 1]) if "--trials" in args else None
    out = args[args.index("--out") + 1] if "--out" in args else os.path.join(HERE, "junction_strategies_more_results")
    os.makedirs(out, exist_ok=True); part = os.path.join(out, "partial.jsonl")
    done = st._resume(part)
    todo = sorted([u for u in ul if u["id"] not in done], key=lambda u: -(u["N"] * u["a"]) ** 2)
    print(f"{len(done)} geometries already in {part}; {len(todo)} to run", flush=True)
    nproc = int(args[args.index("--parallel") + 1]) if "--parallel" in args else 1
    with open(part, "a") as fh:
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                for i, (uid, row) in enumerate(pool.imap_unordered(_job, [(u, trials) for u in todo]), 1):
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[{i}/{len(todo)}] {uid} n={row['n']} skipped={row['skipped_too_large']} {row['seconds']:.0f}s", flush=True)
        else:
            for i, u in enumerate(todo, 1):
                row = run_unit(u, trials)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{i}/{len(todo)}] {u['id']} n={row['n']} skipped={row['skipped_too_large']} {row['seconds']:.0f}s", flush=True)
    print(f"done -> {part}")


if __name__ == "__main__":
    main()
