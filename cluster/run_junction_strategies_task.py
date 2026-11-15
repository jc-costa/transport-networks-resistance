"""
E2: what do different measurement strategies teach about a two-dimensional stick film whose junction
resistance is unknown? (PREREGISTRATION_JUNCTION_STRATEGIES.md)

For every film (geometry, network, junction ratio rho, equal junctions) the script simulates readings of
  one   a fixed electrode (the source strip) and one movable tip: R between the strip and the tip node
  two   two movable tips: R between the two tip nodes
  four  a four-point probe: current in at the outer tips, voltage between the inner tips (the transresistance
        (e_a - e_d)' L+ (e_b - e_c)); contact resistance does not enter
on the graph with the two strips shorted to their electrodes, and the all-pairs profile P of the graph with the
terminals not merged as an upper reference.

A tip lands on a wire: at the nearest point of a retained stick within a footprint f of the target (after a position
error sigma_pos), attached to the nearer of the two crossings that bound that segment (the one inside the film
region), or it misses.
Designs are m positions (shells) x r readings per position: 3x1, 3x3, 9x1, 9x3, 9x9, 9x27 attempts of 3, 9, 9,
27, 81, 243. Landing configurations: base (f = 0.02, sigma_pos = 0.01), f = 0.01, f = 0.04, sigma_pos = 0.03.

What is STORED is the noise-free, tip-resistance-free mean reading per position with the number of successful
readings; the analysis adds the tip contact resistance, the multiplicative reading noise (as the mean of r
independent noises) and the errors in R0, a and N, so that these can be swept without new simulation. The prior
errors are stored as u_c, u_a, u_N, uniform in [-1, 1] (a relative error is margin x u).

Output: <out>/partial.jsonl, one row per geometry; a resubmitted job skips the geometries already there.
Usage:
  python run_junction_strategies_task.py --validate
  python run_junction_strategies_task.py --parallel 16
  python run_junction_strategies_task.py --units S_N100_a0.4 --trials 1 --out /tmp/st      # quick check
"""
import sys, os, json, time
if "--parallel" in sys.argv:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils")); sys.path.insert(0, HERE)
import run_junction_pairwise_task as jp  # noqa: E402

R0 = jp.R0
TAG = "junction_strategies"
N_GEOM, N_TRIALS, N_RHO = 80, 30, 4
RHO_LOG = (float(np.log10(0.03)), 2.0)
DESIGNS = {"3x1": (3, 1), "3x3": (3, 3), "9x1": (9, 1), "9x3": (9, 3), "9x9": (9, 9), "9x27": (9, 27)}
CONFIGS = {"base": (0.02, 0.01), "f01": (0.01, 0.01), "f04": (0.04, 0.01), "p03": (0.02, 0.03)}   # (footprint, position error)
LO, HI = jp.R0 and 0.1, 0.9                       # interior region in x (the strips are x < 0.1 and x > 0.9)
SEPS = {3: [0.2, 0.5, 0.8], 9: [round(0.1 * k, 2) for k in range(1, 10)]}
SPACINGS = {3: [0.06, 0.15, 0.24], 9: [round(0.03 * k, 2) for k in range(1, 10)]}
RETRIES = 10


def units(n_geom=N_GEOM, trials=N_TRIALS, seed=1):
    """Off-grid geometries, as in the transfer run but with their own seed and tag."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n_geom:
        a = round(float(10 ** rng.uniform(np.log10(0.2), 0.0)), 3)
        N = int(round(float(rng.uniform(5.0, 16.0)) / a ** 2))
        uid = f"S_N{N}_a{a:g}"
        if uid in seen:
            continue
        seen.add(uid); out.append({"id": uid, "N": N, "a": a, "x": N * a * a, "trials": trials})
    return out


# ------------------------------------------------------------------ the graph with the strips shorted
def merged_inverse(net, Rj):
    """Pseudoinverse (as the inverse of L + 11'/k) of the Laplacian with every copy inside a strip tied to its
    electrode. Returns Lp, the map of copies to merged nodes, and the merged source and drain nodes."""
    n = len(net["coords"])
    label = np.arange(n)
    label[net["source_nodes"]] = -1; label[net["drain_nodes"]] = -2
    uniq = {}
    m = np.array([uniq.setdefault(int(l), len(uniq)) for l in label])
    s, d, k = uniq[-1], uniq[-2], len(uniq)
    eu = np.concatenate([net["seg_u"], net["jun_u"]]); ev = np.concatenate([net["seg_v"], net["jun_v"]])
    eR = np.concatenate([R0 * net["seg_L"], Rj])
    mu, mv = m[eu], m[ev]
    keep = mu != mv
    g = 1.0 / eR[keep]; mu, mv = mu[keep], mv[keep]
    L = np.zeros((k, k))
    np.add.at(L, (mu, mu), g); np.add.at(L, (mv, mv), g); np.add.at(L, (mu, mv), -g); np.add.at(L, (mv, mu), -g)
    J = 1.0 / k
    return np.linalg.inv(L + J) - J, m, s, d


class Tips:
    """Landing of tips and the three readings, on one film."""

    STEP = 0.004                       # spacing of the points that stand for the wires

    def __init__(self, net, Lp, m, s, rng):
        self.Lp, self.s, self.rng, self.m = Lp, s, rng, m
        xy = net["coords"]; su, sv = net["seg_u"], net["seg_v"]
        self.xy, self.su, self.sv = xy, su, sv
        pts, seg, tpar = [], [], []
        for e in range(len(su)):
            p, q = xy[su[e]], xy[sv[e]]
            L = float(np.hypot(*(q - p)))
            n = max(int(np.ceil(L / self.STEP)), 1) + 1
            t = np.linspace(0.0, 1.0, n)
            P = p[None, :] + t[:, None] * (q - p)[None, :]
            ok = (P[:, 0] >= LO) & (P[:, 0] <= HI)
            pts.append(P[ok]); seg.append(np.full(ok.sum(), e)); tpar.append(t[ok])
        self.pts = np.vstack(pts); self.seg = np.concatenate(seg); self.tpar = np.concatenate(tpar)
        self.tree = cKDTree(self.pts)
        self.diag = np.diag(Lp)

    def land(self, pt, f, sig):
        p = pt + self.rng.normal(0.0, sig, 2) if sig > 0 else pt
        if not (LO <= p[0] <= HI and 0.0 <= p[1] <= 1.0):
            return None
        dist, i = self.tree.query(p)
        if dist > f:
            return None
        e, t = self.seg[i], self.tpar[i]
        first, second = (self.su[e], self.sv[e]) if t < 0.5 else (self.sv[e], self.su[e])
        for c in (first, second):                               # the nearer crossing, if it is inside the film region
            if LO <= self.xy[c, 0] <= HI:
                return int(self.m[c])
        return None

    def R(self, u, v):
        L = self.Lp
        return self.diag[u] + self.diag[v] - 2.0 * L[u, v]

    def reading(self, kind, k, m, f, sig):
        """One attempt at position k of m; returns the reading in ohms or None (a miss)."""
        rng = self.rng
        if kind == "one":
            x = LO + (HI - LO) * (k + 0.5) / m
            u = self.land(np.array([x, rng.uniform()]), f, sig)
            return None if u is None else self.R(self.s, u)
        if kind == "two":
            sep = SEPS[m][k]
            for _ in range(RETRIES):
                p1 = rng.uniform([LO, 0.0], [HI, 1.0]); th = rng.uniform(0, 2 * np.pi)
                p2 = p1 + sep * np.array([np.cos(th), np.sin(th)])
                if LO <= p2[0] <= HI and 0.0 <= p2[1] <= 1.0:
                    break
            else:
                return None
            u, v = self.land(p1, f, sig), self.land(p2, f, sig)
            return None if (u is None or v is None or u == v) else self.R(u, v)
        sp = SPACINGS[m][k]                                         # four-point probe
        for _ in range(RETRIES):
            p0 = rng.uniform([LO, 0.0], [HI, 1.0]); th = rng.uniform(0, 2 * np.pi)
            dv = sp * np.array([np.cos(th), np.sin(th)])
            end = p0 + 3 * dv
            if LO <= end[0] <= HI and 0.0 <= end[1] <= 1.0:
                break
        else:
            return None
        a, b, c, d = (self.land(p0 + j * dv, f, sig) for j in range(4))
        if None in (a, b, c, d) or len({a, b, c, d}) < 4:
            return None
        L = self.Lp
        T = L[a, b] - L[a, c] - L[d, b] + L[d, c]                   # (e_a - e_d)' L+ (e_b - e_c)
        return float(T) if T > 0 else None


def sample_design(tips, kind, design, cfg):
    mpos, r = DESIGNS[design]; f, sig = CONFIGS[cfg]
    means, cnt = [], []
    for k in range(mpos):
        vals = [v for v in (tips.reading(kind, k, mpos, f, sig) for _ in range(r)) if v is not None]
        cnt.append(len(vals))
        means.append(float(np.mean(vals)) / R0 if vals else None)    # in units of R0 L
    return {"mean": [None if v is None else float(f"{v:.5g}") for v in means], "cnt": cnt}


def pairs_profile(net, Rj, keep, D):
    """The all-pairs profile P (nine shells), as in run_junction_pairwise_task.features_at."""
    Lp, mm = jp._pairwise_inverse(net, Rj, keep)
    dg = np.diag(Lp)[mm]
    n = len(mm)
    Rm = dg[:, None] + dg[None, :] - 2.0 * Lp[np.ix_(mm, mm)]
    iu = np.triu(np.ones((n, n), dtype=bool), 1)
    P = []
    for dl in jp.LAYERS:
        sel = (np.abs(D - dl) <= jp.TOL) & iu
        P.append(float(f"{Rm[sel].mean() / R0:.5g}") if sel.any() else None)
    return P


def run_unit(u, trials=None):
    t0 = time.time()
    N, a = int(u["N"]), float(u["a"]); trials = trials or u["trials"]
    np.random.seed(jp.stable_seed(N, round(a, 3), TAG))
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
        keep = jp.main_component(net)
        D = np.linalg.norm(net["coords"][keep][:, None, :] - net["coords"][keep][None, :, :], axis=2)
        rr = np.random.default_rng(jp.stable_seed(N, round(a, 3), TAG, k, "rho"))
        rhos = 10 ** rr.uniform(RHO_LOG[0], RHO_LOG[1], size=N_RHO)
        films = []
        for j, rho in enumerate(rhos):
            Rj = np.full(net["n_cross"], rho * R0 * a)
            Lp, m, s, d = merged_inverse(net, Rj)
            Rsd = float(Lp[s, s] + Lp[d, d] - 2 * Lp[s, d])
            ref = jp.solve_two_terminal(net, R0, Rj)["R_eff"]
            if abs(Rsd - ref) > 1e-7 * ref:
                raise RuntimeError(f"{u['id']} network {k}: merged-graph resistance {Rsd!r} != solve {ref!r}")
            rng = np.random.default_rng(jp.stable_seed(N, round(a, 3), TAG, k, "tips", j))
            tips = Tips(net, Lp, m, s, rng)
            strat = {kind: {cfg: {dz: sample_design(tips, kind, dz, cfg) for dz in DESIGNS} for cfg in CONFIGS}
                     for kind in ("one", "two", "four")}
            uu = rng.uniform(-1, 1, 3)
            films.append({"rho": float(rho), "S": float(f"{Rsd / R0:.6g}"), "P": pairs_profile(net, Rj, keep, D),
                          "u_c": float(uu[0]), "u_a": float(uu[1]), "u_N": float(uu[2]), "strat": strat})
        nets.append({"net": k, "n_nodes": int(len(net["coords"])), "n_main": int(len(keep)), "films": films})
    return {"id": u["id"], "N": N, "a": a, "x": u["x"], "tag": TAG, "n": len(nets), "skipped_too_large": skipped,
            "attempts_total": att_total, "seconds": time.time() - t0, "networks": nets}


def _job(args):
    u, t = args
    return u["id"], run_unit(u, t)


def _resume(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line); done[r["id"]] = r
    return done


def validate():
    """The readings against direct solves on a few films: the merged-graph resistance against the solver of
    Appendix F, the symmetry of a two-tip reading, and a four-point reading against a direct current solve."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    worst = 0.0
    for (N, a) in ((60, 0.5), (200, 0.25), (30, 0.8)):
        np.random.seed(jp.stable_seed(N, a, TAG, "validate"))
        res, _ = jp.generate_percolating_network(N, a, k_uniform=R0, max_attempts=200, boundary=jp.BOUNDARY)
        net = jp.build_junction_network(res)
        Rj = np.full(net["n_cross"], 1.3 * R0 * a)
        Lp, m, s, d = merged_inverse(net, Rj)
        ref = jp.solve_two_terminal(net, R0, Rj)["R_eff"]
        e1 = abs(Lp[s, s] + Lp[d, d] - 2 * Lp[s, d] - ref) / ref
        # direct solve of the merged graph: inject 1 A at node p, withdraw at q, read the potential difference
        nodes = np.unique(m)
        eu = np.concatenate([net["seg_u"], net["jun_u"]]); ev = np.concatenate([net["seg_v"], net["jun_v"]])
        eR = np.concatenate([R0 * net["seg_L"], Rj]); mu, mv = m[eu], m[ev]; keep = mu != mv
        k = len(nodes); g = 1 / eR[keep]
        L = sp.csc_matrix((np.concatenate([g, g, -g, -g]), (np.concatenate([mu[keep], mv[keep], mu[keep], mv[keep]]),
                                                            np.concatenate([mu[keep], mv[keep], mv[keep], mu[keep]]))), shape=(k, k))
        rng = np.random.default_rng(0)
        p, q, r_, t_ = rng.choice(k, 4, replace=False)
        def potential(i, j):
            b = np.zeros(k); b[i] = 1; b[j] = -1
            Lr = L.tolil(); Lr[k - 1, :] = 0; Lr[k - 1, k - 1] = 1; bb = b.copy(); bb[k - 1] = 0
            return spla.spsolve(Lr.tocsc(), bb)
        v = potential(p, q)
        direct_two = v[p] - v[q]
        two = Lp[p, p] + Lp[q, q] - 2 * Lp[p, q]
        T_direct = v[r_] - v[t_]                                       # voltage between r and t, current p -> q
        T = Lp[p, r_] - Lp[p, t_] - Lp[q, r_] + Lp[q, t_]
        e2 = abs(direct_two - two) / abs(two); e3 = abs(T_direct - T) / max(abs(T), 1e-12)
        worst = max(worst, e1, e2, e3)
        print(f"N={N} a={a}: merged R_sd vs solver {e1:.1e}; two-tip vs direct solve {e2:.1e}; four-point vs direct solve {e3:.1e}")
    if worst > 1e-6:
        raise SystemExit("validation FAILED")
    print("validate: the readings reproduce direct solves")


def main():
    args = sys.argv[1:]
    if "--validate" in args:
        validate(); return
    ul = units()
    if "--units" in args:
        want = args[args.index("--units") + 1].split(",")
        ul = [u for u in ul if u["id"] in want]
    trials = int(args[args.index("--trials") + 1]) if "--trials" in args else None
    out = args[args.index("--out") + 1] if "--out" in args else os.path.join(HERE, "junction_strategies_results")
    os.makedirs(out, exist_ok=True); part = os.path.join(out, "partial.jsonl")
    done = _resume(part)
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
