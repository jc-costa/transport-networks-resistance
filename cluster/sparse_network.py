"""
Sparse builder and solver for 2D random stick networks, for films too large
for the dense pipeline (utils/pipeline_utils.py solves a dense system and
tests every pair of sticks, which limits it to a few thousand nodes).

It reproduces the pipeline's network rules exactly:
  * sticks come from utils/point_generator.generate_points (clipping rule),
    so the same seed gives the same sticks;
  * every crossing of two sticks is a node; a node with x < 0.1 belongs to the
    source strip and one with x > 0.9 to the drain strip;
  * the film percolates if a chain of crossings joins a source node to a drain
    node; only the connected components that touch both strips are kept;
  * consecutive crossings along a stick are joined by an edge of resistance
    R0 * (length of the stretch), measured on the original geometry;
  * all source nodes are merged into one terminal, all drain nodes into the
    other; edges inside a strip disappear, parallel edges add conductances.
Dead ends are not pruned: they carry no current and do not change R_eff.

What changes is the cost: crossings are found with a uniform grid of cell
size a (only sticks sharing a cell are tested), and the grounded Laplacian is
solved with a sparse LU factorization. Both scale roughly linearly with the
number of sticks for these planar networks.

validate() replays stored networks of the near-threshold sweep from their
seeds and checks that every stored R_eff is reproduced.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

SRC_X, DRN_X = 0.1, 0.9


def _pipeline_dir():
    """ColabProject root: PIPELINE_DIR if set, else the parent of this file's folder
    (the scripts are meant to sit in ColabProject/cluster/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("PIPELINE_DIR", os.path.normpath(os.path.join(here, "..")))


sys.path.insert(0, os.path.join(_pipeline_dir(), "utils"))
from point_generator import generate_points  # noqa: E402
from pipeline_utils import stable_seed  # noqa: E402,F401  (re-exported for the task scripts)


def draw_sticks(N, a):
    """One draw with the pipeline's generator (consumes the global numpy RNG
    exactly as generate_percolating_network does per attempt)."""
    p1, p2, ang = [], [], []
    generate_points(N, p1, p2, ang, a, enforce_bounds=True, boundary="clip")
    return np.asarray(p1, float).reshape(-1, 2), np.asarray(p2, float).reshape(-1, 2)


def find_crossings(P1, P2, cell):
    """All proper crossings between segments P1[i]-P2[i]. Returns arrays
    (i, j, t_i, t_j, x, y) with i < j and t the fraction along each stick."""
    n = len(P1)
    if n < 2:
        return [np.empty(0)] * 6
    lo = np.minimum(P1, P2); hi = np.maximum(P1, P2)
    c0 = np.floor(lo / cell).astype(np.int64); c1 = np.floor(hi / cell).astype(np.int64)
    ncell = int(np.ceil(1.0 / cell)) + 2
    # every stick is registered in each cell its bounding box covers (<= 4 for length <= cell)
    rows, cells = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            cx = c0[:, 0] + dx; cy = c0[:, 1] + dy
            ok = (cx <= c1[:, 0]) & (cy <= c1[:, 1])
            rows.append(np.flatnonzero(ok)); cells.append((cx[ok] + 1) * ncell + (cy[ok] + 1))
    rows = np.concatenate(rows); cells = np.concatenate(cells)
    order = np.lexsort((rows, cells)); rows = rows[order]; cells = cells[order]
    starts = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]]); ends = np.r_[starts[1:], len(cells)]
    I, J = [], []
    for s, e in zip(starts, ends):
        m = e - s
        if m < 2:
            continue
        idx = rows[s:e]
        a_, b_ = np.triu_indices(m, 1)
        I.append(idx[a_]); J.append(idx[b_])
    if not I:
        return [np.empty(0)] * 6
    I = np.concatenate(I); J = np.concatenate(J)
    key = np.unique(np.minimum(I, J) * n + np.maximum(I, J))   # pairs sharing two cells counted once
    I = key // n; J = key % n
    p = P1[I]; r = P2[I] - p; q = P1[J]; s = P2[J] - q
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / den
        u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
    ok = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    I, J, t, u = I[ok], J[ok], t[ok], u[ok]
    xy = P1[I] + t[:, None] * (P2[I] - P1[I])
    return I, J, t, u, xy[:, 0], xy[:, 1]


def build_network(P1, P2, a, R0=1000.0):
    """Percolating resistor network of one draw, or None if it does not percolate.
    Returns a dict with the merged sparse Laplacian blocks and edge data."""
    I, J, t, u, x, y = find_crossings(P1, P2, cell=max(a, 1e-3))
    nc = len(x)
    if nc == 0:
        return None
    # node k is crossing k; edges between consecutive crossings on each stick
    stick = np.r_[I, J]; frac = np.r_[t, u]; node = np.r_[np.arange(nc), np.arange(nc)]
    o = np.lexsort((frac, stick)); stick, frac, node = stick[o], frac[o], node[o]
    same = stick[1:] == stick[:-1]
    eu, ev = node[:-1][same], node[1:][same]
    seglen = np.linalg.norm(P2 - P1, axis=1)
    elen = (frac[1:] - frac[:-1])[same] * seglen[stick[:-1][same]]
    src = x < SRC_X; drn = x > DRN_X
    if not src.any() or not drn.any() or len(eu) == 0:
        return None
    A = sp.coo_matrix((np.ones(len(eu)), (eu, ev)), shape=(nc, nc))
    ncomp, lab = connected_components(A, directed=False)
    good = np.intersect1d(np.unique(lab[src]), np.unique(lab[drn]))
    if len(good) == 0:
        return None
    keep = np.isin(lab, good)
    # merge terminals: source -> 0, drain -> 1, interior kept nodes -> 2..
    newid = np.full(nc, -1, dtype=np.int64)
    interior = keep & ~src & ~drn
    newid[keep & src] = 0; newid[keep & drn] = 1
    newid[interior] = 2 + np.arange(interior.sum())
    ek = keep[eu] & keep[ev]
    U, V, Le = newid[eu[ek]], newid[ev[ek]], elen[ek]
    nz = U != V                                    # edges inside one strip vanish
    U, V, Le = U[nz], V[nz], Le[nz]
    if np.any(Le <= 0):
        Le = np.maximum(Le, 1e-15)                 # coincident crossings: a negligible resistor
    g = 1.0 / (R0 * Le)
    n = int(2 + interior.sum())
    return dict(U=U, V=V, g=g, length=Le, n=n, n_crossings=nc,
                xy=np.c_[x, y], old_of_new=np.r_[[-1, -1], np.flatnonzero(interior)])


def solve(net, red_tol=1e-8):
    """Terminal resistance (source at 1 V, drain at 0 V) and the red bonds,
    the edges that carry the whole current."""
    U, V, g, n = net["U"], net["V"], net["g"], net["n"]
    L = sp.coo_matrix((np.r_[g, g, -g, -g], (np.r_[U, V, U, V], np.r_[U, V, V, U])), shape=(n, n)).tocsr()
    # unknowns: interior nodes 2..n-1, with V_s = 1, V_d = 0
    Mi = L[2:, 2:].tocsc(); b = -L[2:, 0].toarray().ravel()
    Vint = spla.splu(Mi).solve(b) if n > 2 else np.empty(0)
    pot = np.r_[1.0, 0.0, Vint]
    I_e = g * (pot[U] - pot[V])
    I_tot = float(np.sum(I_e[U == 0]) - np.sum(I_e[V == 0]))
    R = 1.0 / I_tot
    red = np.abs(I_e) >= (1.0 - red_tol) * abs(I_tot)
    return dict(R_eff=R, n_nodes=n, n_edges=len(U), n_red=int(red.sum()),
                red_R_fraction=float(np.sum(1.0 / g[red]) / R))


def percolating_draw(N, a, R0=1000.0, max_attempts=500):
    """Like pipeline_utils.generate_percolating_network: draw until a film
    percolates. Returns (net, attempts) or (None, max_attempts)."""
    for k in range(1, max_attempts + 1):
        P1, P2 = draw_sticks(N, a)
        net = build_network(P1, P2, a, R0)
        if net is not None:
            return net, k
    return None, max_attempts


def validate(points=(0, 4, 12, 16, 24), n_check=10, verbose=True):
    """Replay stored near-threshold networks (critical_sweep_results) from their
    seeds with this builder; every stored R_eff must be reproduced to 1e-9."""
    import json
    root = _pipeline_dir()
    sys.path.insert(0, os.path.join(root, "cluster"))
    from run_critical_sweep_task import build_plan, MAX_ATTEMPTS  # noqa: E402
    plan = build_plan(); worst = 0.0
    for tid in points:
        pt = plan[tid]; N, a = pt["N"], pt["a"]
        ref = json.load(open(os.path.join(root, "cluster", "critical_sweep_results", f"point_{tid:02d}.json")))["R_eff_values"]
        np.random.seed(stable_seed(N, round(a, 3), "critical"))
        for k in range(min(n_check, len(ref))):
            net, _ = percolating_draw(N, a, max_attempts=MAX_ATTEMPTS)
            while net is None:
                net, _ = percolating_draw(N, a, max_attempts=MAX_ATTEMPTS)
            R = solve(net)["R_eff"]
            rel = abs(R - ref[k]) / ref[k]; worst = max(worst, rel)
            if verbose:
                print(f"  point {tid:2d} (a={a}, N={N}) network {k}: R = {R:.6f}  stored {ref[k]:.6f}  rel {rel:.1e}", flush=True)
            if rel > 1e-9:
                raise RuntimeError(f"point {tid} network {k}: sparse builder gives {R}, stored {ref[k]}")
    print(f"validate: all checked networks reproduced, worst relative difference {worst:.1e}")
    return worst
