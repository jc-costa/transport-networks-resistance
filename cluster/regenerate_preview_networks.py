"""
Regenerate the 150 networks behind the Part A preview onion curves
(cluster/preview_curves_results/, Fig. "onion-preview" in the thesis) from their
seeds, and split every shell mean through the Laplacian-pseudoinverse identity

    R_eff(i, j) = L+_ii + L+_jj - 2 L+_ij .

Why this exists: the chunk files store only R_eff and the per-shell means, not
the geometry. The thesis subsection on the distance dependence of R_eff,ij
(Sec. "Dependence of R_eff,ij on probe separation") and its Appendix D quote
per-network quantities that need the geometry back: node counts, degrees, the
degree of the reference node, the L+ terms per shell and their variance shares.
Because run_preview_curves_task.py seeds every (combo, realization) unit
independently with stable_seed(N, a, d, 'ext', r), each network can be redrawn
on its own, and this script checks that every one reproduces the stored R_eff
and shell means before using it (it does, to a relative ~1e-13).

This is a local analysis script, not a SLURM job. Under the clipping boundary
rule the rarest combination, (125, 0.2, 0.10), percolates on about 1 draw in 20,
so the whole set takes a few minutes with --workers 7 on a laptop.

Usage:
  python regenerate_preview_networks.py [--workers K]   regenerate, verify, write
                                                        preview_curves_decomposition.json
  python regenerate_preview_networks.py --tables        print the two LaTeX tables of
                                                        Appendix D and the resistor-chain
                                                        control, from the JSON above

The output JSON lives next to this script, not inside preview_curves_results/,
which holds data synced back from Apuana and must not be written to locally.
"""
import sys
import os
import json
import glob
import argparse
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
import numpy as np  # noqa: E402
from pipeline_utils_3d import generate_percolating_network_3d  # noqa: E402
from pipeline_utils import distance_dependent_impedance, stable_seed  # noqa: E402
from laplacian_utils import _weighted_laplacian_distance_dependent, _bfs_reachable  # noqa: E402
from multimeter import onion_layers  # noqa: E402

# Must match run_preview_curves_task.py exactly, or the seeds reproduce nothing.
CONTACT_DISTANCE = 0.03
R0 = 1000.0
LAYER_DISTANCES = [round(0.1 * n, 2) for n in range(1, 10)]
PREVIEW_COMBOS = [(150, 0.2, 0.1), (150, 0.2, 0.15), (125, 0.2, 0.1)]
N_TRIALS = 50
MAX_ATTEMPTS = 100000

RESULTS_DIR = os.path.join(HERE, "preview_curves_results")
OUT_PATH = os.path.join(HERE, "preview_curves_decomposition.json")
KEYS = [str(d) for d in LAYER_DISTANCES]


def load_chunks():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "chunk_*.json"))):
        rows.extend(json.load(open(p))["results"])
    if len(rows) != len(PREVIEW_COMBOS) * N_TRIALS:
        sys.exit(f"expected {len(PREVIEW_COMBOS) * N_TRIALS} units in {RESULTS_DIR}, found {len(rows)}")
    return {(r["combo_index"], r["realization"]): r for r in rows}


def regenerate_one(args):
    """Redraw one (combo, realization) unit and decompose its shell means."""
    ci, r, stored = args
    N, a, d = PREVIEW_COMBOS[ci]
    np.random.seed(stable_seed(N, round(a, 3), round(d, 3), 'ext', r))
    res, attempts = generate_percolating_network_3d(
        N=N, a=a, contact_distance=CONTACT_DISTANCE,
        box_dims=(1.0, 1.0, d), k_uniform=R0, max_attempts=MAX_ATTEMPTS
    )
    if res is None:
        return {"combo": ci, "realization": r, "percolated": False}

    R_eff = float(distance_dependent_impedance(res, R0)['impedance'])
    inter = np.asarray(res['pre_merge_intersections'], float)
    s2n = res['pre_merge_segment_to_nodes']
    i = res['pre_merge_source_nodes'][0]

    L = _weighted_laplacian_distance_dependent(inter, s2n, R0=R0)
    Lp = np.linalg.pinv(L)
    dg = np.diag(Lp)
    reach = _bfs_reachable(len(inter), s2n, i)
    deg = np.zeros(len(inter), int)
    for (u, v) in s2n.values():
        if u != v:
            deg[u] += 1
            deg[v] += 1

    rec = {"combo": ci, "realization": r, "percolated": True, "attempts": attempts,
           "n_nodes": int(len(inter)), "mean_degree": float(deg.mean()),
           "R_eff": R_eff, "R_eff_stored": stored["R_eff"],
           "Lii": float(dg[i]), "deg_i": int(deg[i]), "shells": {}}
    for ld, cands in onion_layers(inter, i, layer_distances=LAYER_DISTANCES).items():
        js = np.array([j for j in cands if reach[j]])
        if js.size == 0:
            continue
        rec["shells"][str(ld)] = {
            "n": int(js.size),
            "Ljj": float(dg[js].mean()),
            "pair": float((-2.0 * Lp[i, js]).mean()),
            "r_eff": float((dg[i] + dg[js] - 2.0 * Lp[i, js]).mean()),
            "r_eff_stored": stored["layer_means"].get(str(ld)),
            "deg_j": float(deg[js].mean()),
        }
    return rec


def cmd_regenerate(workers):
    stored = load_chunks()
    units = [(ci, r, stored[(ci, r)]) for ci in range(len(PREVIEW_COMBOS)) for r in range(N_TRIALS)]
    # The rare combination dominates the runtime; interleave so no worker gets it all.
    units.sort(key=lambda u: u[1])
    with Pool(workers) as pool:
        recs = pool.map(regenerate_one, units, chunksize=1)

    failed = [x for x in recs if not x["percolated"]]
    if failed:
        sys.exit(f"{len(failed)} units did not percolate; seeds do not reproduce the run")
    worst_R = max(abs(x["R_eff"] - x["R_eff_stored"]) / x["R_eff_stored"] for x in recs)
    worst_shell = max(abs(s["r_eff"] - s["r_eff_stored"]) / s["r_eff_stored"]
                      for x in recs for s in x["shells"].values() if s["r_eff_stored"])
    print(f"{len(recs)} networks regenerated; worst relative deviation from the chunk files: "
          f"R_eff {worst_R:.1e}, shell means {worst_shell:.1e}")
    if worst_R > 1e-9 or worst_shell > 1e-9:
        sys.exit("reproduction failed: the regenerated networks are not the stored ones")

    for ci, (N, a, d) in enumerate(PREVIEW_COMBOS):
        R = [x for x in recs if x["combo"] == ci]
        print(f"  ({N}, {a}, {d:.2f}): nodes {np.mean([x['n_nodes'] for x in R]):.0f}, "
              f"mean degree {np.mean([x['mean_degree'] for x in R]):.2f}, "
              f"reference-node degree {np.mean([x['deg_i'] for x in R]):.2f}")
    json.dump(recs, open(OUT_PATH, "w"), indent=1)
    print(f"wrote {OUT_PATH}")


def _shell_matrix(R, field):
    return np.array([[x["shells"][k][field] if k in x["shells"] else np.nan for k in KEYS] for x in R])


def cmd_tables():
    if not os.path.exists(OUT_PATH):
        sys.exit(f"{OUT_PATH} not found; run without --tables first")
    recs = json.load(open(OUT_PATH))
    terms, shares = {}, {}
    for ci in range(len(PREVIEW_COMBOS)):
        R = [x for x in recs if x["combo"] == ci]
        tot, Ljj, pair = _shell_matrix(R, "r_eff"), _shell_matrix(R, "Ljj"), _shell_matrix(R, "pair")
        Lii = np.array([x["Lii"] for x in R])
        n = np.sum(~np.isnan(tot), 0)
        terms[ci] = (Lii.mean(), Lii.std(ddof=1) / np.sqrt(len(R)),
                     np.nanmean(Ljj, 0), np.nanstd(Ljj, 0, ddof=1) / np.sqrt(n),
                     np.nanmean(pair, 0), np.nanstd(pair, 0, ddof=1) / np.sqrt(n))
        # Variance shares of the normalized shape (each realization's curve divided by its
        # own r = 0.9 value, the normalization the ANOVA uses): cov(term, shape)/var(shape).
        ok = ~np.isnan(tot[:, -1])
        norm = tot[ok, -1][:, None]
        shape = tot[ok] / norm
        parts = {"Lii": np.repeat(Lii[ok][:, None], 9, 1) / norm,
                 "Ljj": Ljj[ok] / norm, "pair": pair[ok] / norm}
        sh = {}
        for nm, P in parts.items():
            v = []
            for k in range(8):
                m = ~np.isnan(shape[:, k])
                v.append(np.cov(P[m, k], shape[m, k], ddof=1)[0, 1] / np.var(shape[m, k], ddof=1))
            sh[nm] = v
        shares[ci] = (sh, int(ok.sum()))

    def combo_label(c):
        return f"$({c[0]},{c[1]},{c[2]:.2f})$"

    print("% ---- Table: shell means of the L+ terms ----")
    print(r"\begin{tabular}{r" + "rr" * 3 + "}")
    print(r"\toprule")
    print("Shell $r$ & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{combo_label(c)}}}" for c in PREVIEW_COMBOS) + r" \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    print(" & " + " & ".join(["$L^{+}_{jj}$ & $-2L^{+}_{ij}$"] * 3) + r" \\")
    print(r"\midrule")
    for k in range(9):
        print(f"{KEYS[k]} & " + " & ".join(
            f"${terms[ci][2][k]:.0f}\\pm{terms[ci][3][k]:.0f}$ & ${terms[ci][4][k]:.0f}\\pm{terms[ci][5][k]:.0f}$"
            for ci in range(3)) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("% L+_ii per combination: " + ", ".join(
        f"{combo_label(c)} ${terms[ci][0]:.0f}\\pm{terms[ci][1]:.0f}$" for ci, c in enumerate(PREVIEW_COMBOS)))

    print("\n% ---- Table: variance shares of the normalized shape ----")
    print(r"\begin{tabular}{r" + "rrr" * 3 + "}")
    print(r"\toprule")
    print("Shell $r$ & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{combo_label(c)}, $n={shares[ci][1]}$}}" for ci, c in enumerate(PREVIEW_COMBOS)) + r" \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    print(" & " + " & ".join(["$L^{+}_{ii}$ & $L^{+}_{jj}$ & $-2L^{+}_{ij}$"] * 3) + r" \\")
    print(r"\midrule")
    for k in range(8):
        print(f"{KEYS[k]} & " + " & ".join(
            f"${shares[ci][0]['Lii'][k]:.2f}$ & ${shares[ci][0]['Ljj'][k]:.2f}$ & ${shares[ci][0]['pair'][k]:.2f}$"
            for ci in range(3)) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # The control that disqualifies the decomposition as a diagnostic: a homogeneous chain
    # probed from one end gives the same saturating pair term and U-shaped node term.
    n = 101
    pos = np.linspace(0, 1, n)
    g = 1.0 / (R0 * (1.0 / (n - 1)))
    L = np.zeros((n, n))
    for k in range(n - 1):
        L[k, k] += g; L[k + 1, k + 1] += g; L[k, k + 1] -= g; L[k + 1, k] -= g
    Lp = np.linalg.pinv(L)
    dg = np.diag(Lp)
    rows = []
    for r in LAYER_DISTANCES:
        js = np.where(np.abs(pos - r) <= 0.05)[0]
        js = js[js != 0]
        rows.append((dg[js].mean(), (-2 * Lp[0, js]).mean(), (dg[0] + dg[js] - 2 * Lp[0, js]).mean()))
    rows = np.array(rows)
    print("\n% ---- Control: 101-node chain on [0,1], R0 per unit length, probed from node 0 ----")
    print("% L+_jj   :", np.round(rows[:, 0]).astype(int).tolist())
    print("% -2L+_ij :", np.round(rows[:, 1]).astype(int).tolist())
    print("% total   :", np.round(rows[:, 2]).astype(int).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="print the Appendix D tables from the JSON")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()
    if args.tables:
        cmd_tables()
    else:
        cmd_regenerate(args.workers)
