"""
Regenerate the network library embedded in the thesis' interactive supplement
(MSc---Thesis/docs/index.html, <script id="netdata">), which the page solves in
the browser and probe_distance_figure.py reads for the 2D probe-distance panel.

Five (N, a) combinations x four seeds. For each, np.random.seed(seed) and draw
with generate_percolating_network until a percolating network appears; the
record keeps the pre-merge geometry (nodes, edges, source and drain node
indices) and the whole-face distance-dependent resistance the pipeline
recorded, which the page checks itself against.

Usage:  python export_supplement_networks.py --html ../../MSc---Thesis/docs/index.html
        python export_supplement_networks.py --json out.json      (data only)
"""
import os, sys, re, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utils"))
from pipeline_utils import generate_percolating_network, distance_dependent_impedance  # noqa: E402

R0 = 1000.0
BOUNDARY = "clip"
COMBOS = [(40, 0.6), (30, 0.7), (55, 0.5), (25, 0.9), (70, 0.45)]
SEEDS = [11, 3, 7, 21]
MAX_ATTEMPTS = 200


def export_one(N, a, seed):
    np.random.seed(seed)
    res, attempts = generate_percolating_network(N, a, k_uniform=R0, max_attempts=MAX_ATTEMPTS, boundary=BOUNDARY)
    if res is None:
        raise RuntimeError(f"N={N} a={a} seed={seed}: no percolating network in {MAX_ATTEMPTS} draws")
    whole = distance_dependent_impedance(res, R0)["impedance"]
    nodes = [[round(float(x), 9), round(float(y), 9)] for x, y in res["pre_merge_intersections"]]
    edges = [[int(u), int(v)] for _, (u, v) in sorted(res["pre_merge_segment_to_nodes"].items())]
    return {"N": N, "a": a, "seed": seed, "attempts": attempts, "boundary": BOUNDARY,
            "n_sticks": res["n_sticks"], "nodes": nodes, "edges": edges,
            "source": [int(i) for i in res["pre_merge_source_nodes"]],
            "drain": [int(i) for i in res["pre_merge_drain_nodes"]],
            "whole_face": round(float(whole), 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--html"); ap.add_argument("--json"); args = ap.parse_args()
    nets = []
    for N, a in COMBOS:
        for s in SEEDS:
            rec = export_one(N, a, s); nets.append(rec)
            print(f"N={N:3d} a={a:.2f} seed={s:2d}: {rec['attempts']} draws, {rec['n_sticks']} sticks, "
                  f"{len(rec['nodes'])} nodes, {len(rec['edges'])} edges, whole-face {rec['whole_face']:.2f} ohm", flush=True)
    data = {"R0": R0, "boundary": BOUNDARY, "networks": nets}
    if args.json:
        json.dump(data, open(args.json, "w"), separators=(",", ":")); print("wrote", args.json)
    if args.html:
        html = open(args.html, encoding="utf-8").read()
        new = re.sub(r'(<script[^>]*id="netdata"[^>]*>).*?(</script>)',
                     lambda m: m.group(1) + json.dumps(data, separators=(",", ":")) + m.group(2), html, count=1, flags=re.S)
        assert new != html
        open(args.html, "w", encoding="utf-8").write(new); print("replaced netdata in", args.html)


if __name__ == "__main__":
    main()
