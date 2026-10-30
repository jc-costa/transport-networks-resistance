"""
Multimeter Module

"Touch two probes to the network and read the resistance": measures
R_eff(i, j) between two arbitrary, pre-existing nodes i and j of an already
built network, without altering the network itself (unlike
merge_boundary_nodes / merge_terminal_groups, which physically fuse a whole
region of nodes into one terminal for the standard source/drain impedance
calculation).

Two pieces:
  - measure_r_eff(): the probe itself, R_eff between one chosen pair.
  - onion_layers() / run_onion_protocol(): the experimental protocol for
    choosing WHICH pairs to probe. Probing all O(n^2) node pairs on a real
    network is wasteful and mostly redundant, so instead we fix one probe
    (reference_node) and sweep the other probe outward through concentric
    Euclidean-distance shells around it (d1=0.1, d2=0.2, d3=0.3, ... like
    the layers of an onion), which is enough to see how R_eff grows with
    separation distance.
"""

import numpy as np

from impedance_calculator import ImpedanceCalculator
from laplacian_utils import effective_resistance_from_reference


def measure_r_eff(intersections, graph, segment_to_nodes, node_i, node_j,
                   k=1.0, edge_conductance=None, distance_dependent=False,
                   verbose=False):
    """
    Measure the effective resistance R_eff(node_i, node_j) between two
    arbitrary nodes already present in the network: the multimeter probe.

    Reuses ImpedanceCalculator.calculate() exactly as-is: it already solves
    "V=1 at one boundary node, V=0 at another, everything else floating",
    which is precisely a two-point probe. We just point source_node/
    drain_node at the two chosen nodes instead of letting __init__ pick them
    by min/max x.

    Parameters
    ----------
    intersections : list
        [x, y] (or [x, y, z]) coordinates for each node.
    graph : list
        graph[i] = segments touching node i.
    segment_to_nodes : dict
        edge_idx -> (node_a, node_b).
    node_i, node_j : int
        The two probed node indices (local indices into `intersections`).
    k : float
        Resistance per segment (or R0 per unit length if distance_dependent).
    edge_conductance : dict, optional
        edge_idx -> parallel-edge multiplier (as produced by
        merge_boundary_nodes/merge_terminal_groups), if probing a merged
        network.
    distance_dependent : bool
        If True, R_edge = k * L_edge instead of a uniform k.
    verbose : bool
        Print ImpedanceCalculator's step-by-step output.

    Returns
    -------
    result : dict
        Same shape as ImpedanceCalculator.calculate(), plus:
        'node_i', 'node_j', 'probe_distance' (Euclidean distance between the
        two probed nodes).
    """
    n_nodes = len(intersections)
    if not (0 <= node_i < n_nodes) or not (0 <= node_j < n_nodes):
        raise IndexError(f"node_i={node_i}, node_j={node_j} out of range "
                         f"for a network with {n_nodes} nodes")
    if node_i == node_j:
        raise ValueError("node_i and node_j must be different nodes")

    calc = ImpedanceCalculator(intersections, graph, segment_to_nodes,
                               edge_conductance=edge_conductance)
    calc.source_node = node_i
    calc.drain_node = node_j

    result = calc.calculate(k=k, distance_dependent=distance_dependent, verbose=verbose)
    result['node_i'] = node_i
    result['node_j'] = node_j
    result['probe_distance'] = float(np.linalg.norm(
        np.array(intersections[node_j], dtype=float) - np.array(intersections[node_i], dtype=float)
    ))
    return result


def onion_layers(intersections, reference_node, layer_distances=None,
                 tolerance=None, candidate_nodes=None):
    """
    Group nodes into concentric Euclidean-distance shells around
    `reference_node`: the "onion layers" used to pick which node j to
    probe against a fixed node i, instead of trying every pair.

    Parameters
    ----------
    intersections : list
        [x, y] (or [x, y, z]) coordinates for each node.
    reference_node : int
        The fixed probe (index into `intersections`); shells are measured
        from here.
    layer_distances : sequence of float, optional
        Target shell radii d1, d2, d3, ... Defaults to [0.1, 0.2, ..., 0.9]
        (the network lives in the unit square/cube, so this covers it).
    tolerance : float, optional
        Shell half-width: a node qualifies for layer d_k if its distance to
        reference_node falls in [d_k - tolerance, d_k + tolerance]. Defaults
        to half the spacing between consecutive layer_distances, so shells
        partition distance space without gaps.
    candidate_nodes : iterable of int, optional
        Restrict candidates to this subset (e.g. only nodes on the
        percolating path). Defaults to every node except reference_node.

    Returns
    -------
    dict {d_k: [node indices]}: one list per requested layer distance
    (reference_node itself never included; empty lists mean no node fell in
    that shell).
    """
    if layer_distances is None:
        layer_distances = [round(0.1 * n, 2) for n in range(1, 10)]
    layer_distances = sorted(layer_distances)

    if tolerance is None:
        if len(layer_distances) > 1:
            spacing = min(b - a for a, b in zip(layer_distances, layer_distances[1:]))
        else:
            spacing = layer_distances[0]
        tolerance = spacing / 2

    if candidate_nodes is None:
        candidate_nodes = range(len(intersections))

    ref_pos = np.array(intersections[reference_node], dtype=float)
    layers = {d: [] for d in layer_distances}
    for n in candidate_nodes:
        if n == reference_node:
            continue
        dist = float(np.linalg.norm(np.array(intersections[n], dtype=float) - ref_pos))
        for d in layer_distances:
            if abs(dist - d) <= tolerance:
                layers[d].append(n)

    return layers


def run_onion_protocol(intersections, graph, segment_to_nodes, reference_node,
                       layer_distances=None, tolerance=None, k=1.0,
                       edge_conductance=None, distance_dependent=False,
                       candidate_nodes=None, max_probes_per_layer=None,
                       seed=None, verbose=False):
    """
    Run the onion-layer protocol: probe R_eff(reference_node, j) for j drawn
    from each concentric distance shell around reference_node (see
    onion_layers), building up R_eff as a function of separation distance
    without measuring all O(n^2) node pairs.

    Parameters
    ----------
    reference_node : int
        The fixed probe every layer is measured against.
    layer_distances, tolerance, candidate_nodes : see onion_layers().
    k, edge_conductance, distance_dependent : see measure_r_eff().
    max_probes_per_layer : int, optional
        Cap on how many candidates to probe per shell. If a shell has more
        candidates than this, a random sample of this size is probed
        instead of all of them (bounds the number of linear solves on dense
        networks where a shell can contain many nodes).
    seed : int, optional
        Seed for the random sampling above (reproducibility).
    verbose : bool
        Print per-layer progress.

    Returns
    -------
    list of dict, one per (reference_node, j) probe actually measured:
        {'layer_distance': d_k, 'node_j': j, 'probe_distance': actual
         Euclidean distance to reference_node, 'r_eff': R_eff (may be inf if
         j is not connected to reference_node), 'result': full
         measure_r_eff() result dict}
    """
    rng = np.random.default_rng(seed)
    layers = onion_layers(intersections, reference_node, layer_distances=layer_distances,
                          tolerance=tolerance, candidate_nodes=candidate_nodes)

    records = []
    for d, candidates in layers.items():
        if not candidates:
            if verbose:
                print(f"  layer d={d}: no candidate nodes, skipping")
            continue

        probes = candidates
        if max_probes_per_layer is not None and len(candidates) > max_probes_per_layer:
            probes = list(rng.choice(candidates, size=max_probes_per_layer, replace=False))

        if verbose:
            print(f"  layer d={d}: probing {len(probes)}/{len(candidates)} candidate node(s)")

        for j in probes:
            result = measure_r_eff(intersections, graph, segment_to_nodes,
                                   reference_node, j, k=k,
                                   edge_conductance=edge_conductance,
                                   distance_dependent=distance_dependent)
            records.append({
                'layer_distance': d,
                'node_j': j,
                'probe_distance': result['probe_distance'],
                'r_eff': result['impedance'],
                'result': result,
            })

    return records


def run_onion_protocol_pinv(intersections, segment_to_nodes, reference_node,
                            layer_distances=None, tolerance=None, R0=1000.0,
                            edge_conductance=None, candidate_nodes=None):
    """
    Fast onion-layer protocol: same idea as run_onion_protocol (probe
    R_eff(reference_node, j) for j drawn from each concentric distance shell
    around reference_node), but instead of one dense O(n^3) solve per probed
    pair (measure_r_eff), computes R_eff from reference_node to EVERY node in
    one shot via effective_resistance_from_reference (one Laplacian
    pseudoinverse per network, O(n^3) total regardless of probe count).

    That also removes any reason to sample/cap probes per shell -- every
    candidate node in a shell is "free" once the pinv is computed, so every
    one of them is included (unlike run_onion_protocol's max_probes_per_layer).

    Only supports the distance-dependent resistance model (R_edge = R0 *
    edge_length), since that's what effective_resistance_from_reference
    implements -- there is no uniform-k variant of this fast path.

    Parameters
    ----------
    intersections, segment_to_nodes : network geometry (see onion_layers /
        effective_resistance_from_reference).
    reference_node : int
        The fixed probe every layer is measured against.
    layer_distances, tolerance, candidate_nodes : see onion_layers().
    R0, edge_conductance : see effective_resistance_from_reference().

    Returns
    -------
    list of dict, one per (reference_node, j) pair in a non-empty shell:
        {'layer_distance': d_k, 'node_j': j, 'probe_distance': actual
         Euclidean distance to reference_node, 'r_eff': R_eff (inf if j is
         not connected to reference_node)}
    """
    layers = onion_layers(intersections, reference_node, layer_distances=layer_distances,
                          tolerance=tolerance, candidate_nodes=candidate_nodes)

    r_eff_vec = effective_resistance_from_reference(
        intersections, segment_to_nodes, reference_node,
        R0=R0, edge_conductance=edge_conductance
    )
    ref_pos = np.array(intersections[reference_node], dtype=float)

    records = []
    for d, candidates in layers.items():
        for j in candidates:
            probe_distance = float(np.linalg.norm(
                np.array(intersections[j], dtype=float) - ref_pos
            ))
            records.append({
                'layer_distance': d,
                'node_j': j,
                'probe_distance': probe_distance,
                'r_eff': float(r_eff_vec[j]),
            })

    return records


def series_path_resistance(intersections, segment_to_nodes, node_i, node_j,
                           R0=1.0, distance_dependent=True):
    """
    Cheapest single series path between two nodes: Dijkstra with edge weight
    R_edge = R0 * L_edge (or R0 per edge under the uniform model).

    A single path carrying all the current is an UPPER BOUND on R_eff, so
    comparing it against the measured value says roughly how many comparable
    parallel routes the network offers. It is also the one number a reader can
    re-derive by hand from the edge model alone, which makes it the natural
    sanity check to print next to a probe result.

    Returns
    -------
    (resistance, hops, length) : the path's total resistance, its edge count and
    its total Euclidean length; (inf, None, None) if j is unreachable from i.
    """
    import heapq

    inter = np.asarray(intersections, dtype=float)
    adjacency = [[] for _ in range(len(inter))]
    for n1, n2 in segment_to_nodes.values():
        if n1 == n2:
            continue
        length = float(np.linalg.norm(inter[n2] - inter[n1]))
        weight = R0 * length if distance_dependent else R0
        adjacency[n1].append((n2, weight, length))
        adjacency[n2].append((n1, weight, length))

    best = {node_i: 0.0}
    hops = {node_i: 0}
    travelled = {node_i: 0.0}
    queue = [(0.0, node_i)]
    seen = set()
    while queue:
        cost, u = heapq.heappop(queue)
        if u in seen:
            continue
        seen.add(u)
        if u == node_j:
            return cost, hops[u], travelled[u]
        for v, weight, length in adjacency[u]:
            new_cost = cost + weight
            if new_cost < best.get(v, np.inf):
                best[v] = new_cost
                hops[v] = hops[u] + 1
                travelled[v] = travelled[u] + length
                heapq.heappush(queue, (new_cost, v))
    return float('inf'), None, None


def probe_summary(intersections, graph, segment_to_nodes, node_i, node_j,
                  R0=1000.0, distance_dependent=True):
    """
    Measure R_eff(node_i, node_j) and run every self-check that goes with it,
    returning numbers rather than printing them (see format_probe_summary for
    the text, edge_color_field for the plot).

    Both resistance models are solved on every call, so the distance-dependent
    value can always be read against the uniform one for the same pair.

    Checks included, and what each one would catch:
      pinv       R_eff recomputed as e^T L+ e from the weighted Laplacian
                 pseudoinverse, which shares no code with ImpedanceCalculator's
                 boundary-reduction solve. Catches a wrong solve.
      kcl        largest net current left at a non-probe node. Must be ~0 by
                 charge conservation; catches a linear system that did not
                 converge.
      model      R/L == R0 on every edge (or R == R0 under the uniform model).
                 Catches the edge model silently not being applied.
      series     cheapest single series path, an upper bound on R_eff. Catches
                 a result that is not physically plausible for the geometry.

    Returns a dict; 'r_eff' is the selected model's value, and every check is
    reported as a relative deviation so the caller can threshold it.
    """
    probe_dd = measure_r_eff(intersections, graph, segment_to_nodes, node_i, node_j,
                             k=R0, distance_dependent=True)
    probe_uniform = measure_r_eff(intersections, graph, segment_to_nodes, node_i, node_j,
                                  k=R0, distance_dependent=False)
    probe = probe_dd if distance_dependent else probe_uniform

    R_edges = np.asarray(probe['edge_resistances'], dtype=float)
    L_edges = np.asarray(probe['edge_lengths'], dtype=float)
    real = L_edges > 0

    summary = {
        'node_i': node_i,
        'node_j': node_j,
        'distance_dependent': distance_dependent,
        'R0': R0,
        'probe': probe,
        'r_eff': probe['impedance'],
        'r_eff_dd': probe_dd['impedance'],
        'r_eff_uniform': probe_uniform['impedance'],
        'probe_distance': probe['probe_distance'],
        'connected': np.isfinite(probe['impedance']),
        'n_edges': int(real.sum()),
    }

    # charge conservation at every node that is not a probe
    inject = probe['A'].T @ np.asarray(probe['currents'], dtype=float)
    internal = [n for n in range(len(intersections)) if n not in (node_i, node_j)]
    summary['kcl_residual'] = float(np.max(np.abs(inject[internal]))) if internal else 0.0

    # the edge model actually applied to the solve
    if real.any():
        R_real, L_real = R_edges[real], L_edges[real]
        if distance_dependent:
            summary['model_deviation'] = float(np.max(np.abs(R_real / L_real - R0)) / R0)
        else:
            summary['model_deviation'] = float(np.max(np.abs(R_real - R0)) / R0)
        summary['edge_stats'] = {
            'R_min': float(R_real.min()), 'R_max': float(R_real.max()),
            'R_mean': float(R_real.mean()), 'L_min': float(L_real.min()),
            'L_max': float(L_real.max()), 'L_mean': float(L_real.mean()),
            'spread': float(R_real.max() / R_real.min()),
        }
    else:
        summary['model_deviation'] = None
        summary['edge_stats'] = None

    # independent recomputation of the distance-dependent value
    if np.isfinite(probe_dd['impedance']) and probe_dd['impedance'] > 0:
        from laplacian_utils import effective_resistance_pinv
        r_pinv = effective_resistance_pinv(intersections, segment_to_nodes,
                                           node_i, node_j, R0=R0)
        summary['pinv'] = r_pinv
        summary['pinv_deviation'] = abs(r_pinv - probe_dd['impedance']) / probe_dd['impedance']
    else:
        summary['pinv'] = None
        summary['pinv_deviation'] = None

    path_r, path_hops, path_len = series_path_resistance(
        intersections, segment_to_nodes, node_i, node_j,
        R0=R0, distance_dependent=distance_dependent)
    summary['series_path'] = {'resistance': path_r, 'hops': path_hops, 'length': path_len}
    summary['parallel_factor'] = (path_r / probe['impedance']
                                  if np.isfinite(path_r) and probe['impedance'] > 0 else None)
    return summary


def format_probe_summary(summary, verbose=False, tolerance=1e-6):
    """
    Render probe_summary() as text: four lines by default, the full audit when
    verbose=True. Kept deliberately short because this runs on every click of an
    interactive widget, where a wall of numbers is harder to read than one line
    per question.
    """
    i, j, R0 = summary['node_i'], summary['node_j'], summary['R0']
    model = 'distance-dependent, R_edge = R0 * L' if summary['distance_dependent'] else 'uniform, R_edge = R0'
    lines = []

    if not summary['connected']:
        lines.append(f"R_eff({i}, {j}) = inf   (the two nodes are in different components, "
                     f"so no current can flow: correct, not a bug)")
        return "\n".join(lines)

    other = summary['r_eff_uniform'] if summary['distance_dependent'] else summary['r_eff_dd']
    other_name = 'uniform' if summary['distance_dependent'] else 'distance-dependent'
    lines.append(f"R_eff({i}, {j}) = {summary['r_eff']:.2f} Ω     [{model}, R0 = {R0:g}]")
    lines.append(f"  probe distance {summary['probe_distance']:.4f}"
                 f"     {other_name} model would read {other:.2f} Ω")

    def verdict(value):
        return "OK" if (value is not None and value < tolerance) else "CHECK"

    checks = [f"pinv {verdict(summary['pinv_deviation'])}",
              f"edge model {verdict(summary['model_deviation'])}",
              f"KCL {'OK' if summary['kcl_residual'] < 1e-9 else 'CHECK'}"]
    lines.append("  checks: " + "   ".join(checks) +
                 "      (tick 'detailed checks' for the numbers behind these)")

    path = summary['series_path']
    if summary['parallel_factor']:
        lines.append(f"  context: cheapest single path {path['resistance']:.0f} Ω over "
                     f"{path['hops']} edges, so the measured value is "
                     f"{summary['parallel_factor']:.1f}x lower, i.e. roughly that many "
                     f"comparable parallel routes")

    if not verbose:
        return "\n".join(lines)

    lines.append("")
    lines.append("  detailed checks")
    if summary['pinv'] is not None:
        lines.append(f"    independent pinv (e^T L+ e): {summary['pinv']:.4f} Ω, "
                     f"relative difference {summary['pinv_deviation']:.2e}")
    lines.append(f"    max KCL residual at a non-probe node: {summary['kcl_residual']:.2e} A "
                 f"(charge conservation, must be ~0)")
    if summary['edge_stats']:
        st = summary['edge_stats']
        invariant = "R/L == R0" if summary['distance_dependent'] else "R == R0"
        lines.append(f"    {invariant} on all {summary['n_edges']} edges: "
                     f"max relative deviation {summary['model_deviation']:.2e}")
        lines.append(f"    edge length  L: min={st['L_min']:.4g}  max={st['L_max']:.4g}  "
                     f"mean={st['L_mean']:.4g}")
        lines.append(f"    edge resist. R: min={st['R_min']:.4g}  max={st['R_max']:.4g}  "
                     f"mean={st['R_mean']:.4g}   spread {st['spread']:.1f}x "
                     f"(1.0x would mean length was ignored)")
    if summary['probe']['I_total'] > 0:
        lines.append(f"    I_total = {summary['probe']['I_total']:.4e} A, and R_eff = 1/I_total "
                     f"= {1.0 / summary['probe']['I_total']:.2f} Ω by definition (1 V applied)")
    return "\n".join(lines)


def edge_color_field(probe, quantity='current', floor_fraction=1e-4):
    """
    Per-edge values to colour a network plot with, for the three quantities worth
    looking at after a probe.

    'resistance' is a property of the NETWORK (R_edge = R0 * L_edge), so it looks
    identical for every probe pair; 'current' and 'power' are properties of the
    MEASUREMENT and are what change when the probes move.

    Current and power span many decades, since most edges carry essentially
    nothing, so they are floored at floor_fraction of the maximum: without that,
    the few live paths are the only distinguishable colour on a log scale and
    everything else collapses into one.

    Returns (values, colorbar_label, note).
    """
    R_edges = np.asarray(probe['edge_resistances'], dtype=float)
    I_edges = np.abs(np.asarray(probe['currents'], dtype=float))

    if quantity == 'resistance':
        return (R_edges, 'Edge resistance R0 * L (Ω)',
                'a property of the network: identical for every probe pair')

    finite_R = np.where(np.isfinite(R_edges), R_edges, 0.0)
    if quantity == 'current':
        values, label = I_edges, 'Edge current |I| (A)'
    elif quantity == 'power':
        values, label = I_edges ** 2 * finite_R, 'Power dissipated I^2 R (W)'
    else:
        raise ValueError(f"quantity must be 'current', 'resistance' or 'power', got {quantity!r}")

    peak = values.max() if values.size else 0.0
    if peak > 0:
        values = np.maximum(values, peak * floor_fraction)
    return values, label, f'changes with the probes; floored at {floor_fraction:g} of the maximum'
