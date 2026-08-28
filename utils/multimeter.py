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
