#!/usr/bin/env python3
"""Test measure_r_eff() and the onion-layer protocol against known networks."""

from multimeter import measure_r_eff, onion_layers, run_onion_protocol, run_onion_protocol_pinv
from laplacian_utils import effective_resistance_from_reference

# ------------------------------------------------------------------
# Test 1: plain 4-node ring (0-1-2-3-0), unit resistors (k=1).
# Two paths of length 2 in parallel between opposite nodes:
#   R_eff(0,2) = (1+1)*(1+1) / (2+2) = 1.0
#   R_eff(0,1) = adjacent nodes: 1 || 3 = 3/4 = 0.75
# ------------------------------------------------------------------
intersections = [
    [0.0, 0.5],  # 0
    [0.5, 1.0],  # 1
    [1.0, 0.5],  # 2
    [0.5, 0.0],  # 3
]
graph = [[0, 3], [0, 1], [1, 2], [2, 3]]
segment_to_nodes = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}

print("=" * 60)
print("TEST 1: 4-node ring, unit resistors")
print("=" * 60)

r02 = measure_r_eff(intersections, graph, segment_to_nodes, 0, 2, k=1.0)
print(f"R_eff(0,2) = {r02['impedance']:.4f}  (expected 1.0000)")
assert abs(r02['impedance'] - 1.0) < 1e-9

r01 = measure_r_eff(intersections, graph, segment_to_nodes, 0, 1, k=1.0)
print(f"R_eff(0,1) = {r01['impedance']:.4f}  (expected 0.7500)")
assert abs(r01['impedance'] - 0.75) < 1e-9

print(f"probe_distance(0,2) = {r02['probe_distance']:.4f}  (expected 1.0000, diagonal)")
assert abs(r02['probe_distance'] - 1.0) < 1e-9

print("PASSED")

# ------------------------------------------------------------------
# Test 2: disconnected pair -> infinite resistance.
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 2: disconnected node pair")
print("=" * 60)
intersections_disc = intersections + [[2.0, 2.0]]  # node 4, isolated
graph_disc = graph + [[]]
r_disc = measure_r_eff(intersections_disc, graph_disc, segment_to_nodes, 0, 4, k=1.0)
print(f"R_eff(0,4) = {r_disc['impedance']}  (expected inf)")
assert r_disc['impedance'] == float('inf')
print("PASSED")

# ------------------------------------------------------------------
# Test 3: onion_layers groups nodes on a straight line by distance from
# node 0 at the origin, into the right shells.
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 3: onion_layers on a straight line of nodes")
print("=" * 60)
line_intersections = [[0.1 * i, 0.0] for i in range(10)]  # nodes 0..9 at x=0.0..0.9
layers = onion_layers(line_intersections, reference_node=0,
                      layer_distances=[0.1, 0.2, 0.3], tolerance=0.01)
print(f"layers = {layers}")
assert layers[0.1] == [1]
assert layers[0.2] == [2]
assert layers[0.3] == [3]
print("PASSED")

# ------------------------------------------------------------------
# Test 4: run_onion_protocol on a resistor ladder (nodes 0..9 in a line,
# unit resistors between consecutive nodes). R_eff(0, n) should equal n
# (series resistors), so R_eff vs. distance should come back linear.
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 4: run_onion_protocol on a resistor ladder")
print("=" * 60)
ladder_graph = [[] for _ in range(10)]
ladder_segments = {}
for i in range(9):
    ladder_segments[i] = (i, i + 1)
    ladder_graph[i].append(i)
    ladder_graph[i + 1].append(i)

records = run_onion_protocol(line_intersections, ladder_graph, ladder_segments,
                             reference_node=0,
                             layer_distances=[0.1, 0.2, 0.3, 0.4, 0.5],
                             tolerance=0.01, k=1.0)
for rec in sorted(records, key=lambda r: r['layer_distance']):
    n = rec['node_j']
    expected_r = n  # n unit resistors in series from node 0 to node n
    print(f"  d={rec['layer_distance']:.1f}  node_j={n}  "
          f"r_eff={rec['r_eff']:.4f}  (expected {expected_r})")
    assert abs(rec['r_eff'] - expected_r) < 1e-9
print("PASSED")


# ------------------------------------------------------------------
# Test 5: effective_resistance_from_reference (the pinv fast path) on the
# same disconnected pair as Test 2 -> inf, same as the solve-based path.
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 5: effective_resistance_from_reference on a disconnected pair")
print("=" * 60)
r_eff_vec = effective_resistance_from_reference(intersections_disc, segment_to_nodes, 0, R0=1.0)
print(f"r_eff[4] = {r_eff_vec[4]}  (expected inf)")
assert r_eff_vec[4] == float('inf')
print("PASSED")

# ------------------------------------------------------------------
# Test 6: run_onion_protocol_pinv vs. run_onion_protocol on the same
# resistor ladder. The ladder's nodes are evenly spaced (0.1 apart), so the
# distance-dependent model R_edge = R0 * length reduces to a uniform
# per-edge resistance of R0 * 0.1 -- picking R0 = 10 reproduces Test 4's
# k=1.0 ladder exactly, so both the pinv fast path's own values (r_eff = n)
# and its agreement with the slow solve-based path (distance_dependent=True)
# can be checked against the same closed-form expectation.
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 6: run_onion_protocol_pinv vs. run_onion_protocol (resistor ladder)")
print("=" * 60)
R0 = 10.0  # R_edge = R0 * 0.1 = 1.0, matching Test 4's k=1.0

fast_records = run_onion_protocol_pinv(line_intersections, ladder_segments,
                                       reference_node=0,
                                       layer_distances=[0.1, 0.2, 0.3, 0.4, 0.5],
                                       tolerance=0.01, R0=R0)
slow_records = run_onion_protocol(line_intersections, ladder_graph, ladder_segments,
                                  reference_node=0,
                                  layer_distances=[0.1, 0.2, 0.3, 0.4, 0.5],
                                  tolerance=0.01, k=R0, distance_dependent=True)

fast_by_j = {rec['node_j']: rec['r_eff'] for rec in fast_records}
slow_by_j = {rec['node_j']: rec['r_eff'] for rec in slow_records}
assert set(fast_by_j) == set(slow_by_j)

for j in sorted(fast_by_j):
    expected_r = j  # n unit resistors in series from node 0 to node n
    fast_r, slow_r = fast_by_j[j], slow_by_j[j]
    print(f"  node_j={j}  fast={fast_r:.6f}  slow={slow_r:.6f}  (expected {expected_r})")
    assert abs(fast_r - expected_r) < 1e-6
    assert abs(fast_r - slow_r) < 1e-6 * max(1.0, abs(slow_r))
print("PASSED")

print()
print("ALL TESTS PASSED")
