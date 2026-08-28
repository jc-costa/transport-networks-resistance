#!/usr/bin/env python3
"""Test the merge_boundary_nodes function."""

from impedance_calculator import ImpedanceCalculator, merge_boundary_nodes
import numpy as np

# Test: 2x2 grid network
#   0 -┬- 2
#     ╳
#   1 -┴- 3
#
intersections = [
    [0.05, 0.6],  # Node 0 (source region)
    [0.05, 0.4],  # Node 1 (source region)
    [0.95, 0.6],  # Node 2 (drain region)
    [0.95, 0.4],  # Node 3 (drain region)
]

graph = [
    [0, 1],  # Node 0
    [2, 3],  # Node 1
    [0, 2],  # Node 2
    [1, 3],  # Node 3
]

segment_to_nodes = {
    0: (0, 2),  # 0 - 2
    1: (0, 3),  # 0 - 3
    2: (1, 2),  # 1 - 2  
    3: (1, 3),  # 1 - 3
}

source_boundary = [0, 1]
drain_boundary = [2, 3]
k = 1000

print('='*60)
print('ORIGINAL NETWORK')
print('='*60)
print(f'Nodes: {len(intersections)}')
print(f'Edges: {len(segment_to_nodes)}')
print(f'Source boundary: {source_boundary}')
print(f'Drain boundary: {drain_boundary}')

print()
print('='*60)
print('MERGING BOUNDARY NODES')
print('='*60)

(merged_int, merged_graph, merged_seg_to_nodes,
 src_node, drn_node, mapping, edge_conductance) = merge_boundary_nodes(
    intersections, graph, segment_to_nodes,
    source_boundary, drain_boundary
)

print(f'Merged nodes: {len(merged_int)}')
print(f'Merged edges: {len(merged_seg_to_nodes)}')
print(f'Source node: {src_node}')
print(f'Drain node: {drn_node}')
print(f'Intersections: {merged_int}')
print(f'Segment to nodes: {merged_seg_to_nodes}')
print(f'Graph: {merged_graph}')
print(f'Node mapping: {mapping}')
print(f'Edge conductance multipliers: {edge_conductance}')

print()
print('='*60)
print('IMPEDANCE CALCULATION ON MERGED NETWORK')
print('='*60)

# Pass edge_conductance to account for parallel resistors!
calc = ImpedanceCalculator(merged_int, merged_graph, merged_seg_to_nodes,
                           edge_conductance=edge_conductance)
result = calc.calculate(k=k, verbose=False)

print(f'Source: {calc.source_node}, Drain: {calc.drain_node}')
print(f'Impedance: R = {result["impedance"]:.2f} Ohms')

# Expected: 4 parallel resistors from merged source to merged drain
# R_parallel = k/4 = 250 Ohm
print()
print(f'Expected (4 parallel paths): R = k/4 = {k/4:.2f} Ohms')
