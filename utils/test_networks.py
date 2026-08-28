"""
Predefined test networks for self-loop validation.
All networks are assumed to be already merged (i.e., after merge_boundary_nodes)
so that source/drain are single nodes at min/max x.
"""

# ----------------------------------------------------------------------
# Test 1 – Linear chain (3 nodes)
# Node 0: internal (x=0.5)
# Node 1: source   (x=0.05)
# Node 2: drain    (x=0.95)
# ----------------------------------------------------------------------
intersections_t1 = [
    [0.50, 0.50],   # node 0 internal
    [0.05, 0.50],   # node 1 source
    [0.95, 0.50],   # node 2 drain
]
graph_t1 = [
    [0, 1],  # node 0: edges 0 and 1
    [0],     # node 1: edge 0
    [1],     # node 2: edge 1
]
seg_t1 = {0: (0, 1), 1: (0, 2)}
src_t1 = 1   # min-x
drn_t1 = 2   # max-x

# ----------------------------------------------------------------------
# Test 2 – Ladder network (4 nodes)
# Node 0: I1 (internal, x=0.35)
# Node 1: I2 (internal, x=0.65)
# Node 2: source (x=0.05)
# Node 3: drain (x=0.95)
# ----------------------------------------------------------------------
intersections_t2 = [
    [0.35, 0.70],   # node 0 I1
    [0.65, 0.30],   # node 1 I2
    [0.05, 0.50],   # node 2 Source
    [0.95, 0.50],   # node 3 Drain
]
seg_t2 = {
    0: (0, 2),   # I1 ↔ Source
    1: (1, 2),   # I2 ↔ Source
    2: (0, 3),   # I1 ↔ Drain
    3: (1, 3),   # I2 ↔ Drain
    4: (0, 1),   # I1 ↔ I2   (rung)
}
graph_t2 = [
    [0, 2, 4],   # node 0
    [1, 3, 4],   # node 1
    [0, 1],      # node 2
    [2, 3],      # node 3
]
src_t2 = 2
drn_t2 = 3

# ----------------------------------------------------------------------
# Test 7 – 5‑node serial chain (post‑merge layout)
# Node 0: I1 (x=0.30)
# Node 1: I2 (x=0.50)
# Node 2: I3 (x=0.70)
# Node 3: Source (x=0.05)
# Node 4: Drain  (x=0.95)
# ----------------------------------------------------------------------
intersections_t7 = [
    [0.30, 0.50],   # node 0 I1
    [0.50, 0.50],   # node 1 I2
    [0.70, 0.50],   # node 2 I3
    [0.05, 0.50],   # node 3 Source
    [0.95, 0.50],   # node 4 Drain
]
seg_t7 = {
    0: (3, 0),   # Source ↔ I1
    1: (0, 1),   # I1     ↔ I2
    2: (1, 2),   # I2     ↔ I3
    3: (2, 4),   # I3     ↔ Drain
}
graph_t7 = [
    [0, 1],    # node 0 I1
    [1, 2],    # node 1 I2
    [2, 3],    # node 2 I3
    [0],       # node 3 Source
    [3],       # node 4 Drain
]
src_t7 = 3
drn_t7 = 4

# ----------------------------------------------------------------------
# Common test resistance per segment
k_test = 1000.0   # Ohms