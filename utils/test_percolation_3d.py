"""
Validation suite for the 3D stick-percolation pipeline: hand-computed
geometry checks and hand-built network topologies with known equivalent
resistance, mirroring the pattern of utils/test_merge.py and
utils/test_networks.py for the 2D pipeline.

`run_*` functions execute and assert entirely on their own (no pipeline
needed) and print their own OK/FAIL lines. `build_*_case` functions instead
return (points_1, points_2, expected_R_eff, label) for a hand-built stick
network -- the caller (typically a notebook cell) runs it through
run_full_pipeline_3d, asserts against expected_R_eff, and visualizes it.
Keeping the geometry-design logic here, separate from execution/plotting,
is what lets the notebook stay a thin "run it and look at it" layer.
"""
import numpy as np

from intersection_utils_3d import closest_points_between_segments, identify_contacts_3d


def run_geometry_checks(verbose=True):
    """
    Hand-computed closest-distance cases for closest_points_between_segments:
    the general (skew) case, the near-threshold boundary, the parallel/
    degenerate case (with and without overlapping extents), and the collinear
    case. Raises AssertionError on failure.
    """
    def check(name, dist, expected, tol=1e-9):
        ok = abs(dist - expected) < tol
        if verbose:
            print(f"{'OK  ' if ok else 'FAIL'} {name}: got {dist:.6f}, expected {expected:.6f}")
        assert ok, name

    _, _, d = closest_points_between_segments((0, 0, 0), (1, 0, 0), (0.5, -1, 1), (0.5, 1, 1))
    check("skew, far apart", d, 1.0)

    for h, label in [(0.05, "below tol"), (0.15, "above tol")]:
        _, _, d = closest_points_between_segments((0, 0, 0), (1, 0, 0), (0.5, 0, h), (0.5, 2, h))
        check(f"perpendicular near-touch, {label} (h={h})", d, h)

    _, _, d = closest_points_between_segments((0, 0, 0), (1, 0, 0), (0.2, 0, 0.3), (0.8, 0, 0.3))
    check("parallel, overlapping extents, offset 0.3", d, 0.3)

    _, _, d = closest_points_between_segments((0, 0, 0), (1, 0, 0), (1.5, 0.4, 0), (2.5, 0.4, 0))
    expected = np.linalg.norm(np.array([1, 0, 0]) - np.array([1.5, 0.4, 0]))
    check("parallel, non-overlapping extents (endpoint clamp)", d, expected)

    _, _, d = closest_points_between_segments((0, 0, 0), (1, 0, 0), (1.0, 0, 0), (2, 0, 0))
    check("collinear, touching end-to-end", d, 0.0)
    _, _, d = closest_points_between_segments((0, 0, 0), (1, 0, 0), (1.5, 0, 0), (2, 0, 0))
    check("collinear, with a gap", d, 0.5)

    if verbose:
        print("\nAll geometry unit tests passed.")


def run_symmetry_check(verbose=True):
    """A contact must not depend on which stick is processed first."""
    p1 = [(0, 0, 0), (0.5, -1, 0.05)]
    p2 = [(1, 0, 0), (0.5, 1, 0.05)]
    nodes_a, _, _ = identify_contacts_3d(p1, p2, contact_distance=0.2)

    p1_swapped, p2_swapped = [p1[1], p1[0]], [p2[1], p2[0]]
    nodes_b, _, _ = identify_contacts_3d(p1_swapped, p2_swapped, contact_distance=0.2)

    if verbose:
        print("node from (i,j) order:", nodes_a[0])
        print("node from (j,i) order:", nodes_b[0])
    assert np.allclose(nodes_a[0], nodes_b[0]), "contact node depends on stick order!"
    if verbose:
        print("OK: contact node is order-independent")


def run_isotropy_check(n_samples=20000, seed=0, verbose=True):
    """
    Draws n_samples directions from random_unit_vector_3d and returns them,
    printing the isotropy diagnostics (mean direction ~0, norm ~1, mean z^2
    ~1/3 for a truly uniform sphere -- pole-biased sampling would push this
    above 1/3).
    """
    from point_generator_3d import random_unit_vector_3d

    np.random.seed(seed)
    dirs = np.array([random_unit_vector_3d() for _ in range(n_samples)])
    norms = np.linalg.norm(dirs, axis=1)

    if verbose:
        print(f"norm mean/std (expect ~1, ~0):        {norms.mean():.6f} / {norms.std():.2e}")
        print(f"mean direction (expect ~0,0,0):        {dirs.mean(axis=0)}")
        print(f"mean z^2 (expect ~0.3333 for uniform): {(dirs[:, 2] ** 2).mean():.4f}")

    return dirs


def build_single_edge_case():
    """One backbone stick + two tabs crossing it near x=0 and x=1: one edge,
    R_eff = k exactly."""
    points_1 = [(0.05, 0.0, 0.0), (0.05, -0.1, 0.0), (0.95, -0.1, 0.0)]
    points_2 = [(0.95, 0.0, 0.0), (0.05, 0.1, 0.0), (0.95, 0.1, 0.0)]
    return points_1, points_2, 1000.0, "Single edge"


def build_ladder_case():
    """Two rails offset in z, joined by two rungs: two k-resistors in
    parallel, R_eff = k/2."""
    points_1 = [
        (0.05, 0.0, 0.0), (0.05, 0.0, 0.2),
        (0.05, 0.0, 0.0),
        (0.95, 0.0, 0.0),
    ]
    points_2 = [
        (0.95, 0.0, 0.0), (0.95, 0.0, 0.2),
        (0.05, 0.0, 0.2),
        (0.95, 0.0, 0.2),
    ]
    return points_1, points_2, 500.0, "3D ladder"


def build_triple_parallel_case():
    """Three rails offset in z, joined by two long rungs spanning all three:
    three k-resistors in parallel, R_eff = k/3."""
    points_1 = [
        (0.05, 0, 0), (0.05, 0, 0.2), (0.05, 0, 0.4),
        (0.05, 0, -0.05),
        (0.95, 0, -0.05),
    ]
    points_2 = [
        (0.95, 0, 0), (0.95, 0, 0.2), (0.95, 0, 0.4),
        (0.05, 0, 0.45),
        (0.95, 0, 0.45),
    ]
    return points_1, points_2, 1000.0 / 3, "Triple-parallel rails"


def build_three_branch_star_case():
    """One direct backbone plus two independent 2-edge paths through
    separate waypoints: R1=k, R2=R3=2k in parallel, R_eff = k/2."""
    points_1 = [
        (0.05, -0.3, 0), (0.95, -0.3, 0),
        (0.05, 0, 0),
        (0.05, 0.15, 0), (0.5, 0.15, 0.3),
        (0.05, -0.15, 0), (0.5, -0.15, -0.3),
    ]
    points_2 = [
        (0.05, 0.3, 0), (0.95, 0.3, 0),
        (0.95, 0, 0),
        (0.5, 0.15, 0.3), (0.95, 0.15, 0),
        (0.5, -0.15, -0.3), (0.95, -0.15, 0),
    ]
    return points_1, points_2, 500.0, "Three-branch star"


def build_dangling_branch_case():
    """
    Returns (points_1_clean, points_2_clean, points_1_dangling, points_2_dangling, expected_R_eff).
    Clean: backbone split into 2 series edges by one crossing stick (R=2k).
    Dangling: same, plus one stick that touches the backbone at the same mid
    point and continues to a dead end capped by a stick touching nothing
    else -- a genuine degree-1 internal node that pruning must remove. Both
    variants must give the SAME expected R_eff (the dead branch changes
    nothing).
    """
    points_1_clean = [(0.05, -0.1, 0), (0.95, -0.1, 0), (0.05, 0, 0), (0.5, -0.1, 0)]
    points_2_clean = [(0.05, 0.1, 0), (0.95, 0.1, 0), (0.95, 0, 0), (0.5, 0.1, 0)]

    points_1_dangling = points_1_clean[:3] + [(0.5, 0, 0), (0.4, 0.3, 0.3)]
    points_2_dangling = points_2_clean[:3] + [(0.5, 0.3, 0.3), (0.6, 0.3, 0.3)]

    return points_1_clean, points_2_clean, points_1_dangling, points_2_dangling, 2000.0


def build_wheatstone_bridge_case():
    """
    Genuine cycle topology (S-A, S-B, A-D, B-D, plus bridging A-B) -- not
    series/parallel reducible. Two degree-3 junctions (A, B) are built from
    pairwise stick contacts via short "hub" sticks (length ~0.001), touched by
    their three arms at three distinct but very close points; the hub's own
    tiny internal edges are negligible under the distance-dependent model
    (R0=1000 -> well under 1 Ohm, vs. ~300-700 Ohm arms).

    Arm lengths are deliberately asymmetric, so there's no shortcut hand
    formula -- verify the returned network against an independent method
    (e.g. laplacian_utils.effective_resistance_pinv), not a fixed expected value.
    """
    tab_S = [(0.05, -0.3, 0), (0.05, 0.3, 0)]
    tab_D = [(0.95, -0.3, 0), (0.95, 0.3, 0)]
    hub_A = [(0.35, -0.0005, 0), (0.35, 0.0005, 0)]
    hub_B = [(0.35, -0.0005, 0.3), (0.35, 0.0005, 0.3)]
    arm_SA = [(0.05, 0.2, 0), (0.35, -0.0003, 0)]
    arm_AD = [(0.35, 0.0003, 0), (0.95, 0.2, 0)]
    arm_SB = [(0.05, -0.2, 0), (0.35, -0.0003, 0.3)]
    arm_BD = [(0.35, 0.0003, 0.3), (0.95, -0.2, 0)]
    arm_AB = [(0.35, 0, 0), (0.35, 0, 0.3)]

    sticks = [tab_S, tab_D, hub_A, hub_B, arm_SA, arm_AD, arm_SB, arm_BD, arm_AB]
    points_1 = [s[0] for s in sticks]
    points_2 = [s[1] for s in sticks]
    return points_1, points_2, "Wheatstone bridge"


if __name__ == "__main__":
    # The self-contained checks (geometry, symmetry, isotropy) assert on their own
    # and are safe to run here. The build_*_case functions instead return sticks
    # for a caller to push through the pipeline, so they are exercised from
    # notebooks/percolation_3d.ipynb rather than from this entry point.
    run_geometry_checks()
    run_symmetry_check()
    run_isotropy_check()
    print("\nStandalone checks complete. For the hand-built R_eff cases, "
          "see notebooks/percolation_3d.ipynb section 3.")
