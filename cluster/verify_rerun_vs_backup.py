"""
Check a rerun of the grid arrays against a pre-rerun backup snapshot.

Context: `pipeline_utils.distance_dependent_impedance` used to measure edge
lengths on the MERGED geometry, where merge_boundary_nodes parks each terminal
at the average position of the face it replaces, inflating the resistance of
every edge touching a terminal. Fixed 2026-08-31 to measure lengths on the
pre-merge geometry.

That fix has a sharp, checkable signature, and this script tests exactly it:

  * the SCALAR R_eff column (R_eff_values / mean_ohm) must change, downward,
    since the old values were biased high;
  * the ONION CURVES must not move at all. run_onion_protocol_pinv never
    merges anything, and the seeds are unchanged, so identical networks must
    produce identical curves. Any change here means something OTHER than the
    merge fix altered the pipeline, and the rerun should not be trusted;
  * the network draws themselves (n, perc_rate) must be identical too, for
    the same reason.

Usage:
    python verify_rerun_vs_backup.py backups/<timestamp>
    python verify_rerun_vs_backup.py backups/<timestamp> --dir onion_grid_results

Exit code 0 if everything matches expectations, 1 otherwise.
"""
import sys
import os
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

RESULT_DIRS = [
    "effective_resistance_d_results",
    "onion_grid_results",
    "onion_grid_boost_results",
]

CURVE_TOL = 1e-9      # relative; same machine + same seeds should be far tighter
SCALAR_EPS = 1e-12


def max_rel_deviation(old, new, path=""):
    """Largest relative difference between two nested JSON structures.

    The onion curve payload is a dict of per-layer dicts ('mean', 'std',
    'n_realizations', 'per_realization_means'), so this walks dicts and lists
    rather than assuming a flat list of numbers.

    Returns (worst_relative_deviation, note): note is a string describing a
    structural mismatch (different keys, different lengths, one side None) and
    is None when the two structures line up.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            return 0.0, f"different keys at {path or 'top level'}"
        worst = 0.0
        for k in old:
            w, note = max_rel_deviation(old[k], new[k], f"{path}/{k}")
            if note:
                return 0.0, note
            worst = max(worst, w)
        return worst, None

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return 0.0, f"length {len(old)} -> {len(new)} at {path}"
        worst = 0.0
        for i, (o, n) in enumerate(zip(old, new)):
            w, note = max_rel_deviation(o, n, f"{path}[{i}]")
            if note:
                return 0.0, note
            worst = max(worst, w)
        return worst, None

    if old is None or new is None:
        return (0.0, None) if old == new else (0.0, f"None vs value at {path}")

    if isinstance(old, bool) or isinstance(new, bool) or isinstance(old, str):
        return (0.0, None) if old == new else (0.0, f"{old!r} -> {new!r} at {path}")

    denom = abs(old) if abs(old) > SCALAR_EPS else 1.0
    return abs(new - old) / denom, None


def load_chunks(directory):
    """All records from chunk_*.json in `directory`, keyed by (N, a, d)."""
    records = {}
    if not os.path.isdir(directory):
        return records
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("chunk_") and name.endswith(".json")):
            continue
        with open(os.path.join(directory, name)) as fh:
            for rec in json.load(fh):
                records[(rec["N"], rec["a"], rec["d"])] = rec
    return records


def rel_diff(new, old):
    if old in (None, 0) or new is None:
        return None
    return (new - old) / abs(old)


def compare_dir(name, backup_root):
    old_records = load_chunks(os.path.join(backup_root, name))
    new_records = load_chunks(os.path.join(HERE, name))

    if not old_records:
        print(f"\n[{name}] not present in the backup, skipping")
        return True
    if not new_records:
        print(f"\n[{name}] not present in cluster/, skipping")
        return True

    print(f"\n[{name}]")
    print(f"  combos: {len(old_records)} in backup, {len(new_records)} now")

    ok = True
    missing = set(old_records) - set(new_records)
    added = set(new_records) - set(old_records)
    if missing:
        print(f"  MISMATCH: {len(missing)} combo(s) present in the backup but not in the rerun, "
              f"e.g. {sorted(missing)[:3]}")
        ok = False
    if added:
        print(f"  note: {len(added)} combo(s) new in the rerun, e.g. {sorted(added)[:3]}")

    draw_mismatch = []
    curve_mismatch = []
    scalar_changes = []
    scalar_unchanged = 0

    for key in sorted(set(old_records) & set(new_records)):
        old, new = old_records[key], new_records[key]

        # 1. the networks drawn must be identical (same seeds, untouched generator)
        if old.get("n") != new.get("n") or old.get("perc_rate") != new.get("perc_rate"):
            draw_mismatch.append((key, old.get("n"), new.get("n"),
                                  old.get("perc_rate"), new.get("perc_rate")))

        # 2. onion curves must be identical (no merge involved anywhere)
        old_curves = old.get("onion_layer_curves") or {}
        new_curves = new.get("onion_layer_curves") or {}
        worst, note = max_rel_deviation(old_curves, new_curves)
        if note:
            curve_mismatch.append((key, note))
        elif worst > CURVE_TOL:
            curve_mismatch.append((key, f"max rel. deviation {worst:.2e}"))

        # 3. the scalar column is what the fix was supposed to move
        d = rel_diff(new.get("mean_ohm"), old.get("mean_ohm"))
        if d is None:
            continue
        if abs(d) < SCALAR_EPS:
            scalar_unchanged += 1
        else:
            scalar_changes.append((key, old["mean_ohm"], new["mean_ohm"], d))

    if draw_mismatch:
        ok = False
        print(f"  MISMATCH: {len(draw_mismatch)} combo(s) drew different networks "
              f"(n / perc_rate changed). Seeds should be unchanged, so this is NOT the "
              f"merge fix. First: {draw_mismatch[0]}")
    else:
        print("  OK: identical network draws (n, perc_rate) on every combo")

    if curve_mismatch:
        ok = False
        print(f"  MISMATCH: {len(curve_mismatch)} combo(s) have different onion curves; "
              f"the pinv path does no merging, so this should be impossible. "
              f"First: {curve_mismatch[0]}")
    elif any((r.get("onion_layer_curves") for r in old_records.values())):
        print("  OK: onion curves bit-for-bit unchanged (as the fix predicts)")

    if scalar_changes:
        drops = [d for _, _, _, d in scalar_changes]
        n_down = sum(1 for d in drops if d < 0)
        worst_key, worst_old, worst_new, worst_d = min(scalar_changes, key=lambda t: t[3])
        print(f"  scalar R_eff changed on {len(scalar_changes)} combo(s) "
              f"({scalar_unchanged} unchanged):")
        print(f"    direction: {n_down} down, {len(drops) - n_down} up "
              f"(the fix removes an upward bias, so down is expected)")
        print(f"    median change {sorted(drops)[len(drops)//2]*100:+.2f}%, "
              f"largest {worst_d*100:+.2f}% at (N,a,d)={worst_key} "
              f"({worst_old:.2f} -> {worst_new:.2f} ohm)")
        if len(drops) - n_down:
            print("    note: some combos went UP; possible when a face's centroid happened to "
                  "sit closer than the true endpoints, so this is not by itself an error")
    else:
        print("  WARNING: no scalar R_eff changed at all. Either this directory was not "
              "rerun yet, or the rerun used the old code (check that utils/ on the cluster "
              "is the fixed version).")
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backup", help="backup directory made by backup_results.sh, "
                                       "e.g. backups/20260831_111505")
    parser.add_argument("--dir", action="append", dest="dirs",
                        help="check only this results directory (repeatable)")
    args = parser.parse_args()

    backup_root = args.backup
    if not os.path.isabs(backup_root):
        backup_root = os.path.join(HERE, backup_root)
    if not os.path.isdir(backup_root):
        print(f"ERROR: no such backup directory: {backup_root}", file=sys.stderr)
        return 2

    print(f"Comparing cluster/ against backup {os.path.relpath(backup_root, HERE)}")
    targets = args.dirs or RESULT_DIRS
    all_ok = all([compare_dir(name, backup_root) for name in targets])

    print()
    if all_ok:
        print("PASSED: scalar R_eff moved, everything that should be identical is identical.")
    else:
        print("FAILED: see the mismatches above before trusting the rerun.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
