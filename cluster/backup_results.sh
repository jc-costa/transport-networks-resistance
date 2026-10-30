#!/bin/bash
# Timestamped backup of every cluster results directory, to run BEFORE any
# rerun that overwrites chunk_*.json.
#
# The task scripts always recompute and overwrite their own chunk file (only
# the notebooks skip work when a complete chunk set exists), so a rerun
# silently replaces whatever is in these directories. They hold real results
# synced back from Apuana, which is why this exists: back up first, then
# submit.
#
# Usage:  bash backup_results.sh            (from anywhere)
# Output: cluster/backups/<timestamp>/ plus MANIFEST.sha256 for later diffing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DIRS=(effective_resistance_d_results onion_grid_results onion_grid_boost_results validation_results critical_sweep_results preview_curves_results)
FILES=(onion_grid_boost_plan.json validation_plan.json)

TS="$(date +%Y%m%d_%H%M%S)"
DEST="backups/$TS"

if [ -e "$DEST" ]; then
    echo "ERROR: $DEST already exists, refusing to overwrite." >&2
    exit 1
fi

mkdir -p "$DEST"
echo "Backing up into cluster/$DEST"

for d in "${DIRS[@]}"; do
    if [ -d "$d" ]; then
        cp -R "$d" "$DEST/"
        echo "  $d  ($(find "$d" -type f | wc -l | tr -d ' ') files, $(du -sh "$d" | cut -f1))"
    else
        echo "  $d  (absent, skipped)"
    fi
done

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        cp "$f" "$DEST/"
        echo "  $f"
    fi
done

# Checksums, so a later rerun can be diffed against this snapshot with
# certainty about what changed.
( cd "$DEST" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256 )

echo
echo "Done: $(du -sh "$DEST" | cut -f1) in cluster/$DEST"
echo "Verify a later rerun with:"
echo "  python verify_rerun_vs_backup.py $DEST"
