#!/usr/bin/env bash
# Convertit tous les .ipynb de 1_Old_Notebooks/ipynb/ vers .md dans 1_Old_Notebooks/md/
# Utilise jupytext via uv. Idempotent : écrase la sortie existante.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/1_Old_Notebooks/ipynb"
OUT_DIR="$ROOT/1_Old_Notebooks/md"

mkdir -p "$OUT_DIR"

shopt -s nullglob
fail=0
ok=0
for nb in "$SRC_DIR"/*.ipynb; do
    base="$(basename "$nb" .ipynb)"
    out="$OUT_DIR/$base.md"
    echo "[02_to_md] $base"
    if uv run jupytext --to md --output "$out" "$nb" >/dev/null 2>&1; then
        ok=$((ok + 1))
    else
        echo "  ⚠️  ÉCHEC sur $base"
        fail=$((fail + 1))
    fi
done
echo "[02_to_md] Terminé — OK: $ok, échecs: $fail"
[ "$fail" -eq 0 ]
