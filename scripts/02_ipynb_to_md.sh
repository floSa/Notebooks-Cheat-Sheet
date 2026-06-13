#!/usr/bin/env bash
# Convertit tous les .ipynb de Notebook_2018-2021/ipynb/ vers .md dans Notebook_2018-2021/md/
# Utilise jupytext via uv. Idempotent : écrase la sortie existante.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/Notebook_2018-2021/ipynb"
OUT_DIR="$ROOT/Notebook_2018-2021/md"

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
