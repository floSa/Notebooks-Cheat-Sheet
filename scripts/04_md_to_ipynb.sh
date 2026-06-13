#!/usr/bin/env bash
# Convertit tous les .md de Notebook_2026/md/ vers .ipynb dans Notebook_2026/ipynb/
# Utilise jupytext via uv. Idempotent : écrase la sortie existante.
# Usage: ./04_md_to_ipynb.sh             # tout convertir
#        ./04_md_to_ipynb.sh fichier.md  # un seul fichier

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$ROOT/Notebook_2026/md"
OUT_DIR="$ROOT/Notebook_2026/ipynb"

mkdir -p "$OUT_DIR"

if [ $# -ge 1 ]; then
    targets=("$@")
else
    shopt -s nullglob
    targets=("$SRC_DIR"/*.md)
fi

fail=0
ok=0
for md in "${targets[@]}"; do
    # accepter chemin absolu ou nom relatif à Notebook_2026/md/
    if [ ! -f "$md" ]; then
        md="$SRC_DIR/$md"
    fi
    base="$(basename "$md" .md)"
    out="$OUT_DIR/$base.ipynb"
    echo "[04_to_ipynb] $base"
    if uv run jupytext --to ipynb --output "$out" "$md" >/dev/null 2>&1; then
        ok=$((ok + 1))
    else
        echo "  ⚠️  ÉCHEC sur $base"
        fail=$((fail + 1))
    fi
done
echo "[04_to_ipynb] Terminé — OK: $ok, échecs: $fail"
[ "$fail" -eq 0 ]
