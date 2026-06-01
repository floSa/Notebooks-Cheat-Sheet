#!/usr/bin/env bash
# Dézippe l'archive source des notebooks vers 1_MES_NOTEBOOKS/ipynb/
# Idempotent : si le dossier de sortie existe déjà avec des .ipynb, ne refait rien.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_SRC="${ZIP_SRC:-/home/florian/mes_projets/Notebooks_ok/Notebooks.zip}"
OUT_DIR="$ROOT/1_MES_NOTEBOOKS/ipynb"

if [ -d "$OUT_DIR" ] && [ -n "$(find "$OUT_DIR" -maxdepth 2 -name '*.ipynb' -print -quit 2>/dev/null)" ]; then
    echo "[01_unzip] $OUT_DIR contient déjà des .ipynb — skip (rm -rf si vous voulez forcer)."
    exit 0
fi

mkdir -p "$OUT_DIR"
echo "[01_unzip] Extraction de $ZIP_SRC vers $OUT_DIR ..."
unzip -q -o "$ZIP_SRC" -d "$OUT_DIR.tmp"

# Le zip contient un dossier racine Notebooks/ — on aplatit.
if [ -d "$OUT_DIR.tmp/Notebooks" ]; then
    mv "$OUT_DIR.tmp/Notebooks"/* "$OUT_DIR/"
    rmdir "$OUT_DIR.tmp/Notebooks"
fi
rm -rf "$OUT_DIR.tmp"

n=$(find "$OUT_DIR" -maxdepth 2 -name '*.ipynb' | wc -l)
echo "[01_unzip] OK — $n fichiers .ipynb extraits dans $OUT_DIR"
