#!/usr/bin/env bash
# restore_originaux.sh
# Recrée 1_MES_NOTEBOOKS/ipynb/ depuis le ZIP source des VRAIS originaux.
#
# ⚠️ NE PAS restaurer depuis `main` : la branche main contient des versions GUTÉES
#    (réduites de 200+ à ~20 cellules). La SEULE source de vérité est 1_MES_NOTEBOOKS/Notebooks.zip.
#
# 1_MES_NOTEBOOKS/ipynb/ est gitignoré (lourd, reproductible) — le zip est la vérité versionnée.
# Usage : bash scripts/restore_originaux.sh

set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
ZIP="$REPO_ROOT/1_MES_NOTEBOOKS/Notebooks.zip"
OUT="$REPO_ROOT/1_MES_NOTEBOOKS/ipynb"

if [[ ! -f "$ZIP" ]]; then
  echo "❌ ERREUR : $ZIP introuvable."
  echo "   Place le zip des vrais originaux dans 1_MES_NOTEBOOKS/Notebooks.zip puis relance."
  exit 1
fi

echo "📥 Restauration des originaux depuis $ZIP → $OUT"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
unzip -q "$ZIP" -d "$TMP"

# Le zip contient un dossier Notebooks/ ; on aplatit tous les .ipynb dans OUT (noms d'origine).
find "$TMP" -name '*.ipynb' -exec cp {} "$OUT/" \;
rm -rf "$TMP"

N=$(ls "$OUT"/*.ipynb 2>/dev/null | wc -l)
echo "✅ $N notebooks originaux restaurés dans 1_MES_NOTEBOOKS/ipynb/"
echo "   (source de vérité — ne pas les modifier ; cf. 1_MES_NOTEBOOKS/00_REFERENCE_ORIGINAUX.md)"
