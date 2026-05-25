#!/usr/bin/env bash
# Télécharge les datasets nécessaires aux notebooks "case study".
# Les datasets `sklearn`/`seaborn`/`HF datasets` sont chargés à la volée — pas ici.
# Ce script ne télécharge que ceux qui demandent un fichier persistant.
#
# Usage : bash scripts/download_data.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/_shared"

mkdir -p "$DATA"

# ============================================================================
# 1. NYC Taxi parquet (un mois) — BDD_DuckDB
# ============================================================================
NYC_DIR="$DATA/nyc_taxi"
NYC_FILE="$NYC_DIR/yellow_2024-01.parquet"
mkdir -p "$NYC_DIR"

if [ ! -f "$NYC_FILE" ]; then
    echo "[NYC Taxi] Téléchargement (≈50MB)..."
    URL="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$NYC_FILE" "$URL"
    else
        wget -O "$NYC_FILE" "$URL"
    fi
    echo "[NYC Taxi] OK → $NYC_FILE"
else
    echo "[NYC Taxi] déjà téléchargé → skip"
fi

# README
cat > "$NYC_DIR/README.md" <<'EOF'
# NYC Yellow Taxi — janvier 2024

Source : <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
Format : Parquet
Volume : ~50 MB, ~3M lignes

## Schéma principal

| Colonne | Type | Description |
|---|---|---|
| VendorID | int | Vendor |
| tpep_pickup_datetime | timestamp | Début de course |
| tpep_dropoff_datetime | timestamp | Fin |
| passenger_count | int | Nombre de passagers |
| trip_distance | float | Miles |
| fare_amount | float | Tarif USD |
| tip_amount | float | Pourboire |
| total_amount | float | Total |
| PULocationID, DOLocationID | int | Zones taxi pickup/dropoff |

## Utilisation
- Notebook `BDD_DuckDB` (analytics SQL)
- Notebook futur `DE_PySpark` (gros volume)

Pour télécharger plus de mois : changer la date dans l'URL CloudFront.
EOF

# ============================================================================
# 2. NASA Turbofan C-MAPSS — TS_Maintenance_Predictive
# ============================================================================
TURBOFAN_DIR="$DATA/turbofan"
mkdir -p "$TURBOFAN_DIR"

cat > "$TURBOFAN_DIR/README.md" <<'EOF'
# NASA C-MAPSS Turbofan Engine Degradation

Source officielle : NASA Prognostics Center of Excellence (PCoE).
URL historique : <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
URL miroir Kaggle (souvent plus accessible) :
<https://www.kaggle.com/datasets/behrad3d/nasa-cmaps>

## Téléchargement manuel
1. Aller sur la page Kaggle ci-dessus
2. Télécharger le ZIP (≈30 MB)
3. Extraire dans `data/_shared/turbofan/`

## Structure attendue après extraction
```
turbofan/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── ... (FD002, FD003, FD004)
└── readme.txt
```

## Description rapide
- 4 datasets (FD001-FD004) : conditions opérationnelles différentes
- ~21 capteurs + 3 settings opérationnels
- Train : moteurs run jusqu'à la défaillance
- Test : moteurs run jusqu'à un point arbitraire (RUL à prédire)

## Utilisation
- Notebook `TS_Maintenance_Predictive` (case study RUL prediction)
EOF

echo "[Turbofan] README créé → $TURBOFAN_DIR/README.md (download manuel Kaggle requis)"

# ============================================================================
# 3. PDFs sample — DE_Docling
# ============================================================================
PDFS_DIR="$DATA/pdfs_sample"
mkdir -p "$PDFS_DIR"

if [ ! -f "$PDFS_DIR/attention.pdf" ]; then
    echo "[PDFs] Téléchargement Attention is All You Need (≈2 MB)..."
    URL="https://arxiv.org/pdf/1706.03762"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$PDFS_DIR/attention.pdf" "$URL"
    else
        wget -O "$PDFS_DIR/attention.pdf" "$URL"
    fi
fi

cat > "$PDFS_DIR/README.md" <<'EOF'
# PDFs sample pour DE_Docling

- `attention.pdf` : "Attention is All You Need" (Vaswani 2017) — papier classique pour tester layout, formules, tables.

## Ajout manuel suggéré
- Un PDF avec **tables complexes** (rapport financier).
- Un PDF **scanné** (image, pour tester OCR).
- Un PDF **multi-colonnes** (journal scientifique).
EOF

# ============================================================================
# 4. Wikipedia subset — NLP_Recherche_d_informations / BDD_Vectorielles
# ============================================================================
echo "[Wikipedia] Pas de téléchargement — utiliser HuggingFace datasets à la volée :"
echo "  python -c 'from datasets import load_dataset; load_dataset(\"wikipedia\", \"20220301.simple\", split=\"train[:1%]\")'"

# ============================================================================
# 5. Vérification finale
# ============================================================================
echo ""
echo "=========================="
echo "État des datasets locaux :"
echo "=========================="
du -sh "$DATA"/* 2>/dev/null || echo "(aucun)"
echo ""
echo "Note : Air Passengers, MNIST, Titanic, California Housing, 20 Newsgroups, IMDB, CoNLL"
echo "       sont chargés via sklearn/seaborn/datasets — pas besoin de download manuel."
