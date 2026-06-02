# ⚠️ Refaits bâtis sur base GUTÉE — à revérifier contre les vrais originaux

> **Incident 2026-06-01** : les vrais originaux (`1_Old_Notebooks/ipynb/`, source `data/Notebooks.zip`)
> n'étaient pas présents sur la machine lors des sessions de refonte. Celles-ci ont donc travaillé
> sur des versions **réduites/gutées** (~20 cellules au lieu de 200+). Les notebooks "faits"
> ci-dessous ont probablement **perdu des sections originales** et doivent être re-contrôlés.

## Méthode

Comparaison du nombre de cellules : **vrai original** (zip) vs **base réellement utilisée** (gutée).
Écart majeur ⇒ refonte sur contenu réduit.

## Notebooks "faits" bâtis sur base gutée (à REVÉRIFIER en priorité)

| Refait (2_New_Notebooks/ipynb) | Vrai original | base utilisée | Statut |
|---|---:|---:|---|
| EDA_Analyse_Multivarie | **198** | 20 | 🔴 à revérifier |
| ML_Explication_Feature_Importance_Selection | **287** | 25 | 🔴 à revérifier |
| Structures_DataFrame *(pas encore refait)* | **227** | 35 | 🔴 original à utiliser |
| retrieval→BDD_Vectorielles | **224** | 16 | 🔴 à revérifier |
| INRIA_SKLearn_MOOC | **163** | 16 | 🔴 à revérifier |
| EDA_Visualisation_Introduction | **110** | 23 | 🔴 à revérifier |
| DL (ex DL_Tensorflow_Keras → Keras/TensorFlow) | **112** | 20 | 🔴 à revérifier |
| EDA_Stats_Analyse_Desc_Visual | **83** | 23 | 🔴 à revérifier |
| DL_Deep_Learning_Maths | **78** | 18 | 🔴 à revérifier |
| ML_Bagging_Boosting | **61** | 16 | 🔴 à revérifier |
| DE_Docling | **60** | 10 | 🔴 à revérifier |
| BDD_DuckDB | **52** | 18 | 🔴 à revérifier |
| ML_Optimisation_de_Modeles | **49** | 18 | 🔴 à revérifier |
| Detection_Outliers | **47** | 21 | 🔴 à revérifier |

## Refait OK (original réellement petit, pas de perte)

| Refait | Vrai original | base utilisée | Statut |
|---|---:|---:|---|
| ML_Apprentissage_par_Renforcement | 18 | 16 | 🟢 base ~complète |

## Refaits sans original (sujets neufs — rien à revérifier)

DL_KAN_Kolmogorov_Arnold (original KAN = 4 cellules de théorie), Flask_API (fusion Flask 2c + FastAPI),
DL_JAX, DL_Frameworks_Comparatif, + tous les DS_*/MLOps_*/AI_* nouveaux.

## Action recommandée

Pour chaque ligne 🔴 : relancer le workflow contrat en lisant **le vrai original**
(`1_Old_Notebooks/ipynb/<nom accentué>.ipynb`), pas la base gutée. La refonte existante
peut servir de point de départ, mais il faut **réintégrer les sections originales perdues**.
