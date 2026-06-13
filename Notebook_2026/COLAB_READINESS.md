# Colab-readiness — Notebook_2026/ipynb/

_44 notebooks analysés (lecture seule, aucun notebook modifié)._

**Légende verdict** : `PRÊT` = données récupérées par programme et/ou cellule pip présente ; `FICHIER REQUIS` = lit un fichier local → à uploader sur Colab/Drive ; `PIP À VÉRIFIER` = imports tiers sans cellule pip install visible.

| Notebook | Verdict | Source des données | Fichier local requis | pip install | Imports tiers |
|---|---|---|---|---|---|
| BDD_DuckDB.ipynb | **PIP À VÉRIFIER** | — | — | non | duckdb, polars |
| BDD_Vectorielles.ipynb | **PIP À VÉRIFIER** | sklearn fetch_20newsgroups | — | non | chromadb, faiss, lancedb, psycopg, pymilvus, qdrant_client, sentence_transformers, weaviate |
| DE_Docling.ipynb | **PRÊT** | — | — | oui | docling, docling_core, fitz, rich |
| DL_Deep_Learning_Maths.ipynb | **PRÊT** | sklearn load_digits<br>sklearn make_blobs (synthétique)<br>sklearn make_moons (synthétique) | — | non | — |
| DL_Frameworks_Comparatif.ipynb | **PRÊT** | sklearn load_digits | — | non | — |
| DL_JAX.ipynb | **À VÉRIFIER** | — | — | non | — |
| DL_KAN_Kolmogorov_Arnold.ipynb | **PIP À VÉRIFIER** | sklearn make_moons (synthétique) | — | non | kan |
| DL_Keras.ipynb | **PIP À VÉRIFIER** | keras.datasets (téléchargé)<br>sklearn fetch_california_housing<br>sklearn make_classification (synthétique) | — | non | shap |
| DL_PyTorch.ipynb | **PIP À VÉRIFIER** | sklearn fetch_california_housing<br>sklearn make_classification (synthétique)<br>torchvision.datasets (téléchargé) | — | non | shap |
| DL_TensorFlow.ipynb | **PIP À VÉRIFIER** | keras.datasets (téléchargé)<br>sklearn fetch_california_housing<br>sklearn make_classification (synthétique)<br>tf.keras.datasets (téléchargé) | — | non | shap |
| Detection_Outliers.ipynb | **PIP À VÉRIFIER** | Air Passengers<br>HuggingFace datasets.load_dataset<br>seaborn load_dataset<br>sklearn make_blobs (synthétique) | — | non | pyod, stumpy |
| EDA_Analyse_Multivarie.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | pacmap, prince, umap |
| EDA_Stats_Analyse_Desc_Visual.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | sweetviz, ydata_profiling |
| EDA_Visualisation_Introduction.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | mpl_chord_diagram |
| FastAPI_API.ipynb | **À VÉRIFIER** | — | — | non | — |
| Flask_API.ipynb | **PIP À VÉRIFIER** | — | — | non | fastapi, flask, jwt, pydantic, werkzeug |
| INRIA_SKLearn_MOOC.ipynb | **PRÊT** | HuggingFace datasets.load_dataset<br>seaborn load_dataset<br>sklearn fetch_california_housing<br>sklearn make_classification (synthétique) | — | non | — |
| ML_Apprentissage_par_Renforcement.ipynb | **PIP À VÉRIFIER** | — | — | non | gymnasium, stable_baselines3 |
| ML_Bagging_Boosting.ipynb | **PIP À VÉRIFIER** | sklearn fetch_california_housing<br>sklearn load_breast_cancer | — | non | catboost |
| ML_Explication_Feature_Importance_Selection.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset<br>sklearn fetch_california_housing<br>sklearn load_digits<br>sklearn load_iris<br>sklearn make_classification (synthétique)<br>sklearn make_regression (synthétique) | — | non | catboost, lime, shap |
| ML_MLFlow_Bench.ipynb | **PIP À VÉRIFIER** | sklearn fetch_california_housing | — | non | mlflow |
| ML_Optimisation_de_Modeles.ipynb | **PIP À VÉRIFIER** | sklearn fetch_california_housing | — | non | catboost, optuna |
| ML_Regression_Classification_CV_GridSearch.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>sklearn fetch_california_housing<br>sklearn make_blobs (synthétique)<br>sklearn make_regression (synthétique) | — | non | hyperopt, optuna |
| ML_Regression_Classification_Multiple.ipynb | **PIP À VÉRIFIER** | sklearn make_regression (synthétique) | — | non | catboost |
| NLP_Classification_Smote.ipynb | **PIP À VÉRIFIER** | sklearn fetch_20newsgroups | — | non | imblearn |
| NLP_Classification_Supervisee.ipynb | **PIP À VÉRIFIER** | sklearn fetch_20newsgroups | — | non | sentence_transformers |
| NLP_NER.ipynb | **PRÊT** | HuggingFace datasets.load_dataset | — | oui | datasets, gliner, seqeval, transformers |
| NLP_NER_BiLSTM_CRF.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset | — | non | datasets, seqeval, torchcrf |
| NLP_Recherche_d_informations.ipynb | **PIP À VÉRIFIER** | — | — | non | bm25s, sentence_transformers, transformers |
| NLP_Transformers.ipynb | **PIP À VÉRIFIER** | sklearn fetch_20newsgroups | — | non | accelerate, datasets, evaluate, peft, transformers |
| Streamlit_brique.ipynb | **À VÉRIFIER** | — | — | non | — |
| Structure_BDD_DataFrame.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | adbc_driver_sqlite, duckdb, mongomock, polars, psycopg, pydantic_settings, pymongo, sqlalchemy, tenacity |
| Structure_Generer_Donnees_Classification.ipynb | **PIP À VÉRIFIER** | sklearn make_blobs (synthétique)<br>sklearn make_classification (synthétique)<br>sklearn make_moons (synthétique)<br>sklearn make_regression (synthétique) | — | non | faker |
| Structure_Python.ipynb | **PIP À VÉRIFIER** | — | — | non | helper, pydantic |
| Structures_DataFrame.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | polars |
| Structures_L_T_D_Cheat_Sheet.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | iteration_utilities, xarray |
| Structures_Preprocessing.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset<br>sklearn fetch_20newsgroups<br>sklearn load_iris<br>sklearn make_blobs (synthétique)<br>sklearn make_classification (synthétique) | — | non | imblearn, missingno, pyod |
| TS_ARIMA.ipynb | **PIP À VÉRIFIER** | Air Passengers | — | non | pmdarima, statsforecast |
| TS_Generer_Sequence.ipynb | **PRÊT** | Air Passengers | — | non | — |
| TS_Maintenance_Predictive.ipynb | **PIP À VÉRIFIER** | — | — | non | catboost |
| TS_Time_Series_Intro.ipynb | **PIP À VÉRIFIER** | Air Passengers | — | non | missingno |
| TS_Time_Series_Overview.ipynb | **PIP À VÉRIFIER** | Air Passengers | — | non | pmdarima, statsforecast, utilsforecast |
| TdS_Introduction_Traitement_Signal.ipynb | **PIP À VÉRIFIER** | — | — | non | librosa, pywt, skimage |
| Test_donnees_manquante_modeles.ipynb | **PIP À VÉRIFIER** | HuggingFace datasets.load_dataset<br>seaborn load_dataset | — | non | catboost |

## Récapitulatif

- **PRÊT** : 6
- **PIP À VÉRIFIER** : 35
- **À VÉRIFIER** : 3

## Notes

- Les notebooks `FICHIER REQUIS` lisent un fichier présent dans `data/` localement. Sur Colab il faut uploader ce fichier (ou le monter depuis Drive). Voir `scripts/download_data.sh` pour savoir d'où vient chaque fichier.
- `PIP À VÉRIFIER` ne veut pas dire "cassé" : beaucoup de paquets sont déjà sur Colab. Cela signale les imports tiers à confirmer (ajouter `!pip install <pkg>` en 1re cellule si absent).
- Ce rapport reflète une heuristique statique, pas une exécution end-to-end sur Colab.

## Cas particuliers données (vérifiés à la main)

- **BDD_DuckDB** : lit le parquet NYC Taxi via DuckDB `read_parquet` directement depuis l'URL CloudFront (`httpfs`). Marche sur Colab tant qu'il y a internet — **aucun fichier à uploader**.
- **TS_Maintenance_Predictive** : utilise le dataset NASA C-MAPSS turbofan (`train_FD001.txt` lu via `read_csv`), qui demande un **téléchargement manuel Kaggle** (voir `scripts/download_data.sh`, section Turbofan). Un repli synthétique (`make_`/`np.random`) semble présent, mais pour le vrai cas d'étude il faut uploader le fichier sur Colab/Drive. **Seul notebook nécessitant réellement un fichier data manuel.**
