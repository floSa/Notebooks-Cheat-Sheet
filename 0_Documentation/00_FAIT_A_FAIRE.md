# ✅ Fait / 🔧 À faire — état réel

> Consigne de refonte : `00_consignes_refonte.md` (DL frameworks : `00_consignes_DL.md`).
> Règle : on refait depuis MON vrai original (`1_Old_Notebooks/`), jamais depuis les versions
> VIDÉES (`main` / `3_Sessions_Ratees/`). « Vidé » = mon notebook réduit de ~200 à ~20 cellules.

---

## ✅ FAIT (partis de mon vrai original — à VALIDER) — 5

| Notebook | note |
|---|---|
| `DL_PyTorch` | fait selon le blueprint (mix PyTorch + TF/Keras) |
| `DL_Keras` | fait selon le blueprint |
| `DL_TensorFlow` | fait selon le blueprint |
| `DL_Deep_Learning_Maths` | a utilisé mon original |
| `EDA_Stats_Analyse_Desc_Visual` | a utilisé mon original |

> À valider = vérifier qu'ils respectent bien `00_consignes_refonte.md` (mes graphiques/images
> présents, contenu ≥ original, exécution de bout en bout).

## 🔧 À REFAIRE (faits sur du VIDÉ → mes graphiques/images perdus) — ~26

| Thème | Notebooks |
|---|---|
| Structures | `Structures_DataFrame` · `Structures_Preprocessing` · `Structures_L_T_D_Cheat_Sheet` · `Structure_Python` · `Structure_Generer_Donnees_Classification` · `Structure_BDD_DataFrame` |
| EDA | `EDA_Analyse_Multivarie` · `EDA_Visualisation_Introduction` · `Detection_Outliers` |
| ML | `ML_Explication_Feature_Importance_Selection` · `ML_Regression_Classification_CV_GridSearch` · `ML_Bagging_Boosting` · `ML_Optimisation_de_Modeles` · `ML_Regression_Classification_Multiple` · `ML_Apprentissage_par_Renforcement` · `INRIA_SKLearn_MOOC` |
| NLP | `NLP_NER` · `NLP_NER_BiLSTM_CRF` · `NLP_Classification_Smote` · `NLP_Classification_Supervisee` · `NLP_Recherche_d_informations` · `NLP_Transformers` |
| TS / Signal | `TS_Time_Series_Overview` · `TS_Time_Series_Intro` · `TS_ARIMA` · `TS_Maintenance_Predictive` · `TS_Generer_Sequence` · `TdS_Introduction_Traitement_Signal` |
| BDD / Apps | `BDD_DuckDB` · `BDD_Vectorielles` · `DE_Docling` · `Streamlit_brique` · `Flask_API` |
| DL | `DL_KAN_Kolmogorov_Arnold` |
| Divers | `ML_MLFlow_Bench` · `Test_donnees_manquante_modeles` |

## 🆕 À CRÉER (sujets neufs, n'ont jamais existé chez moi) — 24

| Rôle | Notebooks |
|---|---|
| Data Scientist | `DS_Bayesian` · `DS_Causal_Inference` · `DS_Geospatial` · `DS_Survival_Analysis` · `DS_Recommender_Systems` |
| Data Engineer | `DE_PySpark` · `DE_Airflow_Prefect` · `DE_Kafka_Streaming` · `DE_dbt_Modeling` · `DE_Data_Quality` |
| AI Engineer | `AI_Local_LLMs` · `AI_LLM_Finetuning_PEFT` · `AI_Agents_Tools` · `AI_Prompt_Engineering` · `AI_Multimodal_VLM` · `AI_Speech_Audio` |
| ML Engineer | `MLE_Feature_Store` · `MLE_Online_Inference` · `MLE_Model_Serving` · `MLE_AB_Testing` |
| MLOps | `MLOps_Pipelines_Airflow` · `MLOps_Model_Registry` · `MLOps_Drift_Monitoring` · `MLOps_CICD_GitHub_Actions` |

## 🗑️ À jeter / cas spéciaux

- `AAA_Test_ML`, `Suppr_ML_Bench_…(A finir)` → brouillons, ne pas refaire.
- `Structures_Preprocessing_Function_Utiles` (8 cellules) → fusionner dans `Structures_Preprocessing`.
- `TS_Maintenance_Prédictive` (ancien) → doublon, garder `TS_Maintenance_Predictive`.
- `FastAPI_API` / `DL_JAX` / `DL_Frameworks_Comparatif` → pas d'original (sujets neufs DL/API).

---

## Compteur

**5 faits (à valider) · ~26 à refaire · 24 à créer.**

> Note : la liste « faits » repose sur l'analyse de contenu (sections + techniques de mon
> original présentes). L'ancien indicateur en % (`00_VERIFICATION_BASE.md`) était trompeur,
> surtout pour les DL frameworks (squelette renommé) — ne plus s'y fier seul.
