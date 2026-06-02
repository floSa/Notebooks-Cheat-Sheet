# ✅ Fait / 🔧 À faire — état réel

> Source : `scripts/verifier_base.py` (cf. `00_VERIFICATION_BASE.md`) + structure du repo.
> **Règle d'or** (cf. `00_workflow_contract.md` §0) : un notebook avec original se refait sur
> `1_Old_Notebooks/`, jamais sur les gutés. Les 24 sujets neufs s'inspirent de `2_New_Notebooks/`.

## Légende
- ✅ **bonne base** : le refait a vraiment utilisé ton original → à valider/finaliser (5 critères).
- 🟡 **acceptable** : ton original était déjà minuscule (≈ guté) → refonte ok, à valider vite.
- 🔴 **à refaire** : refait bâti sur un guté → reprendre depuis ton vrai original.
- 🆕 **à implémenter** : sujet neuf, pas d'original → partir du plan + inspiration 2_New_Notebooks.
- 🗑️ **à jeter** : brouillon/obsolète.

---

## A. Refait à FINALISER (bonne base — ont utilisé ton original)

| Notebook | base | action |
|---|---|---|
| `DL_Deep_Learning_Maths` | ✅ 63 % | valider les 5 critères, compléter ce qui manque vs original |
| `EDA_Stats_Analyse_Desc_Visual` | ✅ 27 % | idem |
| `DE_Docling` | 🟠 16 % | compléter depuis l'original (sections manquantes) |
| `Detection_Outliers` | 🟠 14 % | compléter depuis l'original (`Détection D_outliers`) |

## B. Refait ACCEPTABLE (original déjà minuscule ≈ guté — valider vite)

`Flask_API` · `ML_MLFlow_Bench` · `NLP_Transformers` · `Test_donnees_manquante_modeles` · `TS_Generer_Sequence`

## C. Refait À REFAIRE sur le vrai original (priorité — bâtis sur guté)

| Thème | Notebooks |
|---|---|
| Structures | `Structures_DataFrame` (227c orig) · `Structures_Preprocessing` (186c) · `Structures_L_T_D_Cheat_Sheet` (80c) · `Structure_Python` · `Structure_Generer_Donnees_Classification` · `Structure_BDD_DataFrame` |
| EDA | `EDA_Analyse_Multivarie` (198c orig) · `EDA_Visualisation_Introduction` (110c) |
| ML | `ML_Explication_Feature_Importance_Selection` (287c orig) · `ML_Regression_Classification_CV_GridSearch` (123c) · `ML_Bagging_Boosting` · `ML_Optimisation_de_Modeles` · `ML_Regression_Classification_Multiple` · `ML_Apprentissage_par_Renforcement` · `INRIA_SKLearn_MOOC` (163c) |
| NLP | `NLP_NER` · `NLP_NER_BiLSTM_CRF` · `NLP_Classification_Smote` · `NLP_Classification_Supervisee` · `NLP_Recherche_d_informations` |
| TS / Signal | `TS_Time_Series_Overview` (157c orig) · `TS_Time_Series_Intro` · `TS_ARIMA` · `TS_Maintenance_Predictive` (168c) · `TdS_Introduction_Traitement_Signal` (161c) |
| BDD / Apps | `BDD_DuckDB` · `BDD_Vectorielles` · `Streamlit_brique` |
| DL | `DL_Keras` · `DL_TensorFlow` · `DL_KAN_Kolmogorov_Arnold` (depuis `DL_Tensorflow_Keras` / `KAN` originaux) |

## D. Sujets NEUFS à implémenter (24 — depuis plan + inspiration 2_New_Notebooks)

| Rôle | Notebooks |
|---|---|
| Data Scientist | `DS_Bayesian` · `DS_Causal_Inference` · `DS_Geospatial` · `DS_Survival_Analysis` · `DS_Recommender_Systems` |
| Data Engineer | `DE_PySpark` · `DE_Airflow_Prefect` · `DE_Kafka_Streaming` · `DE_dbt_Modeling` · `DE_Data_Quality` |
| AI Engineer | `AI_Local_LLMs` · `AI_LLM_Finetuning_PEFT` · `AI_Agents_Tools` · `AI_Prompt_Engineering` · `AI_Multimodal_VLM` · `AI_Speech_Audio` |
| ML Engineer | `MLE_Feature_Store` · `MLE_Online_Inference` · `MLE_Model_Serving` · `MLE_AB_Testing` |
| MLOps | `MLOps_Pipelines_Airflow` · `MLOps_Model_Registry` · `MLOps_Drift_Monitoring` · `MLOps_CICD_GitHub_Actions` |

## E. Cas spéciaux

| Notebook | cas | action |
|---|---|---|
| `DL_PyTorch` | original présent sur `main` mais **absent du zip** | récupérer le vrai original PyTorch avant de refaire |
| `FastAPI_API` / `DL_JAX` / `DL_Frameworks_Comparatif` | pas d'original | sujets neufs (cf. décision split DL `00_blueprint_DL_frameworks.md`) |
| `AAA_Test_ML` · `Suppr_ML_Bench_…(A finir)` | 🗑️ brouillons | à jeter (ne pas refaire) |
| `Structures_Preprocessing_Function_Utiles` | micro (8c) | fusionner dans `Structures_Preprocessing` |
| `TS_Maintenance_Prédictive` (ancien) | doublon de `_GOOD` | abandonner au profit de `TS_Maintenance_Predictive` |

---

## Compteurs honnêtes

- **0** notebook "fait" au sens de la règle d'or.
- **2** refait sur bonne base à finaliser (A) ; **4** au total avec base ≥ partielle (A).
- **5** acceptables à valider vite (B).
- **~28** à refaire sur l'original (C).
- **24** sujets neufs à implémenter (D).
