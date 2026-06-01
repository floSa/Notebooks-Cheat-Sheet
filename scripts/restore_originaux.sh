#!/usr/bin/env bash
# restore_originaux.sh
# Recrée 01_notebooks_originaux/ depuis la branche main (originaux de référence).
# À lancer une fois après clone ou quand le dossier est absent.
# Le dossier est dans .gitignore (trop lourd) — reproductible avec ce script.

set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
OUT="$REPO_ROOT/01_notebooks_originaux"
mkdir -p "$OUT"

echo "📥 Restauration des originaux depuis main → $OUT"

restore() {
  local src="$1"
  local dst="$2"
  git show "main:$src" > "$OUT/$dst" 2>/dev/null && echo "  ✅ $dst" || echo "  ⚠️  Introuvable: $src"
}

# EDA
restore "EDA_Visualisation_Introduction.ipynb"         "EDA_Visualisation_Introduction.ipynb"
restore "EDA_Stats_Analyse_Desc_Visual.ipynb"          "EDA_Stats_Analyse_Desc_Visual.ipynb"
restore "EDA_Analyse_Multivarié.ipynb"                 "EDA_Analyse_Multivarie.ipynb"
restore "Détection D_outliers.ipynb"                   "Detection_Outliers.ipynb"

# Structures
restore "Structure_Pyhton.ipynb"                       "Structure_Python.ipynb"
restore "Structures_L_T_D_Cheat_Sheet.ipynb"           "Structures_L_T_D_Cheat_Sheet.ipynb"
restore "Structures_DataFrame.ipynb"                   "Structures_DataFrame.ipynb"
restore "Structure_BDD_&_DataFrame.ipynb"              "Structure_BDD_DataFrame.ipynb"
restore "Structures_Preprocessing.ipynb"               "Structures_Preprocessing.ipynb"
restore "Structures_Preprocessing_Function_Utiles.ipynb" "Structures_Preprocessing_Function_Utiles.ipynb"
restore "Structure_Generer_Données_pour_Classification.ipynb" "Structure_Generer_Donnees_Classification.ipynb"
restore "Structures_Polars.ipynb"                      "Structures_Polars.ipynb"

# ML classique
restore "INRIA_SKLearn_MOOC.ipynb"                     "INRIA_SKLearn_MOOC.ipynb"
restore "ML_Regression_Classification_CV_GridSearch.ipynb" "ML_Regression_Classification_CV_GridSearch.ipynb"
restore "ML_Régression_&_Classification_Multiple.ipynb" "ML_Regression_Classification_Multiple.ipynb"
restore "ML_Bagging_Boosting.ipynb"                    "ML_Bagging_Boosting.ipynb"
restore "ML_Optimisation_de_Modèles.ipynb"             "ML_Optimisation_de_Modeles.ipynb"
restore "ML_Explication_Feature_Importance_Selection.ipynb" "ML_Explication_Feature_Importance_Selection.ipynb"
restore "ML_MLFlow_Bench.ipynb"                        "ML_MLFlow_Bench.ipynb"
restore "ML_Apprentissage_par_Renforcement.ipynb"      "ML_Apprentissage_par_Renforcement.ipynb"
restore "ML_AutoML.ipynb"                              "ML_AutoML.ipynb"
restore "Test_données_manquante_modèles.ipynb"         "Test_donnees_manquante_modeles.ipynb"

# DS avancée
restore "DS_Bayesian.ipynb"                            "DS_Bayesian.ipynb"
restore "DS_Recommender_Systems.ipynb"                 "DS_Recommender_Systems.ipynb"
restore "DS_Survival_Analysis.ipynb"                   "DS_Survival_Analysis.ipynb"
restore "DS_Causal_Inference.ipynb"                    "DS_Causal_Inference.ipynb"
restore "DS_Geospatial.ipynb"                          "DS_Geospatial.ipynb"
restore "DS_Graph_Neural_Networks.ipynb"               "DS_Graph_Neural_Networks.ipynb"
restore "DS_Metrics_Reference.ipynb"                   "DS_Metrics_Reference.ipynb"

# DL
restore "DL_Deep_Learning_Maths.ipynb"                 "DL_Deep_Learning_Maths.ipynb"
restore "DL_Tensorflow_Keras.ipynb"                    "DL_Tensorflow_Keras.ipynb"
restore "DL_PyTorch.ipynb"                             "DL_PyTorch.ipynb"
restore "DL_PyTorch_Lightning.ipynb"                   "DL_PyTorch_Lightning.ipynb"
restore "DS_Graph_Neural_Networks.ipynb"               "DS_Graph_Neural_Networks.ipynb"
restore "KAN (Kolmogorov-Arnold Networks).ipynb"        "KAN_Kolmogorov_Arnold_Networks.ipynb"

# NLP
restore "NLP_Classification_Spervisé.ipynb"            "NLP_Classification_Supervisee.ipynb"
restore "NLP_Classification_Smote.ipynb"               "NLP_Classification_Smote.ipynb"
restore "NLP_NER.ipynb"                                "NLP_NER.ipynb"
restore "NLP_NER_BiLSTM_CRF.ipynb"                     "NLP_NER_BiLSTM_CRF.ipynb"
restore "NLP_Recherche_d_informations.ipynb"           "NLP_Recherche_d_informations.ipynb"
restore "NLP_Transformers.ipynb"                       "NLP_Transformers.ipynb"
restore "NLP_LangChain_RAG.ipynb"                      "NLP_LangChain_RAG.ipynb"

# Time Series / Signal
restore "TS_Time_Series_Intro.ipynb"                   "TS_Time_Series_Intro.ipynb"
restore "TS_Time_Series_Overview.ipynb"                "TS_Time_Series_Overview.ipynb"
restore "TS_ARIMAs_Revoir.ipynb"                       "TS_ARIMA.ipynb"
restore "TS_Generer_Sequence.ipynb"                    "TS_Generer_Sequence.ipynb"
restore "TS_Maintenance_Predictive_GOOD.ipynb"         "TS_Maintenance_Predictive_GOOD.ipynb"
restore "TdS_Introduction au Traitement du Signal.ipynb" "TdS_Introduction_Traitement_Signal.ipynb"

# BDD / DE / Web
restore "BDD_DuckDB.ipynb"                             "BDD_DuckDB.ipynb"
restore "BDD_Vectorielles.ipynb"                       "BDD_Vectorielles.ipynb"
restore "retrieval_BDD_Vectorielle.ipynb"              "retrieval_BDD_Vectorielle.ipynb"
restore "DE_Docling.ipynb"                             "DE_Docling.ipynb"
restore "DE_PySpark.ipynb"                             "DE_PySpark.ipynb"
restore "Flask_API.ipynb"                              "Flask_API.ipynb"
restore "FastAPI_API.ipynb"                            "FastAPI_API.ipynb"
restore "Streamlit_brique.ipynb"                       "Streamlit_brique.ipynb"

# MLOps
restore "MLOps_Tracking_DVC_Wandb.ipynb"               "MLOps_Tracking_DVC_Wandb.ipynb"
restore "MLOps_Model_Serving.ipynb"                    "MLOps_Model_Serving.ipynb"
restore "MLOps_Drift_Monitoring.ipynb"                 "MLOps_Drift_Monitoring.ipynb"
restore "MLOps_Pipelines_Airflow_Prefect.ipynb"        "MLOps_Pipelines_Airflow_Prefect.ipynb"
restore "SoftEng_Tests_Quality_ML.ipynb"               "SoftEng_Tests_Quality_ML.ipynb"

# AI Engineering
restore "AI_Prompt_Engineering.ipynb"                  "AI_Prompt_Engineering.ipynb"
restore "AI_Local_LLMs.ipynb"                          "AI_Local_LLMs.ipynb"
restore "AI_LLM_Finetuning_PEFT_LoRA.ipynb"            "AI_LLM_Finetuning_PEFT_LoRA.ipynb"

echo ""
echo "✅ $(ls "$OUT" | wc -l) notebooks restaurés dans $OUT"
