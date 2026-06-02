# 🔍 Critique des 44 notebooks — axes d'amélioration

Document **critique honnête** rédigé après livraison des 44 notebooks de la Phase 4. Pour chaque notebook : ce qui est bien, ce qui manque, prioritisation des améliorations.

**Échelle d'effort des améliorations** : `S` (≤ 1h), `M` (1-3h), `L` (3h+).

**Priorité d'amélioration** : 🔴 (critique), 🟠 (recommandé), 🟢 (nice-to-have).

---

## 🎯 Critique générale (transverse)

Avant le détail par notebook, les **faiblesses systémiques** que la refonte n'a pas pu corriger faute de temps :

1. **🔴 Datasets non téléchargés** : NASA Turbofan (`TS_Maintenance_Predictive`), NYC Taxi (`BDD_DuckDB`), PDFs sample (`DE_Docling`), Wikipedia subset (vector DBs / RAG). Sans ces datasets, plusieurs notebooks restent en mode "code prêt mais non exécuté de bout en bout".
2. **🔴 Aucun notebook exécuté à 100 %** : les smoke tests valident les APIs et les bouts de code critiques, mais aucune section "cas réel" n'a été tournée jusqu'à l'output complet (graphe + métriques visibles). Idéalement il faudrait exécuter chaque notebook en CI.
3. **🟠 Cohérence visuelle** : pas de style de plot uniforme (palette, dpi, taille). À standardiser via un `mpl_style.py` partagé.
4. **🟠 Cellules d'exercices** : aucun notebook ne propose d'**exercice à faire** pour l'utilisateur. Ajouterait beaucoup de valeur pédagogique.
5. **🟠 Tests unitaires** : aucune fonction n'est testée formellement (pytest). Les fonctions utilitaires (`make_windows`, `cramers_v`, `rrf_fuse`, etc.) mériteraient une suite de tests.
6. **🟢 Diagrammes** : très peu de diagrammes explicatifs (mermaid, schémas). Texte uniquement parfois trop dense.
7. **🟢 Liens entre notebooks** : les renvois `[notebook.ipynb]` sont posés mais pas tous valides (certains liens utilisent des noms qui ont changé entre temps).
8. **🟢 Internationalisation** : tout en français — limite la réutilisation pour des collègues anglophones.

---

## 📚 Vague 1 — NLP/LLMs/RAG

### NLP_Transformers
✅ **Points forts** : panorama HF 5.x complet, sections Trainer/PEFT/LLMs/chat templates bien structurées, exemples avec modèles légers (DistilBERT/SmolLM2).
❌ **Manque** :
- 🔴 Pas d'exécution end-to-end du fine-tuning (les cellules sont commentées). À tourner sur GPU et inclure les outputs.
- 🟠 Section "fine-tuning avec PEFT/LoRA" reste pseudo-code → exécuter avec `peft` installé.
- 🟠 Pas de section sur **les multimodaux** en profondeur (vision-language en pratique).
- 🟢 Comparatif Trainer vs Accelerate vs SFTTrainer manquant.

### NLP_Recherche_d_informations (RAG)
✅ **Points forts** : sparse + dense + hybrid + reranking + RAG end-to-end + RAGAS, vue d'ensemble très complète.
❌ **Manque** :
- 🔴 Le pipeline complet sur **Wikipedia subset** (vrai dataset) n'est pas exécuté.
- 🟠 Pas d'exemple **HyDE** ou **multi-query** en code.
- 🟠 Pas de comparaison effective des **embedders** (BGE vs E5 vs GTE) sur même benchmark.
- 🟢 RAGAS pseudo-code uniquement — exécuter sur petit dataset eval.

### NLP_NER
✅ **Points forts** : formats IOB/BIOES bien expliqués, GLiNER positionné comme game-changer, eval seqeval.
❌ **Manque** :
- 🔴 Fine-tuning DistilBERT sur **CoNLL-2003** non exécuté.
- 🟠 Pas d'exemple GLiNER tourné (commenté pour éviter le téléchargement).
- 🟠 Pas de section **NER multilingue** (XLM-R, mDeBERTa).
- 🟢 Manque d'exemple JSON-extraction via LLM (function calling) en code complet.

### NLP_NER_BiLSTM_CRF
✅ **Points forts** : maths CRF claires, contextualisation historique honnête, comparatif final avec alternatives.
❌ **Manque** :
- 🟠 Le code BiLSTM-CRF est **uniquement pseudo-code** — ajouter une vraie impl mini sur ~30 lignes avec `pytorch-crf` exécutable.
- 🟢 Pas d'exemple d'algo Viterbi from scratch.

### NLP_Classification_Supervisee
✅ **Points forts** : matrice de décision utile, baseline TF-IDF avec interprétabilité, smoke F1=0.798 démontrant la chaîne.
❌ **Manque** :
- 🟠 SetFit en pseudo-code uniquement — exécuter sur quelques labels.
- 🟠 Pas d'exemple ZSL/LLM (function calling) en code.
- 🟢 Pas de calibration des probas via `CalibratedClassifierCV`.

### NLP_Classification_Smote
✅ **Points forts** : démo narrative parfaite (Baseline F1=0.52 → CW=0.86 → SMOTE=0.87 → Threshold=0.90). Pédagogie excellente.
❌ **Manque** :
- 🟠 Focal Loss en **maths uniquement**, pas d'impl PyTorch/sklearn.
- 🟠 LLM augmentation en pseudo-code uniquement — code exécutable.
- 🟢 Pas de **calibration** post-resampling (souvent les probas sont déformées après SMOTE).

---

## 📚 Vague 2 — Vector DBs

### BDD_Vectorielles
✅ **Points forts** : matrice 9 vector DBs, FAISS Flat/IVF/HNSW exécuté, LanceDB hands-on, Qdrant et pgvector documentés.
❌ **Manque** :
- 🔴 **Pas de benchmark perf réel** sur même dataset — claims (Qdrant ~40 QPS @ 50M) sont des chiffres publiés non reproduits.
- 🟠 Qdrant en pseudo-code (Docker requis) — fournir un docker-compose dans le repo.
- 🟠 pgvector pareil.
- 🟢 Pas d'exemple **Matryoshka embeddings** en code (tronquer + re-rank).

---

## 📚 Vague 3 — MLOps

### ML_MLFlow_Bench
✅ **Points forts** : aliases @champion/@challenger bien expliqués (vs stages deprecated), bench 5 modèles tracké, MLflow 3.x LLM mentionné.
❌ **Manque** :
- 🔴 MLflow Registry execution dépend du backend → en mode SQLite local, certaines features sont limitées. Préciser le `tracking_uri` Postgres + S3 dans un setup script.
- 🟠 Section CI/CD avec GitHub Actions en exemple concret (yml).
- 🟠 MLflow LLM (tracing/eval) en pseudo-code uniquement.
- 🟢 Pas de comparatif effectif avec W&B (juste tableau).

---

## 📚 Vague 4 — Time Series

### TS_Time_Series_Intro
✅ **Points forts** : tutoriel débutant solide, décomposition + stationnarité + ACF/PACF + lag features + split temporel correct.
❌ **Manque** :
- 🟠 Pas d'exemple **Nixtla statsforecast** alors qu'il est cité comme l'option moderne.
- 🟢 Pas de comparaison naïve (persistence baseline) avant le modèle.

### TS_Time_Series_Overview
✅ **Points forts** : panorama exhaustif 2026 (foundation models, Nixtla, métriques MASE/WAPE).
❌ **Manque** :
- 🔴 La plupart des sections sont en **prose seule** (Nixtla mlforecast / neuralforecast / TimeGPT / Chronos) sans code exécuté.
- 🟠 Pas de **bench comparatif** ETS vs ARIMA vs Theta vs Prophet sur Air Passengers.
- 🟠 Foundation models : il manque l'exemple Chronos exécuté.

### TS_ARIMA
✅ **Points forts** : Box-Jenkins complet, AutoARIMA avec pmdarima, diagnostics résidus, forecast IC 95%.
❌ **Manque** :
- 🟠 Pas d'exemple **SARIMAX** avec exogenous variables (covariables) exécuté.
- 🟢 Pas de comparaison entre AutoARIMA pmdarima et statsforecast (Nixtla, plus rapide).

### TS_Generer_Sequence
✅ **Points forts** : cheat-sheet sliding window propre, NumPy + PyTorch + multi-features + split temporel avec gap.
❌ **Manque** :
- 🟢 Pas d'exemple **LSTM end-to-end** qui consomme les windows produites (renvoi vers DL_PyTorch).

### TS_Maintenance_Predictive
✅ **Points forts** : pipeline complet RUL XGBoost + score asymétrique C-MAPSS + 3 stratégies PCA + LSTM mentionné. Smoke MAE=20 cycles.
❌ **Manque** :
- 🔴 Utilise un dataset **synthétique** (proxy NASA). Idéalement le vrai C-MAPSS chargé.
- 🟠 LSTM en pseudo-code uniquement.
- 🟠 Pas de section **anomaly detection** comme alternative à RUL.

---

## 📚 Vague 5 — EDA

### EDA_Visualisation_Introduction
✅ **Points forts** : cheat-sheet bien équilibré matplotlib/seaborn/plotly, catalogue 12 fonctions seaborn.
❌ **Manque** :
- 🟠 Plotly en pseudo-code seulement (exécuter un exemple).
- 🟠 Pas d'exemple **seaborn objects API** (l'API moderne de seaborn 0.12+).
- 🟢 Pas de section **interactive Bokeh** (mention rapide).

### EDA_Stats_Analyse_Desc_Visual
✅ **Points forts** : workflow standardisé, tests stat (Shapiro/chi²/ANOVA/Kruskal) avec smoke, mention ydata-profiling.
❌ **Manque** :
- 🟠 AutoEDA (ydata-profiling) pseudo-code seulement — exécuter et inclure une capture du rapport.
- 🟢 Pas de section **outlier detection univarié** rapide (renvoi `Detection_Outliers` est OK).

### EDA_Analyse_Multivarie
✅ **Points forts** : PCA exécutée (Iris), scree + cercle correl, MCA/FAMD/CA documentés, UMAP/t-SNE comparés.
❌ **Manque** :
- 🟠 MCA/FAMD/CA en pseudo-code uniquement — exécuter sur Titanic (variables qualitatives).
- 🟠 UMAP/t-SNE en pseudo-code — exécuter sur Iris ou MNIST.
- 🟢 Pas d'exemple `prince` avec output visuel.

### Detection_Outliers
✅ **Points forts** : decision tree complet, 4 algos benchmarkés (Z-score, IQR, EllipticEnv, LOF, IsoForest, OCSVM), autoencoder mentionné. Smoke 11/11 flagged sur tous.
❌ **Manque** :
- 🟠 Autoencoder en pseudo-code uniquement.
- 🟠 STUMPY (TS anomalies) cité mais pas démo.
- 🟢 Pas d'exemple **PyOD** alors qu'il est cité comme lib panoramique.

---

## 📚 Vague 6 — ML core

### ML_Regression_Classification_Multiple
✅ **Points forts** : biais/variance bien expliqué + démontré sur KNN (table train/test vs K), introduction structurée.
❌ **Manque** :
- 🟠 Pas de **figure** des courbes train/test vs K (très visuel).
- 🟢 Pas de comparaison équivalente sur la classification.

### ML_Regression_Classification_CV_GridSearch
✅ **Points forts** : bench 9 + 7 algos exécutés via CV, Optuna en démo, calibration mentionnée.
❌ **Manque** :
- 🟠 Calibration en pseudo-code uniquement → exécuter `CalibratedClassifierCV` + reliability diagram.
- 🟠 Pas de pipeline GridSearch sur hyperparamètres du preprocessing (`prep__num__imp__strategy`).
- 🟢 Pas de comparaison sklearn pipeline vs imblearn pipeline (rappel anti-leak).

### ML_Bagging_Boosting
✅ **Points forts** : maths gradient boosting, trinité XGB/LGBM/CB exécutée, early stopping et stacking documentés.
❌ **Manque** :
- 🟠 Early stopping et stacking en pseudo-code uniquement.
- 🟢 Pas de visualisation comparée RF vs XGB vs LGBM sur même tâche (figure perf vs time).

### ML_Optimisation_de_Modeles
✅ **Points forts** : Optuna deep dive (samplers, pruners, multi-objectif, importance), exemple LGBM exécuté avec 15 trials.
❌ **Manque** :
- 🟠 Pruner en pseudo-code uniquement.
- 🟠 Multi-objectif (NSGA-II) en pseudo-code uniquement.
- 🟢 Pas d'exemple **PBT (Population Based Training)** mentionné pour LLMs.

### ML_Explication_Feature_Importance_Selection
✅ **Points forts** : permutation importance + SHAP TreeExplainer exécutés, RFECV exécuté, Boruta documenté, Display API.
❌ **Manque** :
- 🟠 SHAP summary plot / waterfall plot non générés (juste les valeurs).
- 🟠 LIME / Boruta en pseudo-code.
- 🟢 Pas d'exemple **Fairlearn / aif360** (fairness).

### ML_Apprentissage_par_Renforcement
✅ **Points forts** : excellent panorama (bandits → MDP → Q-Learning → DQN → PPO → RLHF/DPO), Thompson/UCB/ε-greedy exécutés.
❌ **Manque** :
- 🟠 Q-Learning sur Gymnasium en pseudo-code uniquement — exécuter sur FrozenLake.
- 🟠 Stable-Baselines3 PPO en pseudo-code.
- 🟢 Pas d'exemple **DPO** sur HuggingFace TRL (le vrai sujet 2026 pour les LLMs).

### Test_donnees_manquante_modeles
✅ **Points forts** : typologie MCAR/MAR/MNAR claire, bench expérimental sur Titanic (LR + 4 strategies + XGB native).
❌ **Manque** :
- 🟠 missForest non exécuté (juste cité).
- 🟢 Pas de comparaison sur dataset **avec MNAR** simulé (différent de MCAR Titanic).

### INRIA_SKLearn_MOOC
✅ **Points forts** : synthèse 2026 propre, learning + validation curves exécutées.
❌ **Manque** :
- 🟢 C'est volontairement une synthèse — moins de manques structurels. Pourrait pointer vers les modules MOOC originaux par section.

---

## 📚 Vague 7 — DL frameworks

### DL_Deep_Learning_Maths
✅ **Points forts** : MLP from scratch NumPy avec Adam et XOR exécuté. Forward, backprop, optimisation maths claires.
❌ **Manque** :
- 🟠 Pas de visualisation de la **convergence** training (courbes loss).
- 🟠 Pas de comparaison perf SGD vs Adam vs AdamW sur le même problème.
- 🟢 Pas de section vanishing/exploding gradient illustrée.

### DL_PyTorch
✅ **Points forts** : suit blueprint 16 sections, XOR + digits 8x8 + Cal Housing exécutés, classes propres.
❌ **Manque** :
- 🟠 Pas d'exemple `torch.compile` (l'innovation PyTorch 2.x).
- 🟠 TensorBoard en pseudo-code uniquement.
- 🟢 Pas de section **mixed precision** (autocast).

### DL_TensorFlow
✅ **Points forts** : `tf.GradientTape` + `tf.function` exécutés, XOR + digits + Cal Housing.
❌ **Manque** :
- 🟠 Custom training loop pseudo-code seulement pour les variantes (gradient clipping, accumulation).
- 🟠 Pas d'exemple TFLite export pour mobile.
- 🟢 Pas de section **distributed** (`tf.distribute`).

### DL_Keras
✅ **Points forts** : 3 syntaxes (Sequential/Functional/Subclassing), Cal Housing + digits exécutés. Multi-backend bien expliqué.
❌ **Manque** :
- 🟠 Pas de démo réelle du **switch backend** (faire tourner le même notebook avec `KERAS_BACKEND=tensorflow` puis `=torch` puis `=jax`).
- 🟢 Pas d'exemple `keras.utils.PyDataset` custom.

### DL_JAX
✅ **Points forts** : explication fonctionnelle propre, JAX/Flax/Optax/Orbax positionnés.
❌ **Manque** :
- 🔴 **Tout est pseudo-code** (deps `jax/flax` non installées). À exécuter pour de vrai.
- 🟠 Pas de démo `vmap` / `grad-of-grad` (les superpouvoirs JAX).
- 🟢 Pas de comparaison perf JAX vs PyTorch sur le même MLP.

### DL_Frameworks_Comparatif
✅ **Points forts** : matrice de décision, idiome côte-à-côte (modèle MLP en 4 versions), bench PyTorch sur digits exécuté.
❌ **Manque** :
- 🔴 **Bench réel manquant pour TF/Keras/JAX** (seul PyTorch est tourné — pour avoir un vrai comparatif, il faut les 4).
- 🟠 Pas de bench mémoire (juste perf temps).
- 🟢 Pas de comparaison déploiement (export ONNX, TFLite, etc.).

### DL_KAN_Kolmogorov_Arnold
✅ **Points forts** : théorème expliqué, B-splines illustrées, comparatif KAN vs MLP, decision tree d'utilisation.
❌ **Manque** :
- 🟠 pykan en pseudo-code uniquement. Exécuter le symbolic regression mini exemple.
- 🟢 Pas d'exemple **efficient-kan** (la version perf).

---

## 📚 Vague 8 — Structures Python/Pandas

### Structure_Python
✅ **Points forts** : panorama complet Python avancé, type hints 3.10+, pattern matching, async.
❌ **Manque** :
- 🟠 Pydantic en pseudo-code uniquement.
- 🟢 Pas d'exemple **dataclass + slots** ou **__slots__** pour la perf.

### Structures_L_T_D_Cheat_Sheet
✅ **Points forts** : cheat-sheet dense et utilisable, NumPy bien couvert, xarray présenté.
❌ **Manque** :
- 🟠 xarray en pseudo-code uniquement.
- 🟢 Pas d'exemple `collections.abc` (typing structurel).

### Structures_DataFrame
✅ **Points forts** : pandas 2.x complet, GroupBy / Merge / Pivot / Apply, Arrow backend, Polars cité.
❌ **Manque** :
- 🟠 Polars en pseudo-code uniquement → ajouter un exemple court.
- 🟠 Pas de bench pandas vs Polars sur même tâche.
- 🟢 Pas d'exemple `dask.dataframe` (alternative pour > RAM).

### Structure_BDD_DataFrame
✅ **Points forts** : SQLAlchemy 2.x + psycopg3 + Mongo + Parquet bien couvert, bonnes pratiques (pooling, retry).
❌ **Manque** :
- 🟠 Pas d'exemple **fastparquet** vs **pyarrow** comme backend.
- 🟠 SQLAlchemy + DuckDB combo (DuckDB peut être backend SQLA).
- 🟢 Pas d'exemple **Alembic** migrations.

### Structures_Preprocessing
✅ **Points forts** : Pipeline complet exécuté, encoders / scalers comparés visuellement, utils intégrés.
❌ **Manque** :
- 🟠 TargetEncoder exécuté mais pas démonstré sur cardinalité haute.
- 🟢 Pas d'exemple `feature-engine` (alternative sklearn-compatible).

### Structure_Generer_Donnees_Classification
✅ **Points forts** : 6 datasets sklearn visualisés (figure 2x3), make_classification paramétré.
❌ **Manque** :
- 🟠 SDV (CTGAN) en pseudo-code uniquement.
- 🟠 Faker en pseudo-code uniquement.
- 🟢 Pas de section sur la **privacy synthétique** (k-anonymity, differential privacy).

---

## 📚 Vague 9 — BDD / TdS / DE / Apps

### BDD_DuckDB
✅ **Points forts** : SQL moderne exécuté (window, CTE, QUALIFY), pandas/Parquet/Arrow intégration, extensions.
❌ **Manque** :
- 🔴 NYC Taxi parquet en pseudo-code (dataset à télécharger).
- 🟠 Extension `vss` (vector search DuckDB) à exécuter — concurrence avec BDD_Vectorielles.
- 🟢 Pas de comparatif perf DuckDB vs Polars sur même tâche.

### TdS_Introduction_Traitement_Signal
✅ **Points forts** : FFT/STFT/Hilbert/Butterworth/Savgol/médian/MA tous exécutés avec figures, vue d'ensemble cohérente.
❌ **Manque** :
- 🟠 Wavelets en pseudo-code uniquement (PyWavelets non installé).
- 🟠 EMD en pseudo-code uniquement (PyEMD non installé).
- 🟠 librosa en pseudo-code uniquement — exécuter sur `librosa.example("trumpet")`.

### DE_Docling
✅ **Points forts** : architecture Docling expliquée, comparatif 7 outils, intégration LlamaIndex documentée.
❌ **Manque** :
- 🔴 **Tout est pseudo-code** (Docling non installé). À tourner sur un PDF sample.
- 🟠 Pas d'exemple **Surya OCR** (la SOTA 2025-2026).
- 🟢 Pas d'exemple **table extraction** comparé entre Docling, camelot, pdfplumber sur même PDF.

### Flask_API
✅ **Points forts** : tuto complet REST avec CRUD, auth API key, payload variés (form/upload/streaming), tests, déploiement Gunicorn.
❌ **Manque** :
- 🔴 **Tout est pseudo-code** (Flask non installé / pas exécutable depuis le notebook qui tournerait l'app). À porter en `app.py` séparé avec instructions de run.
- 🟠 Pas de section **WebSocket** (Flask-SocketIO).
- 🟢 Pas d'exemple **rate limiting** (Flask-Limiter).

### FastAPI_API
✅ **Points forts** : tuto async complet, Pydantic v2, DI, OAuth2/JWT, OpenAPI auto, streaming SSE.
❌ **Manque** :
- 🔴 **Pseudo-code** (idem Flask) — fournir un `main.py` exécutable.
- 🟠 Pas d'exemple **WebSocket** end-to-end (chat live).
- 🟢 Pas d'exemple intégration **Celery** + Redis pour les tâches longues.

### Streamlit_brique
✅ **Points forts** : cheat-sheet complet (widgets/layout/cache/state/chatbot LLM 2026/dataframes/déploiement).
❌ **Manque** :
- 🔴 Pseudo-code (Streamlit notebook ≠ Streamlit app — fournir `app.py` séparé).
- 🟠 Pas d'exemple **Gradio** en alternative.
- 🟢 Pas d'exemple `st.column_config` (les nouveaux types dataframe interactif).

---

## 🚦 Priorités d'amélioration (Top 10)

| # | Action | Priorité | Notebook(s) |
|---|---|---|---|
| 1 | Télécharger NASA Turbofan + NYC Taxi + Wikipedia subset + PDFs samples + créer scripts `download_data.sh` | 🔴 | TS_Maintenance, BDD_DuckDB, BDD_Vectorielles, NLP_Recherche, DE_Docling |
| 2 | Convertir tous les pseudo-code Apps (Flask, FastAPI, Streamlit) en **fichiers .py exécutables** dans `apps/` | 🔴 | Flask_API, FastAPI_API, Streamlit_brique |
| 3 | Bench DL Frameworks réel sur les 4 backends | 🔴 | DL_Frameworks_Comparatif |
| 4 | Exécuter `DL_JAX` (installer jax/flax/optax) | 🔴 | DL_JAX |
| 5 | Exécuter Docling sur un PDF sample | 🔴 | DE_Docling |
| 6 | Notebook **NLP_RAG_Bench_Embedders** : comparer BGE/E5/GTE/MixedBread sur dataset BEIR mini | 🟠 | NEW |
| 7 | Style matplotlib unifié + figures pour chaque notebook avec courbes | 🟠 | Tous |
| 8 | Suite de tests pytest pour les fonctions utilitaires extraites | 🟠 | Tous |
| 9 | Cellules d'exercices pédagogiques | 🟠 | Tous |
| 10 | Diagrammes Mermaid pour les pipelines complexes (RAG, MLOps, RL) | 🟢 | NLP_Recherche, ML_MLFlow, ML_Apprentissage |

---

## 📅 Plan d'action recommandé

**Phase A (1 jour) — Datasets + Apps exécutables** :
- Script `scripts/download_data.sh` qui télécharge tout
- Convertir Flask/FastAPI/Streamlit en projets `.py` séparés sous `apps/`

**Phase B (1 jour) — Exécution end-to-end** :
- Tourner DL_JAX, DE_Docling, DL_Frameworks_Comparatif (bench réel)
- Inclure les outputs dans les .ipynb (`jupyter nbconvert --execute`)

**Phase C (1 jour) — Polish** :
- Style matplotlib unifié
- Suite pytest sur utils
- Cellules d'exercices

**Phase D (1-2 jours) — Nouveaux sujets** :
- Cf `00_sujets_nouveaux.md`.
