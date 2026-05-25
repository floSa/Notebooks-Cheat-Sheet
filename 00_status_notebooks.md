# 📊 Suivi des notebooks

État de chaque notebook au long du projet — enrichi avec l'audit Phase 3 (2026-05-25).

**Légende statut :**
- `⏳ à faire` — pas encore traité
- `📥 ingéré` — converti en `.md`, lu, prêt pour refonte
- `🚧 en cours` — refonte en cours
- `✅ fait` — `.ipynb` final dans `04_notebooks_finaux/`
- `❌ bloqué` — problème (voir note)
- `🗑️ supprimé` — décidé de retirer du corpus
- `🔀 fusionné` — fusionné avec un autre notebook (indiquer la cible)

**Légende rôle :** `CS` = Cheat-sheet · `TUTO` = Tutoriel · `WIKI` = Wiki technique

**Légende effort :** `S` (≤ 1h) · `M` (1–3h) · `L` (3h+)

**Légende refresh 2026 :** `🌐` = recherche web obligatoire avant refonte (sujet fast-moving)

---

## 🎯 Décisions structurantes issues de l'audit

### Suppressions proposées
| Notebook | Raison |
|---|---|
| `AAA_Test_ML.md` | Préfixe `AAA_`, scratch test pipeline Titanic sans valeur pédagogique |
| `Suppr_ML_Bench_Regression_Classification(A finir).md` | Préfixe `Suppr_` + "(A finir)" — brouillon inachevé |

### Fusions / arbitrages proposés
| Notebooks | Décision |
|---|---|
| `TS_Maintenance_Prédictive.md` vs `_GOOD.md` | Supprimer l'ancienne, garder `_GOOD` (renommée en `TS_Maintenance_Predictive`) |
| `BDD_Vectorielles.md` + `retrieval_BDD_Vectorielle.md` | Fusionner en un seul `BDD_Vectorielles` (forte duplication) |
| `Structures_Preprocessing_Function_Utiles.md` (2 KB) | Fusionner dans `Structures_Preprocessing` ou `Structure_Python` (trop petit pour rester seul) |
| `NLP_NER` + `NLP_NER_BiLSTM_CRF` | **Garder les 2** : NER devient tutoriel transformers-first, BiLSTM_CRF devient wiki historique avec disclaimer |
| `TS_Time_Series_Intro` + `TS_Time_Series_Overview` | **Garder les 2** : Intro = tutoriel débutant, Overview = wiki de référence |

### Renommages proposés (typos / caractères problématiques)
| Avant | Après |
|---|---|
| `Structure_Pyhton.ipynb` | `Structure_Python.ipynb` |
| `Structure_BDD_&_DataFrame.ipynb` | `Structure_BDD_DataFrame.ipynb` |
| `ML_Régression_&_Classification_Multiple.ipynb` | `ML_Regression_Classification_Multiple.ipynb` |
| `Structure_Generer_Données_pour_Classification.ipynb` | `Structure_Generer_Donnees_Classification.ipynb` |
| `Détection D_outliers.ipynb` | `Detection_Outliers.ipynb` |
| `NLP_Classification_Spervisé.ipynb` | `NLP_Classification_Supervisee.ipynb` |
| `KAN (Kolmogorov-Arnold Networks).ipynb` | `DL_KAN_Kolmogorov_Arnold.ipynb` |
| `TdS_Introduction au Traitement du Signal.ipynb` | `TdS_Introduction_Traitement_Signal.ipynb` |
| `TS_Maintenance_Predictive_GOOD.ipynb` | `TS_Maintenance_Predictive.ipynb` (après suppression de l'autre) |
| `TS_ARIMAs_Revoir.ipynb` | `TS_ARIMA.ipynb` (et finaliser) |

> Règle générale appliquée : **pas d'accents, pas d'espaces, pas de `&`, pas de parenthèses** dans les noms de notebook finaux.

---

## Structures / Python de base

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 1 | `Structure_Pyhton.md` | 📥 ingéré | TUTO | S | | Fragmenté (apply, décorateur, f-strings, lambda, typing, yield, pickle, pattern matching, multiprocessing). **Renommer Python**. Ajouter intro structurée, moderniser MP. Candidat absorbtion `Preprocessing_Function_Utiles`. |
| 2 | `Structures_L_T_D_Cheat_Sheet.md` | 📥 ingéré | CS | L | | Volumineux (1.4 MB) — numpy reshape, dict ops, listes texte, xarray. **Manque TOC/index**. Refresh xarray API. |
| 3 | `Structures_DataFrame.md` | 📥 ingéré | TUTO + CS | L | | Solide (chargement, infos, colonnes, NaN, types) mais centré Colab. **Manque merge/join, groupby avancé, pivot**. Refresh pandas 2.0+. |
| 4 | `Structure_BDD_&_DataFrame.md` | 📥 ingéré | TUTO | M | | DataFrame ↔ Postgres (SQLAlchemy/psycopg) / MongoDB. Manque gestion erreurs, transactions, retry. Vérif psycopg3. **Renommer (virer `&`)**. |
| 5 | `Structures_Preprocessing.md` | 📥 ingéré | TUTO + WIKI | L | | Beaucoup de matière mais brouillon (imports, encoding, KNNImputer, PCA, LOF). Manque structure pipeline progressive. |
| 6 | `Structures_Preprocessing_Function_Utiles.md` | 📥 ingéré | WIKI | S | | Micro (2 KB) — `var to string`, recherche liste, dict. **Candidat fusion → absorber dans Preprocessing ou Python**. |
| 7 | `Structure_Generer_Données_pour_Classification.md` | 📥 ingéré | TUTO | L | | Datasets synthétiques (gaussian, moon, blobs, make_classification, JSON). Manque docs paramètres + comparaison. **Renommer**. |

## EDA

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 8 | `EDA_Visualisation_Introduction.md` | 📥 ingéré | TUTO | S | 🌐 | Pure code-dump 20+ viz (sns/plt). Aucune structure pédagogique. Renommer `Seaborn_Matplotlib_Basics` + ajouter plotly. |
| 9 | `EDA_Analyse_Multivarié.md` | 📥 ingéré | WIKI + TUTO | L | 🌐 | Riche : régression, PCA (sklearn/Prince/Fanalysis), CA, MCA, FAMD, Procuste. **Migrer Prince/Fanalysis vers alternatives maintenues**. Ajouter matrice choix-méthode. |
| 10 | `EDA_Stats_Analyse_Desc_Visual.md` | 📥 ingéré | WIKI | M | | Stats desc, distributions, multivariée (qual/quant). Bon squelette. Candidat fusion partielle avec `EDA_Visualisation`. |
| 11 | `Détection D_outliers.md` | 📥 ingéré | TUTO + WIKI | M | | Elliptic envelope, OC-SVM, LOF, Isolation Forest. Manque DBSCAN, decision tree de choix. **Renommer `Detection_Outliers`**. |

## ML — généraliste

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 12 | `ML_Régression_&_Classification_Multiple.md` | 📥 ingéré | TUTO | M | | Bias-variance, KNN, classif, bootstrap, RF, extra-trees. Pédago inégale, manque preprocessing/évaluation. **Renommer (virer `&`/accent)**. |
| 13 | `ML_Regression_Classification_CV_GridSearch.md` | 📥 ingéré | WIKI | L | | **Très complet** (2.4 MB) : Lasso, DT, SGD, SVR, RF, GB, AdaBoost, clustering (KMeans/DBSCAN), classif complète, Optuna/Hyperopt. Format référence idéal. |
| 14 | `ML_Bagging_Boosting.md` | 📥 ingéré | TUTO | M | 🌐 | Bias-variance, bagging vs boosting, RF, Extra-Trees. **XGBoost/LightGBM/CatBoost manquants** — à ajouter. |
| 15 | `ML_Optimisation_de_Modèles.md` | 📥 ingéré | TUTO | M | 🌐 | Optuna multi-framework (sklearn/XGB/CatBoost/LGBM/Keras). Très complet. Manque visualisation Optuna 2026 + Hyperopt vraies recettes. |
| 16 | `ML_Explication_Feature_Importance_Selection.md` | 📥 ingéré | WIKI | L | 🌐 | **Exhaustif** : RFECV, SelectFromModel, GB/RF importance, Eli5, Boruta, **SHAP** (RF/XGB/CB/Ensemble), LIME, DL, Display API. Refresh SHAP ecosystem. |
| 17 | `ML_MLFlow_Bench.md` | 📥 ingéré | WIKI | S | 🌐 | **Bugué** (if incomplet ligne 146). À reprendre de zéro avec MLFlow 2.x : Model Registry, Deployments, intégrations 2026. |
| 18 | `ML_Apprentissage_par_Renforcement.md` | 📥 ingéré | TUTO | S | | Bandits (UCB, Thompson). **30% du sujet seulement** — ajouter Q-learning, DQN, policy gradient, Gymnasium ? Ou recadrer titre. |
| 19 | `Test_données_manquante_modèles.md` | 📥 ingéré | TUTO | S | | Expé SimpleImputer vs KNNImputer + RF/XGB. Manque conclusion/analyse. À compléter ou intégrer dans Preprocessing. |
| 20 | `Suppr_ML_Bench_Regression_Classification(A finir).md` | 🗑️ supprimer | — | — | | Brouillon inachevé, code incomplet. Préfixe `Suppr_` confirme. |
| 21 | `AAA_Test_ML.md` | 🗑️ supprimer | — | — | | Scratch test pipeline Titanic, 3 cellules, sans contexte. |
| 22 | `INRIA_SKLearn_MOOC.md` | 📥 ingéré | TUTO | L | 🌐 | Contenu complet MOOC INRIA (~987 lignes). Refresh sklearn API (validation_curve `grid_scores_` deprecated). |

## DL

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 23 | `DL_Deep_Learning_Maths.md` | 📥 ingéré | TUTO | M | 🌐 | Gradient descent, overflow, généralisation N couches. Manque chapitre intro, momentum/Adam, régularisation. Ajouter JAX/autograd moderne. |
| 24 | `DL_Tensorflow_Keras.md` | 📥 ingéré | TUTO | M | 🌐 | ANN, batch, régularisation, weights, SHAP. **Décision stratégique** : migrer vers Keras 3 (multi-backend) ou archiver et basculer sur PyTorch. |
| 25 | `DL_PyTorch.md` | 📥 ingéré | TUTO | L | | **Excellent** (13 MB, 12+ sections : tenseurs, GPU, Dataset/Loader, training, eval, régression, classif, TensorBoard, ROC/AUC). Ajouter ViT, autograd hooks, distributed. |
| 26 | `KAN (Kolmogorov-Arnold Networks).md` | 📥 ingéré | WIKI | S | 🌐 | Théorème KA, contribution, avantages/limites. Compléter avec impl pratique (pykan/torch-kan), benchmarks 2024-2026. **Renommer**. |

## NLP

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 27 | `NLP_Transformers.md` | 📥 ingéré | TUTO | L | 🌐 | Fragmenté (RoBERTa, configs Colab). **Refonte massive** : Hugging Face 2026, LLMs, instruction-tuning, modèles open-source. |
| 28 | `NLP_NER.md` | 📥 ingéré | TUTO | M | 🌐 | Formats IOB/BILOU/IOBES + LSTM/CRF. Pivoter vers HF transformers-NER, garder formats. |
| 29 | `NLP_NER_BiLSTM_CRF.md` | 📥 ingéré | WIKI historique | M | | GMB dataset + BiLSTM-CRF TF. **Conserver comme référence historique** avec disclaimer + lien vers NER moderne. |
| 30 | `NLP_Classification_Smote.md` | 📥 ingéré | TUTO | L | 🌐 | LSTM IMDB + BoW/TF-IDF + NB/SVM/LR/RF. **SMOTE annoncé mais absent** → ajouter ou renommer. Ajouter DistilBERT. |
| 31 | `NLP_Classification_Spervisé.md` | 📥 ingéré | TUTO | S | 🌐 | 20newsgroups + TF-IDF + baselines. Trop basique. Ajouter zero-shot/few-shot LLM. **Renommer `Supervisee`**. Candidat fusion avec Smote ? |
| 32 | `NLP_Recherche_d_informations.md` | 📥 ingéré | TUTO | M | 🌐 | TF-IDF, Gensim, SpaCy, BERT pour search. **Mise à jour RAG/dense retrievers (ColBERT, SBERT) + Langchain/LlamaIndex**. |

## Time Series

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 33 | `TS_Time_Series_Intro.md` | 📥 ingéré | TUTO | M | 🌐 | Tendance, saison, autocorr, lag, régression. Manque stationnarité (ADF), ACF/PACF complet. Refresh imports Colab. |
| 34 | `TS_Time_Series_Overview.md` | 📥 ingéré | WIKI | L | 🌐 | **Référence** : EDA, feat eng (lag/rolling/décomp), down/up-sampling, ARIMA/SARIMA, ACF/PACF, prédictions. Manque métriques détaillées. |
| 35 | `TS_ARIMAs_Revoir.md` | 📥 ingéré | TUTO | S | 🌐 | Brouillon AR/MA/ARMA/ARIMA + GridSearch. Erreur import dernière cellule. **Renommer `TS_ARIMA`** et finaliser. |
| 36 | `TS_Generer_Sequence.md` | 📥 ingéré | TUTO | S | | Snippet utility génération séquences chevauchantes (prep LSTM). Compléter docstring + cas d'usage, ou déplacer en utility. |
| 37 | `TS_Maintenance_Prédictive.md` | 🗑️ supprimer | — | — | | Version antérieure (25 KB) moins complète que `_GOOD`. |
| 38 | `TS_Maintenance_Predictive_GOOD.md` | 📥 ingéré | CS (case study) | L | 🌐 | **Garder, renommer `TS_Maintenance_Predictive`**. EDA pompe, interpolation, LOF, PCA (3 stratégies), ML/DL/LSTM, optim hyperparam. Refresh Keras/Colab. |

## BDD

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 39 | `BDD_DuckDB.md` | 📥 ingéré | TUTO | S | | DuckDB vs pandas/SQLite. À structurer + ajouter cas d'usage réels. |
| 40 | `BDD_Vectorielles.md` | 📥 ingéré | TUTO | M | 🌐 | SentenceTransformer + FAISS (Flat/IVF/HNSW) + Weaviate partiel. Ajouter **Qdrant, LanceDB, pgvector**. **Fusionner avec retrieval_BDD_Vectorielle**. |
| 41 | `retrieval_BDD_Vectorielle.md` | 🔀 fusionner → 40 | WIKI | L | 🌐 | Forte duplication avec `BDD_Vectorielles`. Consolider en 1 référentiel. |

## Traitement du signal

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 42 | `TdS_Introduction au Traitement du Signal.md` | 📥 ingéré | TUTO | L | | Fourier, ondelettes, Hilbert, filtrage, extraction amplitude. Excellent théorique. Ajouter applis audio/image (librosa). **Renommer**. |

## Data Engineering / Apps

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 43 | `DE_Docling.md` | 📥 ingéré | TUTO | M | 🌐 | Installation, conversion, layout, tables, images. Ajouter comparatifs PyPDF/pdfplumber + OCR scans. Suivre releases mensuelles. |
| 44 | `Flask_API.md` | 📥 ingéré | TUTO | S | 🌐 | CRUD basique. **Décision stratégique** : migrer vers FastAPI ou maintenir comme legacy + ajouter FastAPI à côté ? |
| 45 | `Streamlit_brique.md` | 📥 ingéré | TUTO | M | | Async, autocomplete, chatbot, viz, DataFrame. Ajouter caching/perf + déploiement (HF Spaces, Railway). |

---

## 📈 Compteurs

- Total fichiers ingérés : **45**
- À traiter (refonte) : **42** (après suppression `AAA`, `Suppr_`, et `TS_Maintenance_Prédictive` ancienne version)
- Fusions prévues : **2** (`retrieval_BDD_Vectorielle` → `BDD_Vectorielles`, `Preprocessing_Function_Utiles` → ? )
- Renommages prévus : **~10**
- Refresh 2026 obligatoire (🌐) : **24 notebooks**
- Effort cumulé : S=14 · M=15 · L=13 · Total ≈ **~75-100h** de refonte

## 🗓️ Ordre de traitement suggéré

Pour mutualiser les recherches web :
1. **Vague NLP/RAG/LLMs** (🌐 dense) : Transformers → NER → Classification → Recherche_d_informations
2. **Vague Vector DB** (🌐) : BDD_Vectorielles + retrieval (fusion)
3. **Vague MLOps** : ML_MLFlow_Bench (refresh majeur)
4. **Vague TS** : Intro → Overview → ARIMA → Maintenance_Predictive
5. **Vague EDA/Stats**
6. **Vague ML core** (le plus volumineux mais peu de refresh)
7. **Vague DL** (décision stratégique TF/PyTorch à prendre avant)
8. **Vague Structures**
9. **Vague Apps/DE**
