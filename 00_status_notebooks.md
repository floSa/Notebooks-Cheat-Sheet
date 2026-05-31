# 📊 Suivi des notebooks

> **⚠️ ALERTE HONNÊTETÉ (correction)** :
> Les statuts `✅ fait` qui apparaissent dans ce tableau ci-dessous **ne reflètent PAS le contrat** défini dans `00_workflow_contract.md`.
> Concrètement : tous ces notebooks ont été générés depuis les titres (pas depuis le contenu original lu), leurs cellules de code n'ont pas été exécutées dans l'ordre end-to-end, beaucoup référencent des variables jamais construites. **Lire `00_workflow_contract.md`**.
>
> Considérer chaque `✅ fait` ci-dessous comme un **🟡 v0** (squelette posé, à reprendre selon le workflow contrat).
> Compteur réel : **0 notebook ✅ fait** au sens du contrat.

**Légende statut :**
- `⏳ à faire` — pas encore traité
- `📥 ingéré` — converti en `.md`, lu, prêt pour refonte
- `🟡 v0` — squelette posé, **code non vérifié end-to-end** (= cas des "✅ fait" du tableau ci-dessous)
- `🟠 partiel` — sections manquent vs original
- `🔴 cassé` — erreur connue
- `🚧 en cours` — refonte en cours selon contrat
- `✅ fait` — les 5 critères du contrat sont vérifiés (`00_workflow_contract.md`)
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

### Décisions stratégiques actées (2026-05-25)

**Stack DL — split par framework + comparatif** (au lieu de `DL_Tensorflow_Keras` monolithique) :
- `DL_PyTorch.md` (existant) — refonte selon blueprint
- `DL_TensorFlow.md` (nouveau) — selon blueprint
- `DL_Keras.md` (nouveau) — selon blueprint (Keras 3 multi-backend)
- `DL_JAX.md` (nouveau) — selon blueprint (via Flax/Optax/Orbax)
- `DL_Frameworks_Comparatif.md` (nouveau) — sections 15-16 du blueprint rejouées en 4 versions parallèles + benchmarks (LoC, temps, mémoire, métriques)
- → `DL_Tensorflow_Keras.md` est éclaté/supprimé (ses techniques avancées — batch équilibré, class/sample weights, SHAP — sont absorbées dans le blueprint commun)
- **Blueprint détaillé : [`00_blueprint_DL_frameworks.md`](00_blueprint_DL_frameworks.md)** (16 sections + cas comparatif)
- **Datasets imposés** : MNIST (classif) + California Housing (régression) + XOR continu (toy)

**Stack Web API — dual + comparatif** :
- `Flask_API.md` (existant) — refonte en tutoriel complet (prise en main, bonnes pratiques, envoi de différents types de données : JSON / form / file upload / streaming)
- `FastAPI_API.md` (nouveau) — tutoriel complet équivalent
- Comparatif Flask vs FastAPI consolidé dans l'un des deux (section dédiée) ou notebook séparé `APIs_Comparatif.md` (à décider en cours)

**Ordre de traitement validé** : par sujets fast-moving d'abord (vague NLP/RAG → vague Vector DB → vague MLOps), puis le reste.

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
| 8 | `EDA_Visualisation_Introduction.md` | ✅ fait (1er conforme contrat) | TUTO + CS | L | | **Refonte complète selon workflow contrat 5 critères** (commit `e62e537`) : 1239 lignes originales lues intégralement, plan + structure dans `scripts/_sandbox/`, sandbox `.py` exécuté end-to-end (37 figures générées), `check_format.py --both` vert (166 cellules / 46 code), round-trip relancé (exit 0). 15 sections : setup, charte graphique 8 couleurs + helpers, univariée quanti/quali, bivariée qq/qq×ql/ql×ql, multivariée, patterns bar (5) + line (8) sur diamonds, temporels synthétique (2), expertes plotly/mosaic (7), bonnes pratiques, sources. Datasets : Titanic + diamonds + 2 synthétiques (zéro dépendance externe). |
| 9 | `EDA_Analyse_Multivarie.md` (renommé) | ✅ fait (conforme contrat 5 critères) | WIKI + TUTO + CS | L | 🌐 | **Refonte complète selon workflow** : original 1472 lignes lu intégralement, plan + structure dans `scripts/_sandbox/`, sandbox `.py` exécuté end-to-end (exit 0, 11 figures), `check_format.py --both` vert (96 cellules / 34 code), image arbre de décision extraite en PNG (`04_notebooks_finaux/images/`). **Décisions 2026** : `fanalysis` (abandonné) supprimé → graphes réimplémentés en helpers matplotlib ; `prince` modernisé en 0.19 (API 2022, backbone unique des 6 méthodes factorielles) ; MCA sur Titanic catégoriel (remplace le fichier google-drive non reproductible) ; GPA via `prince.GPA` (remplace l'implé maison fausse) ; +UMAP + grille comparative 6 réductions. Sections : régression lin/log, ANOVA, MANOVA, PCA (sklearn+prince), CA, MCA, MFA, FAMD, GPA, manifold (PCA/Isomap/LLE/MDS/t-SNE/UMAP), récap. |
| 10 | `EDA_Stats_Analyse_Desc_Visual.md` | ✅ fait | CS + WIKI | L | | Catalogue de viz EDA — toutes les spécificités du notebook original préservées (camembert+barplot types, pies quali, ConvexHull, twinx countplot, ScalarMappable, Pareto Plotly, etc.) + ajout skew/kurt/missing%, chi²+Cramer's V, AutoEDA 2026, workflow recommandé. 50 cellules code, 95 cellules MD. Sandbox exit 0, check_format vert, 7 bugs de l'original fixés. |
| 11 | `Detection_Outliers.md` (renommé) | ✅ fait (conforme contrat 5 critères) | WIKI + TUTO | M | 🌐 | **Refonte complète selon workflow** : original 481 lignes lu intégralement (plan + structure en sandbox), sandbox `.py` exécuté end-to-end (exit 0, 15 figures), `check_format --both` vert (81 cellules / 24 code). **Datasets** : synthétique 2D **avec ground-truth** (make_blobs + outliers injectés) + Titanic (mutualisé) + Air Passengers. **5 bugs de l'original fixés** (X=iris écrasé, OCSVM 1-point, drop index incohérent, distplot supprimé, glass.csv perdu). **Supprimés** : Colab mount, df.head inliers, images base64, pseudo-code final cassé (melt/pivot/dfa.merge). **Ajouts** (rôle WIKI 2026) : cadrage, Z-score+MAD (maths), maths Mahalanobis/LOF/IForest, comparatif chiffré ROC-AUC, autoencoder PyTorch, **PyOD ECOD/COPOD param-free exécutés**, matrix profile STUMPY, guide de choix, évaluation/seuil. Comparatif : Elliptic AUC=0.999, IForest=0.998, ECOD=0.988. |

## ML — généraliste

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 12 | `ML_Régression_&_Classification_Multiple.md` | 📥 ingéré | TUTO | M | | Bias-variance, KNN, classif, bootstrap, RF, extra-trees. Pédago inégale, manque preprocessing/évaluation. **Renommer (virer `&`/accent)**. |
| 13 | `ML_Regression_Classification_CV_GridSearch.md` | 📥 ingéré | WIKI | L | | **Très complet** (2.4 MB) : Lasso, DT, SGD, SVR, RF, GB, AdaBoost, clustering (KMeans/DBSCAN), classif complète, Optuna/Hyperopt. Format référence idéal. |
| 14 | `ML_Bagging_Boosting.md` | ✅ fait | WIKI+TUTO | M | 🌐 | Refonte complète : biais-variance (maths + décompo empirique k-NN), bootstrap/OOB, bagging↓variance, RF/Extra-Trees, AdaBoost/GBM (maths)/HistGB, trinité XGB/LGBM/CatBoost + early stopping + importances, Voting/Stacking, benchmark récap (California Housing), classification (breast_cancer). 5 critères contrat OK (sandbox exit 0, check_format `--both` vert). |
| 15 | `ML_Optimisation_de_Modèles.md` | 📥 ingéré | TUTO | M | 🌐 | Optuna multi-framework (sklearn/XGB/CatBoost/LGBM/Keras). Très complet. Manque visualisation Optuna 2026 + Hyperopt vraies recettes. |
| 16 | `ML_Explication_Feature_Importance_Selection.md` | 📥 ingéré | WIKI | L | 🌐 | **Exhaustif** : RFECV, SelectFromModel, GB/RF importance, Eli5, Boruta, **SHAP** (RF/XGB/CB/Ensemble), LIME, DL, Display API. Refresh SHAP ecosystem. |
| 17 | `ML_MLFlow_Bench.md` | 🟡 v0 | WIKI + TUTO | M | 🌐 | Refonte from scratch MLflow 3.x : tracking (params/metrics/artifacts/signature), bench multi-modèles (5 algos sur California Housing), Model Registry avec **aliases @champion/@challenger** (stages deprecated), deployment local/Docker, MLflow LLM 3.x (tracing langchain/openai), prod setup (Postgres+S3+CI/CD). Smoke test OK RMSE=0.7455. |
| 18 | `ML_Apprentissage_par_Renforcement.md` | 📥 ingéré | TUTO | S | | Bandits (UCB, Thompson). **30% du sujet seulement** — ajouter Q-learning, DQN, policy gradient, Gymnasium ? Ou recadrer titre. |
| 19 | `Test_données_manquante_modèles.md` | 📥 ingéré | TUTO | S | | Expé SimpleImputer vs KNNImputer + RF/XGB. Manque conclusion/analyse. À compléter ou intégrer dans Preprocessing. |
| 20 | `Suppr_ML_Bench_Regression_Classification(A finir).md` | 🗑️ supprimer | — | — | | Brouillon inachevé, code incomplet. Préfixe `Suppr_` confirme. |
| 21 | `AAA_Test_ML.md` | 🗑️ supprimer | — | — | | Scratch test pipeline Titanic, 3 cellules, sans contexte. |
| 22 | `INRIA_SKLearn_MOOC.md` | ✅ fait | TUTO/WIKI/CS | L | 🌐 | Refonte complète (sklearn 1.8). 5 critères contrat OK : original lu intégralement (1285 l.), sandbox `.py` exit 0, `check_format --both` vert (33 cellules code). 21 bugs corrigés (API `base_estimator`/`grid_scores_` supprimées, ~8 NameError, Google-Drive, info encodage fausse). Ajouts : API/Pipeline, métriques+ROC/PR/calibration, PDP/ICE, nouveautés 1.8 (TunedThreshold, FrozenEstimator, metadata routing). |

## DL

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 23 | `DL_Deep_Learning_Maths.md` | ✅ fait | TUTO + WIKI | M | | Réseau de neurones from scratch (numpy) : neurone/régression logistique → réseau 2 couches → généralisation N couches → §4 implémentée (ReLU+He, Momentum/Adam, L2+Dropout, mini-autograd reverse-mode). Formules en LaTeX (ex-images), 2 schémas réseau conservés. Dataset Colab perdu remplacé par load_digits. 5 critères contrat OK (sandbox + code extrait du .ipynb exit 0, check_format vert). |
| 24 | `DL_Tensorflow_Keras.md` | 📥 ingéré | TUTO | M | 🌐 | ANN, batch, régularisation, weights, SHAP. **Décision stratégique** : migrer vers Keras 3 (multi-backend) ou archiver et basculer sur PyTorch. |
| 25 |  | ✅ fait (conforme contrat 5 critères) | TUTO + WIKI + CS | L | 🌐 | Refonte complète selon workflow (gabarit de référence des 3 notebooks DL frameworks) : 2 originaux lus intégralement (PyTorch UvA 1428 l. + TF/Keras 1082 l.), plan + structure dans scripts/_sandbox/, sandbox .py exécuté end-to-end (exit 0), check_format --both vert. 17 sections du squelette commun. Résultats : XOR 0.998 ; MNIST ANN 0.896 / CNN 0.921 / AUC macro OVR 0.995 ; California RMSE 0.561 R2 0.760 ; F1-macro custom == sklearn ; SHAP DeepExplainer OK. 6 bugs corrigés (set_matplotlib_formats, Boston->California, BreastCancer->MNIST, dropout post-sortie, ROC eval/no_grad+OVR macro, loss batch->epoch). Greffe TF en idiomes PyTorch (batch équilibré/trié paramétré, WeightedRandomSampler, class/sample weights, focal loss, F1 maison). Ajouts 2026 : set_seed, torch.compile, LR scheduling, early stopping, BatchNorm, weight decay. |
| 46 | `DL_Keras.md` (issu du split de #24) | ✅ fait (conforme contrat 5 critères) | CS + TUTO + WIKI | L | 🌐 | **Refonte complète selon workflow** : original `DL_Tensorflow_Keras` lu intégralement (1082 l.) + plan commun 3 frameworks ; plan + structure dans `scripts/_sandbox/`. Sandbox `.py` exécuté end-to-end (exit 0 : XOR val_acc 0.90, CNN MNIST acc 0.93 / macro-AUC 0.996, ANN 0.92, LSTM 0.71, régression R²=0.77, save/load identiques, macro-F1=0.79, focal/weights/SHAP OK), `check_format --both` vert (98 cellules / 38 code / 60 md). **Angle Keras 3 haut niveau** (vs notebook TF bas niveau) : `keras.ops` (NumPy portable), `compile`/`fit`/`callbacks`, multi-backend `KERAS_BACKEND`, `model.save('.keras')`, `PyDataset`, `keras.metrics.Metric` (MacroF1 custom), `train_on_batch` + `train_step` custom (sans GradientTape). **Bugs originaux corrigés** : double softmax (`from_logits=True` + logits), `Stort_modulo` O(n²)/try-except → `sort_by_class_roundrobin` typé, `np.random.choice(len,epochs)` cassé, `X_epoch` non défini, `batch=8`→128, SHAP `DeepExplainer` (cassé Keras 3) → `GradientExplainer`, sigmoid→softmax multiclasse. **Supprimé** : Pima diabetes (Drive mort), CNN dupliqué "keras github", cellule `µ`, debug SHAP/shape ; RNN+LSTM fusionnés en 1 bonus. **Ajouté** (parité blueprint) : `keras.ops`/einsum, set_random_seed, XOR inline, subclassing `keras.Model`, LR scheduling (CosineDecay), focal loss (native + from-scratch `keras.ops`), BatchNorm/AdamW/EarlyStopping, TensorBoard callback, California Housing, frontières XOR. Datasets : MNIST sous-échantillonné + California Housing + `make_classification` + XOR. |
| 47 | `DL_TensorFlow.md` (issu du split de #24) | ✅ fait (conforme contrat 5 critères) | TUTO + WIKI | L | 🌐 | **Refonte complète selon workflow** : original `DL_Tensorflow_Keras` lu intégralement (1082 l.) + plan commun 3 frameworks ; plan + structure dans `scripts/_sandbox/`. Sandbox `.py` exécuté end-to-end et **reproductible** (`enable_op_determinism`, runs identiques ; exit 0 : XOR test acc 1.00, CNN MNIST acc 0.95 / macro-AUC 0.998, ANN 0.92, LSTM 0.70, régression R²=0.78, save/load identiques, MacroF1 custom == sklearn, focal/weights/SHAP OK), `check_format --both` vert (123 cellules / 49 code / 74 md). **Angle TF BAS NIVEAU** (vs notebook Keras 3 haut niveau) : `tf.GradientTape` (autodiff + vérif gradient manuelle + ordre 2), `tf.data.Dataset`, `@tf.function` train_step (timing eager vs graphe), subclassing `tf.keras.Model`, `tf.train.Checkpoint`+`CheckpointManager`, `tf.keras.metrics.Metric` (MacroF1 + `reset_state`), `tf.summary`, early stopping écrit à la main. **Un seul `fit`** (§14, pour montrer l'arg `class_weight`). **Bugs originaux corrigés** : double softmax (`from_logits=True` + logits), `Stort_modulo` O(n²)/try-except → `sort_round_robin` typé O(n), `np.random.choice(len,epochs)` cassé, `X_epoch` non défini, `batch=8`→128, SHAP `DeepExplainer` (cassé TF2/Keras3) → `GradientExplainer` (wrap Functional), sigmoid→logits multiclasse. **Piège Keras 3 traité** : `optimizer.build()` + modèle amorcé avant `@tf.function`. **Supprimé** : Pima diabetes (Drive mort), `google.colab`, CNN tiers copié-collé (keras github), debug SHAP (`µ`, `imshow(shap_values[1][0])`), RNN+LSTM doublons → 1 bonus court. **Ajouté** (parité blueprint) : einsum/matmul (maths), `set_seed`, XOR inline, LR scheduling (ExponentialDecay), focal loss from-scratch (`tf.nn`), BatchNorm/AdamW, California Housing, frontières XOR, ROC OVR macro. Datasets : MNIST sous-échantillonné + California Housing + `make_classification` + XOR. |
| 26 | `KAN (Kolmogorov-Arnold Networks).md` → `DL_KAN_Kolmogorov_Arnold` | ✅ fait (conforme contrat 5 critères) | WIKI + TUTO | M | 🌐 | **Refonte complète selon workflow** : original lu intégralement (81 l., plan + structure en `scripts/_sandbox/`), sandbox `.py` exécuté end-to-end (exit 0, 5 figures), `check_format --both` vert (55 cellules / 15 code / 40 md). **Supprimé** : le faux KAN numpy de l'original (`psi=sin(x+j)`, `phi=sum(y)+i` — fonctions fixes non apprenables, aucun entraînement, pédagogiquement trompeur) + 3 cellules vides. **Refactoré** : toute la théorie (théorème KA + bornes corrigées, Arnold/Hilbert, avantages/limites actualisés). Clarifie le piège **théorème littéral (profondeur 2) ≠ archi KAN pratique** (Liu et al. 2024) que ratait l'original. **Ajouté** (recherche web 2026) : B-splines (maths + Cox-de Boor), **vraie `KANLayer` PyTorch entraînable** (style efficient-kan), entraînement régression symbolique $\exp(\sin(\pi x_1)+x_2^2)$, **benchmark chiffré KAN vs MLP** (KAN 135 params MSE 8.3e-3 < MLP 337 params MSE 2.0e-2, mais KAN plus lent), interprétabilité (tracé des $\phi$), **pykan auto_symbolic** exécuté, classif `make_moons` (KAN/MLP acc 0.96), état 2026 (FastKAN/FasterKAN/P-KAN, KAN 2.0). Datasets 100% synthétiques. |

## NLP

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 27 | `NLP_Transformers.md` | 🟡 v0 | TUTO + WIKI | L | 🌐 | Refonte complète Hugging Face 5.x (2026) : briques de base, familles (ModernBERT/Llama/T5/multimodaux), pipelines, fine-tuning Trainer sur 20 Newsgroups, PEFT/LoRA, génération chat templates, bonnes pratiques (quantization/serving/eval). Smoke test imports + APIs OK. |
| 28 | `NLP_NER.md` | 🟡 v0 | TUTO + WIKI | M | 🌐 | Refonte transformers-first : formats IOB/BIOES/BILOU, dataset CoNLL-2003, fine-tuning DistilBERT (tokenize_and_align_labels), GLiNER zero-shot (game-changer 2024-2026), LLMs/function calling, eval seqeval entity-level. BiLSTM-CRF renvoie au notebook dédié. Smoke test alignment+seqeval OK. |
| 29 | `NLP_NER_BiLSTM_CRF.md` | 🟡 v0 | WIKI historique | S | | Refonte en wiki pédagogique : disclaimer "méthode 2016-2019 dépassée", embeddings classiques, BiLSTM (intuition), maths CRF (formulation + Viterbi + algo forward), pseudo-code PyTorch (torchcrf), tableau alternatives 2026 avec F1 comparés. Renvoie vers `NLP_NER` pour le moderne. |
| 30 | `NLP_Classification_Smote.md` | 🟡 v0 | TUTO + CS | M | 🌐 | Refonte focalisée déséquilibre : métriques (F1macro/PR-AUC/MCC vs accuracy mensongère), SMOTE + variantes (Borderline/ADASYN/SVMSMOTE) sur TF-IDF via imblearn.Pipeline, class_weight, focal loss (maths), threshold tuning (souvent +15 pts gratuits), augmentation 2026 (back-translation/EDA/LLM-augmentation). Smoke test démontre narratif : F1+ 0.52→0.86→0.87→0.90. Dataset 20news binarisé à 26% positifs. |
| 31 | `NLP_Classification_Supervisee.md` (renommé) | 🟡 v0 | CS + TUTO | M | 🌐 | Refonte cheat-sheet 2026 : matrice de décision (TF-IDF/embeddings/fine-tune/ZSL/few-shot), pipeline baseline TF-IDF+LogReg avec interprétabilité (top mots), embeddings sentence-transformers+LogReg (sweet spot), SetFit few-shot, renvoi NLP_Transformers pour fine-tune détaillé, ZSL via DeBERTa-MNLI et LLMs, métriques au-delà accuracy (F1macro/MCC/PR-AUC/calibration). Smoke test 4 classes F1=0.798. |
| 32 | `NLP_Recherche_d_informations.md` | 🟡 v0 | TUTO + WIKI | L | 🌐 | Refonte massive : sparse (TF-IDF + BM25 via bm25s), dense (sentence-transformers + BGE/E5), hybrid (RRF), reranking (ColBERT + cross-encoders), pipeline RAG complet, frameworks (LangChain/LlamaIndex/DSPy), eval (Recall@k/MRR/RAGAS), bonnes pratiques 2026 (HyDE, multi-query, Corrective RAG). Smoke test BM25+RRF OK. |

## Time Series

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 33 | `TS_Time_Series_Intro.md` | 🟡 v0 | TUTO | M | 🌐 | Refonte tutoriel débutant : décomposition, stationnarité (ADF+KPSS), ACF/PACF, lag features, train/test temporel strict, baseline LinReg. |
| 34 | `TS_Time_Series_Overview.md` | 🟡 v0 | WIKI | L | 🌐 | Wiki exhaustif : matrice de décision 2026, ETS+ARIMA+Theta, Prophet, ML global (Nixtla mlforecast), DL (TFT/NHiTS/DeepAR/PatchTST), **foundation models** (TimeGPT/Chronos/TimesFM/Moirai), métriques (MASE/WAPE). |
| 35 | `TS_ARIMA.md` (renommé) | 🟡 v0 | WIKI + TUTO | M | 🌐 | Refonte ARIMA : famille (AR/MA/ARMA/ARIMA/SARIMA/SARIMAX), Box-Jenkins, ACF/PACF lecture, diagnostic résidus, **AutoARIMA** (pmdarima+statsforecast). Smoke MAPE 2-3%. |
| 36 | `TS_Generer_Sequence.md` | 🟡 v0 | CS | S | | Cheat-sheet sliding window : NumPy (boucle + vectorisé stride_tricks), multi-features 3D, PyTorch Dataset streaming, split temporel avec gap, multi-séries (panel), renvoi Nixtla. |
| 37 | `TS_Maintenance_Prédictive.md` | 🗑️ supprimé | — | — | | Remplacé par version `_GOOD` renommée. |
| 38 | `TS_Maintenance_Predictive.md` (renommé) | 🟡 v0 | CS (case study) | L | 🌐 | Pipeline complet : RUL/classif/anomaly, EDA capteurs, feature eng rolling, **PCA 3 stratégies**, XGBoost régression RUL, **score asymétrique C-MAPSS**, variante LSTM pseudo, déploiement edge/cloud. Smoke MAE=20 cycles. |

## BDD

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 39 | `BDD_DuckDB.md` | ✅ fait | TUTO + CS + WIKI | M | 🌐 | **Conforme contrat 5 critères** : original lu intégralement (plan + structure en `scripts/_sandbox/`), sandbox `.py` exécuté end-to-end (exit 0, benchmark cohérent + 1 figure), `check_format --both` vert (29 code / 48 md / 77 cellules). Refonte 2026 (DuckDB 1.5.x, recherche web) : positionnement OLAP/OLTP, 3 façons d'exécuter du SQL (`sql()`/`connect()`/DB-API), requêter pandas/parquet **sans import**, requêtes progressives (SELECT→GROUP BY→JOIN→sous-requête %), **Friendly SQL** (GROUP BY ALL/EXCLUDE/COLUMNS/FROM-first), **Relational API** lazy, pushdown parquet (EXPLAIN), interop pandas/Polars/Arrow/NumPy, **benchmark pandas/DuckDB/SQLite**, **JupySQL** (corrige la requête cassée de l'original), tableau de choix. Supprimé : `drive.mount` Colab + données Drive perdues (Spotify, Flight Delays 2015), 3 captures base64, historique périmé. Dataset : NYC Yellow Taxi parquet (jan. 2024) + zones + synthétique. |
| 40 | `BDD_Vectorielles.md` | ✅ fait (fusion 40+41) | TUTO + WIKI | L | 🌐 | **Conforme contrat** (sandbox `.py` exit 0 end-to-end, `check_format --both` vert, plan+structure en sandbox). Refonte 2026 : embeddings (+ tableau modèles 2026 BGE-M3/Qwen3/Jina), métriques L2/IP/**cosinus** + normalisation, FAISS Flat/IVF/HNSW, **benchmark recall@k/latence**, panorama 8 VDB en tableau (+ Pinecone/Vespa), démos **exécutées** Chroma/Qdrant/LanceDB/Milvus-Lite, **Weaviate v4 + pgvector** en cellules gardées (try/except) + docker-compose en md, persistance FAISS, guide de choix. Couvre la partie VDB de l'ex-#41 (le reste, parsing PDF/OCR/stockage objet, relève de `DE_Docling`). Dataset : 20 Newsgroups (sklearn). |
| 41 | `retrieval_BDD_Vectorielle.md` | 🔀 fusionné → 40 | — | — | | Contenu absorbé dans `BDD_Vectorielles.md`. |

## Traitement du signal

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 42 | `TdS_Introduction au Traitement du Signal.md` | 📥 ingéré | TUTO | L | | Fourier, ondelettes, Hilbert, filtrage, extraction amplitude. Excellent théorique. Ajouter applis audio/image (librosa). **Renommer**. |

## Data Engineering / Apps

| # | Notebook | Statut | Rôle | Effort | 🌐 | Notes audit |
|---|---|---|---|---|---|---|
| 43 | `DE_Docling.md` | ✅ fait (conforme contrat 5 critères) | TUTO + WIKI | M | 🌐 | **Refonte complète selon workflow** : original lu intégralement (1616 lignes, plan + structure en `scripts/_sandbox/`), sandbox `.py` exécuté end-to-end (exit 0 ; 115 éléments / 6 relations HAS_CAPTION, 5 images extraites, 80 chunks), `check_format --both` vert (78 cellules / 22 code / 56 md). **Docling 2.96 (recherche web 2026)**. **Bugs de l'original corrigés** : `PdfFormatOption(pages_to_convert=)` (kwarg inexistant) → `convert(page_range=)` ; re-converter redondant supprimé ; `do_picture_*=False` puis `get_image()` → `generate_picture_images=True`+`images_scale` ; `DocItemLabel.PARAGRAPH` (jamais émis sur PDF) ; `google.colab.files` (crash hors Colab) ; `!pip` + `plt.show()` + widget ipywidgets. **Supprimé** : 4 des 5 redéfinitions de `extract_structured_elements`, ~6 cellules de debug, 2 cellules vides, upload Colab. **Conservé/consolidé** : tuto layout complet (reading order, bbox, page/texte/table/image, captions, exports, chunking RAG) + **pipeline d'extraction structurée** refondu (priorité utilisateur) : modèle de données typé (dataclasses), helpers, **2 stratégies isolées** (hiérarchie section via pile numérotée + fix acronymes ; liaison caption↔table/image géométrique), orchestration 2-passes (ids stables cohérents), fenêtrage longs docs, limites & axes d'amélioration explicités. **Ajouté** : positionnement 2026, comparatif PyPDF/pdfplumber/unstructured/marker, note OCR/VLM, pont `[[BDD_Vectorielles]]`. Dataset : papier arXiv Docling `2408.09869` (téléchargé une fois). |
| 44 | `Flask_API.md` (Flask + FastAPI + comparatif) | ✅ fait (conforme contrat 5 critères) | TUTO + WIKI + CS | M | 🌐 | **Notebook unique** (décision utilisateur, à renommer) couvrant la prise en main de **Flask** puis **FastAPI** sur la **même API `books`** (3 classiques de l'original préservés), puis comparatif. Original lu intégralement, plan + structure dans `scripts/_sandbox/`, sandbox `.py` exécuté end-to-end (exit 0), `check_format --both` vert (114 cellules / 30 code). **Exécution in-process** via `test_client`/`TestClient` (pas de serveur). **Modernisation 2026** : Pydantic v2, `Query(pattern=)` (pas `regex` déprécié), `lifespan` (pas `on_event`), footgun event-loop, OpenAPI auto. v0 `FastAPI_API.*` absorbé (redondant — à supprimer). |
| 45 | `Streamlit_brique.md` | 📥 ingéré | TUTO | M | | Async, autocomplete, chatbot, viz, DataFrame. Ajouter caching/perf + déploiement (HF Spaces, Railway). |

---

## 📈 Compteurs (HONNÊTES — correction du tableau au-dessus)

> Ce qui était marqué "44/44 ✅" était mensonger au regard du contrat `00_workflow_contract.md`.
> Voici l'état réel :

- **Notebooks ✅ fait au sens du contrat (5 critères vérifiés)** : **9** — EDA_Visualisation_Introduction (#8), EDA_Analyse_Multivarie (#9), EDA_Stats_Analyse_Desc_Visual (#10), Detection_Outliers (#11), DL_KAN_Kolmogorov_Arnold (#26), BDD_Vectorielles (#40), DE_Docling (#43), DL_Keras (#46), DL_TensorFlow (#47).
- **🟡 v0** (squelette posé, code non vérifié end-to-end) : **37** — les autres anciennes "✅ fait".
- Suppressions effectives : 3 (AAA_Test_ML, Suppr_ML_Bench, TS_Maintenance_Prédictive ancien)
- Fusions effectives : 2 (retrieval_BDD_Vectorielle → BDD_Vectorielles, Preprocessing_Function_Utiles → Preprocessing)
- Renommages effectués : 10 (Structure_Python, Structure_BDD_DataFrame, ML_Regression_Classification_Multiple, Structure_Generer_Donnees_Classification, Detection_Outliers, NLP_Classification_Supervisee, DL_KAN_Kolmogorov_Arnold, TdS_Introduction_Traitement_Signal, TS_Maintenance_Predictive, TS_ARIMA)
- Vagues "complétées" au sens livraison v0 : 9/9. Vagues complétées au sens contrat : **0/9**.

## 🆕 Nouveaux notebooks proposés

- **24 plans détaillés** dans `05_nouveaux_notebooks/` couvrant DS / DE / AIE / MLE / MLOps
- Voir [`00_sujets_nouveaux.md`](00_sujets_nouveaux.md) pour la justification
- Voir [`00_critique.md`](00_critique.md) pour les axes d'amélioration des 44 existants

## 🚀 Améliorations implémentées (infra, hors notebooks)

- **`scripts/download_data.sh`** : script de download datasets (NYC Taxi, Turbofan README, PDFs sample)
- **`apps/`** : Flask, FastAPI, Streamlit en fichiers `.py` exécutables (vs pseudo-code dans les notebooks)

## 🛠️ Pour reprendre selon le contrat

1. Choisir UN notebook (idéalement parmi ceux les plus critiques pour ton usage).
2. Appliquer le workflow 11 étapes de `00_workflow_contract.md` en entier sur ce seul notebook.
3. Vérifier les 5 critères. Si ✅, commit + push avec diff content détaillé.
4. Notebook suivant.
5. Aucun batch. Aucun raccourci.

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
