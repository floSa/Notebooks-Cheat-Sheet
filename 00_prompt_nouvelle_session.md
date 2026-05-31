# 🚀 Prompt pour une nouvelle session (1 notebook à la fois)

Copie-colle le bloc ci-dessous dans une **nouvelle conversation**, en remplaçant `<NOM_DU_NOTEBOOK>` par le notebook ciblé.

**Un notebook = une conversation.** Ne pas demander plusieurs notebooks dans la même session (le contexte se charge et la qualité chute — c'est ce qui a foiré les premières fois).

---

## 📋 Prompt à copier

```
Projet : Notebooks_convertion (~/mes_projets/Notebooks_convertion dans WSL ubuntu-24.04).

AVANT TOUT : lis le contrat `00_workflow_contract.md` à la racine du repo + la mémoire
`project_anti_lie_contract.md` (auto-chargée). Ils définissent le workflow obligatoire
en 9 étapes, les 5 critères de "fait", et le vocabulaire de rapporting honnête.

Notebook à refaire cette session : <NOM_DU_NOTEBOOK>

Applique le workflow 9 étapes en entier, sans raccourci :
1. ipynb -> md (si pas déjà dans 02_md_originaux/)
2. Read (PAS Grep) le 02_md_originaux/<NOM>.md EN ENTIER
3. Plan d'amélioration dans scripts/_sandbox/plan_<NOM>.md
   (table sections de l'original : garde/refactore/supprime/ajoute + justif suppressions)
4. Plan détaillé structure du nouveau notebook dans scripts/_sandbox/structure_<NOM>.md (sans code)
5. Code Python dans scripts/_sandbox/notebook_<NOM>.py (toutes les cellules, dans l'ordre)
6. Teste : uv run python scripts/_sandbox/notebook_<NOM>.py jusqu'à exit 0
7. Intègre le code testé dans 03_md_ameliores/<NOM>.md (titre seul, description avant ET après code)
8. Audit : uv run python scripts/check_format.py --both 03_md_ameliores/<NOM>.md 04_notebooks_finaux/<NOM>.ipynb -> doit être [OK] TOUT VERT
9. Convertis md -> ipynb via jupytext

Puis MAJ 00_status_notebooks.md + commit + push (message listant garde/supprime/ajoute).

CONTRAINTES IMPORTANTES :
- TOUTES les commandes uv passent par WSL : wsl -d ubuntu-24.04 -- bash -lc "cd ~/mes_projets/Notebooks_convertion && uv ..."
  (jamais uv Windows, ça casse mon kernel).
- Le sandbox .py doit utiliser matplotlib.use("Agg") + savefig (pas plt.show) pour ne pas
  spammer mon IDE de figures.
- Datasets mutualisés : voir 00_datasets.md (privilégier Titanic / California Housing /
  MNIST / 20News / etc. chargés programmatiquement, pas de fichier Kaggle).
- Recherche web 2026 si le sujet a bougé (LLMs, RAG, vector DBs, MLOps, frameworks).
- Si tu n'arrives pas à finir : dis-le honnêtement avec le vrai statut (🟡 v0 / 🟠 partiel),
  ne marque PAS ✅ fait.
- Avant de committer : montre-moi le plan (étape 3) et la structure (étape 4) pour validation.

Commence par l'étape 1-2 et montre-moi le plan d'amélioration.
```

---

## 📊 Liste des notebooks (état au dernier commit)

### ✅ Fait (conforme contrat — 5 critères)
- `EDA_Visualisation_Introduction`

### 🟡 v0 — à refaire selon le contrat (43)

**NLP** (avaient des smoke tests d'imports, mais pas de vérif end-to-end) :
- `NLP_Transformers`, `NLP_Recherche_d_informations`, `NLP_NER`, `NLP_NER_BiLSTM_CRF`,
  `NLP_Classification_Supervisee`, `NLP_Classification_Smote`

**Vector DB / BDD** :
- `BDD_Vectorielles`, `BDD_DuckDB`, `Structure_BDD_DataFrame`

**MLOps** :
- `ML_MLFlow_Bench`

**Time Series** :
- `TS_Time_Series_Intro`, `TS_Time_Series_Overview`, `TS_ARIMA`, `TS_Generer_Sequence`,
  `TS_Maintenance_Predictive`

**EDA / Stats** :
- `EDA_Analyse_Multivarie`, `EDA_Stats_Analyse_Desc_Visual`, `Detection_Outliers`

**ML core** :
- `ML_Regression_Classification_Multiple`, `ML_Regression_Classification_CV_GridSearch`,
  `ML_Bagging_Boosting`, `ML_Optimisation_de_Modeles`,
  `ML_Explication_Feature_Importance_Selection`, `ML_Apprentissage_par_Renforcement`,
  `Test_donnees_manquante_modeles`, `INRIA_SKLearn_MOOC`

**DL frameworks** (suivre `00_blueprint_DL_frameworks.md`) :
- `DL_Deep_Learning_Maths`, `DL_PyTorch`, `DL_TensorFlow`, `DL_Keras`, `DL_JAX`,
  `DL_Frameworks_Comparatif`, `DL_KAN_Kolmogorov_Arnold`

**Structures / Python** :
- `Structure_Python`, `Structures_L_T_D_Cheat_Sheet`, `Structures_DataFrame`,
  `Structures_Preprocessing`, `Structure_Generer_Donnees_Classification`

**Apps / DE / Signal** :
- `TdS_Introduction_Traitement_Signal`, `DE_Docling`, `Flask_API`, `FastAPI_API`,
  `Streamlit_brique`

---

## 💡 Ordre conseillé

Commence par les notebooks que tu **utilises le plus** ou ceux dont le **format/contenu te déçoit le plus** quand tu les ouvres. Pas besoin de suivre un ordre — chaque notebook est indépendant dans le workflow.

Suggestion si tu veux du concret rapidement : les **EDA / Stats** et **Structures** (peu de dépendances lourdes, exécution rapide, résultat visible vite). Les **DL frameworks** sont les plus longs (téléchargement torch/tf + entraînements).
