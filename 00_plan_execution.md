# 🗺️ Plan d'exécution

Ce document décrit, étape par étape, ce qui va être fait. Aucune étape ne démarre tant que les documents (`00_consignes.md`, ce plan, `00_status_notebooks.md`) ne sont pas validés.

---

## Phase 0 — Préparation (documents)

- [x] `00_consignes.md` — directives de refonte
- [x] `00_plan_execution.md` — ce document
- [x] `00_status_notebooks.md` — tableau de suivi initial (45 entrées)

**Sortie de phase :** validation utilisateur des 3 documents avant de lancer Phase 1.

---

## Phase 1 — Infrastructure technique

### 1.1 Env UV

- Init du projet : `uv init` à la racine, Python 3.12.
- `pyproject.toml` initial avec dépendances de base :
  - `jupytext` (conversion)
  - `jupyter`, `ipykernel` (exécution)
  - Stack ML/Data de base : `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
  - Les dépendances spécifiques (torch, transformers, mlflow, duckdb, etc.) sont ajoutées **au fil des notebooks** quand on les rencontre, pas en bloc au début.
- Enregistrement du kernel : `uv run python -m ipykernel install --user --name=notebooks-refonte`.

### 1.2 Scripts de conversion (idempotents)

Dans `scripts/` :

- `01_unzip.sh` — dézippe `Notebooks_ok/Notebooks.zip` vers `01_notebooks_originaux/`.
- `02_ipynb_to_md.sh` — convertit tout `01_notebooks_originaux/*.ipynb` vers `02_md_originaux/*.md` via `jupytext --to md`.
- `04_md_to_ipynb.sh` — convertit tout `03_md_ameliores/*.md` vers `04_notebooks_finaux/*.ipynb` via `jupytext --to ipynb`.
- `99_test_cell.sh <fichier.py>` — exécute un `.py` temporaire dans l'env UV pour tester un bloc de code avant de l'intégrer dans le `.md`.

Note : on n'écrit **aucun** script qui édite directement du `.ipynb`. Le seul outil d'I/O `.ipynb` est jupytext.

---

## Phase 2 — Ingestion

- Exécution de `01_unzip.sh` → 45 fichiers dans `01_notebooks_originaux/`.
- Vérification : nom étrange `DL_PyTorch` (sans extension `.ipynb`) dans la liste du zip → inspecter (probablement un dossier).
- Exécution de `02_ipynb_to_md.sh` → 45 `.md` dans `02_md_originaux/`.
- Mise à jour de `00_status_notebooks.md` : passage à `📥 ingéré` pour chaque notebook converti avec succès, `⚠️ erreur conversion` sinon.

**Sortie de phase :** tous les notebooks lisibles en `.md`.

---

## Phase 3 — Audit & priorisation

Avant de refondre, je lis chaque `.md` original et je remplis dans le tableau de suivi :

- Rôle(s) du notebook (Cheat-sheet / Tutoriel / Wiki — combinables).
- État estimé (vide / brouillon / abouti).
- Sujets potentiellement obsolètes nécessitant recherche web 2026 (LLMs, vector DBs, MLOps, agents, etc.).
- Sections manquantes évidentes.
- Estimation d'effort (S / M / L).

**Sortie de phase :** `00_status_notebooks.md` enrichi, ordre de traitement décidé (probablement par domaine : on traite tous les NLP ensemble, puis tous les TS, etc., pour mutualiser les recherches).

---

## Phase 4 — Refonte (boucle par notebook)

Pour **chaque** notebook (dans l'ordre décidé en Phase 3) :

1. **Lire** `02_md_originaux/<notebook>.md` en entier.
2. **Recherches web** si le sujet est sensible à l'évolution 2025→2026.
3. **Brouiller** un plan de refonte dans la tête (ou en commentaire en haut du nouveau `.md`) :
   - Rôle(s) du notebook
   - Sections à garder / refondre / ajouter
   - Maths à intégrer
4. **Écrire** `03_md_ameliores/<notebook>.md` en suivant strictement le format du `00_consignes.md`.
5. **Tester** les cellules de code critiques :
   - Extraire dans un `.py` temporaire (`scripts/_sandbox/`)
   - Lancer dans l'env UV
   - Corriger jusqu'à ce que ça tourne
6. **Datasets** : si le notebook a besoin de données, créer `data/<notebook>/` et y placer ce qu'il faut (ou un `README.md` qui indique comment les récupérer si trop volumineux).
7. **Convertir** vers `.ipynb` final : exécution ciblée de `04_md_to_ipynb.sh` sur ce fichier.
8. **Mettre à jour** `00_status_notebooks.md` : passage à `✅ fait` (ou `🚧 partiel` / `❌ bloqué` avec note).

---

## Phase 5 — Conversion finale & vérifications

- Exécution finale de `04_md_to_ipynb.sh` sur tout `03_md_ameliores/`.
- Inspection : ouvrir 2-3 notebooks aléatoires dans Jupyter pour vérifier le rendu.
- Vérification round-trip : `04_notebooks_finaux/<x>.ipynb` → re-convertir en `.md` → diff avec `03_md_ameliores/<x>.md` (doit être nul ou cosmétique).

---

## Règles transverses

- **Pas de raccourci `.ipynb`.** Jamais d'édition directe du JSON `.ipynb`. Si une manip semble nécessiter ça, c'est qu'il faut passer par jupytext.
- **Recherche web obligatoire** sur les sujets fast-moving avant refonte (citer les sources en bas du notebook).
- **Mises à jour atomiques du `00_status_notebooks.md`** dès qu'un notebook change d'état (pas de batch à la fin).
- **Git** (optionnel à confirmer) : si on init un repo, respecter la règle perso "pas de mention d'outil IA" dans les commits.
