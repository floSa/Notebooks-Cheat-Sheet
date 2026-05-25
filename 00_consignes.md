# 📋 Consignes — refonte qualité des notebooks

## Objectif

Mettre à jour et **compléter** un corpus de 45 notebooks personnels couvrant ML / DL / NLP / Time Series / EDA / BDD / TdS / DE.
On est en **2026**, les connaissances d'entraînement du modèle s'arrêtent à fin 2025 → pour tout sujet qui a pu bouger récemment (LLMs, MLOps, vector DBs, agents, frameworks DL/NLP), **vérifier sur internet** avant de proposer une refonte.

---

## Rôles que peut prendre un notebook (cumulables)

Chaque notebook peut tenir 1 à 3 rôles parmi :

- **Cheat-sheet** : recettes courtes et autonomes pour une méthode / fonction d'une lib (snippets prêts à copier).
- **Tutoriel** : comprendre le *quoi*, le *pourquoi*, le *quand* — déroulé pédagogique avec exemples.
- **Wiki technique** : panorama complet d'une techno ou d'un domaine, avec les **maths** et les explications techniques avancées. Ces sections sont vraiment les bienvenues.

Avant de toucher un notebook, identifier dans le plan de refonte quel(s) rôle(s) il joue.

---

## Format imposé du notebook

### Structure globale

```
[Cellule MD] # Titre principal
[Cellule MD] description du notebook (paragraphe)
[Cellule MD] ## Sous-titre
[Cellule MD] description de la section (cellule "description" — voir plus bas)
[Cellule code] ...
[Cellule MD] description / commentaire qui suit
[Cellule code] ...
...
```

### Règles strictes

- **Tous les titres** (`#`, `##`, `###`, …) sont **seuls dans leur cellule MD**. Pas de titre + texte dans la même cellule.
- **Entre une cellule de code et un titre**, il y a **toujours** une cellule MD `description` qui prépare le terrain.

### Implémentation jupytext (très important)

Dans le format `.md` jupytext, **plusieurs blocs markdown consécutifs sont fusionnés en 1 seule cellule** par défaut. Pour que les titres restent seuls dans leur cellule, chaque bloc markdown doit être encadré par des marqueurs de région :

```markdown
<!-- #region -->
# Titre principal
<!-- #endregion -->

<!-- #region -->
Description du notebook.
<!-- #endregion -->

<!-- #region -->
## Section 1
<!-- #endregion -->

<!-- #region -->
Description avant le code.
<!-- #endregion -->

` ` `python
import x
` ` `

<!-- #region -->
Description après le code.
<!-- #endregion -->
```

Test round-trip : `uv run jupytext --to ipynb fichier.md` produit bien 1 cellule par région.
- La cellule `description` peut contenir, au choix selon pertinence :
  - **Ce que fait** le code ci-dessous
  - **Pourquoi / dans quel cas** on l'utilise
  - **Explication technique / mathématique** (formules en LaTeX bienvenues)
  - Tout autre élément utile (pièges, alternatives, complexité, etc.)

### Cellules de code

- **Typage** systématique : annotations sur les entrées et sorties des fonctions.
- **Docstrings** : courts mais clairs (rôle, params, retour).
- **Commentaires** : uniquement quand le *pourquoi* n'est pas évident — pas de paraphrase du code.
- **Snippets réutilisables** : les cellules doivent pouvoir être copiées dans un autre projet sans avoir à réécrire le contexte.

---

## Esprit de la refonte

- **Sois critique.** Si un notebook a un trou (section manquante, méthode obsolète, manque les maths), **ajoute** la section.
- **Augmenter > préserver.** L'objectif n'est pas de garder l'existant intact mais de produire un document de référence personnel solide.
- **Maths bienvenues** sur les notebooks Wiki — pas de pudeur sur les formules.
- **Bonnes pratiques** modernes (2026) sur le code : typage, pyproject.toml, structure claire, pas de cellules-bazar.

---

## Environnement d'exécution

- **UV** pour la gestion d'environnement (un seul env partagé pour tous les notebooks).
- `pyproject.toml` à la racine, dépendances ajoutées au fil des notebooks traités.
- Le code doit être **testable** : avant de valider une cellule dans le `.md` final, possibilité de générer un `.py` temporaire et l'exécuter dans l'env UV pour vérifier que ça tourne.

---

## Données

- Dossier `data/` à la racine.
- Un sous-dossier `data/<nom_notebook>/` par notebook qui a besoin de datasets.
- Les chemins dans les notebooks pointent vers ces sous-dossiers (chemins relatifs).

---

## Workflow technique imposé

- **Jamais** d'édition directe d'un `.ipynb`. Les `.ipynb` sont traités via **jupytext** uniquement.
- Le format de travail est le `.md` jupytext (avec fences ` ```python ` pour les cellules de code).
- Round-trip imposé :
  ```
  .ipynb  --jupytext-->  .md  (édition / refonte)  --jupytext-->  .ipynb
  ```
- Conversion en batch via scripts dédiés (voir `scripts/`).

---

## Livrables attendus

À la fin du projet :

1. Le dossier `04_notebooks_finaux/` contient **45 `.ipynb`** refondus (ou moins si certains sont fusionnés / supprimés — à justifier dans le tableau de suivi).
2. Le fichier `00_status_notebooks.md` est à jour avec l'état de chaque notebook (à faire / en cours / fait / fusionné / supprimé).
3. Le dossier `data/` contient les datasets nécessaires, organisés par notebook.
4. L'env UV (`pyproject.toml` + `uv.lock`) permet de relancer n'importe quel notebook.
