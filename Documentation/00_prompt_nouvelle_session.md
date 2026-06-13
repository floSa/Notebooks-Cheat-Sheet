# 🚀 Prompt pour une nouvelle session (1 notebook à la fois)

Copie-colle le bloc ci-dessous dans une **nouvelle conversation**, en remplaçant `<NOM_DU_NOTEBOOK>` par le notebook ciblé.

**Un notebook = une conversation.** Ne pas demander plusieurs notebooks dans la même session (le contexte se charge et la qualité chute — c'est ce qui a foiré les premières fois).

---

## 📋 Prompt à copier

```
Projet : Notebooks-Cheat-Sheet (~/Projets/Notebooks/Notebooks-Cheat-Sheet dans WSL ubuntu-24.04).

AVANT TOUT : lis le contrat `Documentation/00_workflow_contract.md` (section 0 = RÈGLE D'OR).
Il définit le workflow obligatoire en 9 étapes, les 5 critères de "fait", le rapporting honnête.

🎯 RÈGLE D'OR :
- Si le notebook a un ORIGINAL (dans Notebook_2018-2021/ipynb/) → je me base PRINCIPALEMENT
  sur MON original. Jamais sur `main` ni 3_Sessions_Ratees/ (versions gutées à jeter).
- Si c'est un SUJET NEUF (un des 24 : DS_/DE_/AI_/MLE_/MLOps_, pas d'original) → je pars du
  plan (Notebook_2026/plans/) et je m'inspire du refait existant (Notebook_2026/ipynb/).
- Notebook_2026/ n'est une base QUE pour les 24 sujets neufs, jamais pour un notebook qui a un original.

D'abord : bash scripts/restore_originaux.sh (régénère Notebook_2018-2021/ipynb/ depuis le zip).

Notebook à refaire cette session : <NOM_DU_NOTEBOOK>

Applique le workflow 9 étapes en entier, sans raccourci :
1. S'assurer que Notebook_2018-2021/ipynb/ existe (sinon restore_originaux.sh)
2. Read (PAS Grep) le Notebook_2018-2021/md/<NOM>.md EN ENTIER (le VRAI original)
   — OU, si sujet neuf : Notebook_2026/plans/<NOM>.md + Notebook_2026/ipynb/<NOM>.ipynb
3. Plan d'amélioration dans scripts/_sandbox/plan_<NOM>.md
   (table sections de l'original : garde/refactore/supprime/ajoute + justif suppressions)
4. Plan détaillé structure du nouveau notebook dans scripts/_sandbox/structure_<NOM>.md (sans code)
5. Code Python dans scripts/_sandbox/notebook_<NOM>.py (toutes les cellules, dans l'ordre)
6. Teste : uv run python scripts/_sandbox/notebook_<NOM>.py jusqu'à exit 0
7. Intègre le code testé dans Notebook_2026/md/<NOM>.md (titre seul, description avant ET après code)
8. Audit : uv run python scripts/check_format.py --both Notebook_2026/md/<NOM>.md Notebook_2026/ipynb/<NOM>.ipynb -> doit être [OK] TOUT VERT
9. Convertis md -> ipynb via jupytext

Puis MAJ 00_status_notebooks.md + commit + push (message listant garde/supprime/ajoute).

CONTRAINTES IMPORTANTES :
- TOUTES les commandes uv passent par WSL : wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv ..."
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

## 📊 Liste des notebooks et état

➡️ **L'état complet, à jour, par notebook est dans [`00_FAIT_A_FAIRE.md`](00_FAIT_A_FAIRE.md).**
Il indique pour chacun : a-t-il un original, sur quelle base le refait a été fait
(cf. `00_VERIFICATION_BASE.md`), et l'action à mener.

En résumé : **aucun** notebook n'est "fait" au sens de la règle d'or. Presque tous les refait
ont été bâtis sur des gutés → à refaire sur le vrai original. Les 24 sujets neufs sont à
implémenter depuis leur plan.

## 💡 Ordre conseillé

Commence par les notebooks que tu **utilises le plus**. Les **EDA / Stats** et **Structures**
sont rapides (peu de dépendances). Les **DL frameworks** sont les plus longs (torch/tf + entraînements).
