# 📘 CONSIGNES DE REFONTE — à suivre scrupuleusement

> Document auto-suffisant. Une session qui le suit produit exactement ce que je veux.
> Lis-le EN ENTIER avant de commencer.

---

## 0. Où on travaille

- Projet : `~/Projets/Notebooks/Notebooks-Cheat-Sheet` (WSL ubuntu-24.04).
- Branche : `feat/restart-jupytext-workflow` (**jamais** `main`).
- Toutes les commandes `uv` passent par WSL :
  `wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv ..."`
- D'abord : `bash scripts/restore_originaux.sh` (régénère mes vrais originaux).

## 1bis. Les toutes premières actions (lecture)

1. `bash scripts/restore_originaux.sh` = **dézippe mes originaux** (`1_Old_Notebooks/Notebooks.zip`
   → `1_Old_Notebooks/ipynb/`). Ce ne sont PAS dans git directement (trop lourds), d'où le dézip.
2. **`ipynb → md`** : la version markdown lisible de l'original existe déjà dans
   `1_Old_Notebooks/md/<NOM>.md` (générée par **jupytext**). Si elle manque :
   `uv run jupytext --to md 1_Old_Notebooks/ipynb/<NOM>.ipynb`.
   (jupytext récupère TOUTES les cellules — markdown ET code.)
3. **Étudier** ce `.md` original en entier.
4. **Pareil pour la refonte déjà faite** si elle existe : lire `2_New_Notebooks/md/<NOM>.md`.
5. **Réfléchir au plan en repartant de l'existant** (cf. §1 et §2).

## 1. Les sources — dans cet ordre

**(A) BASE PRINCIPALE = mon vrai original** : `1_Old_Notebooks/ipynb/<NOM>.ipynb`
   (version lisible : `1_Old_Notebooks/md/<NOM>.md`).
   - **Lis-le EN ENTIER** (`Read`, pas `Grep`) avant toute chose.
   - C'est lui qui fixe le contenu de référence, mes graphiques et mes images à préserver.

**(B) SOURCE SECONDAIRE = la refonte déjà faite**, SI elle existe :
   `2_New_Notebooks/ipynb/<NOM>.ipynb` (+ `2_New_Notebooks/md/<NOM>.md`).
   - Du vrai travail y a souvent déjà été fait (modernisation 2026, code testé end-to-end).
   - **Réutilise ce qui est bon** (mises à jour 2026, code qui tourne, nouvelles sections)
     plutôt que de repartir de zéro.
   - ⚠️ Mais elle a pu **perdre des graphiques/sections** de mon original → c'est (A) qui
     tranche en cas de manque : on **réintègre** depuis l'original ce que (B) a oublié.

**Résultat = (A) pour le fond et mes graphiques + (B) pour la modernisation déjà faite.**

❌ **JAMAIS** partir de `main` ni de `3_Sessions_Ratees/` : ce sont des versions VIDÉES
   (mes notebooks réduits de ~200 cellules à ~20 par une mauvaise session).

## 2. Ce que je veux que tu fasses

1. **Mettre à jour pour 2026** : libs, API, bonnes pratiques actuelles (recherche web si le
   sujet a bougé : LLMs, RAG, vector DBs, MLOps, frameworks ML).
2. **Enrichir la pédagogie** : PLUS de commentaires, PLUS de formules (en LaTeX), PLUS de détails.
3. **Augmenter** : ajouter des concepts qui **complètent** mon original (sections nouvelles bienvenues).
4. **Réorganiser** si ça améliore la lecture (changer l'ordre, regrouper).

## 3. Ce que tu ne dois JAMAIS faire

- ❌ Supprimer mes **graphiques / visualisations** (mes plots, mes schémas).
- ❌ Supprimer mes **images**.
- ❌ Réduire le contenu : on **complète**, on ne **retire** pas. Le résultat doit contenir
  **AU MOINS tout** ce qu'il y avait dans mon original, **plus** les améliorations.
- ❌ Mettre du pseudo-code / du code non exécuté / du code commenté à la place du vrai code.
- Si tu penses devoir retirer quelque chose : **demande-moi avant**, justifie.

## 4. Qualité technique exigée

- Le notebook doit **s'exécuter de bout en bout** (test sandbox `.py` → exit 0), cf.
  `00_workflow_contract.md` (workflow 9 étapes + 5 critères + `check_format.py`).
- Code refactoré en fonctions typées + docstrings.
- Datasets chargés programmatiquement (`00_datasets.md`), reproductibles (seed=42).
- Figures via `matplotlib Agg` + `savefig` dans le sandbox (pas de `plt.show`).

## 5. Où mettre le résultat

- `2_New_Notebooks/md/<NOM>.md` puis conversion → `2_New_Notebooks/ipynb/<NOM>.ipynb`.
- Commit + push sur `feat/restart-jupytext-workflow` (SSH `github.com-perso`).

## 6. Avant de coder

**Montre-moi le plan** (table des sections de l'original : gardé / amélioré / ajouté, et où
sont préservés mes graphiques/images) et **attends ma validation**.

## 7. Si tu n'arrives pas à finir

Statut honnête (🟡 v0 / 🟠 partiel). Tu ne dis JAMAIS « fait » si les 5 critères du contrat
ne sont pas vérifiés. Tu listes ce qui manque.

## Cas particulier — frameworks DL

`DL_PyTorch` / `DL_TensorFlow` / `DL_Keras` : voir `00_consignes_DL.md`
(base = mix `DL_PyTorch.ipynb` + `DL_Tensorflow_Keras.ipynb`, blueprint 16 sections).

---

## ▶️ Comment me solliciter (dans une nouvelle conversation)

J'écris seulement ces 2 lignes :

```
Les consignes à suivre scrupuleusement sont dans ce document : 0_Documentation/00_consignes_refonte.md (lis-le en entier d'abord).
Voici le notebook à traiter : <NOM_DU_NOTEBOOK>
```
