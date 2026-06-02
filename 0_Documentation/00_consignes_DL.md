# 🧠 Consignes — générer les 3 notebooks frameworks DL

> **Usage** : ouvrir **3 sessions séparées** (une par framework). Dans chacune, copier le prompt
> ci-dessous en remplaçant `<FRAMEWORK>` par **PyTorch**, **TensorFlow** ou **Keras**.
> Un framework = une session. Pas les 3 dans la même session.

---

## Contexte (commun aux 3)

Les 3 notebooks `DL_PyTorch`, `DL_TensorFlow`, `DL_Keras` doivent être générés à partir d'un
**MIX de mes 2 originaux** :
- `1_Old_Notebooks/ipynb/DL_PyTorch.ipynb` (143 cellules) → fournit le **squelette** (intro,
  tenseurs, GPU, XOR, Dataset/DataLoader, loss, train, eval, save/load, régression, classif, ROC/AUC).
- `1_Old_Notebooks/ipynb/DL_Tensorflow_Keras.ipynb` (112 cellules) → fournit les **techniques
  avancées** (batch manuel équilibré/trié, `class_weight`, `sample_weight`, SHAP DeepExplainer).

Le squelette commun imposé (16 sections) est dans `0_Documentation/00_blueprint_DL_frameworks.md`.

---

## 📋 Prompt à copier (remplacer `<FRAMEWORK>`)

```
Projet : Notebooks-Cheat-Sheet (~/Projets/Notebooks/Notebooks-Cheat-Sheet, WSL ubuntu-24.04).
Branche : feat/restart-jupytext-workflow (PAS main).

OBJECTIF : générer LE notebook DL pour le framework <FRAMEWORK> uniquement.

AVANT TOUT :
- bash scripts/restore_originaux.sh   (régénère mes vrais originaux dans 1_Old_Notebooks/ipynb/)
- Lis 0_Documentation/00_workflow_contract.md (§0 RÈGLE D'OR) et 00_blueprint_DL_frameworks.md.

BASE OBLIGATOIRE (mes vrais originaux, JAMAIS main ni 3_Sessions_Ratees) :
- Read EN ENTIER 1_Old_Notebooks/md/DL_PyTorch.md          → squelette
- Read EN ENTIER 1_Old_Notebooks/md/DL_Tensorflow_Keras.md → techniques avancées
Le notebook <FRAMEWORK> = ces 2 originaux fusionnés, exprimés dans les idiomes de <FRAMEWORK>,
en suivant les 16 sections du blueprint (mêmes sections que les autres frameworks pour comparaison).

EXIGENCES DE CONTENU (doivent figurer dans le notebook <FRAMEWORK>) :
- Les 16 sections du blueprint, dans l'ordre.
- Section synthétique XOR continu (même fonction de génération que les autres frameworks).
- Cas régression (California Housing) + cas classification (MNIST sous-échantillonné).
- Techniques avancées venues de mon original TF/Keras : batch équilibré/trié, class_weight,
  sample_weight, focal loss, SHAP — adaptées à <FRAMEWORK>.
- ROC/AUC macro OVR, save/load, early stopping, BatchNorm, dropout, weight decay, LR scheduling.
- Datasets chargés programmatiquement (00_datasets.md), pas de fichier Kaggle.

WORKFLOW (zéro raccourci, cf. contrat) :
1. Plan d'amélioration scripts/_sandbox/plan_DL_<FRAMEWORK>.md (table sections des 2 originaux :
   garde/refactore/supprime/ajoute + justif).
2. Structure scripts/_sandbox/structure_DL_<FRAMEWORK>.md (16 sections, sans code).
3. Code scripts/_sandbox/notebook_DL_<FRAMEWORK>.py (toutes les cellules, dans l'ordre),
   matplotlib Agg + savefig.
4. Teste : uv run python scripts/_sandbox/notebook_DL_<FRAMEWORK>.py → exit 0 + résultats cohérents.
5. Intègre dans 2_New_Notebooks/md/DL_<FRAMEWORK>.md (titre seul, description avant ET après code).
6. Audit : uv run python scripts/check_format.py --both 2_New_Notebooks/md/DL_<FRAMEWORK>.md 2_New_Notebooks/ipynb/DL_<FRAMEWORK>.ipynb → [OK] TOUT VERT.
7. Convertis md → ipynb (jupytext).

CONTRAINTES :
- Toutes les commandes uv via WSL : wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv ..."
- Si tu ne finis pas : statut honnête (🟡 v0 / 🟠 partiel), ne marque PAS ✅ fait.
- Montre-moi le plan (étape 1) et la structure (étape 2) avant de coder.
- Commit + push sur feat/restart-jupytext-workflow (SSH github.com-perso).

Commence par restore_originaux.sh + lecture des 2 originaux, puis montre-moi le plan.
```

---

## Vérifier après coup

Les 3 notebooks doivent avoir les 16 sections et contenir `sample_weight` (preuve qu'ils
viennent de mon original TF/Keras, pas du vidé qui ne l'avait pas).
