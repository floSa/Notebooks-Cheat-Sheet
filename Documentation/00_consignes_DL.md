# 🧠 CONSIGNES — les 3 notebooks frameworks DL (en UNE conversation)

> Les 3 frameworks (**PyTorch**, **TensorFlow**, **Keras**) sont générés dans **une seule
> conversation**, l'un après l'autre (budget ~1M tokens). On NE PART PAS de zéro : on **fusionne
> mes 2 originaux** et on **améliore** (cf. `00_consignes_refonte.md`).

---

## 0. Où on travaille + git (portable ↔ fixe)

- `~/Projets/Notebooks/Notebooks-Cheat-Sheet` (**WSL ubuntu-24.04**), branche `feat/restart-jupytext-workflow` (jamais `main`).
- Toutes les commandes `uv` via WSL : `wsl -d ubuntu-24.04 -- bash -lc "cd ~/Projets/Notebooks/Notebooks-Cheat-Sheet && uv ..."`.
- **Au début** : `git checkout feat/restart-jupytext-workflow` → `git pull` → `bash scripts/restore_originaux.sh`.
- **À la fin** (après les 3) : `git add` + `commit` + `git push` (SSH `github.com-perso`, ma clé perso).
- Une seule machine à la fois sur ces notebooks.

## 1. La base — fusion de mes 2 vrais originaux

Les 3 notebooks sont le **MIX** de :
- `Notebook_2018-2021/ipynb/DL_PyTorch.ipynb` (143 cellules) → **squelette** (intro, tenseurs, GPU,
  XOR, Dataset/DataLoader, loss, train, eval, save/load, régression, classification, ROC/AUC).
- `Notebook_2018-2021/ipynb/DL_Tensorflow_Keras.ipynb` (112 cellules) → **techniques avancées**
  (batch manuel équilibré/trié, `class_weight`, `sample_weight`, SHAP DeepExplainer).

Squelette commun imposé (16 sections) : `Documentation/00_blueprint_DL_frameworks.md`.
S'il existe déjà une refonte dans `Notebook_2026/ipynb/DL_<Framework>.ipynb`, **s'appuyer dessus**
(modernisation 2026 déjà faite) sans repartir de zéro, mais réintégrer ce qu'elle aurait perdu.

## 2. Ce qu'on fait pour CHAQUE framework

- Les **16 sections du blueprint**, dans l'ordre, en idiomes du framework.
- **Mise à jour 2026** + **plus de commentaires / formules (LaTeX) / détails** (cf. `00_consignes_refonte.md`).
- Techniques avancées de mon original TF/Keras : batch équilibré/trié, `class_weight`,
  `sample_weight`, focal loss, SHAP — adaptées au framework.
- XOR continu (même générateur partout), régression California Housing, classification MNIST
  sous-échantillonné, ROC/AUC macro OVR, save/load, early stopping, BatchNorm, dropout, LR scheduling.
- Datasets chargés programmatiquement (`00_datasets.md`), seed=42.
- ❌ Ne supprime aucun de mes graphiques / images ; contenu ≥ originaux.

## 3. Ordre dans la conversation (les 3 à la suite)

Pour CHAQUE framework, dans l'ordre **PyTorch → TensorFlow → Keras** :

1. Plan `scripts/_sandbox/plan_DL_<Framework>.md` (table des sections des 2 originaux : garde/améliore/ajoute).
2. Structure `scripts/_sandbox/structure_DL_<Framework>.md` (16 sections, sans code).
3. Code `scripts/_sandbox/notebook_DL_<Framework>.py` (toutes les cellules, dans l'ordre), matplotlib Agg + savefig.
4. Test : `uv run python scripts/_sandbox/notebook_DL_<Framework>.py` → exit 0 + résultats cohérents.
5. Intègre dans `Notebook_2026/md/DL_<Framework>.md` (titre seul, description avant ET après code).
6. Audit : `uv run python scripts/check_format.py --both Notebook_2026/md/DL_<Framework>.md Notebook_2026/ipynb/DL_<Framework>.ipynb` → **[OK] TOUT VERT**.
7. Convertir md → ipynb (jupytext).

Puis passer au framework suivant. **Commit + push** à la fin des 3 (ou après chacun, au choix).

## 4. Vérif finale

Les 3 notebooks doivent avoir les **16 sections** et contenir **`sample_weight`** (preuve que ça
vient de mon vrai original TF/Keras, pas du vidé qui ne l'avait pas). check_format vert pour les 3.

---

## ▶️ Mini-prompt (à copier dans UNE conversation)

```
Les consignes à suivre scrupuleusement sont dans ce document : Documentation/00_consignes_DL.md (lis-le en entier d'abord).
Génère les 3 notebooks frameworks DL — PyTorch, TensorFlow, Keras — dans cette conversation,
l'un après l'autre, par fusion de mes 2 originaux (DL_PyTorch + DL_Tensorflow_Keras).
tu valides les plans directement et tu finalises tout, puis commit + push.
```
