# 🧩 Blueprint commun — Notebooks frameworks DL

Squelette pédagogique partagé par les notebooks `DL_PyTorch.md`, `DL_TensorFlow.md`, `DL_Keras.md`, `DL_JAX.md`.
Objectif : pouvoir lire les 4 notebooks côte-à-côte et comparer les idiomes section par section.

Le notebook `DL_Frameworks_Comparatif.md` rejoue les sections 14-15 sur **les mêmes datasets** avec tous les frameworks pour benchmark.

---

## 🧬 Origine du blueprint

Union des sections présentes dans les notebooks existants (Phase 2 audit) :

- **De `DL_PyTorch.md`** : architecture claire (intro → tenseurs → GPU → XOR → Dataset/DataLoader → loss → train → eval → save/load → TensorBoard → cas régression → cas classification → ROC/AUC)
- **De `DL_Tensorflow_Keras.md`** : pratiques avancées d'entraînement (batch manuel équilibré/trié, class weights, sample weights, SHAP DeepExplainer)

Le blueprint prend le **squelette PyTorch** et y greffe les **techniques avancées TF**.

---

## 🦴 Squelette imposé (16 sections)

Chaque section est une `##` dans le notebook. Toutes les cellules `description` (entre code et titre) restent obligatoires comme défini dans `00_consignes.md`.

| # | Section | Pourquoi | Notes par framework |
|---|---|---|---|
| 1 | **Présentation du framework** | Philosophie, écosystème, état 2026, quand le choisir | TF: production/TPU · PyTorch: recherche/flexibilité · Keras 3: multi-backend · JAX: fonctionnel/perf |
| 2 | **Tenseurs** | Création, shape, opérations basiques, conversion ↔ numpy | `torch.Tensor` / `tf.Tensor` / `keras.ops` / `jnp.array` |
| 3 | **GPU / accélérateur** | Détection device, CPU vs GPU timing, reproductibilité | `cuda` / `tf.device` / `jax.devices()` ; seed par framework |
| 4 | **Cas synthétique simple** | XOR continu — pédagogique, fait toucher tous les rouages sans data lourd | Identique partout (même fonction de génération) |
| 5 | **Définition du modèle** | Construction d'une ANN simple, vocabulaire framework | `nn.Module` / `tf.keras.Sequential` & `Model` / `keras.Model` / `flax.linen.Module` |
| 6 | **Données** | Dataset, DataLoader / pipeline `tf.data`, transforms | `Dataset+DataLoader` / `tf.data.Dataset` / `keras.utils.Sequence` ou `tf.data` / `grain` |
| 7 | **Loss & Optimisation** | CE/MSE, SGD/Adam, scheduling de LR | `torch.optim` / `tf.keras.optimizers` / `keras.optimizers` / `optax` |
| 8 | **Boucle d'entraînement standard** | Version idiomatique courte | `for epoch...` / `model.fit` / `model.fit` / `train_step` jitted |
| 9 | **Boucle manuelle + gestion batch** | Batches équilibrés / triés / variantes (du TF original) | Custom train loop · `tf.GradientTape` · custom loop Keras · `jax.grad + optax.apply_updates` |
| 10 | **Save / Load** | Persistance modèle + reproductibilité | `torch.save/load` · `model.save('.keras')` · idem Keras · `orbax-checkpoint` |
| 11 | **Évaluation** | Accuracy / loss sur test set, gestion `drop_last` | Boucle eval · `model.evaluate` · idem · custom |
| 12 | **Gestion du déséquilibre** | Class weights, sample weights, focal loss | `WeightedRandomSampler` · `class_weight` arg · idem Keras · custom loss |
| 13 | **Régularisation** | Dropout, weight decay, early stopping, batch norm | `nn.Dropout` / `keras.layers.Dropout` / idem / `flax.linen.Dropout` |
| 14 | **Visualisation / Logging** | Courbes loss/acc, TensorBoard ou équivalent | `torch.utils.tensorboard` · `tf.summary` · `keras.callbacks.TensorBoard` · `aim` ou `wandb` |
| 15 | **Cas réel — Classification (MNIST)** | Pipeline complet : data → modèle → train → conf matrix → ROC/AUC | **Même dataset** dans les 4 notebooks |
| 16 | **Cas réel — Régression (California Housing)** | Pipeline complet : early stopping, dropout, métriques régression | **Même dataset** dans les 4 notebooks |
| 17 | **Explainability (SHAP)** | DeepExplainer / KernelExplainer / GradientExplainer | Le mode SHAP dépend du framework (impl native vs wrapper) |

---

## 🧪 JAX : peut-il rentrer ?

**Réponse courte : oui, via Flax (haut niveau).**

| Section | JAX faisable ? | Notes |
|---|---|---|
| 1-4 | ✅ | Direct |
| 5 (modèle) | ✅ avec **Flax** | Sans Flax, beaucoup de boilerplate — donc Flax obligatoire pour aligner avec les autres |
| 6 (données) | ✅ avec **Grain** ou conversion depuis `tf.data` / `torch.DataLoader` | JAX n'a pas de DataLoader natif |
| 7 (optim) | ✅ via **Optax** | Standard |
| 8-9 (train loop) | ✅ | Toujours custom (pas de `.fit`) — c'est l'idiome JAX |
| 10 (save/load) | ✅ via **Orbax** | Standard |
| 11 (eval) | ✅ | Custom |
| 12 (déséquilibre) | ✅ | Custom loss / pondération manuelle |
| 13 (régul) | ✅ | Via Flax modules |
| 14 (logging) | ✅ via `aim` ou `wandb` (pas de TensorBoard natif idéal) | À démontrer |
| 15-16 (cas réels) | ✅ | OK |
| 17 (SHAP) | ⚠️ partiel | SHAP est numpy/torch/TF-centric — soit on wrappe via `jax2tf`, soit on remplace par **Integrated Gradients via Captum-like** (`shap.KernelExplainer` marche sur n'importe quelle fonction Python pure) |

**Décision : on intègre JAX dans le blueprint complet**, en notant les écarts (pas de `.fit`, SHAP plus contraint). Le notebook sera un peu plus dense parce que JAX = explicite par design.

---

## 📊 Notebook comparatif (`DL_Frameworks_Comparatif.md`)

Refait uniquement les sections **15 (MNIST classif)** et **16 (California Housing regr)** en 4 versions parallèles, puis :

- Tableau **lignes de code** par framework, section par section
- Tableau **temps d'entraînement** (mêmes hyperparamètres : epochs, batch, optim, lr)
- Tableau **mémoire GPU pic** (`nvidia-smi` ou hooks)
- Tableau **accuracy / RMSE finales**
- Discussion **quand choisir quoi** (production / recherche / portabilité / TPU)

---

## 📦 Datasets utilisés (cf. `00_datasets.md`)

- **MNIST** → `sklearn.datasets.fetch_openml('mnist_784')` ou loader natif par framework (`torchvision.datasets.MNIST`, `tf.keras.datasets.mnist`, etc.) — on documente les 4 chemins de chargement
- **California Housing** → `sklearn.datasets.fetch_california_housing()` (utilisé pour le cas régression, identique partout)
- **XOR continu** → fonction de génération inline, identique partout (≈ 10 lignes)
