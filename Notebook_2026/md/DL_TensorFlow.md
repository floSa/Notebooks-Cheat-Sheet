---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python (notebooks-refonte)
    language: python
    name: notebooks-refonte
---

<!-- #region -->
# Deep Learning avec TensorFlow (bas niveau)
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Wiki** sur **TensorFlow vu par en dessous**. L'objectif n'est pas de réutiliser `model.compile()` / `model.fit()` — ça, c'est le rôle du notebook **Keras 3** (haut niveau, multi-backend). Ici on ouvre le capot et on écrit nous-mêmes la mécanique que `fit` cache :

- **`tf.GradientTape`** — la différentiation automatique explicite ;
- **`tf.data.Dataset`** — le pipeline de données performant ;
- **`@tf.function`** — la compilation en graphe (AutoGraph + XLA) ;
- une **boucle d'entraînement** écrite à la main (`GradientTape` → `apply_gradients`) ;
- une **métrique custom** (`tf.keras.metrics.Metric`), une **focal loss** maison, un **checkpoint** bas niveau, et l'**explicabilité** via SHAP.

Ce notebook fait partie d'une série de trois sur le même squelette, à lire côte à côte :

| Notebook | Angle |
|---|---|
| `DL_PyTorch` | recherche, *define-by-run*, `nn.Module` |
| **`DL_TensorFlow`** (ici) | **bas niveau** : `GradientTape`, `tf.data`, `@tf.function`, production |
| `DL_Keras` | haut niveau, `keras.ops`, multi-backend, portabilité |

Fil rouge pédagogique : un **XOR continu** (jeu jouet) pour toucher tous les rouages, puis deux **cas réels** (classification MNIST, régression California Housing) et une section **explicabilité**.
<!-- #endregion -->

<!-- #region -->
## 0. Setup
<!-- #endregion -->

<!-- #region -->
Imports, configuration du déterminisme et graine globale. `enable_op_determinism()` + `set_random_seed()` rendent les exécutions **reproductibles** (utile pour un support pédagogique) ; la graine doit être posée **avant** toute opération aléatoire. On est sur **TensorFlow 2.21** où `tf.keras` est **Keras 3**.
<!-- #endregion -->

```python
import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from sklearn.datasets import fetch_california_housing, make_classification
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.experimental.enable_op_determinism()  # résultats reproductibles run-à-run
tf.keras.utils.set_random_seed(42)  # seed requis AVANT toute op aléatoire (déterminisme)
print(f"tensorflow {tf.__version__} | keras {tf.keras.__version__}")
```

<!-- #region -->
## 1. Présentation de TensorFlow (2026)
<!-- #endregion -->

<!-- #region -->
**Philosophie.** TensorFlow a été conçu autour du **graphe de calcul** : on décrit les opérations, puis on les exécute (et on les optimise) sous forme de graphe. Depuis la 2.x, le mode **eager** (impératif, comme NumPy) est par défaut, mais le graphe reste accessible via `@tf.function` — c'est là que TF tire sa force en **production** : un graphe se sérialise (`SavedModel`), se sert (**TF Serving**), s'embarque (**TFLite**, **TF.js**), et se compile (**XLA**) pour CPU/GPU/**TPU**.

**État 2026.** TensorFlow **2.21**, avec **Keras 3** intégré comme `tf.keras` (depuis 2.16). Le bas niveau idiomatique tient en quatre briques :

| Brique | Rôle |
|---|---|
| `tf.GradientTape` | enregistre les opérations pour la rétropropagation |
| `tf.data.Dataset` | pipeline de données (streaming, shuffle, batch, prefetch) |
| `@tf.function` | compile une fonction Python en graphe (AutoGraph) |
| `tf.train.Checkpoint` | persistance fine de l'état (poids **+** optimiseur) |

**Quand choisir quoi ?**

| Besoin | Outil conseillé |
|---|---|
| Recherche, prototypage, flexibilité | PyTorch |
| Production, mobile/edge, TPU, MLOps | **TensorFlow bas niveau** |
| Portabilité multi-backend, API concise | Keras 3 |
<!-- #endregion -->

<!-- #region -->
## 2. Tenseurs
<!-- #endregion -->

<!-- #region -->
Le tenseur est la structure de base. Deux types à distinguer :

- **`tf.constant`** : tenseur **immuable** (une valeur figée) ;
- **`tf.Variable`** : tenseur **mutable**, dont l'état est suivi par `GradientTape` — c'est le support des **poids** d'un modèle.

Chaque tenseur a un `dtype` et une `shape`.
<!-- #endregion -->

```python
# Création : tf.constant (immuable) vs tf.Variable (mutable -> support de l'autodiff)
scalar = tf.constant(3.14)
vector = tf.constant([1.0, 2.0, 3.0])
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
weights = tf.Variable(tf.random.normal((2, 3)), name="w")
print("scalar", scalar.numpy(), "| matrix shape", matrix.shape, "| Variable", weights.shape)
```

<!-- #region -->
La conversion avec NumPy est immédiate (`.numpy()` dans un sens, `tf.convert_to_tensor` dans l'autre, sans copie côté CPU). Les opérations élémentaires, le `reshape` et le **broadcasting** suivent les mêmes règles que NumPy.
<!-- #endregion -->

```python
# Conversion <-> numpy (zero-copy côté CPU), opérations, reshape, broadcasting
np_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
t = tf.convert_to_tensor(np_arr)
back = t.numpy()  # tf.Tensor -> ndarray
elementwise = matrix * 2 + 1  # ops élémentaires
reshaped = tf.reshape(vector, (3, 1))
broadcasted = matrix + tf.constant([10.0, 20.0])  # broadcasting sur les colonnes
print("reshape", reshaped.shape, "| broadcast row0", broadcasted.numpy()[0])
```

<!-- #region -->
## 3. Multiplication matricielle
<!-- #endregion -->

<!-- #region -->
Le produit matriciel $C = AB$ avec $A \in \mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$ donne $C \in \mathbb{R}^{m\times n}$, $C_{ij} = \sum_p A_{ip} B_{pj}$.

TensorFlow propose trois écritures équivalentes : `tf.matmul`, l'opérateur `@`, et `tf.einsum` (notation d'Einstein, plus expressive pour les contractions complexes). Le **batch matmul** applique le produit sur la dernière paire d'axes d'un tenseur de rang ≥ 3.
<!-- #endregion -->

```python
a = tf.random.normal((3, 4))
b = tf.random.normal((4, 2))
prod = tf.matmul(a, b)  # (3,4) @ (4,2) -> (3,2)
prod_op = a @ b  # opérateur équivalent

batch_a = tf.random.normal((8, 3, 4))  # 8 matrices (3,4)
batch_b = tf.random.normal((8, 4, 2))
batch_prod = tf.matmul(batch_a, batch_b)  # batch matmul -> (8,3,2)

einsum_prod = tf.einsum("ij,jk->ik", a, b)  # produit matriciel via einsum
print("matmul", prod.shape, "| batch", batch_prod.shape, "| einsum==matmul",
      bool(tf.reduce_all(tf.abs(einsum_prod - prod) < 1e-5)))
```

<!-- #region -->
## 4. Autodiff avec `tf.GradientTape`
<!-- #endregion -->

<!-- #region -->
**`tf.GradientTape`** enregistre, dans son contexte (`with`), toutes les opérations appliquées aux tenseurs **suivis** (les `tf.Variable` le sont automatiquement ; pour un `tf.constant`, utiliser `tape.watch(x)`). On obtient ensuite le gradient par `tape.gradient(sortie, entrées)`.

C'est l'équivalent TF du `backward()` de PyTorch, mais **explicite** : on choisit ce qui est enregistré et quand.

**Vérification manuelle.** Pour $f(x) = x^3 + 2x$, la dérivée analytique est $f'(x) = 3x^2 + 2$. On compare au gradient automatique pour se convaincre que la *tape* applique bien la règle de dérivation en chaîne.
<!-- #endregion -->

```python
def f(x: tf.Tensor) -> tf.Tensor:
    """f(x) = x^3 + 2x  ->  f'(x) = 3x^2 + 2 (vérif analytique)."""
    return x**3 + 2 * x


x0 = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = f(x0)
grad_auto = tape.gradient(y, x0)
grad_manual = 3 * x0**2 + 2  # dérivée analytique
print(f"f'(2) auto={float(grad_auto):.4f} manuel={float(grad_manual):.4f} "
      f"ok={abs(float(grad_auto) - float(grad_manual)) < 1e-5}")
```

<!-- #region -->
On peut imbriquer deux *tapes* pour obtenir une **dérivée seconde** : la *tape* externe différencie le gradient produit par la *tape* interne. Pour $f$, $f''(x) = 6x$.
<!-- #endregion -->

```python
# Gradient d'ordre 2 via tapes imbriqués
with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = f(x0)
    d1 = t1.gradient(y, x0)  # f'(x)
d2 = t2.gradient(d1, x0)  # f''(x) = 6x
print(f"f''(2) auto={float(d2):.4f} manuel={float(6 * x0):.4f}")
```

<!-- #region -->
## 5. GPU / accélérateur et reproductibilité
<!-- #endregion -->

<!-- #region -->
`tf.config.list_physical_devices("GPU")` liste les accélérateurs. Le placement se contrôle avec `with tf.device("/CPU:0")` ou `"/GPU:0"`. Sur une machine sans GPU, tout s'exécute sur CPU (ce notebook tourne ainsi).
<!-- #endregion -->

```python
gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs détectés: {len(gpus)} (CPU-only sur ce poste si 0)")
```

<!-- #region -->
Pour la **reproductibilité**, on factorise une fonction `set_seed` typée : `tf.keras.utils.set_random_seed` fixe d'un coup les graines Python, NumPy et TensorFlow. Combinée à `enable_op_determinism()` (posé au setup), elle garantit des résultats identiques d'un run à l'autre.
<!-- #endregion -->

```python
def set_seed(seed: int) -> None:
    """Fixe les graines pour la reproductibilité (Python + numpy + TF en un appel)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)  # seed Python, numpy et TF d'un coup


set_seed(42)
big = tf.random.normal((512, 512))
_t = time.perf_counter()
for _ in range(20):
    _ = tf.matmul(big, big)
print(f"20x matmul 512x512 : {time.perf_counter() - _t:.3f}s")
```

<!-- #region -->
## 6. Cas synthétique : XOR continu
<!-- #endregion -->

<!-- #region -->
Le **XOR** est le « hello world » des réseaux : il n'est **pas linéairement séparable**, donc il force le modèle à apprendre une représentation non triviale. On génère une version continue : quatre gaussiennes centrées sur les coins du carré unité, avec le label XOR (coins opposés = même classe). Ce jeu sert de fil rouge des sections 6 à 16 — il est **identique** dans les trois notebooks de la série.
<!-- #endregion -->

```python
def generate_xor(n: int = 400, noise: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Génère un jeu XOR continu : 2 gaussiennes par coin, label = XOR des signes.

    Args:
        n: nombre total de points.
        noise: écart-type du bruit gaussien autour de chaque coin.
        seed: graine numpy.
    Returns:
        (X, y) avec X de shape (n, 2) float32 et y de shape (n,) int.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    labels = np.array([0, 1, 1, 0])  # XOR
    idx = rng.integers(0, 4, size=n)
    X = centers[idx] + rng.normal(0, noise, size=(n, 2)).astype(np.float32)
    y = labels[idx].astype(np.int32)
    return X, y


set_seed(42)
X_xor, y_xor = generate_xor(n=800, noise=0.15, seed=0)
Xtr, Xte, ytr, yte = train_test_split(X_xor, y_xor, test_size=0.25, random_state=42)
print(f"XOR train={Xtr.shape} test={Xte.shape} classes={np.bincount(y_xor)}")
```

<!-- #region -->
Visualisons les deux classes : on voit les quatre paquets, les coins opposés partageant la même couleur.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], s=12, label="classe 0", alpha=0.7)
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], s=12, label="classe 1", alpha=0.7)
ax.set_title("XOR continu")
ax.legend()
plt.show()
```

<!-- #region -->
## 7. Définition du modèle (sous-classe `tf.keras.Model`)
<!-- #endregion -->

<!-- #region -->
À bas niveau, l'idiome est le **subclassing** : on hérite de `tf.keras.Model`, on déclare les couches dans `__init__`, et on décrit le passage avant dans `call`. C'est plus explicite que `Sequential` et permet des architectures arbitraires.

Point important : la sortie est un **logit brut** (pas d'activation finale). On laissera la fonction de perte appliquer la sigmoïde en interne (`from_logits=True`), ce qui est **numériquement stable**. C'est exactement le bug à éviter de l'ancien notebook, où un `softmax` final était combiné à `from_logits=True` → double application.
<!-- #endregion -->

```python
class MLP(tf.keras.Model):
    """MLP 2-8-1 sortie LOGITS (pas d'activation finale -> BCE from_logits=True)."""

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden, activation="relu")
        self.out = tf.keras.layers.Dense(1)  # logit brut

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.out(self.d1(x))
```

<!-- #region -->
Un modèle sous-classé n'a pas de forme d'entrée connue à la construction : ses poids sont créés au **premier appel**. On l'« amorce » donc sur un petit échantillon avant d'inspecter `summary()`.
<!-- #endregion -->

```python
set_seed(42)
mlp = MLP(hidden=8)
_ = mlp(Xtr[:2])  # build (instancie les poids)
mlp.summary()
```

<!-- #region -->
## 8. Données : pipeline `tf.data`
<!-- #endregion -->

<!-- #region -->
`tf.data.Dataset` construit un pipeline de données **performant** et **streamable** : `from_tensor_slices` découpe les tableaux en exemples, `shuffle` mélange (sur un buffer), `batch` regroupe, et `prefetch(AUTOTUNE)` recouvre la préparation du batch suivant avec le calcul du batch courant. C'est l'équivalent TF du `DataLoader` de PyTorch.
<!-- #endregion -->

```python
def make_dataset(X: np.ndarray, y: np.ndarray, bs: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """Construit un pipeline tf.data : slices -> shuffle -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(Xtr, ytr, bs=32, shuffle=True)
test_ds = make_dataset(Xte, yte, bs=32, shuffle=False)
xb, yb = next(iter(train_ds))
print(f"un batch : X={xb.shape} y={yb.shape}")
```

<!-- #region -->
## 9. Loss, optimiseur et *learning rate scheduling*
<!-- #endregion -->

<!-- #region -->
- **Loss** : `BinaryCrossentropy(from_logits=True)` — la sigmoïde est appliquée en interne, de façon stable.
- **Optimiseur** : `Adam`.
- **Scheduling du LR** : on fait décroître le pas d'apprentissage au fil des steps. `ExponentialDecay` suit $\text{lr}(t) = \text{lr}_0 \cdot r^{\,t/d}$ (ici $r=0.9$, $d=200$). Un LR qui décroît stabilise la fin de l'entraînement.
<!-- #endregion -->

```python
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=200, decay_rate=0.9, staircase=False
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

steps = np.arange(0, 2000)
lrs = [float(lr_schedule(s)) for s in steps]
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(steps, lrs)
ax.set_title("Learning rate (ExponentialDecay)")
ax.set_xlabel("step")
ax.set_ylabel("lr")
plt.show()
print(f"lr step0={lrs[0]:.4f} step1000={lrs[1000]:.5f}")
```

<!-- #region -->
## 10. Boucle d'entraînement standard (`@tf.function`)
<!-- #endregion -->

<!-- #region -->
Le cœur du bas niveau : on écrit **soi-même** le pas d'entraînement.

1. `with tf.GradientTape() as tape:` enregistre le passage avant ;
2. on calcule la perte ;
3. `tape.gradient(loss, model.trainable_variables)` donne les gradients ;
4. `optimizer.apply_gradients(...)` met à jour les poids.

Le décorateur **`@tf.function`** compile cette fonction en graphe (via AutoGraph) : le premier appel *trace* la fonction, les suivants exécutent le graphe optimisé — d'où un gain de vitesse.

> **Piège Keras 3** : un optimiseur crée ses variables internes (moments d'Adam) au **premier** `apply_gradients`. Or créer une variable *dans* une `@tf.function` est interdit. On construit donc le modèle (un appel eager) **et** l'optimiseur (`optimizer.build(...)`) **avant** d'entrer dans le graphe.
<!-- #endregion -->

```python
@tf.function
def train_step(model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor,
               loss_obj: tf.keras.losses.Loss, opt: tf.keras.optimizers.Optimizer) -> tf.Tensor:
    """Un pas d'entraînement : forward + backward + mise à jour. Compilé en graphe."""
    with tf.GradientTape() as tape:
        logits = tf.squeeze(model(x, training=True), axis=-1)
        loss = loss_obj(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

<!-- #region -->
La boucle parcourt les epochs et, à chaque epoch, tous les batchs du `Dataset`, en accumulant la perte moyenne.
<!-- #endregion -->

```python
def train_loop(model, ds, loss_obj, opt, epochs: int = 30) -> list[float]:
    """Boucle d'entraînement standard sur un tf.data.Dataset."""
    history = []
    for epoch in range(epochs):
        losses = [float(train_step(model, x, y, loss_obj, opt)) for x, y in ds]
        history.append(float(np.mean(losses)))
    return history
```

<!-- #region -->
On (ré)instancie un MLP un peu plus large (8 → 16 neurones cachés, pour une convergence robuste sur le XOR), on construit l'optimiseur, puis on entraîne.
<!-- #endregion -->

```python
set_seed(42)
mlp = MLP(hidden=16)
_ = mlp(Xtr[:2])  # build des poids (eager) AVANT toute @tf.function
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer.build(mlp.trainable_variables)  # build des slots optim hors graphe
hist_xor = train_loop(mlp, train_ds, loss_fn, optimizer, epochs=60)
print(f"XOR loss epoch0={hist_xor[0]:.4f} -> epochN={hist_xor[-1]:.4f}")
```

<!-- #region -->
Mesurons l'intérêt de `@tf.function` : le même passage avant, en mode *eager* (Python pur) vs *graph* (compilé). Le graphe est généralement plus rapide sur des appels répétés.
<!-- #endregion -->

```python
@tf.function
def _fwd_graph(m, x):
    return m(x)


def _fwd_eager(m, x):
    return m(x)


_t = time.perf_counter()
for _ in range(200):
    _ = _fwd_eager(mlp, xb)
t_eager = time.perf_counter() - _t
_ = _fwd_graph(mlp, xb)  # trace
_t = time.perf_counter()
for _ in range(200):
    _ = _fwd_graph(mlp, xb)
t_graph = time.perf_counter() - _t
print(f"200 forwards : eager={t_eager:.3f}s graph={t_graph:.3f}s")
```

<!-- #region -->
La **frontière de décision** confirme que le réseau a bien appris le XOR : les zones rouge/bleu épousent les coins opposés.
<!-- #endregion -->

```python
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
probs = tf.sigmoid(tf.squeeze(mlp(grid), axis=-1)).numpy().reshape(xx.shape)
fig, ax = plt.subplots(figsize=(5, 5))
ax.contourf(xx, yy, probs, levels=20, cmap="RdBu_r", alpha=0.6)
ax.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=10, cmap="RdBu", edgecolors="k", linewidths=0.2)
ax.set_title("Frontière de décision (XOR)")
plt.show()
```

<!-- #region -->
## 11. Boucle manuelle et gestion fine du batch
<!-- #endregion -->

<!-- #region -->
Quand les classes sont **déséquilibrées**, la composition des batchs influence l'apprentissage. On reconstruit ici, proprement et typés, les outils de l'ancien notebook (qui étaient buggés : `Stort_modulo` en $O(n^2)$ avec un `try/except` masquant les erreurs, tirages d'indices cassés). On démontre sur un jeu synthétique à 3 classes déséquilibrées (50 / 30 / 20 %).
<!-- #endregion -->

```python
set_seed(42)
X_syn, y_syn = make_classification(
    n_samples=5000, n_features=10, n_informative=4, n_redundant=0, n_repeated=0,
    n_classes=3, n_clusters_per_class=1, weights=[0.5, 0.3, 0.2], random_state=42,
)
print(f"synthétique déséquilibré : {np.bincount(y_syn)}")
```

<!-- #region -->
Trois helpers :

- **`down_sample_balanced`** : génère `k` *folds* d'indices **équilibrés** (autant d'exemples par classe que la classe minoritaire), chacun mélangé.
- **`shuffle_xy`** : permutation **synchronisée** de `X` et `y` (sans casser l'appariement).
- **`sort_round_robin`** : trie en **round-robin** par classe (`0,1,2,0,1,2,…`) en $O(n)$ — remplace l'ancien `Stort_modulo`.
<!-- #endregion -->

```python
def down_sample_balanced(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Génère k folds d'indices équilibrés (n_min par classe), mélangés."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    per_class = {c: np.where(y == c)[0] for c in classes}
    folds = []
    for _ in range(k):
        idx = np.concatenate([rng.choice(per_class[c], n_min, replace=False) for c in classes])
        rng.shuffle(idx)
        folds.append(idx)
    return folds


def shuffle_xy(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Permutation synchronisée de X et y (mélange sans casser l'appariement)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    return X[p], y[p]


def sort_round_robin(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trie les échantillons en round-robin par classe (0,1,2,0,1,2,...). O(n)."""
    classes = np.unique(y)
    queues = {c: list(np.where(y == c)[0]) for c in classes}
    order = []
    while any(queues[c] for c in classes):
        for c in classes:
            if queues[c]:
                order.append(queues[c].pop(0))
    order = np.array(order)
    return X[order], y[order]
```

<!-- #region -->
Démonstration : un fold équilibré contient bien le même nombre d'exemples par classe, et le tri round-robin alterne les labels.
<!-- #endregion -->

```python
folds = down_sample_balanced(y_syn, k=3, seed=0)
print(f"fold0 équilibré : {np.bincount(y_syn[folds[0]])}")
Xs, ys = sort_round_robin(*shuffle_xy(X_syn[:9], y_syn[:9], seed=1))
print(f"round-robin labels (9 premiers triés) : {ys}")
```

<!-- #region -->
On définit un petit classifieur multiclasse (logits) et le pas d'entraînement associé (`SparseCategoricalCrossentropy(from_logits=True)`). `make_clf` utilise `Sequential` avec une couche `Input` : ses poids existent dès la construction, donc pas besoin d'amorçage.
<!-- #endregion -->

```python
def make_clf(n_features: int, n_classes: int) -> tf.keras.Model:
    """Petit MLP de classification (logits en sortie)."""
    set_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input((n_features,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes),  # logits
    ])


scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
scce_none = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")


@tf.function
def train_step_multi(model, x, y, opt) -> tf.Tensor:
    """Pas d'entraînement multiclasse (logits, SCCE from_logits)."""
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = scce(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

<!-- #region -->
La boucle manuelle paramétrée : à chaque epoch, soit on tire un fold **équilibré**, soit on **mélange puis trie** le jeu. On entraîne sur les deux variantes pour comparer la dynamique de la perte.
<!-- #endregion -->

```python
def manual_train(X: np.ndarray, y: np.ndarray, mode: str, epochs: int = 8, bs: int = 64) -> list[float]:
    """Boucle manuelle GradientTape. mode='balanced' (folds équilibrés) ou 'sorted' (tri round-robin)."""
    model = make_clf(X.shape[1], len(np.unique(y)))
    opt = tf.keras.optimizers.Adam(1e-3)
    opt.build(model.trainable_variables)  # slots optim hors @tf.function
    folds = down_sample_balanced(y, epochs, seed=0) if mode == "balanced" else None
    history = []
    for epoch in range(epochs):
        if mode == "balanced":
            Xe, ye = X[folds[epoch]], y[folds[epoch]]
        else:  # sorted
            Xe, ye = sort_round_robin(*shuffle_xy(X, y, seed=epoch))
        batch_losses = []
        for i in range(0, len(Xe), bs):
            xb_ = tf.constant(Xe[i:i + bs], tf.float32)
            yb_ = tf.constant(ye[i:i + bs], tf.int32)
            batch_losses.append(float(train_step_multi(model, xb_, yb_, opt)))
        history.append(float(np.mean(batch_losses)))
    return history


h_bal = manual_train(X_syn, y_syn, mode="balanced", epochs=8)
h_sorted = manual_train(X_syn, y_syn, mode="sorted", epochs=8)
print(f"équilibré loss {h_bal[0]:.3f}->{h_bal[-1]:.3f} | trié {h_sorted[0]:.3f}->{h_sorted[-1]:.3f}")
```

<!-- #region -->
## 12. Save / Load (`tf.train.Checkpoint`)
<!-- #endregion -->

<!-- #region -->
À bas niveau, la persistance passe par **`tf.train.Checkpoint`**, qui sauve l'état de **n'importe quels objets suivis** — modèle **et** optimiseur (utile pour reprendre un entraînement). `CheckpointManager` gère la rotation des fichiers. On vérifie qu'un modèle rechargé produit des prédictions **identiques**.

> Pour la **production**, on exporterait plutôt un `SavedModel` via `model.export("dir")` (servable par TF Serving), ou le format Keras natif `model.save("modele.keras")`.
<!-- #endregion -->

```python
ckpt_dir = Path(tempfile.mkdtemp(prefix="tf_ckpt_"))
ckpt = tf.train.Checkpoint(model=mlp, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=1)
save_path = manager.save()
pred_before = tf.sigmoid(tf.squeeze(mlp(Xte[:5]), -1)).numpy()
print(f"checkpoint sauvé -> {Path(save_path).name}")
```

<!-- #region -->
Rechargement dans un modèle neuf (amorcé avant le `restore`), puis comparaison des prédictions.
<!-- #endregion -->

```python
set_seed(0)
restored = MLP(hidden=16)
_ = restored(Xte[:2])  # build avant restore
tf.train.Checkpoint(model=restored).restore(manager.latest_checkpoint).expect_partial()
pred_after = tf.sigmoid(tf.squeeze(restored(Xte[:5]), -1)).numpy()
print(f"prédictions identiques après reload: {np.allclose(pred_before, pred_after, atol=1e-6)}")
```

<!-- #region -->
## 13. Évaluation et métrique custom
<!-- #endregion -->

<!-- #region -->
La boucle d'évaluation parcourt le `Dataset` de test ; comme `tf.data` gère naturellement le **dernier batch partiel**, pas de gestion spéciale du `drop_last`. On calcule perte et exactitude.
<!-- #endregion -->

```python
def evaluate_binary(model, ds) -> tuple[float, float]:
    """Boucle d'évaluation : loss BCE + accuracy. Le dernier batch partiel est géré naturellement."""
    losses, correct, total = [], 0, 0
    for x, y in ds:
        logits = tf.squeeze(model(x, training=False), -1)
        losses.append(float(loss_fn(y, logits)))
        preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
        correct += int(tf.reduce_sum(tf.cast(preds == y, tf.float32)))
        total += int(y.shape[0])
    return float(np.mean(losses)), correct / total


val_loss, val_acc = evaluate_binary(mlp, test_ds)
print(f"XOR test : loss={val_loss:.4f} acc={val_acc:.4f}")
```

<!-- #region -->
**Métrique custom.** On implémente le **F1 macro** en sous-classant `tf.keras.metrics.Metric` : on accumule TP/FP/FN par classe sur les batchs (`update_state`), on calcule le F1 moyen (`result`), et on remet à zéro entre deux epochs (`reset_state`).

Rappel : $\text{precision}_c = \frac{TP_c}{TP_c+FP_c}$, $\text{recall}_c = \frac{TP_c}{TP_c+FN_c}$, $F1_c = \frac{2\,P_c R_c}{P_c+R_c}$, et le **macro** moyenne les $F1_c$ (chaque classe pèse pareil, ce qui est pertinent en cas de déséquilibre).

> Le `reset_state()` en début de chaque epoch est un point clé du bas niveau TF : sans lui, la métrique cumulerait sur tout l'historique.
<!-- #endregion -->

```python
class MacroF1(tf.keras.metrics.Metric):
    """F1 macro implémentée à la main (accumulation TP/FP/FN par classe)."""

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        yt = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes)
        yp = tf.one_hot(tf.cast(y_pred, tf.int32), self.num_classes)
        self.tp.assign_add(tf.reduce_sum(yt * yp, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - yt) * yp, axis=0))
        self.fn.assign_add(tf.reduce_sum(yt * (1 - yp), axis=0))

    def result(self) -> tf.Tensor:
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1)

    def reset_state(self) -> None:
        for v in self.variables:
            v.assign(tf.zeros_like(v))
```

<!-- #region -->
On entraîne brièvement le classifieur synthétique et on confronte notre métrique au `f1_score` de scikit-learn : les deux doivent coïncider.
<!-- #endregion -->

```python
clf_syn = make_clf(X_syn.shape[1], 3)
opt_syn = tf.keras.optimizers.Adam(1e-3)
opt_syn.build(clf_syn.trainable_variables)
for _ in range(15):
    for i in range(0, len(X_syn), 128):
        train_step_multi(clf_syn, tf.constant(X_syn[i:i + 128], tf.float32),
                         tf.constant(y_syn[i:i + 128], tf.int32), opt_syn)
y_pred_syn = np.argmax(clf_syn(X_syn).numpy(), axis=1)
m = MacroF1(num_classes=3)
m.reset_state()
m.update_state(y_syn, y_pred_syn)
print(f"MacroF1 custom={float(m.result()):.4f} | sklearn={f1_score(y_syn, y_pred_syn, average='macro'):.4f}")
```

<!-- #region -->
## 14. Gestion du déséquilibre : class weights, sample weights, focal loss
<!-- #endregion -->

<!-- #region -->
Trois leviers contre le déséquilibre :

- **Class weights** : un poids par classe (inversement proportionnel à sa fréquence) ;
- **Sample weights** : un poids par échantillon (plus général) ;
- **Focal loss** : modifie la perte pour **atténuer les exemples faciles** et concentrer l'apprentissage sur les difficiles.

`compute_class_weight` / `compute_sample_weight` (scikit-learn) calculent ces poids.
<!-- #endregion -->

```python
class_weights = compute_class_weight("balanced", classes=np.unique(y_syn), y=y_syn)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
sample_weights = compute_sample_weight("balanced", y_syn).astype(np.float32)
print(f"class weights={ {k: round(v, 3) for k, v in class_weight_dict.items()} }")
```

<!-- #region -->
À bas niveau, on applique la pondération **directement dans le `GradientTape`** : on calcule la perte par échantillon (`reduction="none"`), on la multiplie par les poids, puis on moyenne. C'est l'équivalent explicite de ce que `fit` fait en interne.
<!-- #endregion -->

```python
@tf.function
def weighted_train_step(model, x, y, w, opt) -> tf.Tensor:
    """Pas d'entraînement avec pondération MANUELLE par échantillon (dans le GradientTape)."""
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        per_sample = scce_none(y, logits)
        loss = tf.reduce_mean(per_sample * w)  # pondération manuelle
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


clf_w = make_clf(X_syn.shape[1], 3)
opt_w = tf.keras.optimizers.Adam(1e-3)
opt_w.build(clf_w.trainable_variables)
for i in range(0, len(X_syn), 128):
    weighted_train_step(clf_w, tf.constant(X_syn[i:i + 128], tf.float32),
                        tf.constant(y_syn[i:i + 128], tf.int32),
                        tf.constant(sample_weights[i:i + 128]), opt_w)
print("pondération manuelle (GradientTape) OK")
```

<!-- #region -->
**Pont avec l'API haut niveau** (pour référence) : l'argument `class_weight` de `model.fit` fait exactement ce travail. C'est le **seul** `fit` du notebook — on le montre pour faire le lien, mais le reste reste en boucle manuelle, conformément à l'angle bas niveau.
<!-- #endregion -->

```python
clf_fit = make_clf(X_syn.shape[1], 3)
clf_fit.compile(optimizer="adam", loss=scce, metrics=["accuracy"])
clf_fit.fit(X_syn, y_syn, epochs=2, batch_size=128, class_weight=class_weight_dict, verbose=0)
print("démo fit(class_weight=...) OK (raccourci haut niveau)")
```

<!-- #region -->
**Focal loss** (Lin et al., 2017) : $FL(p_t) = -\alpha\,(1 - p_t)^{\gamma}\,\log(p_t)$, où $p_t$ est la probabilité de la bonne classe. Le facteur $(1-p_t)^\gamma$ écrase la contribution des exemples bien classés (grand $p_t$), forçant le modèle à travailler sur les cas durs. On l'implémente à la main avec `tf.nn.log_softmax`.
<!-- #endregion -->

```python
def focal_loss(y_true: tf.Tensor, logits: tf.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> tf.Tensor:
    """Focal loss multiclasse : FL = -alpha (1-p_t)^gamma log(p_t). Atténue les exemples faciles."""
    y_oh = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(logits)[-1])
    log_p = tf.nn.log_softmax(logits, axis=-1)
    p = tf.exp(log_p)
    fl = -alpha * tf.pow(1 - p, gamma) * log_p * y_oh
    return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))


clf_focal = make_clf(X_syn.shape[1], 3)
opt_f = tf.keras.optimizers.Adam(1e-3)
for i in range(0, len(X_syn), 128):
    xb_ = tf.constant(X_syn[i:i + 128], tf.float32)
    yb_ = tf.constant(y_syn[i:i + 128], tf.int32)
    with tf.GradientTape() as tape:
        fl = focal_loss(yb_, clf_focal(xb_, training=True))
    g = tape.gradient(fl, clf_focal.trainable_variables)
    opt_f.apply_gradients(zip(g, clf_focal.trainable_variables))
print(f"focal loss finale={float(fl):.4f}")
```

<!-- #region -->
## 15. Régularisation
<!-- #endregion -->

<!-- #region -->
Quatre techniques courantes, combinées ici :

- **Dropout** : éteint aléatoirement une fraction des neurones à l'entraînement (anti-coadaptation) ;
- **Batch normalization** : normalise les activations par batch (entraînement plus stable et rapide) ;
- **Weight decay** : pénalise les poids ($L_2$), via l'optimiseur **`AdamW`** ;
- **Early stopping** : on arrête quand la perte de validation cesse de s'améliorer.

À bas niveau, l'early stopping s'écrit **à la main** : on suit la meilleure perte de validation, on compte la patience, et on **restaure les meilleurs poids** à la fin.
<!-- #endregion -->

```python
class RegularizedMLP(tf.keras.Model):
    """MLP régularisé : Dense + BatchNorm + Dropout, sortie logits."""

    def __init__(self, n_classes: int, p_drop: float = 0.3) -> None:
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.bn = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(p_drop)
        self.out = tf.keras.layers.Dense(n_classes)

    def call(self, x, training: bool = False):
        x = self.drop(self.bn(self.d1(x), training=training), training=training)
        return self.out(x)
```

<!-- #region -->
La boucle d'entraînement avec early stopping manuel : `training=True` active dropout et batch norm pendant l'entraînement, `training=False` les désactive pour l'évaluation.
<!-- #endregion -->

```python
set_seed(42)
Xtr_s, Xval_s, ytr_s, yval_s = train_test_split(X_syn, y_syn, test_size=0.2, random_state=0)
reg_model = RegularizedMLP(n_classes=3)
_ = reg_model(Xtr_s[:2], training=False)  # build poids (eager)
opt_reg = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)  # weight decay
opt_reg.build(reg_model.trainable_variables)


def eval_loss_multi(model, X, y) -> float:
    return float(scce(y, model(X, training=False)))


best_val, patience, wait, best_weights = np.inf, 5, 0, None
for epoch in range(40):
    for i in range(0, len(Xtr_s), 128):
        train_step_multi(reg_model, tf.constant(Xtr_s[i:i + 128], tf.float32),
                         tf.constant(ytr_s[i:i + 128], tf.int32), opt_reg)
    v = eval_loss_multi(reg_model, Xval_s, yval_s)
    if v < best_val - 1e-4:
        best_val, wait, best_weights = v, 0, reg_model.get_weights()
    else:
        wait += 1
        if wait >= patience:
            break
if best_weights is not None:
    reg_model.set_weights(best_weights)  # restaure les meilleurs poids
print(f"early stopping epoch={epoch} best_val_loss={best_val:.4f}")
```

<!-- #region -->
## 16. Visualisation et logging (`tf.summary`)
<!-- #endregion -->

<!-- #region -->
TensorFlow journalise les métriques via **`tf.summary`** : on crée un *file writer* pointant vers un dossier de logs, et on y écrit des scalaires (`tf.summary.scalar`). Ces logs sont lisibles dans **TensorBoard**.
<!-- #endregion -->

```python
log_dir = Path(tempfile.mkdtemp(prefix="tf_logs_"))
writer = tf.summary.create_file_writer(str(log_dir))
with writer.as_default():
    for step, l in enumerate(hist_xor):
        tf.summary.scalar("xor/train_loss", l, step=step)
writer.flush()
print(f"scalaires loggés dans {log_dir.name}")
```

<!-- #region -->
Pour visualiser, dans une cellule **du notebook** (magics Jupyter, hors script) :

```
%load_ext tensorboard
%tensorboard --logdir $log_dir
```

À défaut de TensorBoard, on trace simplement les courbes de perte accumulées avec matplotlib.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(hist_xor, label="XOR train loss")
ax.plot(h_bal, label="synthétique équilibré")
ax.set_title("Courbes de loss")
ax.set_xlabel("epoch")
ax.legend()
plt.show()
```

<!-- #region -->
## 17. Cas réel — Classification MNIST
<!-- #endregion -->

<!-- #region -->
Premier cas réel : reconnaissance de chiffres manuscrits **MNIST**. On **sous-échantillonne** (8 000 / 2 000) et on limite les epochs pour que le notebook tourne vite sur CPU (l'original utilisait `batch_size=8` → 264 s/epoch ; on passe à 128). On normalise les pixels dans [0, 1].
<!-- #endregion -->

```python
(Xtr_full, ytr_full), (Xte_full, yte_full) = tf.keras.datasets.mnist.load_data()
set_seed(42)
itr = np.random.default_rng(42).choice(len(Xtr_full), 8000, replace=False)
ite = np.random.default_rng(43).choice(len(Xte_full), 2000, replace=False)
Xtr_m = (Xtr_full[itr] / 255.0).astype(np.float32)
ytr_m = ytr_full[itr].astype(np.int32)
Xte_m = (Xte_full[ite] / 255.0).astype(np.float32)
yte_m = yte_full[ite].astype(np.int32)
print(f"MNIST sous-échantillonné : train={Xtr_m.shape} test={Xte_m.shape}")
```

<!-- #region -->
**ANN** (perceptron multicouche) : on aplatit l'image puis deux couches denses. Entraînement via notre boucle bas niveau (`train_step_multi`).
<!-- #endregion -->

```python
class MLPClassifier(tf.keras.Model):
    """ANN MNIST : Flatten + Dense, sortie logits (10)."""

    def __init__(self) -> None:
        super().__init__()
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.drop = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(10)

    def call(self, x, training: bool = False):
        return self.out(self.drop(self.d1(self.flat(x)), training=training))


def train_mnist(model, X, y, epochs: int = 3, bs: int = 128) -> None:
    """Entraînement bas niveau (boucle @tf.function) sur MNIST."""
    opt = tf.keras.optimizers.Adam(1e-3)
    _ = model(X[:2])  # build poids (eager) avant @tf.function
    opt.build(model.trainable_variables)
    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=42).batch(bs).prefetch(tf.data.AUTOTUNE)
    for epoch in range(epochs):
        for xb_, yb_ in ds:
            train_step_multi(model, xb_, yb_, opt)


set_seed(42)
ann = MLPClassifier()
train_mnist(ann, Xtr_m, ytr_m, epochs=3)
acc_ann = float(np.mean(np.argmax(ann(Xte_m).numpy(), axis=1) == yte_m))
print(f"ANN MNIST acc={acc_ann:.4f}")
```

<!-- #region -->
**CNN** : architecture naturelle pour les images — deux blocs `Conv2D`/`MaxPooling` avant les couches denses. Il faut ajouter une dimension de canal aux images (`(N, 28, 28, 1)`).
<!-- #endregion -->

```python
class CNNClassifier(tf.keras.Model):
    """CNN MNIST : Conv2D/MaxPool x2 + Dense, sortie logits (10)."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")
        self.p1 = tf.keras.layers.MaxPooling2D()
        self.c2 = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")
        self.p2 = tf.keras.layers.MaxPooling2D()
        self.flat = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.out = tf.keras.layers.Dense(10)

    def call(self, x, training: bool = False):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        return self.out(self.d1(self.flat(x)))


Xtr_cnn = Xtr_m[..., None]  # canal -> (N,28,28,1)
Xte_cnn = Xte_m[..., None]
set_seed(42)
cnn = CNNClassifier()
train_mnist(cnn, Xtr_cnn, ytr_m, epochs=2)
y_pred_cnn = np.argmax(cnn(Xte_cnn).numpy(), axis=1)
acc_cnn = float(np.mean(y_pred_cnn == yte_m))
print(f"CNN MNIST acc={acc_cnn:.4f}")
```

<!-- #region -->
La **matrice de confusion** montre quelles classes sont confondues (idéalement, tout sur la diagonale).
<!-- #endregion -->

```python
cm = confusion_matrix(yte_m, y_pred_cnn)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Matrice de confusion (CNN MNIST)")
ax.set_xlabel("prédit")
ax.set_ylabel("réel")
fig.colorbar(im)
plt.show()
```

<!-- #region -->
**ROC / AUC** en *one-vs-rest* (chaque classe contre toutes les autres), puis on moyenne les AUC (**macro**). On binarise les labels en 10 colonnes et on trace une courbe par classe.
<!-- #endregion -->

```python
proba_cnn = tf.nn.softmax(cnn(Xte_cnn), axis=-1).numpy()
y_bin = label_binarize(yte_m, classes=list(range(10)))
fig, ax = plt.subplots(figsize=(5, 5))
aucs = []
for c in range(10):
    fpr, tpr, _ = roc_curve(y_bin[:, c], proba_cnn[:, c])
    a = auc(fpr, tpr)
    aucs.append(a)
    ax.plot(fpr, tpr, lw=0.8, alpha=0.6)
ax.plot([0, 1], [0, 1], "k--", lw=0.6)
ax.set_title(f"ROC OVR (AUC macro={np.mean(aucs):.3f})")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
plt.show()
print(f"AUC macro OVR={np.mean(aucs):.4f}")
```

<!-- #region -->
**Bonus (gimmick).** On peut traiter une image comme une **séquence de lignes** et la donner à un **LSTM**. Ce n'est pas l'architecture naturelle pour des images (un CNN fait mieux et plus vite), mais c'est un exercice instructif sur les RNN. Conservé court, en un seul entraînement.
<!-- #endregion -->

```python
set_seed(42)
lstm = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(10),
])
opt_lstm = tf.keras.optimizers.Adam(1e-3)
opt_lstm.build(lstm.trainable_variables)
ds_lstm = tf.data.Dataset.from_tensor_slices((Xtr_m[:4000], ytr_m[:4000])).batch(128)
for _ in range(3):
    for xb_, yb_ in ds_lstm:
        train_step_multi(lstm, xb_, yb_, opt_lstm)
acc_lstm = float(np.mean(np.argmax(lstm(Xte_m).numpy(), axis=1) == yte_m))
print(f"LSTM (gimmick image-séquence) acc={acc_lstm:.4f}")
```

<!-- #region -->
## 18. Cas réel — Régression California Housing
<!-- #endregion -->

<!-- #region -->
Second cas réel : prédire le prix médian des logements en Californie (**California Housing** remplace l'ancien dataset Boston, retiré de scikit-learn pour des raisons éthiques). On standardise les features (`StandardScaler`).
<!-- #endregion -->

```python
data = fetch_california_housing()
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(Xtr_c)
Xtr_c = scaler.transform(Xtr_c).astype(np.float32)
Xte_c = scaler.transform(Xte_c).astype(np.float32)
ytr_c = ytr_c.astype(np.float32)
yte_c = yte_c.astype(np.float32)
print(f"California : train={Xtr_c.shape} test={Xte_c.shape}")
```

<!-- #region -->
Modèle de régression : sortie **linéaire** à une unité, perte **MSE**. On réutilise le pattern boucle manuelle + early stopping + dropout.
<!-- #endregion -->

```python
class MLPRegressor(tf.keras.Model):
    """MLP régression : Dense + Dropout, sortie linéaire (1)."""

    def __init__(self, p_drop: float = 0.2) -> None:
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dense(32, activation="relu")
        self.drop = tf.keras.layers.Dropout(p_drop)
        self.out = tf.keras.layers.Dense(1)

    def call(self, x, training: bool = False):
        return self.out(self.drop(self.d2(self.d1(x)), training=training))


mse_fn = tf.keras.losses.MeanSquaredError()


@tf.function
def reg_step(model, x, y, opt) -> tf.Tensor:
    with tf.GradientTape() as tape:
        pred = tf.squeeze(model(x, training=True), -1)
        loss = mse_fn(y, pred)
    g = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    return loss
```

<!-- #region -->
Entraînement avec early stopping, puis métriques de régression : **RMSE** (erreur quadratique moyenne, dans l'unité cible), **MAE** (erreur absolue moyenne), **R²** (part de variance expliquée).
<!-- #endregion -->

```python
set_seed(42)
reg = MLPRegressor()
_ = reg(Xtr_c[:2])  # build poids (eager)
opt_c = tf.keras.optimizers.Adam(1e-3)
opt_c.build(reg.trainable_variables)
ds_c = tf.data.Dataset.from_tensor_slices((Xtr_c, ytr_c)).shuffle(len(Xtr_c), seed=42).batch(256).prefetch(tf.data.AUTOTUNE)
best, wait, best_w = np.inf, 0, None
for epoch in range(60):
    for xb_, yb_ in ds_c:
        reg_step(reg, xb_, yb_, opt_c)
    vpred = tf.squeeze(reg(Xte_c, training=False), -1).numpy()
    vmse = mean_squared_error(yte_c, vpred)
    if vmse < best - 1e-4:
        best, wait, best_w = vmse, 0, reg.get_weights()
    else:
        wait += 1
        if wait >= 6:
            break
if best_w is not None:
    reg.set_weights(best_w)
pred_c = tf.squeeze(reg(Xte_c, training=False), -1).numpy()
rmse = float(np.sqrt(mean_squared_error(yte_c, pred_c)))
mae = float(mean_absolute_error(yte_c, pred_c))
r2 = float(r2_score(yte_c, pred_c))
print(f"California : RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} (early stop epoch={epoch})")
```

<!-- #region -->
Le nuage **prédit vs réel** : plus les points serrent la diagonale, meilleure est la régression.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(yte_c, pred_c, s=6, alpha=0.3)
ax.plot([yte_c.min(), yte_c.max()], [yte_c.min(), yte_c.max()], "r--")
ax.set_title(f"Prédit vs réel (R2={r2:.3f})")
ax.set_xlabel("réel")
ax.set_ylabel("prédit")
plt.show()
```

<!-- #region -->
## 19. Explicabilité — SHAP
<!-- #endregion -->

<!-- #region -->
**SHAP** attribue à chaque pixel sa contribution à la prédiction. Pour un modèle TensorFlow/Keras 3, on utilise **`GradientExplainer`** (méthode des *expected gradients*) : `DeepExplainer` ne fonctionne **plus** sur TF2/Keras 3 — c'était un autre bug de l'ancien notebook.

`GradientExplainer` exige un modèle exposant `.inputs`/`.outputs` (Functional). Comme notre CNN est sous-classé, on l'enveloppe dans un modèle Functional **partageant les poids** (aucun ré-entraînement).
<!-- #endregion -->

```python
# DeepExplainer est cassé sur TF2/Keras3 -> GradientExplainer (expected gradients).
# shap exige un modèle Functional exposant .inputs/.outputs : on enveloppe le CNN
# sous-classé dans un tf.keras.Model functional (poids partagés, pas de ré-entraînement).
_inp = tf.keras.Input((28, 28, 1))
cnn_functional = tf.keras.Model(_inp, cnn(_inp))
background = Xtr_cnn[:100]
to_explain = Xte_cnn[:4]
explainer = shap.GradientExplainer(cnn_functional, background)
shap_values = explainer.shap_values(to_explain)
print(f"GradientExplainer OK | type={type(shap_values).__name__}")
```

<!-- #region -->
`shap.image_plot` superpose les attributions aux images : en rouge ce qui pousse vers la classe, en bleu ce qui l'éloigne. La forme exacte de `shap_values` dépend de la version de SHAP, d'où le petit aiguillage.
<!-- #endregion -->

```python
# GradientExplainer (shap 0.51) renvoie un ndarray (n, H, W, C, n_classes).
# On le découpe en une carte par classe (canal conservé en dernier) :
# chaque ligne = une image de test, les 10 colonnes = attributions des classes 0-9.
if isinstance(shap_values, list):
    shap_numpy = shap_values  # versions plus anciennes : déjà une liste par classe
else:
    sv = np.asarray(shap_values)
    shap_numpy = [sv[..., k] for k in range(sv.shape[-1])]
shap.image_plot(shap_numpy, to_explain, show=False)
plt.show()
```

<!-- #region -->
## Conclusion
<!-- #endregion -->

<!-- #region -->
On a parcouru **TensorFlow par le bas niveau**, en écrivant explicitement ce que `fit` masque :

| Idiome | Section |
|---|---|
| `tf.GradientTape` (autodiff) | 4, 10 |
| `tf.data.Dataset` (pipeline) | 8 |
| `@tf.function` (graphe) | 10 |
| boucle d'entraînement maison | 10, 11 |
| batch équilibré / trié | 11 |
| `tf.train.Checkpoint` | 12 |
| métrique custom (`Metric`) | 13 |
| class/sample weights, focal loss | 14 |
| régularisation + early stopping manuel | 15 |
| `tf.summary` / TensorBoard | 16 |
| cas réels MNIST / California | 17, 18 |
| SHAP `GradientExplainer` | 19 |

**Pour aller plus loin :**

- `DL_Keras` — les mêmes tâches en **haut niveau** (`compile`/`fit`/`callbacks`, `keras.ops`, multi-backend) ;
- `DL_PyTorch` — l'approche *define-by-run* et `nn.Module` ;
- en production : `model.export()` → **SavedModel**, **TF Serving**, **TFLite**, et `torch.compile`/**XLA** pour la performance.
<!-- #endregion -->
