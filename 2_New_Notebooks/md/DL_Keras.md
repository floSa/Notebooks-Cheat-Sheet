---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python (notebooks-refonte)
    name: notebooks-refonte
---

<!-- #region -->
# Deep Learning avec Keras 3 — API haut niveau & multi-backend
<!-- #endregion -->

<!-- #region -->
Ce notebook est le **3/3** de la stack frameworks DL (après `DL_PyTorch` et `DL_TensorFlow`). Il
suit le **même squelette en 17 sections** que les deux autres, mais en idiomes **Keras 3**.

**Rôle** : cheat-sheet (snippets prêts à copier) + tutoriel (le *quoi/pourquoi/quand*) + wiki
(quelques maths). **Fil rouge** : un XOR continu pour toucher tous les rouages, puis deux cas
réels — **MNIST** (classification) et **California Housing** (régression).

**Valeur ajoutée vs le notebook TensorFlow** (qui montre le bas niveau `tf.GradientTape`/`tf.data`) :
Keras 3 est l'API **haut niveau et portable**. Ce notebook insiste sur ce qui le distingue —
`compile`/`fit`/`callbacks`, le namespace **`keras.ops`** (API NumPy identique quel que soit le
backend), la variable d'environnement **`KERAS_BACKEND`** (TensorFlow / JAX / PyTorch / OpenVINO),
et `model.save('.keras')`. On ne redescend **pas** au `GradientTape` ici.
<!-- #endregion -->

<!-- #region -->
On commence par fixer le backend **avant** d'importer `keras` (c'est la seule contrainte du
multi-backend : le choix se fait à l'import). On fixe aussi une graine globale pour la
reproductibilité.
<!-- #endregion -->

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # à définir AVANT import keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # réduit le bruit de logs TensorFlow

import math

import matplotlib.pyplot as plt
import numpy as np

import keras
from keras import ops

SEED = 42
keras.utils.set_random_seed(SEED)  # seed unique python / numpy / backend

print(f"keras {keras.__version__} | backend = {keras.backend.backend()}")
```

<!-- #region -->
## 1. Présentation de Keras 3
<!-- #endregion -->

<!-- #region -->
**Keras 3** (≥ 3.0, ici 3.14) est une API de deep learning **multi-backend** : le même code
s'exécute sur **TensorFlow**, **JAX**, **PyTorch** ou **OpenVINO**. C'est le défaut depuis
TensorFlow 2.16 (l'ancien Keras 2 est désormais le paquet legacy `tf-keras`).

Deux piliers de portabilité :

- **`keras.ops`** : une API tableau quasi identique à NumPy, qui produit des tenseurs du backend
  actif — on écrit une seule fois le code mathématique.
- **`KERAS_BACKEND`** : on change de moteur de calcul sans toucher au modèle.

**Quand choisir Keras 3 ?** Prototypage rapide, portabilité entre moteurs, pipeline
`compile`/`fit`/`callbacks` standardisé. Pour un contrôle bas niveau fin de la boucle
d'entraînement, on préférera l'idiome natif du backend (cf le notebook TensorFlow).
<!-- #endregion -->

```python
def show_backend_info() -> None:
    """Affiche le backend actif et les backends supportés par Keras 3."""
    print("Backend actif :", keras.backend.backend())
    print("Backends supportés : tensorflow, jax, torch, openvino")
    print("Pour changer : os.environ['KERAS_BACKEND'] = 'jax'  (AVANT import keras)")


show_backend_info()
```

<!-- #region -->
## 2. Tenseurs : keras.ops (API NumPy portable)
<!-- #endregion -->

<!-- #region -->
`keras.ops` expose les opérations tableau (création, `matmul`, `reshape`, `sum`, `einsum`, …)
avec une signature **calquée sur NumPy**. Un `keras.ops` renvoie un tenseur du **backend actif**
(ici un `EagerTensor` TensorFlow) ; on revient à NumPy avec `ops.convert_to_numpy`.

L'intérêt : le code tenseur ci-dessous est **identique** que le backend soit TF, JAX ou PyTorch.
<!-- #endregion -->

```python
a = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
b = ops.reshape(ops.arange(4, dtype="float32"), (2, 2))

prod = ops.matmul(a, b)
s = ops.sum(a)
a_np = ops.convert_to_numpy(a)

print("a @ b =\n", ops.convert_to_numpy(prod))
print("sum(a) =", float(s))
print("type retour ops (backend) :", type(prod).__name__, "| <-> numpy :", type(a_np).__name__)
```

<!-- #region -->
`einsum` couvre les produits tensoriels par lot (parité avec la section matmul/bmm du notebook
PyTorch) : ici un produit matriciel appliqué sur une dimension de batch.
<!-- #endregion -->

```python
batch = ops.convert_to_tensor(np.random.randn(8, 3, 4).astype("float32"))
weight = ops.convert_to_tensor(np.random.randn(4, 5).astype("float32"))
batch_out = ops.einsum("bij,jk->bik", batch, weight)
print("einsum batch shape :", tuple(batch_out.shape))
```

<!-- #region -->
## 3. Accélérateur (GPU/CPU) & reproductibilité
<!-- #endregion -->

<!-- #region -->
La détection de l'accélérateur dépend du backend (ici on interroge TensorFlow). Pour la
reproductibilité, **`keras.utils.set_random_seed`** fixe d'un coup les graines de Python, NumPy et
du backend — préférable à les régler séparément.
<!-- #endregion -->

```python
import time


def detect_devices() -> str:
    """Retourne une description courte des accélérateurs disponibles (backend-dependent)."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return f"TF backend — GPU(s): {len(gpus)}" if gpus else "TF backend — CPU only"
    except Exception as exc:
        return f"device inconnu ({exc})"


print(detect_devices())
keras.utils.set_random_seed(SEED)

_m = ops.convert_to_tensor(np.random.randn(512, 512).astype("float32"))
_t0 = time.perf_counter()
for _ in range(10):
    _m = ops.matmul(_m, _m) * 1e-3
print(f"10 matmul 512x512 : {time.perf_counter() - _t0:.3f}s")
```

<!-- #region -->
## 4. Cas synthétique : XOR continu
<!-- #endregion -->

<!-- #region -->
Le **XOR continu** est notre fil rouge : un problème 2D **non linéairement séparable** (la classe
est `(x0 > 0) XOR (x1 > 0)`), donc un perceptron simple échoue mais un MLP réussit. La fonction de
génération est **identique** dans les trois notebooks frameworks pour pouvoir comparer.
<!-- #endregion -->

```python
def make_xor(n: int = 1000, noise: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Génère un XOR continu 2D.

    Args:
        n: nombre de points.
        noise: écart-type du bruit gaussien ajouté.
        seed: graine locale.
    Returns:
        (X, y) avec X float32 (n, 2) et y int64 (n,), classe = (x0>0) XOR (x1>0).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2)).astype("float32")
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype("int64")
    X += noise * rng.standard_normal(X.shape).astype("float32")
    return X, y


X_xor, y_xor = make_xor(1200, noise=0.15, seed=SEED)
print("XOR:", X_xor.shape, "classes:", np.bincount(y_xor))
```

<!-- #region -->
Visualisation des deux classes : on reconnaît le motif en damier caractéristique du XOR.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap="coolwarm", s=10, alpha=0.7)
ax.set_title("XOR continu")
plt.show()
```

<!-- #region -->
## 5. Définition du modèle : Sequential et subclassing
<!-- #endregion -->

<!-- #region -->
Keras 3 offre deux styles de construction de modèle :

- **`Sequential`** : empilement déclaratif de couches — concis, idéal pour les architectures
  linéaires.
- **Subclassing `keras.Model`** : on définit `__init__` (les couches) et `call` (le forward) —
  plus flexible (branches, logique conditionnelle).

**Point important** : la dernière couche renvoie des **logits** (pas d'activation `softmax`). On
appliquera ensuite la loss avec `from_logits=True` — c'est numériquement plus stable, et ça évite
le bug classique du **double softmax** (softmax dans le modèle *et* dans la loss).
<!-- #endregion -->

```python
def build_mlp_sequential(input_dim: int = 2, n_classes: int = 2) -> keras.Model:
    """MLP via l'API Sequential. Sortie en LOGITS (pas de softmax final)."""
    return keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(n_classes),  # logits
        ],
        name="mlp_sequential",
    )


mlp_seq = build_mlp_sequential()
mlp_seq.summary()
```

<!-- #region -->
La même architecture en subclassing. Un premier appel sur un batch construit (lazy-build) les
couches, ce qui rend `summary()` informatif.
<!-- #endregion -->

```python
class MLPSubclass(keras.Model):
    """Même MLP via subclassing keras.Model (API flexible). Sortie LOGITS."""

    def __init__(self, n_classes: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d1 = keras.layers.Dense(16, activation="relu")
        self.d2 = keras.layers.Dense(16, activation="relu")
        self.out = keras.layers.Dense(n_classes)

    def call(self, x):
        return self.out(self.d2(self.d1(x)))


mlp_sub = MLPSubclass()
_ = mlp_sub(ops.convert_to_tensor(X_xor[:1]))  # un forward construit les couches
mlp_sub.summary()
```

<!-- #region -->
## 6. Données : keras.utils.PyDataset
<!-- #endregion -->

<!-- #region -->
**`keras.utils.PyDataset`** est l'abstraction de chargement par lots de Keras 3 (elle remplace
`keras.utils.Sequence`, déprécié). On implémente `__len__` (nombre de batches) et `__getitem__`
(le batch d'indice `idx`). Elle gère le multiprocessing (`workers`, `use_multiprocessing`) et
s'accepte directement par `model.fit`. Alternative : un pipeline `tf.data.Dataset` (interopérable
sur backend TF).
<!-- #endregion -->

```python
class ArrayDataset(keras.utils.PyDataset):
    """Batche un couple (X, y) numpy. Remplace keras.utils.Sequence (déprécié)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, **kwargs) -> None:
        super().__init__(**kwargs)
        self.X, self.y, self.batch_size = X, y, batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.X[sl], self.y[sl]


xor_ds = ArrayDataset(X_xor, y_xor, batch_size=32)
_xb, _yb = xor_ds[0]
print(f"PyDataset : {len(xor_ds)} batches, 1er batch X={_xb.shape} y={_yb.shape}")
```

<!-- #region -->
## 7. Loss, optimisation & scheduling du learning rate
<!-- #endregion -->

<!-- #region -->
On compile le modèle avec une **loss** (`SparseCategoricalCrossentropy(from_logits=True)` pour des
labels entiers + sortie logits) et un **optimiseur**. Le **learning rate** peut suivre un
*schedule* : ici un **cosine decay**

$$\eta_t = \eta_0 \cdot \tfrac{1}{2}\left(1 + \cos\left(\pi \tfrac{t}{T}\right)\right)$$

qui décroît doucement de $\eta_0$ vers 0 sur $T$ steps. On passe le schedule directement comme
`learning_rate` de l'optimiseur.
<!-- #endregion -->

```python
steps_per_epoch = len(xor_ds)
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2, decay_steps=steps_per_epoch * 20
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

mlp_seq.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

<!-- #region -->
On peut visualiser le profil du learning rate programmé sur la durée de l'entraînement.
<!-- #endregion -->

```python
_steps = np.arange(steps_per_epoch * 20)
_lrs = [float(lr_schedule(s)) for s in _steps]
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(_steps, _lrs)
ax.set(title="CosineDecay LR", xlabel="step", ylabel="lr")
plt.show()
```

<!-- #region -->
## 8. Entraînement standard : model.fit
<!-- #endregion -->

<!-- #region -->
`model.fit` est l'idiome haut niveau de Keras 3 : il gère la boucle d'epochs, le batching, la
validation (`validation_split` ou `validation_data`) et renvoie un objet **`History`** contenant
les métriques par epoch.
<!-- #endregion -->

```python
history = mlp_seq.fit(X_xor, y_xor, validation_split=0.2, epochs=20, batch_size=32, verbose=0)
print("XOR fit — val_accuracy finale :", round(float(history.history["val_accuracy"][-1]), 3))
```

<!-- #region -->
On trace les courbes de loss et d'accuracy (train vs validation) depuis `history.history`.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set(title="loss", xlabel="epoch")
axes[0].legend()
axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set(title="accuracy", xlabel="epoch")
axes[1].legend()
plt.show()
```

<!-- #region -->
## 9. Boucle manuelle & gestion fine du batch
<!-- #endregion -->

<!-- #region -->
Parfois on veut **contrôler la composition des batches** (déséquilibre de classes, ordre des
exemples). Keras 3 offre deux voies haut niveau, **sans descendre au `GradientTape`** :

1. **`model.train_on_batch(x, y)`** : un pas de gradient sur un batch fourni à la main.
2. **Sous-classe `keras.Model` avec `train_step` custom** : on personnalise un pas
   d'entraînement tout en laissant Keras gérer le gradient (on délègue à `super().train_step`).

On illustre sur un dataset **synthétique déséquilibré** (3 classes, proportions 50/30/20 %).
<!-- #endregion -->

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

N_FEATURES, N_CLASSES = 10, 3
X_imb, y_imb = make_classification(
    n_samples=5000,
    n_features=N_FEATURES,
    n_informative=4,
    n_redundant=0,
    n_classes=N_CLASSES,
    n_clusters_per_class=1,
    weights=[0.5, 0.3, 0.2],
    random_state=SEED,
)
X_imb = X_imb.astype("float32")
Xtr, Xval, ytr, yval = train_test_split(X_imb, y_imb, test_size=0.33, random_state=SEED)
print("Synthétique déséquilibré:", np.bincount(y_imb))
```

<!-- #region -->
Trois utilitaires de composition de batch : down-sampling équilibré (folds avec autant
d'exemples par classe), mélange synchronisé de `(X, y)`, et tri par classe en **round-robin**
(`0,1,2,0,1,2,…`). Ce dernier est une réécriture vectorisée et typée de l'ancien `Stort_modulo`
(qui était en O(n²) avec un `try/except: pass` masquant les erreurs).
<!-- #endregion -->

```python
def downsample_balanced(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    """Construit k folds d'indices équilibrés par classe (down-sampling sur la classe min)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts.min()
    per_class = [np.where(y == c)[0] for c in classes]
    folds = []
    for _ in range(k):
        idx = np.concatenate([rng.choice(ci, n_min, replace=False) for ci in per_class])
        rng.shuffle(idx)
        folds.append(idx)
    return folds


def shuffle_xy(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mélange X et y de façon synchronisée."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(y))
    return X[p], y[p]


def sort_by_class_roundrobin(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Réordonne (X, y) en round-robin par classe : 0,1,2,0,1,2,...

    Réécriture vectorisée et typée de l'ancien `Stort_modulo` (O(n^2) + try/except:pass).
    Les classes sont prises à tour de rôle jusqu'à épuisement.
    """
    classes = np.unique(y)
    buckets = [list(np.where(y == c)[0]) for c in classes]
    order: list[int] = []
    while any(buckets):
        for b in buckets:
            if b:
                order.append(b.pop(0))
    order_arr = np.array(order, dtype=int)
    return X[order_arr], y[order_arr]
```

<!-- #region -->
On consolide les quatre variantes de l'original (équilibré/trié × nombre-de-batches/taille-de-batch)
en **une seule fonction paramétrée**, alimentée par `train_on_batch`.
<!-- #endregion -->

```python
def make_imb_model() -> keras.Model:
    """Petit MLP 3 classes, sortie LOGITS."""
    m = keras.Sequential(
        [
            keras.Input(shape=(N_FEATURES,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(N_CLASSES),
        ]
    )
    m.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def train_manual_batches(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    epochs: int = 5,
    mode: str = "balanced",
    by: str = "n_batches",
    n_batches: int = 9,
    batch_size: int = 64,
    seed: int = 0,
) -> keras.Model:
    """Boucle d'entraînement manuelle via model.train_on_batch.

    Args:
        mode: "balanced" (down-sampling équilibré par epoch) ou "sorted" (tri round-robin).
        by:   "n_batches" (nb fixe de batches) ou "batch_size" (découpe par taille).
    Returns:
        le modèle entraîné.
    """
    model = make_imb_model()
    folds = downsample_balanced(ytr, epochs, seed=seed) if mode == "balanced" else None
    for epoch in range(epochs):
        if mode == "balanced":
            Xe, ye = Xtr[folds[epoch]], ytr[folds[epoch]]
        else:  # sorted
            Xe, ye = shuffle_xy(Xtr, ytr, seed=seed + epoch)
            Xe, ye = sort_by_class_roundrobin(Xe, ye)

        losses = []
        if by == "n_batches":
            for _ in range(n_batches):
                metrics = model.train_on_batch(Xe, ye, return_dict=True)
                losses.append(metrics["loss"])
        else:  # batch_size
            for i in range(0, len(Xe), batch_size):
                metrics = model.train_on_batch(
                    Xe[i : i + batch_size], ye[i : i + batch_size], return_dict=True
                )
                losses.append(metrics["loss"])

        val = model.evaluate(Xval, yval, verbose=0, return_dict=True)
        print(
            f"  [{mode}/{by}] epoch {epoch + 1}/{epochs} - loss {np.mean(losses):.3f} - val_acc {val['accuracy']:.3f}"
        )
    return model


print("Variante équilibré / nb_batches :")
train_manual_batches(Xtr, ytr, Xval, yval, mode="balanced", by="n_batches", epochs=3)
print("Variante trié / batch_size :")
train_manual_batches(Xtr, ytr, Xval, yval, mode="sorted", by="batch_size", epochs=3, batch_size=64)
```

<!-- #region -->
La voie idiomatique Keras 3 : une sous-classe qui **surcharge `train_step`**. Ici on injecte à la
volée un `sample_weight` (inverse de fréquence de classe) puis on **délègue le calcul du gradient**
à `super().train_step` — backend-agnostique, sans manipuler le tape. C'est l'équivalent haut niveau
d'une boucle d'entraînement custom.
<!-- #endregion -->

```python
class WeightedTrainStepModel(keras.Model):
    """Sous-classe keras.Model avec train_step custom — idiome haut niveau Keras 3.

    On NE descend PAS au GradientTape : on délègue le calcul du gradient à
    super().train_step() après avoir injecté un sample_weight calculé à la volée
    (inverse de fréquence de classe). Backend-agnostique.
    """

    def __init__(self, n_classes: int, class_freq: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_classes = n_classes
        inv = (1.0 / class_freq).astype("float32")
        self._class_w = ops.convert_to_tensor(inv / inv.mean())
        self.d1 = keras.layers.Dense(64, activation="relu")
        self.d2 = keras.layers.Dense(32, activation="relu")
        self.out = keras.layers.Dense(n_classes)

    def call(self, x):
        return self.out(self.d2(self.d1(x)))

    def train_step(self, data):
        x, y = data
        sw = ops.take(self._class_w, ops.cast(ops.reshape(y, (-1,)), "int32"))
        return super().train_step((x, y, sw))


_freq = np.bincount(ytr, minlength=N_CLASSES) / len(ytr)
custom_model = WeightedTrainStepModel(N_CLASSES, _freq)
custom_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
custom_model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=3, batch_size=64, verbose=0)
print(
    "train_step custom — val_acc :",
    round(float(custom_model.evaluate(Xval, yval, verbose=0, return_dict=True)["accuracy"]), 3),
)
```

<!-- #region -->
## 10. Sauvegarde & chargement : model.save('.keras')
<!-- #endregion -->

<!-- #region -->
Le format **`.keras`** est le format natif portable de Keras 3 : il sérialise l'architecture, les
poids et la configuration de compilation dans un seul fichier. On recharge avec
`keras.models.load_model` et on vérifie que les prédictions sont identiques. (Alternatives :
`model.save_weights(...)` pour les poids seuls, ou l'export SavedModel pour le déploiement TF.)
<!-- #endregion -->

```python
save_path = "mlp_xor.keras"
mlp_seq.save(save_path)
reloaded = keras.models.load_model(save_path)

_p1 = ops.convert_to_numpy(mlp_seq(X_xor[:5]))
_p2 = ops.convert_to_numpy(reloaded(X_xor[:5]))
print("save/load — prédictions identiques :", np.allclose(_p1, _p2, atol=1e-5))
```

<!-- #region -->
## 11. Évaluation & métrique personnalisée
<!-- #endregion -->

<!-- #region -->
`model.evaluate` calcule loss et métriques sur un jeu de test. Pour une métrique non fournie
nativement, on sous-classe **`keras.metrics.Metric`** : on accumule des états (`update_state`),
on les agrège (`result`) et on les remet à zéro entre epochs (`reset_state`). Ici un **F1 macro**
écrit avec `keras.ops` (donc portable), via les comptes TP/FP/FN par classe.
<!-- #endregion -->

```python
class MacroF1(keras.metrics.Metric):
    """F1 macro calculée à la main via keras.ops (états TP/FP/FN par classe)."""

    def __init__(self, n_classes: int, name: str = "macro_f1", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.n_classes = n_classes
        self.tp = self.add_weight(name="tp", shape=(n_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(n_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(n_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = ops.cast(ops.reshape(y_true, (-1,)), "int32")
        y_pred = ops.argmax(y_pred, axis=-1)
        t = ops.one_hot(y_true, self.n_classes)
        p = ops.one_hot(ops.cast(y_pred, "int32"), self.n_classes)
        self.tp.assign_add(ops.sum(t * p, axis=0))
        self.fp.assign_add(ops.sum((1 - t) * p, axis=0))
        self.fn.assign_add(ops.sum(t * (1 - p), axis=0))

    def result(self):
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return ops.mean(f1)

    def reset_state(self) -> None:
        for v in (self.tp, self.fp, self.fn):
            v.assign(ops.zeros((self.n_classes,)))


f1_model = make_imb_model()
f1_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy", MacroF1(N_CLASSES)],
)
f1_model.fit(Xtr, ytr, epochs=3, batch_size=64, verbose=0)
_eval = f1_model.evaluate(Xval, yval, verbose=0, return_dict=True)
print("évaluation — accuracy:", round(_eval["accuracy"], 3), "| macro_f1:", round(_eval["macro_f1"], 3))
```

<!-- #region -->
## 12. Gestion du déséquilibre : class/sample weights & focal loss
<!-- #endregion -->

<!-- #region -->
Trois leviers contre le déséquilibre de classes :

- **`class_weight`** : un poids par classe (passé à `fit`), pour pénaliser davantage les erreurs
  sur les classes rares.
- **`sample_weight`** : un poids par échantillon (plus granulaire).
- **Focal loss** : module la cross-entropy pour se concentrer sur les exemples difficiles —
  $\mathrm{FL} = (1 - p_t)^{\gamma}\,(-\log p_t)$, où $p_t$ est la probabilité de la vraie classe.
  Plus $\gamma$ est grand, plus les exemples bien classés (grand $p_t$) sont atténués.

On commence par `class_weight` (calculé par `compute_class_weight`).
<!-- #endregion -->

```python
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

classes = np.unique(ytr)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
print("class_weight :", {k: round(v, 2) for k, v in class_weight.items()})

m_cw = make_imb_model()
m_cw.fit(Xtr, ytr, class_weight=class_weight, epochs=3, batch_size=64, verbose=0)
```

<!-- #region -->
`sample_weight` : un poids par échantillon, calculé ici par `compute_sample_weight` (équilibrage),
passé directement à `fit`.
<!-- #endregion -->

```python
sw = compute_sample_weight(class_weight="balanced", y=ytr).astype("float32")
m_sw = make_imb_model()
m_sw.fit(Xtr, ytr, sample_weight=sw, epochs=3, batch_size=64, verbose=0)
print("class/sample weights : fit OK")
```

<!-- #region -->
Focal loss en **deux versions** : (1) la version **native** de Keras 3,
`keras.losses.CategoricalFocalCrossentropy` (attend des labels one-hot) ; (2) une version
**écrite à la main en `keras.ops`** pour des labels entiers (il n'existe pas de variante "sparse"
native) — ce qui montre au passage la portabilité du code mathématique.
<!-- #endregion -->

```python
from keras.utils import to_categorical

ytr_oh = to_categorical(ytr, N_CLASSES)
m_focal_native = keras.Sequential(
    [keras.Input(shape=(N_FEATURES,)), keras.layers.Dense(64, activation="relu"), keras.layers.Dense(N_CLASSES)]
)
m_focal_native.compile(
    optimizer="adam",
    loss=keras.losses.CategoricalFocalCrossentropy(from_logits=True, gamma=2.0),
    metrics=["accuracy"],
)
m_focal_native.fit(Xtr, ytr_oh, epochs=3, batch_size=64, verbose=0)


class SparseFocalLoss(keras.losses.Loss):
    """Focal loss multiclasse pour labels entiers, écrite en keras.ops (portable).

    FL = (1 - p_t)^gamma * (-log p_t), p_t = proba de la vraie classe.
    """

    def __init__(self, n_classes: int, gamma: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_classes, self.gamma = n_classes, gamma

    def call(self, y_true, y_pred_logits):
        probs = ops.softmax(y_pred_logits, axis=-1)
        y_oh = ops.one_hot(ops.cast(ops.reshape(y_true, (-1,)), "int32"), self.n_classes)
        p_t = ops.sum(probs * y_oh, axis=-1)
        ce = -ops.log(p_t + 1e-7)
        return ops.power(1.0 - p_t, self.gamma) * ce


m_focal_scratch = make_imb_model()
m_focal_scratch.compile(optimizer="adam", loss=SparseFocalLoss(N_CLASSES, gamma=2.0), metrics=["accuracy"])
m_focal_scratch.fit(Xtr, ytr, epochs=3, batch_size=64, verbose=0)
print("focal native + from-scratch keras.ops : fit OK")
```

<!-- #region -->
## 13. Régularisation : dropout, batch norm, weight decay, early stopping
<!-- #endregion -->

<!-- #region -->
Quatre techniques pour limiter le sur-apprentissage :

- **Dropout** : désactive aléatoirement une fraction des neurones à l'entraînement (masque de
  Bernoulli) — un ensembling implicite.
- **Batch Normalization** : normalise les activations par batch, ce qui stabilise et accélère
  l'entraînement.
- **Weight decay** : pénalité L2 sur les poids, ici via **`AdamW`** (decay découplé du gradient).
- **Early stopping** : un *callback* qui arrête l'entraînement quand la `val_loss` cesse de
  s'améliorer (`patience` epochs) et restaure les meilleurs poids.
<!-- #endregion -->

```python
reg_model = keras.Sequential(
    [
        keras.Input(shape=(N_FEATURES,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(N_CLASSES),
    ]
)
reg_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reg_hist = reg_model.fit(
    Xtr, ytr, validation_data=(Xval, yval), epochs=30, batch_size=64, verbose=0, callbacks=[early]
)
print(f"régularisé — {len(reg_hist.history['loss'])}/30 epochs entraînées (EarlyStopping patience=3)")
```

<!-- #region -->
Courbe train/val : l'early stopping conserve les poids du minimum de `val_loss`.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(reg_hist.history["loss"], label="train")
ax.plot(reg_hist.history["val_loss"], label="val")
ax.set(title="EarlyStopping (val_loss)", xlabel="epoch")
ax.legend()
plt.show()
```

<!-- #region -->
## 14. Visualisation & logging
<!-- #endregion -->

<!-- #region -->
Deux outils : (a) la **visualisation des frontières de décision** (on évalue le modèle sur une
grille 2D), utile pour le XOR ; (b) le **logging TensorBoard** via le callback
`keras.callbacks.TensorBoard`.
<!-- #endregion -->

```python
def plot_decision_boundary(model: keras.Model, X: np.ndarray, y: np.ndarray) -> None:
    """Trace les frontières de décision d'un modèle 2D (sortie logits)."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].astype("float32")
    logits = model.predict(grid, verbose=0)
    zz = np.argmax(logits, axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.2)
    ax.set_title("Frontières de décision (XOR)")
    plt.show()


plot_decision_boundary(mlp_seq, X_xor, y_xor)
```

<!-- #region -->
Le callback `TensorBoard` écrit les logs (loss, métriques, graphe) dans un dossier. On l'ajoute à
`fit(callbacks=[...])`.
<!-- #endregion -->

```python
tb_cb = keras.callbacks.TensorBoard(log_dir="tb_logs", histogram_freq=0)
mlp_seq.fit(X_xor, y_xor, epochs=2, batch_size=32, verbose=0, callbacks=[tb_cb])
print("TensorBoard callback : logs écrits dans tb_logs/")
```

<!-- #region -->
Pour visualiser ces logs dans le notebook, exécuter manuellement les magics Jupyter suivants
(non inclus comme cellule de code car ce ne sont pas du Python) :

```text
%load_ext tensorboard
%tensorboard --logdir tb_logs
```
<!-- #endregion -->

<!-- #region -->
## 15. Cas réel — Classification MNIST
<!-- #endregion -->

<!-- #region -->
Pipeline complet de classification sur **MNIST** (chiffres manuscrits 28×28). On charge via
`keras.datasets.mnist`, on **sous-échantillonne** (~8k train / 2k test, pour tourner vite sur CPU)
et on normalise dans `[0, 1]`. On compare une **ANN** (Dense) et un **CNN** (le modèle principal,
réutilisé pour SHAP en section 17), puis matrice de confusion et ROC/AUC. Un bonus **LSTM**
traite l'image comme une séquence de lignes.
<!-- #endregion -->

```python
(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()

rng = np.random.default_rng(SEED)
tr_idx = rng.choice(len(X_train_full), 8000, replace=False)
te_idx = rng.choice(len(X_test_full), 2000, replace=False)
X_train = (X_train_full[tr_idx] / 255.0).astype("float32")
y_train = y_train_full[tr_idx].astype("int64")
X_test = (X_test_full[te_idx] / 255.0).astype("float32")
y_test = y_test_full[te_idx].astype("int64")
X_train_cnn = X_train[..., np.newaxis]  # channels-last
X_test_cnn = X_test[..., np.newaxis]
print("MNIST sous-échantillonné:", X_train.shape, X_test.shape)
```

<!-- #region -->
Aperçu de 10 images d'entraînement.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    axes[i].imshow(X_train[i], cmap="gray")
    axes[i].axis("off")
plt.show()
```

<!-- #region -->
**ANN** : `Flatten` + couches denses, sortie logits. Compilée avec `from_logits=True` (pas de
softmax dans le modèle — correction du bug de double softmax de l'original).
<!-- #endregion -->

```python
ann = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10),
    ]
)
ann.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
ann.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=128, verbose=0)
print("ANN MNIST — test acc:", round(float(ann.evaluate(X_test, y_test, verbose=0, return_dict=True)["accuracy"]), 3))
```

<!-- #region -->
**CNN** : `Conv2D` + `MaxPooling2D`, l'architecture naturelle pour des images. C'est le modèle
principal, conservé pour l'explicabilité (section 17).
<!-- #endregion -->

```python
cnn = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ]
)
cnn.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
cnn.fit(X_train_cnn, y_train, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
print("CNN MNIST — test acc:", round(float(cnn.evaluate(X_test_cnn, y_test, verbose=0, return_dict=True)["accuracy"]), 3))
```

<!-- #region -->
**Matrice de confusion** : on applique le softmax **une seule fois** sur les logits (via
`keras.ops.softmax`) pour obtenir les probabilités, puis `argmax`.
<!-- #endregion -->

```python
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

logits_test = cnn.predict(X_test_cnn, verbose=0)
proba_test = ops.convert_to_numpy(ops.softmax(ops.convert_to_tensor(logits_test), axis=-1))
y_pred = np.argmax(proba_test, axis=1)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set(title="Matrice de confusion (CNN MNIST)", xlabel="prédit", ylabel="réel")
fig.colorbar(im, ax=ax)
plt.show()
```

<!-- #region -->
**ROC / AUC en one-vs-rest** : pour un problème à 10 classes, on binarise les labels et on trace
une courbe ROC par classe ; l'AUC macro est la moyenne des AUC par classe.
<!-- #endregion -->

```python
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=list(range(10)))
macro_auc = roc_auc_score(y_test_bin, proba_test, multi_class="ovr", average="macro")

fig, ax = plt.subplots(figsize=(6, 5))
for c in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, c], proba_test[:, c])
    ax.plot(fpr, tpr, lw=0.8, alpha=0.6)
ax.plot([0, 1], [0, 1], "k--", lw=0.5)
ax.set(title=f"ROC OVR — macro AUC = {macro_auc:.3f}", xlabel="FPR", ylabel="TPR")
plt.show()
print("CNN MNIST — macro AUC (OVR):", round(float(macro_auc), 3))
```

<!-- #region -->
**Bonus — LSTM** : on traite chaque image comme une **séquence de 28 lignes** de 28 pixels. C'est
un usage détourné (les images ne sont pas des séquences temporelles), mais il illustre une couche
récurrente sur des données 2D.
<!-- #endregion -->

```python
rnn = keras.Sequential(
    [
        keras.Input(shape=(28, 28)),
        keras.layers.LSTM(48),
        keras.layers.Dense(10),
    ]
)
rnn.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
rnn.fit(X_train, y_train, epochs=2, batch_size=128, verbose=0)
print("LSTM bonus MNIST — test acc:", round(float(rnn.evaluate(X_test, y_test, verbose=0, return_dict=True)["accuracy"]), 3))
```

<!-- #region -->
## 16. Cas réel — Régression California Housing
<!-- #endregion -->

<!-- #region -->
Pipeline de **régression** sur **California Housing** (prix médian des logements). On standardise
les features (`StandardScaler`), on construit un MLP à **sortie linéaire** (1 neurone, **pas de
dropout terminal** — une erreur classique), on entraîne avec early stopping, puis on évalue avec
**RMSE / MAE / R²**.
<!-- #endregion -->

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

cal = fetch_california_housing()
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    cal.data.astype("float32"), cal.target.astype("float32"), test_size=0.2, random_state=SEED
)
scaler = StandardScaler()
Xc_tr = scaler.fit_transform(Xc_tr).astype("float32")
Xc_te = scaler.transform(Xc_te).astype("float32")

reg = keras.Sequential(
    [
        keras.Input(shape=(Xc_tr.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),  # sortie linéaire (pas de dropout terminal)
    ]
)
reg.compile(optimizer="adam", loss="mse", metrics=["mae"])
reg.fit(
    Xc_tr,
    yc_tr,
    validation_split=0.1,
    epochs=50,
    batch_size=128,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
)
yc_pred = reg.predict(Xc_te, verbose=0).ravel()
rmse = math.sqrt(mean_squared_error(yc_te, yc_pred))
print(f"Régression — RMSE {rmse:.3f} | MAE {mean_absolute_error(yc_te, yc_pred):.3f} | R2 {r2_score(yc_te, yc_pred):.3f}")
```

<!-- #region -->
Nuage prédit vs réel : plus les points sont proches de la diagonale, meilleure est la régression.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(yc_te, yc_pred, s=5, alpha=0.3)
ax.plot([yc_te.min(), yc_te.max()], [yc_te.min(), yc_te.max()], "r--", lw=1)
ax.set(title="Prédit vs réel (California Housing)", xlabel="réel", ylabel="prédit")
plt.show()
```

<!-- #region -->
## 17. Explainability — SHAP GradientExplainer
<!-- #endregion -->

<!-- #region -->
**SHAP** attribue à chaque pixel sa contribution à la prédiction. Sur Keras 3, l'ancien
`DeepExplainer` est **cassé** ; on utilise **`GradientExplainer`** (expected gradients), qui
estime les attributions à partir des gradients du modèle sur un jeu de fond (*background*). On
explique le **CNN** entraîné en section 15 (réutilisé, pas de ré-entraînement).
<!-- #endregion -->

```python
import shap

background = X_train_cnn[rng.choice(len(X_train_cnn), 100, replace=False)]
test_images = X_test_cnn[:4]

explainer = shap.GradientExplainer(cnn, background)
shap_values = explainer.shap_values(test_images)
print("shap_values :", np.array(shap_values).shape)
```

<!-- #region -->
`shap.image_plot` affiche, pour chaque image test, les attributions par classe (colonnes 0→9).
Les pixels en rouge poussent vers la classe, en bleu l'inverse.
<!-- #endregion -->

```python
# GradientExplainer renvoie un ndarray (n, H, W, C, n_classes). image_plot attend
# une LISTE d'une carte par classe ; sans ce découpage, une seule colonne s'affiche.
if isinstance(shap_values, list):
    shap_per_class = shap_values  # versions plus anciennes : déjà une liste
else:
    sv = np.asarray(shap_values)
    shap_per_class = [sv[..., k] for k in range(sv.shape[-1])]
# chaque ligne = une image de test ; les 10 colonnes = attributions des classes 0-9.
shap.image_plot(shap_per_class, -test_images)
```

<!-- #region -->
## Conclusion
<!-- #endregion -->

<!-- #region -->
Ce notebook a déroulé les 17 sections du squelette commun en idiomes **Keras 3 haut niveau**. Ce
qui le distingue des notebooks PyTorch et TensorFlow :

- **`keras.ops`** : un seul code mathématique, portable sur tous les backends.
- **`compile` / `fit` / `callbacks`** : la boucle d'entraînement standardisée (vs le
  `tf.GradientTape` bas niveau du notebook TensorFlow).
- **`KERAS_BACKEND`** : on change de moteur (TF / JAX / PyTorch / OpenVINO) sans toucher au modèle.
- **`model.save('.keras')`** : un format de sérialisation natif et portable.

Correspondance rapide des idiomes vus :

| Besoin | PyTorch | TensorFlow (bas niveau) | **Keras 3** |
|---|---|---|---|
| Tenseurs | `torch.Tensor` | `tf.constant` | **`keras.ops`** |
| Modèle | `nn.Module` | `tf.Module` / subclass | **`Sequential` / `keras.Model`** |
| Données | `DataLoader` | `tf.data.Dataset` | **`PyDataset`** |
| Entraînement | boucle custom | `@tf.function` + `GradientTape` | **`model.fit`** |
| Boucle manuelle | sampler + loop | `GradientTape` | **`train_on_batch` / `train_step`** |
| Save | `state_dict` | Checkpoint / SavedModel | **`.keras`** |
| Métrique custom | manuelle | `tf.keras.metrics.Metric` | **`keras.metrics.Metric`** |
| SHAP | `DeepExplainer` | `GradientExplainer` | **`GradientExplainer`** |

Pour aller plus loin : le notebook `DL_Frameworks_Comparatif` rejoue MNIST et California Housing
en parallèle sur les frameworks, et `DL_JAX` couvre l'approche fonctionnelle (Flax/Optax).
<!-- #endregion -->
