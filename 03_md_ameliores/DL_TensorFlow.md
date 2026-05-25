---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 🟧 TensorFlow — Framework DL
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **TensorFlow 2.x** (2026). Suit le **blueprint commun** des notebooks DL framework (cf `00_blueprint_DL_frameworks.md`) — 16 sections pour comparer côte à côte avec `DL_PyTorch`, `DL_Keras`, `DL_JAX`.

**Datasets utilisés (mutualisés) :** XOR continu (toy), MNIST (classification, section 15), California Housing (régression, section 16).

> **Note 2026** : Keras 3 multi-backend (TF/JAX/PyTorch) supplante largement l'API TF "pure" pour le code applicatif — voir `DL_Keras`. Ce notebook focus sur l'usage des **APIs bas niveau TF** : `tf.Variable`, `tf.GradientTape`, `tf.function`, `tf.data` — utiles pour le custom training, le TPU, ou de la R&D framework-spécifique.
<!-- #endregion -->

<!-- #region -->
## 1. Présentation de TensorFlow (2026)
<!-- #endregion -->

<!-- #region -->
**TensorFlow** (Google, 2015) — historiquement *défini par graphe* (TF1.x), passé à l'eager execution + graphes compilés via `tf.function` (TF2.x). Reste pertinent en 2026 pour :

- **Production à grande échelle** : TFX (pipelines), TFLite (edge), TF Serving, **TPU** (le seul framework natif Google Cloud TPU).
- **Mobile** : TFLite est le standard Android/iOS.
- **Recommendation systems** : TensorFlow Recommenders, TFRS.

**Évolutions 2024-2026** :

- **Keras 3** (multi-backend) prend la place de l'API haut niveau. TF "pur" devient une option backend.
- `tf.function` + XLA pour compiler à la volée.
- **JAX dans TPU** est devenu équivalent — TF perd du terrain en R&D, mais résiste en infra historique Google.
<!-- #endregion -->

```python
import tensorflow as tf
import numpy as np

print(f"TF version : {tf.__version__}")
print(f"GPU dispo  : {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Eager mode : {tf.executing_eagerly()}")
```

<!-- #region -->
## 2. Tenseurs
<!-- #endregion -->

```python
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v = tf.Variable([[1.0, 2.0], [3.0, 4.0]])   # trainable

print(f"x shape : {x.shape}  dtype : {x.dtype}  device : {x.device}")
print(f"NumPy interop : x.numpy() type = {type(x.numpy())}")
print(f"x @ x.T =\n{x @ tf.transpose(x)}")
```

<!-- #region -->
## 3. GPU / accélérateur
<!-- #endregion -->

```python
# Contrôle device explicite
with tf.device("/CPU:0"):
    a = tf.random.normal([3, 3])
print(f"a sur : {a.device}")

# Reproductibilité
tf.random.set_seed(42)
np.random.seed(42)
```

<!-- #region -->
## 4. Cas synthétique simple — XOR continu
<!-- #endregion -->

```python
def gen_xor(n=300):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int32)
    return tf.constant(X), tf.constant(y)

X_xor, y_xor = gen_xor(300)
print(f"X={X_xor.shape}  y={y_xor.shape}")
```

<!-- #region -->
## 5. Définition de modèle
<!-- #endregion -->

<!-- #region -->
En TF, 3 conventions :

- **Sequential** : empilement linéaire.
- **Functional API** : pour graphes complexes (skip, multi-input/output).
- **Subclassing** : `tf.keras.Model` custom (équivalent `nn.Module`).
<!-- #endregion -->

```python
class MLP(tf.keras.Model):
    def __init__(self, d_hidden: int, d_out: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.d1 = tf.keras.layers.Dense(d_hidden, activation="relu")
        self.drop = tf.keras.layers.Dropout(dropout)
        self.d2 = tf.keras.layers.Dense(d_hidden, activation="relu")
        self.out = tf.keras.layers.Dense(d_out)  # logits

    def call(self, x, training=False):
        x = self.d1(x)
        x = self.drop(x, training=training)
        x = self.d2(x)
        return self.out(x)


model = MLP(d_hidden=32, d_out=2)
_ = model(X_xor[:1])  # build pour avoir les shapes
model.summary()
```

<!-- #region -->
## 6. Données — `tf.data.Dataset`
<!-- #endregion -->

```python
ds = tf.data.Dataset.from_tensor_slices((X_xor, y_xor))
ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
print(f"Dataset : {ds}")
xb, yb = next(iter(ds))
print(f"Batch : x={xb.shape}  y={yb.shape}")
```

<!-- #region -->
## 7. Loss & Optimisation
<!-- #endregion -->

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
print(f"Optimiseur : {type(optimizer).__name__}")
```

<!-- #region -->
## 8. Boucle d'entraînement standard (`model.fit`)
<!-- #endregion -->

```python
model_fit = MLP(d_hidden=32, d_out=2)
model_fit.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model_fit.fit(X_xor, y_xor, batch_size=32, epochs=15, verbose=0)
loss, acc = model_fit.evaluate(X_xor, y_xor, verbose=0)
print(f"Train acc XOR (fit) : {acc:.3f}")
```

<!-- #region -->
## 9. Boucle manuelle + `tf.GradientTape`
<!-- #endregion -->

<!-- #region -->
Pour du contrôle fin (custom loss, gradient clipping, gradient accumulation, training adversarial), on écrit la boucle à la main avec `tf.GradientTape`. C'est l'équivalent du `loss.backward()` de PyTorch.

L'idiome : décorer avec `@tf.function` pour compiler le graphe (gain de 2-10× vs eager).
<!-- #endregion -->

```python
model_manual = MLP(d_hidden=32, d_out=2)
optimizer = tf.keras.optimizers.AdamW(1e-2)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits = model_manual(xb, training=True)
        loss = loss_fn(yb, logits)
    grads = tape.gradient(loss, model_manual.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_manual.trainable_variables))
    return loss


for epoch in range(15):
    for xb, yb in ds:
        train_step(xb, yb)

acc_manual = (tf.argmax(model_manual(X_xor), 1).numpy() == y_xor.numpy()).mean()
print(f"Train acc XOR (manual) : {acc_manual:.3f}")
```

<!-- #region -->
## 10. Save / Load
<!-- #endregion -->

```python
import tempfile, os
ckpt_dir = os.path.join(tempfile.gettempdir(), "tf_demo.keras")

# Format Keras (recommandé depuis Keras 3 / TF 2.13+)
model_manual.save(ckpt_dir)
loaded = tf.keras.models.load_model(ckpt_dir, compile=False)
print(f"Model rechargé, output sample : {loaded(X_xor[:1]).numpy()}")
```

<!-- #region -->
## 11. Évaluation
<!-- #endregion -->

```python
# Avec model.evaluate (si compilé)
model_fit.evaluate(X_xor, y_xor, verbose=0)

# Manuellement
preds = tf.argmax(model_manual(X_xor), 1).numpy()
acc = (preds == y_xor.numpy()).mean()
print(f"Manual eval acc : {acc:.3f}")
```

<!-- #region -->
## 12. Gestion du déséquilibre
<!-- #endregion -->

<!-- #region -->
TF supporte 3 mécanismes :

- **`class_weight`** dans `model.fit(class_weight={0: w0, 1: w1})`.
- **`sample_weight`** par observation.
- **Pondération dans la loss** manuelle (custom loss subclass).

```python
# class_weight
# model.fit(X, y, class_weight={0: 1.0, 1: 5.0})

# sample_weight
# weights = np.where(y == 1, 5.0, 1.0)
# model.fit(X, y, sample_weight=weights)
```
<!-- #endregion -->

<!-- #region -->
## 13. Régularisation
<!-- #endregion -->

```python
# Dropout : tf.keras.layers.Dropout
# BatchNorm : tf.keras.layers.BatchNormalization
# LayerNorm : tf.keras.layers.LayerNormalization
# L1/L2 sur poids : kernel_regularizer=tf.keras.regularizers.L2(1e-4)
# Early stopping : callback dédié

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# model.fit(..., callbacks=[es], validation_split=0.2)
```

<!-- #region -->
## 14. Visualisation — TensorBoard
<!-- #endregion -->

```python
# Natif et excellent en TF
# tb = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
# model.fit(..., callbacks=[tb])
# Lancer : tensorboard --logdir logs
```

<!-- #region -->
## 15. Cas réel — Classification MNIST (digits 8x8)
<!-- #endregion -->

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_d = digits.data.astype(np.float32) / 16.0
y_d = digits.target.astype(np.int32)
X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10),
])
mnist_model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
mnist_model.fit(X_tr, y_tr, batch_size=64, epochs=10, verbose=0)
loss, acc = mnist_model.evaluate(X_te, y_te, verbose=0)
print(f"Test acc digits : {acc:.3f}")
```

<!-- #region -->
## 16. Cas réel — Régression California Housing
<!-- #endregion -->

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing(as_frame=True)
sc = StandardScaler()
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    sc.fit_transform(data.data).astype(np.float32),
    data.target.values.astype(np.float32),
    test_size=0.2, random_state=42,
)

reg = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_tr_r.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1),
])
reg.compile(optimizer=tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4), loss="mse")

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reg.fit(X_tr_r, y_tr_r, validation_split=0.1, batch_size=128, epochs=30, verbose=0, callbacks=[es])
preds = reg.predict(X_te_r, verbose=0).flatten()
rmse = float(np.sqrt(((preds - y_te_r) ** 2).mean()))
print(f"Test RMSE : {rmse:.4f}")
```

<!-- #region -->
## 17. Explainability (SHAP)
<!-- #endregion -->

<!-- #region -->
**SHAP `DeepExplainer`** supporte TF/Keras nativement (via gradients backprop). C'était son cas d'usage historique.

```python
# import shap
# bg = X_tr[:100]
# explainer = shap.DeepExplainer(mnist_model, bg)
# shap_values = explainer.shap_values(X_te[:5])
```
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- [TensorFlow — docs officielles](https://www.tensorflow.org/api_docs/python/tf)
- [TF Guide — tf.function](https://www.tensorflow.org/guide/function)
- [TFX — Production ML pipelines](https://www.tensorflow.org/tfx)
- [TFLite — Edge/Mobile deployment](https://www.tensorflow.org/lite)
- Notebooks liés : `DL_PyTorch`, `DL_Keras`, `DL_JAX`, `DL_Frameworks_Comparatif`, `DL_Deep_Learning_Maths`.
<!-- #endregion -->
