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
# 💎 Keras 3 — Framework DL multi-backend
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **Keras 3** (2024+, état 2026). Suit le **blueprint commun** des notebooks DL framework (cf `00_blueprint_DL_frameworks.md`).

**Datasets utilisés (mutualisés) :** XOR continu (toy), MNIST (digits 8x8 pour la démo), California Housing (régression).
<!-- #endregion -->

<!-- #region -->
## 1. Présentation de Keras 3 (2026)
<!-- #endregion -->

<!-- #region -->
**Keras** (François Chollet, 2015) — initialement surcouche haut-niveau pour TensorFlow. **Keras 3** (2024) est devenue **multi-backend** : un seul code Keras tourne sur TensorFlow, PyTorch, ou JAX au choix (changement de backend par variable d'env `KERAS_BACKEND`).

**Avantages 2026** :

- **Portabilité** : tu écris une fois, tu déploies sur tous les backends.
- **API stable et lisible** : `model.compile(...)` + `model.fit(...)` reste l'idiome le plus simple du DL.
- **`keras.ops`** : un namespace numpy-like commun aux 3 backends, permet d'écrire des layers custom portables.
- **Migration facile** : ton code Keras 2.x marche encore.

**Quand l'utiliser** :

- Recherche rapide / prototypage / pédagogie.
- Pour partager du code dans une équipe mixte (certains préfèrent torch, d'autres TF).
- Pour déployer le même modèle dans plusieurs environnements (TFLite côté mobile, TorchServe côté serveur).
<!-- #endregion -->

```python
# Choisir le backend AVANT d'importer keras (ici torch par exemple — pourrait être "tensorflow" ou "jax")
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
print(f"Keras  version : {keras.__version__}")
print(f"Backend        : {keras.config.backend()}")
```

<!-- #region -->
## 2. Tenseurs (`keras.ops`)
<!-- #endregion -->

<!-- #region -->
`keras.ops` est un namespace qui mime NumPy. Le code écrit avec lui marche sur les 3 backends sans changement.
<!-- #endregion -->

```python
import keras.ops as K

x = K.array([[1.0, 2.0], [3.0, 4.0]])
print(f"shape : {K.shape(x)}")
print(f"x @ x.T = {K.matmul(x, K.transpose(x))}")
print(f"Conversion NumPy : {K.convert_to_numpy(x)}")
```

<!-- #region -->
## 3. GPU / accélérateur
<!-- #endregion -->

<!-- #region -->
Géré par le backend. En général aucun code Keras à changer. Les hyper-decorators (`jax.jit`, `tf.function`, `torch.compile`) sont appliqués automatiquement par Keras quand pertinent.
<!-- #endregion -->

<!-- #region -->
## 4. Cas synthétique simple — XOR continu
<!-- #endregion -->

```python
def gen_xor(n=300):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int32)
    return X, y

X_xor, y_xor = gen_xor(300)
print(f"X={X_xor.shape}  y={y_xor.shape}")
```

<!-- #region -->
## 5. Définition de modèle
<!-- #endregion -->

```python
from keras import layers, models

# 3 manières équivalentes :

# 1. Sequential (le plus simple)
model_seq = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(32, activation="relu"),
    layers.Dense(2),
])

# 2. Functional API (pour graphes non-linéaires)
inp = layers.Input(shape=(2,))
h = layers.Dense(32, activation="relu")(inp)
h = layers.Dropout(0.1)(h)
h = layers.Dense(32, activation="relu")(h)
out = layers.Dense(2)(h)
model_func = models.Model(inp, out)

# 3. Subclassing (pour modèles très custom)
class MLP(models.Model):
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(32, activation="relu")
        self.drop = layers.Dropout(0.1)
        self.d2 = layers.Dense(32, activation="relu")
        self.out = layers.Dense(2)
    def call(self, x, training=False):
        x = self.d1(x); x = self.drop(x, training=training); x = self.d2(x); return self.out(x)

model_seq.summary()
```

<!-- #region -->
## 6. Données
<!-- #endregion -->

<!-- #region -->
Keras accepte les arrays NumPy directement, mais on peut aussi passer un `tf.data.Dataset` (backend tf), un `torch.utils.data.DataLoader` (backend torch), ou un générateur Python.

Le standard cross-backend en 2026 : utiliser **NumPy arrays** ou un **`keras.utils.PyDataset`** custom (équivalent de `tf.data.Dataset` mais portable).
<!-- #endregion -->

<!-- #region -->
## 7. Loss & Optimisation
<!-- #endregion -->

```python
from keras import optimizers, losses, metrics

model_seq.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-2, weight_decay=1e-4),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[metrics.SparseCategoricalAccuracy(name="acc")],
)
print(f"Optimizer : {type(model_seq.optimizer).__name__}")
```

<!-- #region -->
## 8. Boucle d'entraînement standard
<!-- #endregion -->

```python
hist = model_seq.fit(X_xor, y_xor, batch_size=32, epochs=15, verbose=0)
print(f"Train acc XOR : {hist.history['acc'][-1]:.3f}")
```

<!-- #region -->
## 9. Boucle manuelle + custom training
<!-- #endregion -->

<!-- #region -->
Pour overrider la boucle, on subclasse `Model` et on override `train_step`. C'est l'équivalent portable du `tf.GradientTape` ou du `loss.backward()`.

```python
class CustomMLP(models.Model):
    def train_step(self, data):
        x, y = data
        # Backend-specific gradient handling
        # Sur tf:    avec GradientTape ; sur torch: avec autograd ; sur jax: via grad_fn
        # mais Keras t'en abstrait via self.optimizer.apply_gradients / equivalent
        # En général : pas besoin sauf cas custom (GAN, distillation, ...)
        ...
```
<!-- #endregion -->

<!-- #region -->
## 10. Save / Load
<!-- #endregion -->

```python
import tempfile, os
path = os.path.join(tempfile.gettempdir(), "demo_keras3.keras")
model_seq.save(path)
loaded = keras.models.load_model(path)
print(f"Reloaded model preds : {loaded.predict(X_xor[:2], verbose=0).shape}")
```

<!-- #region -->
## 11. Évaluation
<!-- #endregion -->

```python
loss, acc = model_seq.evaluate(X_xor, y_xor, verbose=0)
print(f"Eval : loss={loss:.4f}  acc={acc:.3f}")
```

<!-- #region -->
## 12. Gestion du déséquilibre
<!-- #endregion -->

```python
# class_weight ou sample_weight directement à fit
# model.fit(X, y, class_weight={0: 1.0, 1: 5.0})
# model.fit(X, y, sample_weight=weights)
```

<!-- #region -->
## 13. Régularisation
<!-- #endregion -->

```python
# Dropout : layers.Dropout(rate)
# BatchNorm : layers.BatchNormalization()
# LayerNorm : layers.LayerNormalization()
# L2 sur poids : layers.Dense(64, kernel_regularizer=keras.regularizers.L2(1e-4))
# Early stopping : keras.callbacks.EarlyStopping(...)
```

<!-- #region -->
## 14. Visualisation — TensorBoard / callbacks
<!-- #endregion -->

```python
# TensorBoard callback portable
# tb = keras.callbacks.TensorBoard(log_dir="logs/")
# Reduce LR on plateau
# rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
# ModelCheckpoint
# cp = keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss", save_best_only=True)
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

mnist_model = models.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(64, activation="relu"),
    layers.Dense(10),
])
mnist_model.compile(
    optimizer=optimizers.AdamW(1e-3),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["sparse_categorical_accuracy"],
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

reg = models.Sequential([
    layers.Input(shape=(X_tr_r.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(64, activation="relu"),
    layers.Dense(1),
])
reg.compile(optimizer=optimizers.AdamW(1e-3, weight_decay=1e-4), loss="mse", metrics=["mae"])

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reg.fit(X_tr_r, y_tr_r, validation_split=0.1, batch_size=128, epochs=30, verbose=0, callbacks=[es])
preds = reg.predict(X_te_r, verbose=0).flatten()
rmse = float(np.sqrt(((preds - y_te_r) ** 2).mean()))
print(f"Test RMSE : {rmse:.4f}")
```

<!-- #region -->
## 17. Explainability
<!-- #endregion -->

<!-- #region -->
Selon le backend choisi, SHAP utilise `DeepExplainer` (TF) ou `GradientExplainer` (PyTorch).

```python
# import shap
# bg = X_tr[:100]
# explainer = shap.GradientExplainer(model, bg)  # marche pour Keras 3 + torch backend
# shap_values = explainer.shap_values(X_te[:10])
```
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- [Keras 3 — docs officielles](https://keras.io/)
- [Keras 3 — migration guide from Keras 2](https://keras.io/keras_3/)
- [Keras Core — annonce multi-backend](https://keras.io/keras_core/)
- Notebooks liés : `DL_PyTorch`, `DL_TensorFlow`, `DL_JAX`, `DL_Frameworks_Comparatif`, `DL_Deep_Learning_Maths`.
<!-- #endregion -->
