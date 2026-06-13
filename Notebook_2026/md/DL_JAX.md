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
# ⚡ JAX (+ Flax) — Framework DL fonctionnel
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **JAX** et **Flax** (l'API haut niveau de modèles neuronaux). Suit le **blueprint commun** des notebooks DL framework (cf `00_blueprint_DL_frameworks.md`).

**Datasets utilisés (mutualisés) :** XOR continu, MNIST (digits 8x8), California Housing.

> **Note d'exécution** : ce notebook utilise **JAX + Flax + Optax**. Si pas installés, la plupart des cellules sont en **pseudo-code** ou explication conceptuelle (les sections code "réelles" sont en NumPy équivalent quand JAX absent). Pour l'install complète : `uv add jax flax optax`.
<!-- #endregion -->

<!-- #region -->
## 1. Présentation de JAX (2026)
<!-- #endregion -->

<!-- #region -->
**JAX** (Google Research, 2018+) — librairie de **calcul numérique fonctionnel** avec :

- **`jax.numpy`** : drop-in remplacement de NumPy.
- **`jax.grad`** : différenciation automatique (autograd).
- **`jax.jit`** : compilation JIT vers XLA (CPU/GPU/TPU).
- **`jax.vmap`** : vectorisation automatique (batching d'une fonction scalaire).
- **`jax.pmap`** : parallélisme multi-device.

**Philosophie** : **fonctionnel pur** — pas d'état mutable, les modèles sont des **fonctions**, les paramètres sont des **arguments**. Cela ouvre des superpouvoirs (vmap, grad-of-grad, transformations composables) mais demande un changement de paradigme.

**Écosystème 2026** :

- **Flax** (Google) — l'API haut-niveau dominante pour les modèles neuronaux JAX. Module-based, gère les params/buffers proprement.
- **Optax** (DeepMind) — optimiseurs (Adam, SGD, AdamW, Lion, ...).
- **Orbax** (Google) — sérialisation/checkpointing.
- **Equinox** (alternative à Flax) — encore plus fonctionnelle (PyTree-based).
- **PennyLane / NumPyro / Diffrax** — écosystème scientifique étendu.
- **JAX dans Keras 3** : on peut faire `KERAS_BACKEND=jax` et utiliser l'API Keras.

**Quand utiliser JAX** :

- Recherche en **optim avancée** (composer grad, vmap, jit).
- **Bayesian deep learning** (NumPyro).
- **Très grande échelle TPU** (les TPUs Google sont 1st class).
- Modèles très **fonctionnels** (réseau bayésien, méta-learning).

**Quand NE PAS** : si tu débutes en DL ou tu veux juste fine-tuner un BERT. PyTorch est plus pédagogique et a beaucoup plus de tutoriels.
<!-- #endregion -->

<!-- #region -->
## 2. Tenseurs (`jax.numpy`)
<!-- #endregion -->

```python
# Démo conceptuelle — exécution réelle si JAX installé
"""
import jax
import jax.numpy as jnp

x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
print(f"shape : {x.shape}  dtype : {x.dtype}")
print(f"x @ x.T =\n{x @ x.T}")
print(f"JAX device : {x.device()}")
"""
# Note : les tableaux JAX sont IMMUTABLES.
# `x[0, 0] = 5` ne marche pas. Utiliser : `x = x.at[0, 0].set(5)`
```

<!-- #region -->
## 3. GPU / accélérateur
<!-- #endregion -->

<!-- #region -->
JAX détecte automatiquement le device : CPU / GPU / TPU. On peut forcer via `jax.device_put(x, device=...)`.

```python
# jax.devices()  -> liste des devices disponibles
# jax.local_device_count()  -> nombre de devices sur la machine
```

**Reproductibilité** : JAX utilise une **PRNG key explicite** (`jax.random.PRNGKey(seed)`) au lieu d'un état global. Chaque opération qui consomme de l'aléatoire prend une `key` en argument. Cela rend le code parfaitement reproductible et parallélisable.
<!-- #endregion -->

<!-- #region -->
## 4. Cas synthétique simple — XOR continu
<!-- #endregion -->

```python
import numpy as np

def gen_xor(n=300):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int32)
    return X, y

X_xor, y_xor = gen_xor(300)
print(f"X={X_xor.shape}  y={y_xor.shape}")
```

<!-- #region -->
## 5. Définition de modèle (Flax `nn.Module`)
<!-- #endregion -->

<!-- #region -->
Flax modules sont **purs** : la `__call__` ne porte pas de state, et les paramètres sont stockés séparément (en `FrozenDict`).

```python
"""
import flax.linen as nn

class MLP(nn.Module):
    d_hidden: int
    d_out: int
    dropout_rate: float = 0.0

    @nn.compact   # def des layers à l'intérieur de __call__
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.d_hidden)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.d_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.d_out)(x)
        return x


# Initialisation des paramètres : on passe une key et un input "dummy" pour shape inference
import jax
key = jax.random.PRNGKey(42)
model = MLP(d_hidden=32, d_out=2)
params = model.init(key, X_xor[:1])  # FrozenDict des poids
"""
```
<!-- #endregion -->

<!-- #region -->
## 6. Données
<!-- #endregion -->

<!-- #region -->
JAX n'a pas de DataLoader natif. Solutions :

- **`grain`** (Google) — DataLoader JAX-friendly.
- **`tf.data.Dataset`** comme source.
- Réutiliser **`torch.utils.data.DataLoader`** + conversion en JAX arrays à la volée.
- Pour datasets en RAM : juste un Python generator qui split en batches.

```python
def get_batches(X, y, batch_size: int, key):
    n = len(X)
    perm = jax.random.permutation(key, n)
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        yield X[idx], y[idx]
```
<!-- #endregion -->

<!-- #region -->
## 7. Loss & Optimisation (Optax)
<!-- #endregion -->

<!-- #region -->
````python
import optax

# Loss : combine softmax + CE
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


# Optimiseur Optax
optimizer = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)
opt_state = optimizer.init(params)
````
<!-- #endregion -->

<!-- #region -->
## 8. Boucle d'entraînement — pattern fonctionnel
<!-- #endregion -->

<!-- #region -->
**L'idiome JAX** : la `train_step` est une fonction pure JIT-compilée, qui prend params + batch et renvoie new params + loss. Pas de `model.fit()`.

```python
"""
@jax.jit
def train_step(params, opt_state, x, y, dropout_key):
    def loss_fn(params):
        logits = model.apply(params, x, training=True, rngs={"dropout": dropout_key})
        return cross_entropy_loss(logits, y)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# Boucle Python pure
key = jax.random.PRNGKey(0)
for epoch in range(20):
    key, sub_key, drop_key = jax.random.split(key, 3)
    for xb, yb in get_batches(X_xor, y_xor, 32, sub_key):
        params, opt_state, loss = train_step(params, opt_state, xb, yb, drop_key)
"""
```

**Pattern à retenir** : `jit + value_and_grad + functional optimizer update`. C'est le DL fonctionnel pur.
<!-- #endregion -->

<!-- #region -->
## 9. Custom training (déjà manuel — pas de fit)
<!-- #endregion -->

<!-- #region -->
JAX **n'a pas de `model.fit()`** — tout est manuel par design. Pour la gestion du déséquilibre, des batches custom, on modifie la `loss_fn` ou le `get_batches`.

Pour aller plus haut niveau : utiliser **Keras 3 avec backend JAX** (`KERAS_BACKEND=jax`) — tu retrouves `model.fit()` mais avec la perf JAX.
<!-- #endregion -->

<!-- #region -->
## 10. Save / Load (Orbax)
<!-- #endregion -->

<!-- #region -->
```python
"""
import orbax.checkpoint as ocp
ckptr = ocp.PyTreeCheckpointer()
ckptr.save("/tmp/jax_ckpt", params)
restored_params = ckptr.restore("/tmp/jax_ckpt")
"""
```
<!-- #endregion -->

<!-- #region -->
## 11. Évaluation
<!-- #endregion -->

<!-- #region -->
```python
"""
@jax.jit
def eval_step(params, x, y):
    logits = model.apply(params, x, training=False)
    preds = jnp.argmax(logits, axis=-1)
    return (preds == y).mean()


acc = eval_step(params, X_xor, y_xor)
"""
```
<!-- #endregion -->

<!-- #region -->
## 12. Gestion du déséquilibre
<!-- #endregion -->

<!-- #region -->
- **Class weights dans la loss** : pondérer la cross-entropy.
- **Sampling pondéré** dans `get_batches` (pondérer la `permutation`).
- **Focal loss** : implémentable en 5 lignes en `jnp`.
<!-- #endregion -->

<!-- #region -->
## 13. Régularisation
<!-- #endregion -->

<!-- #region -->
- **Dropout** : `nn.Dropout(rate, deterministic=not training)` dans le module Flax. Demande de passer une PRNG key `rngs={"dropout": key}` dans `apply`.
- **Weight decay** : géré par `optax.adamw`.
- **LayerNorm** : `nn.LayerNorm()`.
- **BatchNorm** : `nn.BatchNorm(use_running_average=True/False)` — plus de gymnastique (running stats à gérer manuellement).
<!-- #endregion -->

<!-- #region -->
## 14. Visualisation / Logging
<!-- #endregion -->

<!-- #region -->
Pas de TensorBoard natif JAX. Solutions :

- **`aim`** (open source, full-featured).
- **`wandb`** (SaaS standard).
- **`tensorboardX`** (écrit dans le format TB).
- **Print + matplotlib** pour les cas simples.
<!-- #endregion -->

<!-- #region -->
## 15. Cas réel — Classification MNIST (digits 8x8)
<!-- #endregion -->

<!-- #region -->
Pseudo-code complet (transposé du blueprint) — l'idée est de **construire un Flax MLP, le train avec un train_step JIT, et évaluer en eval_step JIT**.

```python
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_d = digits.data.astype(jnp.float32) / 16.0
y_d = digits.target.astype(jnp.int32)
X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

class MNIST_MLP(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(128)(x); x = nn.relu(x)
        x = nn.Dropout(0.1, deterministic=not training)(x)
        x = nn.Dense(64)(x);  x = nn.relu(x)
        x = nn.Dense(10)(x);  return x

model_mnist = MNIST_MLP()
key = jax.random.PRNGKey(0)
params = model_mnist.init(key, X_tr[:1])
opt = optax.adamw(1e-3); opt_state = opt.init(params)

# (boucle train_step / eval_step comme section 8 / 11)
"""
```

> Note : tester ce code nécessite `jax + flax + optax` installés. Pour rester dans l'esprit "blueprint comparable", l'archi et les hyperparams sont identiques à `DL_PyTorch` / `DL_TensorFlow` / `DL_Keras`.
<!-- #endregion -->

<!-- #region -->
## 16. Cas réel — Régression California Housing
<!-- #endregion -->

<!-- #region -->
Même structure : remplacer la loss par MSE, la dernière couche par `Dense(1)`, et le dataset par California Housing. Voir notebooks frères pour le code complet équivalent.
<!-- #endregion -->

<!-- #region -->
## 17. Explainability
<!-- #endregion -->

<!-- #region -->
SHAP **`KernelExplainer`** marche puisqu'il prend n'importe quelle fonction `f(X) -> y`. Pour les méthodes basées sur les gradients, JAX a `jax.grad` qui peut servir à implémenter manuellement des **Integrated Gradients** ou **SHAP-like** custom.

Lib spécialisée : **`captum-jax`** (port partiel) ou **`shap-jax`** (communautaire, moins mature qu'en torch/tf).
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- [JAX — docs](https://jax.readthedocs.io/) · [JAX cookbook](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Flax — docs](https://flax.readthedocs.io/)
- [Optax — docs](https://optax.readthedocs.io/)
- [Orbax — docs](https://orbax.readthedocs.io/)
- [JAX 101 — par DeepMind/Google](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- Notebooks liés : `DL_PyTorch`, `DL_TensorFlow`, `DL_Keras`, `DL_Frameworks_Comparatif`, `DL_Deep_Learning_Maths`.
<!-- #endregion -->
