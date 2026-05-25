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
# 🧮 Deep Learning — Mathématiques fondamentales
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur les **mathématiques** sous-jacentes au Deep Learning, implémentées **from scratch** en NumPy.

L'objectif : comprendre **précisément** ce qui se passe dans `model.fit()` avant d'utiliser PyTorch/TensorFlow comme boîtes noires.

Couvre :

1. **Le neurone** : combinaison linéaire + non-linéarité.
2. **Fonctions d'activation** : sigmoid, tanh, ReLU, GELU, Swish, Softmax.
3. **Forward pass** d'un MLP.
4. **Loss functions** : MSE (régression), Cross-Entropy (classification).
5. **Backpropagation** : règle de la chaîne appliquée aux NN.
6. **Optimiseurs** : SGD, Momentum, Adam, AdamW.
7. **Régularisation** : weight decay, dropout, batch norm (maths).
8. **Implémentation from scratch** : MLP NumPy capable d'apprendre XOR + MNIST tabulaire.
9. **Problèmes numériques** : overflow softmax, exploding/vanishing gradients.
<!-- #endregion -->

<!-- #region -->
## 1. Le neurone
<!-- #endregion -->

<!-- #region -->
Un **neurone** calcule une **combinaison linéaire** de ses entrées plus un biais, puis passe le résultat dans une **fonction d'activation** non-linéaire :

$$
z = w^T x + b, \quad a = \sigma(z)
$$

- `x ∈ ℝ^d` : entrées.
- `w ∈ ℝ^d, b ∈ ℝ` : paramètres apprenables.
- `σ` : activation (sigmoid, tanh, ReLU, ...).
- `a` : sortie du neurone.

Sans non-linéarité, empiler des couches reviendrait à composer des fonctions linéaires = encore une fonction linéaire. C'est l'activation qui donne au réseau sa **capacité d'approximation universelle** (Cybenko 1989).
<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Un neurone simple : combinaison linéaire + sigmoid
def neuron(x, w, b):
    z = np.dot(x, w) + b
    return 1.0 / (1.0 + np.exp(-z))  # sigmoid

x = np.array([0.5, -0.3, 0.8])
w = np.array([0.2, -0.5, 0.7])
print(f"Neurone output : {neuron(x, w, 0.1):.4f}")
```

<!-- #region -->
## 2. Fonctions d'activation
<!-- #endregion -->

```python
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
def swish(z): return z * sigmoid(z)
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # stabilité numérique (anti-overflow)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


z = np.linspace(-5, 5, 100)
fig, ax = plt.subplots(figsize=(10, 5))
for name, f in [("sigmoid", sigmoid), ("tanh", tanh), ("ReLU", relu),
                ("Leaky ReLU", leaky_relu), ("GELU", gelu), ("Swish", swish)]:
    ax.plot(z, f(z), label=name)
ax.set_title("Fonctions d'activation"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
```

<!-- #region -->
**Recommandation 2026** :

- **ReLU** : standard pour MLP/CNN, simple et efficace.
- **GELU** : standard dans les Transformers (BERT, GPT).
- **Swish/SiLU** : alternative compétitive (PaLM, EfficientNet).
- **Sigmoid** : uniquement en sortie pour la classification binaire (proba ∈ [0,1]).
- **Softmax** : uniquement en sortie pour la classification multiclass (distrib de probas).
<!-- #endregion -->

<!-- #region -->
## 3. Forward pass d'un MLP
<!-- #endregion -->

<!-- #region -->
Un **MLP** (Multi-Layer Perceptron) à `L` couches :

```
x ── W₁,b₁ ──► z₁ ── σ ──► a₁ ── W₂,b₂ ──► z₂ ── σ ──► a₂ ── ... ──► aₗ = ŷ
```

Pour chaque couche `l` :

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})
$$

(avec `a^(0) = x`). En batch : remplacer `a` par une matrice `A ∈ ℝ^{n×d}`.
<!-- #endregion -->

```python
def init_params(layer_sizes):
    """Initialisation He pour ReLU (variance = 2/fan_in)."""
    params = []
    rng = np.random.default_rng(42)
    for d_in, d_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.normal(0, np.sqrt(2 / d_in), size=(d_in, d_out))
        b = np.zeros(d_out)
        params.append({"W": W, "b": b})
    return params


def forward(X, params, hidden_act=relu):
    """X : (n, d_in). Renvoie sortie + cache pour backprop."""
    cache = {"A0": X}
    A = X
    for i, layer in enumerate(params, start=1):
        Z = A @ layer["W"] + layer["b"]
        if i == len(params):  # dernière couche : pas d'activation (ou softmax en classif)
            A = Z
        else:
            A = hidden_act(Z)
        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A
    return A, cache


# Démo : MLP 3→5→2
params = init_params([3, 5, 2])
X_demo = np.random.randn(4, 3)
out, _ = forward(X_demo, params)
print(f"Output shape : {out.shape}")
```

<!-- #region -->
## 4. Loss functions
<!-- #endregion -->

<!-- #region -->
### 4.1 MSE (régression)
<!-- #endregion -->

<!-- #region -->
$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Gradient par rapport à `ŷ` : `dL/dŷ = 2(ŷ - y) / n`.
<!-- #endregion -->

<!-- #region -->
### 4.2 Cross-Entropy (classification)
<!-- #endregion -->

<!-- #region -->
Pour la classification multiclass avec softmax :

$$
L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
$$

avec `y` one-hot et `ŷ = softmax(z)`. **Astuce numérique magnifique** : si on combine softmax + cross-entropy, le gradient par rapport au pré-softmax `z` simplifie à :

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$

C'est ce qu'utilisent tous les frameworks (PyTorch `CrossEntropyLoss`, TF `SparseCategoricalCrossentropy(from_logits=True)`) — d'où le conseil de **passer les logits, pas la softmax**, à ces fonctions.
<!-- #endregion -->

```python
def softmax_cross_entropy(logits, y):
    """Combinaison stable softmax + CE. y : (n,) labels entiers."""
    n = logits.shape[0]
    # log-softmax stable (log-sum-exp trick)
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), y].mean()
    # Gradient analytique : softmax(z) - y_onehot
    probs = np.exp(log_probs)
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    return loss, dlogits


logits = np.random.randn(4, 3)
y = np.array([0, 2, 1, 1])
loss, grad = softmax_cross_entropy(logits, y)
print(f"Loss = {loss:.4f}  grad shape = {grad.shape}")
```

<!-- #region -->
## 5. Backpropagation
<!-- #endregion -->

<!-- #region -->
**Idée** : utiliser la **règle de la chaîne** pour calculer `∂L/∂W^(l)` et `∂L/∂b^(l)` à chaque couche, en partant de la fin (output) vers le début (input).

Pour une couche `l` :

$$
\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^T \cdot \frac{\partial L}{\partial z^{(l)}}, \quad \frac{\partial L}{\partial b^{(l)}} = \sum_{i} \frac{\partial L}{\partial z^{(l)}}_i
$$

Et pour propager :

$$
\frac{\partial L}{\partial a^{(l-1)}} = \frac{\partial L}{\partial z^{(l)}} \cdot (W^{(l)})^T
$$

Puis on multiplie par la **dérivée de l'activation** : `∂L/∂z^(l-1) = ∂L/∂a^(l-1) ⊙ σ'(z^(l-1))`.

Dérivées utiles :

- `sigmoid'(z) = sigmoid(z) (1 - sigmoid(z))`
- `tanh'(z) = 1 - tanh(z)²`
- `ReLU'(z) = 1[z > 0]`
<!-- #endregion -->

```python
def backward(dlogits, cache, params, hidden_act_grad=lambda z: (z > 0).astype(float)):
    """Backpropagation. Renvoie les gradients pour chaque couche."""
    grads = []
    dA = dlogits  # gradient sortant de la dernière couche (logits)
    L = len(params)
    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        if l == L:
            dZ = dA  # pas d'activation sur la dernière (combinée avec softmax dans la loss)
        else:
            dZ = dA * hidden_act_grad(cache[f"Z{l}"])
        dW = A_prev.T @ dZ
        db = dZ.sum(axis=0)
        grads.insert(0, {"dW": dW, "db": db})
        if l > 1:
            dA = dZ @ params[l-1]["W"].T
    return grads
```

<!-- #region -->
## 6. Optimiseurs
<!-- #endregion -->

<!-- #region -->
| Optimiseur | Update rule | Quand |
|---|---|---|
| **SGD** | `θ ← θ - η ∇L` | Baseline, simple |
| **SGD + Momentum** | `v ← μv + ∇L; θ ← θ - ηv` | Stabilise, accélère |
| **Nesterov** | Variante avec "anticipation" | Souvent meilleur que momentum |
| **RMSProp** | Normalise par moyenne mobile des grads² | Bon pour RNN |
| **Adam** | Combine Momentum + RMSProp | Standard 2026 |
| **AdamW** | Adam + weight decay correct | **Standard pour Transformers** |
| **Lion** (2023) | Plus efficace en mémoire qu'Adam, perf comparable | Émergent |
| **Sophia** (2024) | Précondiitionneur Hessienne approchée | Pour LLMs |

**Recommandation 2026** : **AdamW** par défaut sauf raison contraire. Lion pour les très gros modèles.
<!-- #endregion -->

```python
def sgd_step(params, grads, lr=0.01):
    for p, g in zip(params, grads):
        p["W"] -= lr * g["dW"]
        p["b"] -= lr * g["db"]


def adam_step(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """Adam update. m, v sont des listes de dict initialisés à zéro."""
    for p, g, m_p, v_p in zip(params, grads, m, v):
        for key in ("dW", "db"):
            wkey = "W" if key == "dW" else "b"
            m_p[key] = b1 * m_p[key] + (1 - b1) * g[key]
            v_p[key] = b2 * v_p[key] + (1 - b2) * g[key] ** 2
            m_hat = m_p[key] / (1 - b1 ** t)
            v_hat = v_p[key] / (1 - b2 ** t)
            p[wkey] -= lr * m_hat / (np.sqrt(v_hat) + eps)
```

<!-- #region -->
## 7. Régularisation
<!-- #endregion -->

<!-- #region -->
### 7.1 Weight decay (L2)
<!-- #endregion -->

<!-- #region -->
Ajoute `λ‖W‖² / 2` à la loss → gradient additionnel `λW` → poids tendent à 0. Implémentation directe : `dW += λ * W` avant l'update.

**Note importante** : dans Adam, `weight_decay` natif n'est pas équivalent à `L2 regularization` (subtilité numérique). C'est pour ça qu'**AdamW** (Loshchilov 2018) sépare proprement les deux.
<!-- #endregion -->

<!-- #region -->
### 7.2 Dropout
<!-- #endregion -->

<!-- #region -->
À chaque pas de training, masque aléatoirement une fraction `p` des activations (les met à 0). En inference : pas de dropout, on multiplie par `(1-p)` (inverted dropout : on divise par `(1-p)` en train, rien en inference). Effet : régularisation par bruit + ensembling implicite.
<!-- #endregion -->

```python
def dropout(A, p=0.5, training=True):
    if not training or p == 0:
        return A
    mask = (np.random.rand(*A.shape) > p) / (1 - p)  # inverted dropout
    return A * mask
```

<!-- #region -->
### 7.3 Batch Normalization
<!-- #endregion -->

<!-- #region -->
Normalise chaque feature sur la batch (mean=0, std=1) puis apprend `γ, β` qui ré-échelonnent. Accélère le training et permet des learning rates plus élevés. **LayerNorm** est l'équivalent dans les Transformers (normalise sur les features d'une obs, pas sur la batch).
<!-- #endregion -->

<!-- #region -->
## 8. MLP from scratch — XOR continu
<!-- #endregion -->

```python
# Génération XOR continu : 2 spirales entrelacées
def gen_xor(n=300):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # classes : XOR sur le signe
    return X, y


X, y = gen_xor(300)
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7)
ax.set_title("XOR continu — 2 classes (rouge / bleu)")
plt.tight_layout()
```

```python
# MLP 2-16-16-2 from scratch, entraîné en Adam
params = init_params([2, 16, 16, 2])
m = [{"dW": np.zeros_like(p["W"]), "db": np.zeros_like(p["b"])} for p in params]
v = [{"dW": np.zeros_like(p["W"]), "db": np.zeros_like(p["b"])} for p in params]

losses = []
for t in range(1, 501):
    logits, cache = forward(X, params)
    loss, dlogits = softmax_cross_entropy(logits, y)
    grads = backward(dlogits, cache, params)
    adam_step(params, grads, m, v, t, lr=0.01)
    losses.append(loss)

# Évaluation
logits, _ = forward(X, params)
preds = logits.argmax(axis=1)
acc = (preds == y).mean()
print(f"Train accuracy après 500 steps Adam : {acc:.3f}")
plt.figure(figsize=(8, 3)); plt.plot(losses); plt.title("Loss"); plt.grid(alpha=0.3)
```

<!-- #region -->
## 9. Problèmes numériques
<!-- #endregion -->

<!-- #region -->
### 9.1 Overflow softmax
<!-- #endregion -->

<!-- #region -->
`exp(z)` overflow si `z > 700`. Solution : soustraire le max **avant** l'exp : `softmax(z) = softmax(z - max(z))` (mathématiquement identique, numériquement stable).
<!-- #endregion -->

<!-- #region -->
### 9.2 Vanishing / Exploding gradients
<!-- #endregion -->

<!-- #region -->
Dans un réseau profond, les gradients se multiplient en backprop. Si les facteurs sont < 1, ils tendent vers 0 (vanishing — réseau n'apprend plus). Si > 1, ils explosent.

**Solutions** :

- **Initialisation appropriée** : Xavier (tanh), He (ReLU).
- **Activations sans saturation** : ReLU et variantes.
- **BatchNorm / LayerNorm** : normalisent les activations à chaque couche.
- **Gradient clipping** : capper la norme du gradient (`clip_grad_norm`).
- **Architectures spécifiques** : LSTM/GRU pour les RNN, **residual connections** pour les CNN/Transformers profonds.
<!-- #endregion -->

<!-- #region -->
## 10. Prochaines étapes
<!-- #endregion -->

<!-- #region -->
- Tu sais maintenant ce qui se passe sous le capot. Pour la pratique en production :
  - **PyTorch** (`DL_PyTorch`) — flexible, recherche.
  - **TensorFlow** (`DL_TensorFlow`) — production, TPU.
  - **Keras** (`DL_Keras`) — high-level multi-backend.
  - **JAX** (`DL_JAX`) — fonctionnel, perf.
  - **Comparatif** (`DL_Frameworks_Comparatif`) — quand choisir quoi.
- Architectures spécialisées : voir notebooks DL_PyTorch (Datasets, training loops, modèles).
- Transformers / LLMs : voir `NLP_Transformers`.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Deep Learning Book — Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- [3Blue1Brown — Neural Networks playlist](https://www.3blue1brown.com/lessons/neural-networks)
- [CS231n — Stanford Convolutional Neural Networks](https://cs231n.github.io/)
- [Karpathy — micrograd](https://github.com/karpathy/micrograd) (autograd from scratch)
- [Loshchilov & Hutter (2018) — AdamW](https://arxiv.org/abs/1711.05101)
- Notebooks liés : `DL_PyTorch`, `DL_TensorFlow`, `DL_Keras`, `DL_JAX`, `DL_Frameworks_Comparatif`.
<!-- #endregion -->
