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
# 🔥 PyTorch — Framework DL
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **PyTorch** (2.x, 2026). Suit le **blueprint commun** des notebooks DL framework (cf `00_blueprint_DL_frameworks.md`) — 16 sections pour pouvoir comparer côte à côte avec `DL_TensorFlow`, `DL_Keras`, `DL_JAX`.

**Datasets utilisés (mutualisés) :**
- **XOR continu** (toy) — sections 4, 5, 8, 9
- **MNIST** — section 15 (classification)
- **California Housing** — section 16 (régression)
<!-- #endregion -->

<!-- #region -->
## 1. Présentation de PyTorch (2026)
<!-- #endregion -->

<!-- #region -->
**PyTorch** (Meta AI, 2016) — framework DL pythonique, **define-by-run** (graphe dynamique). Domine la recherche depuis 2019 et a largement gagné la production en 2024-2026.

**État 2026** :

- **PyTorch 2.x** : `torch.compile()` accélère l'inference et le training (jusqu'à 2× sur certaines archi) en compilant le graphe via Triton.
- **FSDP** (Fully Sharded Data Parallel) : standard pour le training distribué de gros modèles.
- **TorchScript / ExecuTorch** : déploiement edge (mobile, embedded).
- **Écosystème immense** : `torchvision`, `torchaudio`, `transformers` (HF), `lightning`, `pyg` (graphes), `torchrl`.

**Philosophie** : graphe construit à l'exécution, debugging Python natif, contrôle total. La courbe d'apprentissage est plus raide que Keras mais le plafond est plus haut.
<!-- #endregion -->

```python
import torch
import numpy as np

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA disponible : {torch.cuda.is_available()}")
print(f"MPS  disponible : {torch.backends.mps.is_available()}")
```

<!-- #region -->
## 2. Tenseurs
<!-- #endregion -->

```python
# Création
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros = torch.zeros(3, 4)
ones = torch.ones_like(x)
rand = torch.rand(2, 3)
from_np = torch.from_numpy(np.array([1, 2, 3]))  # zero-copy

print(f"x shape : {x.shape}  dtype : {x.dtype}  device : {x.device}")
print(f"Op : x @ x.T =\n{x @ x.T}")
print(f"Conversion vers NumPy : {x.numpy()}")
```

<!-- #region -->
## 3. GPU / accélérateur
<!-- #endregion -->

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")

# Déplacer un tenseur
x_gpu = x.to(device)
print(f"x_gpu device : {x_gpu.device}")

# Reproductibilité
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

<!-- #region -->
## 4. Cas synthétique simple — XOR continu
<!-- #endregion -->

```python
def gen_xor(n=300):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

X_xor, y_xor = gen_xor(300)
print(f"X={X_xor.shape}  y={y_xor.shape}  classes={y_xor.unique().tolist()}")
```

<!-- #region -->
## 5. Définition de modèle
<!-- #endregion -->

<!-- #region -->
En PyTorch, on hérite de `nn.Module`. Deux conventions :

- **Sequential** pour empilement simple.
- **Module custom** dès qu'on a des skip connections, branches, paramètres spéciaux.
<!-- #endregion -->

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = MLP(2, 32, 2)
n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"Total params : {n_params}")
```

<!-- #region -->
## 6. Données — Dataset + DataLoader
<!-- #endregion -->

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_xor, y_xor)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Itération
xb, yb = next(iter(loader))
print(f"Batch : x={xb.shape}  y={yb.shape}")
```

<!-- #region -->
## 7. Loss & Optimisation
<!-- #endregion -->

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()           # logits + labels (combine softmax + CE)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
print(f"Optimiseur : {type(optimizer).__name__}, lr={optimizer.param_groups[0]['lr']}")
```

<!-- #region -->
## 8. Boucle d'entraînement standard
<!-- #endregion -->

```python
model = MLP(2, 32, 2)
optimizer = optim.AdamW(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()        # reset des gradients
        logits = model(xb)            # forward
        loss = criterion(logits, yb)  # loss
        loss.backward()               # backprop (autograd)
        optimizer.step()              # update
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(dataset)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}  loss={avg_loss:.4f}")

# Accuracy
model.eval()
with torch.no_grad():
    preds = model(X_xor).argmax(1)
    acc = (preds == y_xor).float().mean().item()
print(f"Train acc XOR : {acc:.3f}")
```

<!-- #region -->
## 9. Boucle manuelle + batch équilibré
<!-- #endregion -->

<!-- #region -->
Pour les datasets déséquilibrés, deux approches PyTorch :

- **WeightedRandomSampler** : tire chaque batch avec probabilité proportionnelle au poids.
- **Class weights dans la loss** : `nn.CrossEntropyLoss(weight=...)`.
<!-- #endregion -->

```python
from torch.utils.data import WeightedRandomSampler
from collections import Counter

# Calcul des poids par échantillon (inverse de la fréquence de classe)
counts = Counter(y_xor.tolist())
sample_weights = torch.tensor([1.0 / counts[int(l)] for l in y_xor])
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

balanced_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
print(f"Loader balancé : {len(balanced_loader)} batches/epoch")
```

<!-- #region -->
## 10. Save / Load
<!-- #endregion -->

```python
import tempfile
from pathlib import Path

# Sauvegarder UNIQUEMENT le state_dict (pas le module entier — best practice)
ckpt_path = Path(tempfile.gettempdir()) / "demo_mlp.pt"
torch.save(model.state_dict(), ckpt_path)

# Reload
model2 = MLP(2, 32, 2)
model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
model2.eval()
print(f"Modèle rechargé. Acc identique : {(model2(X_xor).argmax(1) == y_xor).float().mean().item():.3f}")
```

<!-- #region -->
## 11. Évaluation
<!-- #endregion -->

```python
def evaluate(model, X, y) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.CrossEntropyLoss()(logits, y).item()
        preds = logits.argmax(1)
        acc = (preds == y).float().mean().item()
    return {"loss": loss, "acc": acc}


print(evaluate(model, X_xor, y_xor))
```

<!-- #region -->
## 12. Gestion du déséquilibre
<!-- #endregion -->

```python
# Class weights dans la loss
counts_t = torch.tensor([counts[c] for c in sorted(counts.keys())], dtype=torch.float32)
class_weights = (counts_t.sum() / counts_t) / 2  # normalisation simple
criterion_w = nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights : {class_weights.tolist()}")
```

<!-- #region -->
## 13. Régularisation
<!-- #endregion -->

<!-- #region -->
- **Dropout** : `nn.Dropout(p=0.5)` — déjà inclus dans le MLP plus haut.
- **Weight decay** : passé à l'optimizer (`AdamW(..., weight_decay=1e-4)`).
- **BatchNorm** : `nn.BatchNorm1d(d)` pour des features tabulaires, `BatchNorm2d` pour CNN.
- **LayerNorm** : `nn.LayerNorm(d)` standard dans les Transformers.
- **Early stopping** : à coder soi-même ou via `lightning.pytorch.callbacks.EarlyStopping`.
<!-- #endregion -->

<!-- #region -->
## 14. Visualisation — TensorBoard
<!-- #endregion -->

```python
# TensorBoard natif via SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="runs/exp1")
# writer.add_scalar("loss/train", loss.item(), step)
# writer.add_histogram("weights/layer0", model.net[0].weight, step)
# writer.close()
# Lancer ensuite : tensorboard --logdir runs
```

<!-- #region -->
## 15. Cas réel — Classification MNIST
<!-- #endregion -->

```python
# Démo MNIST en tabulaire via sklearn (évite torchvision download). Subset pour vitesse.
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Subset pour test rapide (decommit pour full)
# mnist = fetch_openml("mnist_784", as_frame=False, parser="auto", n_retries=0)
# X_mnist = mnist.data.astype(np.float32) / 255.0
# y_mnist = mnist.target.astype(np.int64)
# X_tr, X_te, y_tr, y_te = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

# Pour la démo on prend digits 8x8 sklearn (plus léger)
from sklearn.datasets import load_digits
digits = load_digits()
X_d = digits.data.astype(np.float32) / 16.0
y_d = digits.target.astype(np.int64)
X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

X_tr_t, X_te_t = torch.from_numpy(X_tr), torch.from_numpy(X_te)
y_tr_t, y_te_t = torch.from_numpy(y_tr), torch.from_numpy(y_te)

mnist_model = MLP(d_in=64, d_hidden=128, d_out=10, dropout=0.1)
optimizer = optim.AdamW(mnist_model.parameters(), lr=1e-3)
loader_mnist = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

for epoch in range(10):
    mnist_model.train()
    for xb, yb in loader_mnist:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(mnist_model(xb), yb)
        loss.backward()
        optimizer.step()

mnist_model.eval()
with torch.no_grad():
    test_acc = (mnist_model(X_te_t).argmax(1) == y_te_t).float().mean().item()
print(f"Test acc digits 8x8 : {test_acc:.3f}")
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
Xr_tr_t = torch.from_numpy(X_tr_r); Xr_te_t = torch.from_numpy(X_te_r)
yr_tr_t = torch.from_numpy(y_tr_r); yr_te_t = torch.from_numpy(y_te_r)


class RegressorMLP(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


reg_model = RegressorMLP(X_tr_r.shape[1])
opt = optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
loader_reg = DataLoader(TensorDataset(Xr_tr_t, yr_tr_t), batch_size=128, shuffle=True)

best_val = float("inf"); patience = 5; bad = 0
for epoch in range(30):
    reg_model.train()
    for xb, yb in loader_reg:
        opt.zero_grad()
        loss = loss_fn(reg_model(xb), yb)
        loss.backward()
        opt.step()
    reg_model.eval()
    with torch.no_grad():
        val = loss_fn(reg_model(Xr_te_t), yr_te_t).item()
    if val < best_val:
        best_val = val; bad = 0
    else:
        bad += 1
        if bad >= patience: break

print(f"Best val RMSE : {np.sqrt(best_val):.4f}")
```

<!-- #region -->
## 17. Explainability (SHAP)
<!-- #endregion -->

<!-- #region -->
PyTorch est supporté par **`shap.DeepExplainer`** (pour CNN/MLP en classification) et **`shap.GradientExplainer`** (plus rapide, marche sur n'importe quel modèle PyTorch).

```python
# import shap
# bg = X_tr_t[:100]  # background samples
# explainer = shap.GradientExplainer(model, bg)
# shap_values = explainer.shap_values(X_te_t[:10])
# shap.summary_plot(shap_values, X_te[:10])
```
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- [PyTorch — docs officielles](https://pytorch.org/docs/stable/)
- [PyTorch Lightning — High-level wrapper](https://lightning.ai/docs/pytorch/stable/)
- [Karpathy — Deep Learning Course](https://karpathy.ai/zero-to-hero.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- Notebooks liés : `DL_TensorFlow`, `DL_Keras`, `DL_JAX`, `DL_Frameworks_Comparatif`, `DL_Deep_Learning_Maths`.
<!-- #endregion -->
