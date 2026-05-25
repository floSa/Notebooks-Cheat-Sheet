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
# 🔢 Générer des séquences temporelles
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet** : utilitaire pour transformer une **série temporelle** en **tenseur de séquences** prêt à être consommé par un **LSTM, GRU, Transformer, TCN, N-BEATS**, etc.

C'est la brique de prep des données la plus utilisée en DL pour les séries temporelles.

Couvre :

1. **Sliding window** : `(window_size, horizon, stride)`.
2. Génération avec **NumPy** (rapide, pour ingestion en RAM).
3. Génération avec **PyTorch Dataset** (streaming, large datasets).
4. Cas **multi-features** (covariables) et **multi-séries**.
5. **Train/test split** correct (temporel, pas de leak).
6. Renvois vers utilitaires Nixtla (`mlforecast`, `neuralforecast`) qui industrialisent ça.

> Pour les modèles DL TS, voir `TS_Time_Series_Overview` section 5.
> Pour la maintenance prédictive (case study), voir `TS_Maintenance_Predictive`.
<!-- #endregion -->

<!-- #region -->
## 1. Le concept de sliding window
<!-- #endregion -->

<!-- #region -->
À partir d'une série `[y_1, y_2, ..., y_N]`, on génère des paires `(X, Y)` :

- **X** : fenêtre de `window_size` observations passées.
- **Y** : `horizon` observations futures à prédire.
- **stride** : pas entre 2 fenêtres consécutives (1 = chevauchement max).

```
y :    [y0 y1 y2 y3 y4 y5 y6 y7 y8 y9]
                  ─────────────  ─────
                  X (window=5)   Y (h=2)
                       (stride=1 → fenêtre suivante décale de 1)
```

**Paramètres typiques** :

| Paramètre | Valeur typique | Notes |
|---|---|---|
| `window_size` | 24-168 (1j-1sem en hourly), 12-36 (mois) | Doit capturer au moins 1 saisonnalité complète |
| `horizon` | 1-24 | 1 = one-step, > 1 = multi-step |
| `stride` | 1 | Chevauchement max = plus d'échantillons |
<!-- #endregion -->

<!-- #region -->
## 2. Implémentation NumPy (chargement en RAM)
<!-- #endregion -->

```python
import numpy as np

def make_windows(
    series: np.ndarray,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère X (samples, window_size) et Y (samples, horizon) depuis une série 1D.

    Args:
        series: array 1D de longueur N.
        window_size: nombre d'observations dans X.
        horizon: nombre d'observations à prédire dans Y.
        stride: pas entre deux fenêtres consécutives.
    Returns:
        (X, Y) avec X.shape = (n_samples, window_size), Y.shape = (n_samples, horizon).
    """
    series = np.asarray(series).ravel()
    n_samples = (len(series) - window_size - horizon) // stride + 1
    X = np.zeros((n_samples, window_size), dtype=series.dtype)
    Y = np.zeros((n_samples, horizon), dtype=series.dtype)
    for i in range(n_samples):
        start = i * stride
        X[i] = series[start : start + window_size]
        Y[i] = series[start + window_size : start + window_size + horizon]
    return X, Y


# Démo
series = np.arange(100)
X, Y = make_windows(series, window_size=10, horizon=3, stride=1)
print(f"X shape = {X.shape}  Y shape = {Y.shape}")
print(f"Premier exemple : X={X[0]}  Y={Y[0]}")
print(f"Dernier exemple  : X={X[-1]}  Y={Y[-1]}")
```

<!-- #region -->
### 2.1 Variante vectorisée (plus rapide)
<!-- #endregion -->

```python
def make_windows_vec(series: np.ndarray, window_size: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Version vectorisée via stride_tricks (zero-copy, ultra rapide pour grandes séries)."""
    series = np.asarray(series).ravel()
    n_samples = len(series) - window_size - horizon + 1
    # Vue stride sans copie
    shape = (n_samples, window_size + horizon)
    strides = (series.strides[0], series.strides[0])
    full = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides, writeable=False)
    return full[:, :window_size].copy(), full[:, window_size:].copy()


X2, Y2 = make_windows_vec(series, window_size=10, horizon=3)
assert np.array_equal(X2, X)
assert np.array_equal(Y2, Y)
print("Versions équivalentes ✓")
```

<!-- #region -->
## 3. Multi-features (covariables)
<!-- #endregion -->

<!-- #region -->
En pratique on a souvent plusieurs séries en parallèle : la cible + des covariables (températures, prix, indicateurs).

`X.shape = (n_samples, window_size, n_features)` — tenseur 3D, format attendu par `nn.LSTM(input_size=n_features)`.
<!-- #endregion -->

```python
def make_windows_multi(
    data: np.ndarray,
    target_col: int,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-features → tenseur 3D.

    Args:
        data: array 2D (timesteps, n_features). `data[:, target_col]` = cible.
        target_col: index de la colonne cible (la seule prédite dans Y).
    Returns:
        X.shape = (n_samples, window_size, n_features)
        Y.shape = (n_samples, horizon)
    """
    n_steps, n_features = data.shape
    n_samples = (n_steps - window_size - horizon) // stride + 1
    X = np.zeros((n_samples, window_size, n_features), dtype=data.dtype)
    Y = np.zeros((n_samples, horizon), dtype=data.dtype)
    for i in range(n_samples):
        start = i * stride
        X[i] = data[start : start + window_size]
        Y[i] = data[start + window_size : start + window_size + horizon, target_col]
    return X, Y


# Démo : 3 features (target + 2 cov)
data = np.column_stack([np.arange(100), np.arange(100) * 0.5, np.arange(100) ** 2])
X3, Y3 = make_windows_multi(data, target_col=0, window_size=10, horizon=3)
print(f"X3 shape = {X3.shape}  Y3 shape = {Y3.shape}")
```

<!-- #region -->
## 4. PyTorch Dataset (streaming)
<!-- #endregion -->

<!-- #region -->
Quand la série est trop grande pour tenir en RAM, on génère les fenêtres **à la demande** via un `Dataset` PyTorch.
<!-- #endregion -->

```python
import torch
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    """Génère des fenêtres à la demande, zero-copy via index."""

    def __init__(self, series: np.ndarray, window_size: int, horizon: int = 1, stride: int = 1):
        self.series = torch.as_tensor(series, dtype=torch.float32).view(-1)
        self.w = window_size
        self.h = horizon
        self.stride = stride
        self.n = (len(self.series) - window_size - horizon) // stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        X = self.series[start : start + self.w]
        Y = self.series[start + self.w : start + self.w + self.h]
        return X, Y


ds = WindowDataset(np.arange(100), window_size=10, horizon=3)
print(f"Dataset size : {len(ds)}")
X_b, Y_b = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
print(f"Batch X shape = {X_b.shape}  Y shape = {Y_b.shape}")
```

<!-- #region -->
## 5. Train/test split — strictement temporel
<!-- #endregion -->

<!-- #region -->
**Erreur fatale** : `train_test_split(X, Y, shuffle=True)` → leak temporel massif.

**Solution** : on **coupe la série brute** au timestamp `t_split`, puis on génère les windows séparément pour train et test. Sinon, des fenêtres test contiennent des observations train.

Encore plus propre : laisser un **gap = window_size** entre train et test pour éviter l'overlap.
<!-- #endregion -->

```python
def temporal_split(
    series: np.ndarray,
    window_size: int,
    horizon: int,
    train_frac: float = 0.8,
    gap: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split temporel propre. gap=window_size évite tout overlap entre train et test."""
    cut = int(len(series) * train_frac)
    train = series[:cut]
    test = series[cut + gap:]
    X_tr, Y_tr = make_windows_vec(train, window_size, horizon)
    X_te, Y_te = make_windows_vec(test, window_size, horizon)
    return (X_tr, Y_tr), (X_te, Y_te)


(Xt, Yt), (Xe, Ye) = temporal_split(np.arange(200), window_size=10, horizon=3, train_frac=0.8, gap=10)
print(f"Train: X={Xt.shape}  Y={Yt.shape}")
print(f"Test : X={Xe.shape}  Y={Ye.shape}")
print(f"Premier sample test (doit commencer après le gap) : X[0]={Xe[0][:3]}...")
```

<!-- #region -->
## 6. Multi-séries (panel data)
<!-- #endregion -->

<!-- #region -->
Quand on a `N` séries différentes (capteurs IoT, magasins, clients), on génère les fenêtres **par série**, puis on concatène. Chaque sample garde l'**ID** de la série (pour les modèles globaux qui prennent un embedding d'ID).

```python
def make_windows_multi_series(
    df,
    id_col: str,
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère fenêtres pour N séries. Retourne X, Y, ids."""
    Xs, Ys, ids = [], [], []
    for series_id, g in df.groupby(id_col):
        data = g[feature_cols + [target_col]].to_numpy()
        X, Y = make_windows_multi(data, target_col=-1, window_size=window_size, horizon=horizon)
        Xs.append(X)
        Ys.append(Y)
        ids.append(np.full(len(X), series_id))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(ids)
```
<!-- #endregion -->

<!-- #region -->
## 7. Outils 2026 qui industrialisent tout ça
<!-- #endregion -->

<!-- #region -->
Réécrire les fenêtrages soi-même est éducatif mais en prod, utiliser :

- **`mlforecast`** (Nixtla) — lags + transforms + date features automatiques pour LightGBM/XGB.
- **`neuralforecast`** (Nixtla) — DataLoaders prêts pour TFT/N-HiTS/NBEATS/DeepAR.
- **`darts`** (Unit8) — `TimeSeries` + `*ForecastingModel` qui gèrent tout.
- **`gluonts`** (AWS) — historique, encore utilisé pour DeepAR/TFT.

Toutes ces libs gèrent multi-séries, covariables passées/futures, hierarchies, etc.
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- [PyTorch — Dataset and DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [NumPy — stride tricks](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)
- [Nixtla — mlforecast & neuralforecast](https://nixtla.github.io/)
- [Darts — Unit8](https://unit8co.github.io/darts/)
<!-- #endregion -->
