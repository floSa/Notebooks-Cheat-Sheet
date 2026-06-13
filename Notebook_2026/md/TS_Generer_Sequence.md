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
# 🔢 Générer des séquences temporelles (sliding window)
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** : la brique de préparation de données la plus utilisée en deep learning pour les séries temporelles — transformer une série en **tenseur de séquences** `(X, Y)` consommable par un **LSTM, GRU, TCN, Transformer, N-BEATS**, etc.

Ce notebook couvre :

1. Le **concept** de sliding window (`window_size`, `horizon`, `stride`).
2. Génération **NumPy** (boucle lisible + version vectorisée `stride_tricks`).
3. Cas **multi-features** (covariables) → tenseur 3D, et **multi-output** (prédire H pas).
4. **PyTorch `Dataset`** pour le streaming (séries trop grandes pour la RAM).
5. **Split temporel** correct et **scaling sans leakage** (les deux pièges classiques).
6. **Multi-séries** (panel data) et renvois vers les outils 2026 qui industrialisent tout ça.

> Pour le feature engineering TS complet, voir `TS_Time_Series_Overview`.
> Pour un cas d'usage en production, voir `TS_Maintenance_Predictive`.
<!-- #endregion -->

<!-- #region -->
## 1. Le concept de sliding window
<!-- #endregion -->

<!-- #region -->
À partir d'une série `[y_0, y_1, ..., y_N]`, on génère des paires `(X, Y)` :

- **X** : fenêtre de `window_size` observations **passées**.
- **Y** : `horizon` observations **futures** à prédire.
- **stride** : pas entre deux fenêtres consécutives (`1` = chevauchement maximal, plus d'échantillons).

```
y :   [y0 y1 y2 y3 y4 y5 y6 y7 y8 y9]
       └──── X (window=5) ────┘ └Y(h=2)┘
              stride=1 → la fenêtre suivante décale de 1 pas
```

**Valeurs typiques :**

| Paramètre | Valeur typique | Notes |
|---|---|---|
| `window_size` | 24–168 (hourly), 12–36 (mensuel) | Doit capturer ≥ 1 saisonnalité complète |
| `horizon` | 1–24 | `1` = one-step, `> 1` = multi-step |
| `stride` | 1 | Chevauchement max = plus d'échantillons d'entraînement |

> ⚠️ **Anti-pattern fatal** : générer les fenêtres **avant** le split train/test → **data leakage** (des fenêtres de test contiennent des observations de train). Toujours splitter la série brute *d'abord*, fenêtrer *ensuite* (section 6).
<!-- #endregion -->

<!-- #region -->
## 2. Implémentation NumPy (chargement en RAM)
<!-- #endregion -->

<!-- #region -->
Version de référence, lisible : une boucle qui découpe la série. Typée `tuple[np.ndarray, np.ndarray]`, paramètres `window_size / horizon / stride`. Le nombre d'échantillons est `(N - window - horizon) // stride + 1`.
<!-- #endregion -->

```python
import numpy as np


def make_windows(
    series: np.ndarray,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère X (n_samples, window_size) et Y (n_samples, horizon) depuis une série 1D.

    Args:
        series: array 1D de longueur N.
        window_size: nombre d'observations passées dans chaque X.
        horizon: nombre d'observations futures à prédire dans Y.
        stride: pas entre deux fenêtres consécutives (1 = chevauchement max).
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


series = np.arange(100)
X, Y = make_windows(series, window_size=10, horizon=3, stride=1)
print(f"X shape = {X.shape}  Y shape = {Y.shape}")
print(f"Premier exemple : X={X[0]}  ->  Y={Y[0]}")
print(f"Dernier exemple : X={X[-1]}  ->  Y={Y[-1]}")
```

<!-- #region -->
Sur `np.arange(100)` avec `window=10, horizon=3`, on obtient `88` échantillons : `X[0]` voit `[0..9]` et prédit `[10,11,12]`. La sémantique est directe — aucune inversion de série, contrairement aux implémentations « maison » qui reversent puis re-reversent.
<!-- #endregion -->

<!-- #region -->
### 2.1 Variante vectorisée (plus rapide)
<!-- #endregion -->

<!-- #region -->
Pour de grandes séries, `np.lib.stride_tricks.as_strided` crée une **vue** sans copie (zero-copy) sur la mémoire de la série, puis on tranche X et Y. On vérifie l'équivalence stricte avec la version boucle via `assert`. (Stride implicite de 1 ; pour `stride != 1`, garder `make_windows`.)
<!-- #endregion -->

```python
def make_windows_vec(
    series: np.ndarray, window_size: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Version vectorisée via stride_tricks (zero-copy, rapide pour grandes séries).

    Stride=1 implicite. Pour stride != 1, utiliser make_windows.
    """
    series = np.asarray(series).ravel()
    n_samples = len(series) - window_size - horizon + 1
    shape = (n_samples, window_size + horizon)
    strides = (series.strides[0], series.strides[0])
    full = np.lib.stride_tricks.as_strided(
        series, shape=shape, strides=strides, writeable=False
    )
    return full[:, :window_size].copy(), full[:, window_size:].copy()


X2, Y2 = make_windows_vec(series, window_size=10, horizon=3)
assert np.array_equal(X2, X), "X vectorisé != X boucle"
assert np.array_equal(Y2, Y), "Y vectorisé != Y boucle"
print(f"Versions boucle/vectorisée équivalentes — X {X2.shape}")
```

<!-- #region -->
> ⚠️ `as_strided` est puissant mais dangereux : les vues qu'il crée peuvent déborder de la mémoire allouée si les `shape`/`strides` sont faux. On renvoie ici une **copie** (`.copy()`) pour garantir un array sûr et contigu en aval.
<!-- #endregion -->

<!-- #region -->
## 3. Multi-features (covariables) → tenseur 3D
<!-- #endregion -->

<!-- #region -->
En pratique on a souvent la cible **plus des covariables** (températures, prix, promo, indicateurs). On veut alors `X.shape = (n_samples, window_size, n_features)` — le tenseur 3D attendu par `nn.LSTM(input_size=n_features)`. Seule la colonne cible (`target_col`) part dans `Y`.
<!-- #endregion -->

```python
def make_windows_multi(
    data: np.ndarray,
    target_col: int,
    window_size: int,
    horizon: int = 1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-features -> tenseur 3D pour nn.LSTM(input_size=n_features).

    Args:
        data: array 2D (timesteps, n_features). data[:, target_col] = cible.
        target_col: index de la colonne cible (seule colonne prédite dans Y).
    Returns:
        X.shape = (n_samples, window_size, n_features), Y.shape = (n_samples, horizon).
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


# 3 features : cible (col 0) + 2 covariables
data = np.column_stack(
    [np.arange(100, dtype=float), np.arange(100) * 0.5, np.sin(np.arange(100))]
)
X3, Y3 = make_windows_multi(data, target_col=0, window_size=10, horizon=3)
print(f"X3 shape = {X3.shape} (samples, window, features)  Y3 shape = {Y3.shape}")
```

<!-- #region -->
`X3` est un tenseur `(88, 10, 3)` : 88 échantillons, fenêtre de 10 pas, 3 features par pas. C'est exactement la forme `(batch, seq_len, input_size)` qu'attend une couche récurrente PyTorch en mode `batch_first=True`.
<!-- #endregion -->

<!-- #region -->
## 4. Multi-output : prédire H pas
<!-- #endregion -->

<!-- #region -->
Pas besoin d'une fonction dédiée : prédire les `H` prochains pas = appeler `make_windows` avec `horizon=H`. `Y` devient alors `(n_samples, H)`. C'est le format pour un modèle **multi-step direct** (une seule passe prédit tout l'horizon).
<!-- #endregion -->

```python
Xh, Yh = make_windows(series, window_size=12, horizon=6, stride=1)
print(f"Multi-output horizon=6 : X={Xh.shape}  Y={Yh.shape} (prédit 6 pas d'un coup)")
assert Yh.shape[1] == 6
```

<!-- #region -->
Alternative au multi-step direct : la prédiction **récursive** (prédire 1 pas, ré-injecter, recommencer) — plus simple à entraîner mais accumule l'erreur sur l'horizon. Le multi-output direct est généralement plus robuste pour les longs horizons.
<!-- #endregion -->

<!-- #region -->
## 5. PyTorch Dataset (streaming)
<!-- #endregion -->

<!-- #region -->
Quand la série est trop grande pour tenir en RAM (ou pour intégrer un `DataLoader` avec shuffle/batch/num_workers), on génère les fenêtres **à la demande** via un `torch.utils.data.Dataset` : `__getitem__(idx)` ne matérialise qu'une fenêtre, indexée par `idx`.
<!-- #endregion -->

```python
import torch
from torch.utils.data import Dataset, DataLoader


class WindowDataset(Dataset):
    """Génère les fenêtres à la demande (streaming) — adapté aux grandes séries."""

    def __init__(
        self,
        series: np.ndarray,
        window_size: int,
        horizon: int = 1,
        stride: int = 1,
    ) -> None:
        self.series = torch.as_tensor(series, dtype=torch.float32).view(-1)
        self.w = window_size
        self.h = horizon
        self.stride = stride
        self.n = (len(self.series) - window_size - horizon) // stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.series[start : start + self.w]
        y = self.series[start + self.w : start + self.w + self.h]
        return x, y


ds = WindowDataset(np.arange(100), window_size=10, horizon=3)
xb, yb = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
print(f"Dataset size = {len(ds)}  | batch X={tuple(xb.shape)} Y={tuple(yb.shape)}")
```

<!-- #region -->
Le `DataLoader` empile automatiquement 4 fenêtres en un batch `(4, 10)`. Pour le multi-features, retourner un `x` de forme `(window, n_features)` dans `__getitem__` et le batch deviendra `(batch, window, n_features)`.
<!-- #endregion -->

<!-- #region -->
## 6. Split temporel — strictement chronologique
<!-- #endregion -->

<!-- #region -->
**Erreur fatale** : `train_test_split(X, Y, shuffle=True)` → leakage temporel massif (le modèle « voit » le futur). 

**Solution** : couper la **série brute** au point `train_frac`, puis générer les fenêtres séparément pour train et test. Encore plus propre : laisser un **`gap = window_size`** entre les deux pour qu'aucune fenêtre de test ne chevauche le train.
<!-- #endregion -->

```python
def temporal_split(
    series: np.ndarray,
    window_size: int,
    horizon: int,
    train_frac: float = 0.8,
    gap: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split temporel propre : coupe la série brute, génère les fenêtres séparément.

    gap = window_size évite tout chevauchement train/test.
    """
    cut = int(len(series) * train_frac)
    train = series[:cut]
    test = series[cut + gap :]
    X_tr, Y_tr = make_windows_vec(train, window_size, horizon)
    X_te, Y_te = make_windows_vec(test, window_size, horizon)
    return (X_tr, Y_tr), (X_te, Y_te)


(Xt, Yt), (Xe, Ye) = temporal_split(
    np.arange(200), window_size=10, horizon=3, train_frac=0.8, gap=10
)
print(f"Train X={Xt.shape} Y={Yt.shape} | Test X={Xe.shape} Y={Ye.shape}")
print(f"1er sample test commence à {int(Xe[0][0])} (après cut+gap)")
```

<!-- #region -->
Le premier échantillon de test démarre à l'index `170` (cut à `160` + `gap` de `10`) : aucune observation de train ne fuit dans les fenêtres de test. Pour de la validation glissante (walk-forward), répéter ce découpage sur des coupures successives (`TimeSeriesSplit` de scikit-learn fait ça sur les indices).
<!-- #endregion -->

<!-- #region -->
## 7. Scaling sans leakage
<!-- #endregion -->

<!-- #region -->
Deuxième piège classique : **fitter le scaler sur la série entière** fait fuiter les statistiques (moyenne, écart-type) du futur dans le passé. Règle : `fit` sur le **train uniquement**, puis `transform` train/val/test avec ce même scaler.
<!-- #endregion -->

```python
from sklearn.preprocessing import StandardScaler


def scale_no_leak(
    train: np.ndarray, *others: np.ndarray
) -> tuple[StandardScaler, list[np.ndarray]]:
    """Fit le scaler sur TRAIN uniquement, puis transforme train + autres splits.

    Returns:
        (scaler ajusté, [train_scaled, *others_scaled]).
    """
    scaler = StandardScaler().fit(train.reshape(-1, 1))
    out = [scaler.transform(s.reshape(-1, 1)).ravel() for s in (train, *others)]
    return scaler, out


raw = np.arange(200, dtype=float)
train_raw, test_raw = raw[:160], raw[160:]
scaler, (train_s, test_s) = scale_no_leak(train_raw, test_raw)
print(
    f"Scaler fit sur train uniquement : mean={scaler.mean_[0]:.2f} "
    f"std={scaler.scale_[0]:.2f} | train_s[:3]={np.round(train_s[:3], 2)}"
)
X_tr_s, Y_tr_s = make_windows_vec(train_s, window_size=10, horizon=1)
print(f"Fenêtres sur train scalé : X={X_tr_s.shape}")
```

<!-- #region -->
L'ordre est important : **split → fit scaler sur train → transform → fenêtrer**. Inverser (fenêtrer puis scaler) ou fitter sur tout casse l'évaluation. Note : `test_s` peut dépasser `[-1, 1]` puisque ses valeurs n'ont pas servi au fit — c'est normal et souhaité.
<!-- #endregion -->

<!-- #region -->
## 8. Multi-séries (panel data)
<!-- #endregion -->

<!-- #region -->
Quand on a `N` séries (capteurs IoT, magasins, clients), on génère les fenêtres **par série** puis on concatène, en gardant l'**ID** de chaque échantillon (pour les modèles globaux qui prennent un embedding d'ID). Point crucial : une fenêtre ne doit **jamais** traverser la frontière entre deux séries — le `groupby` garantit ça.
<!-- #endregion -->

```python
import pandas as pd


def make_windows_multi_series(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_cols: list[str],
    window_size: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère les fenêtres pour N séries (panel) et garde l'ID de chaque sample.

    Les fenêtres ne traversent jamais la frontière entre deux séries.
    Returns:
        X (n, window, n_features), Y (n, horizon), ids (n,).
    """
    xs, ys, ids = [], [], []
    cols = feature_cols + [target_col]
    target_idx = len(cols) - 1
    for sid, g in df.groupby(id_col):
        arr = g[cols].to_numpy(dtype=float)
        x, y = make_windows_multi(
            arr, target_col=target_idx, window_size=window_size, horizon=horizon
        )
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        ids.append(np.full(len(x), sid))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


panel = pd.DataFrame(
    {
        "store": np.repeat([0, 1, 2], 40),
        "promo": np.tile(np.arange(40) % 2, 3).astype(float),
        "sales": np.concatenate(
            [np.arange(40), np.arange(40) * 2, np.arange(40) + 5]
        ).astype(float),
    }
)
Xp, Yp, ids = make_windows_multi_series(
    panel, id_col="store", target_col="sales", feature_cols=["promo"],
    window_size=8, horizon=2,
)
print(f"Panel : X={Xp.shape} Y={Yp.shape} ids uniques={np.unique(ids)}")
```

<!-- #region -->
On obtient un seul tenseur `X (93, 8, 2)` agrégeant les 3 magasins, avec `ids` qui permet de retrouver à quelle série appartient chaque échantillon. Ce format alimente directement un modèle **global** (un seul modèle pour toutes les séries, avec embedding d'ID), bien plus efficace que `N` modèles locaux.
<!-- #endregion -->

<!-- #region -->
## 9. Démo sur dataset réel : Air Passengers
<!-- #endregion -->

<!-- #region -->
Application sur une vraie série : **Air Passengers** (trafic aérien mensuel 1949-1960, tendance + saisonnalité annuelle de 12 mois), chargée via `statsmodels`. On choisit `window_size=12` pour capturer un cycle annuel complet. *(Fallback synthétique saisonnier si `statsmodels` est absent, pour que le notebook reste exécutable partout.)*
<!-- #endregion -->

```python
def load_air_passengers() -> np.ndarray:
    """Charge Air Passengers (statsmodels). Fallback : série saisonnière synthétique."""
    try:
        import statsmodels.api as sm

        data = sm.datasets.get_rdataset("AirPassengers").data
        return data["value"].to_numpy(dtype=float)
    except Exception:
        t = np.arange(144, dtype=float)
        return 100 + 0.8 * t + 20 * np.sin(2 * np.pi * t / 12)


ap = load_air_passengers()
Xap, Yap = make_windows_vec(ap, window_size=12, horizon=1)
print(f"Air Passengers : série len={len(ap)} -> X={Xap.shape} Y={Yap.shape}")
print(f"X[0] (12 premiers mois) = {np.round(Xap[0], 1)}")
```

<!-- #region -->
La série a 144 points → 132 fenêtres de 12 mois prédisant le mois suivant. `X[0]` correspond aux 12 mois de 1949. Avec un `window_size` égal à la période saisonnière, le modèle dispose d'un cycle complet pour apprendre le motif annuel.
<!-- #endregion -->

<!-- #region -->
## 10. Outils 2026 qui industrialisent tout ça
<!-- #endregion -->

<!-- #region -->
Réécrire le fenêtrage soi-même est éducatif et utile pour comprendre les pièges, mais en production on s'appuie sur des libs qui gèrent multi-séries, covariables passées/futures, hiérarchies et probabilistic forecasting :

- **`mlforecast`** (Nixtla) — lags, rolling features et date features automatiques pour LightGBM/XGBoost.
- **`neuralforecast`** (Nixtla) — DataLoaders et modèles prêts : TFT, N-HiTS, NBEATSx, DeepAR, PatchTST.
- **`pytorch-forecasting`** — `TimeSeriesDataSet` + `.to_dataloader()` : la référence PyTorch-native pour le fenêtrage (encodeur/décodeur, covariables connues/inconnues).
- **`darts`** (Unit8) — objet `TimeSeries` unifié + tout l'éventail de modèles (ARIMA → TFT) avec une API scikit-like.
- **`gluonts`** (AWS) — historique, encore utilisé pour DeepAR/TFT en environnement probabiliste.
<!-- #endregion -->

<!-- #region -->
## 11. Équivalents natifs (à connaître)
<!-- #endregion -->

<!-- #region -->
Avant d'écrire sa propre fonction, vérifier si le framework offre déjà le fenêtrage :

- **Keras / TensorFlow** : `keras.utils.timeseries_dataset_from_array(data, targets, sequence_length=window_size, sequence_stride=stride, batch_size=...)` renvoie un `tf.data.Dataset` de fenêtres glissantes.
- **PyTorch** : pas de helper natif — on subclasse `torch.utils.data.Dataset` (cf. section 5), ou on utilise `pytorch-forecasting`.
- **pandas** : `df.rolling(window)` produit des fenêtres pour des *agrégats* (moyenne mobile…), pas des tenseurs `(X, Y)` — utile pour le feature engineering, pas pour nourrir un LSTM.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Keras — Timeseries data loading (`timeseries_dataset_from_array`)](https://keras.io/api/data_loading/timeseries/)
- [PyTorch — Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [NumPy — `stride_tricks.as_strided`](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)
- [Nixtla — mlforecast & neuralforecast](https://nixtla.github.io/)
- [PyTorch Forecasting — `TimeSeriesDataSet`](https://pytorch-forecasting.readthedocs.io/)
- [Darts — Unit8](https://unit8co.github.io/darts/)
<!-- #endregion -->
