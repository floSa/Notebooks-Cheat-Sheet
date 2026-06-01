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
# 📈 Time Series — Wiki des méthodes de forecasting (stat · ML · DL · foundation models)
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki de référence** du *forecasting* de séries temporelles. Il offre un **panorama exécuté** des grandes familles de méthodes — statistiques classiques, machine learning, deep learning, et **foundation models** (la rupture 2024-2026) — avec, pour chacune, *quand* l'utiliser, le *code testé*, et une *évaluation rigoureuse*.

Positionnement dans le corpus TS :

- `[[TS_Time_Series_Intro]]` — tutoriel **débutant** (décomposition, stationnarité, ACF/PACF, premier modèle).
- **Ce notebook** — **panorama** pour *choisir* une méthode et comparer les familles.
- `[[TS_ARIMA]]` — détail **ARIMA/SARIMA** (Box-Jenkins, diagnostic résidus).
- `[[TS_Generer_Sequence]]` — fabrication de fenêtres (sliding windows) pour le DL.
- `[[TS_Maintenance_Predictive]]` — *case study* industriel (RUL, anomalies).

**Dataset** : *Air Passengers* (mensuel, 1949-1960), chargé programmatiquement via `statsmodels` — mutualisé avec `[[TS_Time_Series_Intro]]` et `[[TS_ARIMA]]`. Un *panel synthétique* déterministe est généré à la section 6 pour illustrer les modèles globaux multi-séries.
<!-- #endregion -->

<!-- #region -->
Plan :

1. Quelle méthode choisir ? (matrice de décision 2026)
2. Dataset & split temporel
3. Baselines naïves
4. Méthodes statistiques (ETS, SARIMA, AutoARIMA, Theta)
5. Prophet
6. Machine Learning (lag features + modèle global)
7. Bench unifié Nixtla (`statsforecast`)
8. Deep Learning (LSTM exécuté + panorama des architectures)
9. Foundation models (zero-shot)
10. Métriques & validation temporelle (+ conformal prediction)
11. Comparatif final
12. Stack 2026 · 13. Pièges · 14. Références
<!-- #endregion -->

<!-- #region -->
## 0. Setup
<!-- #endregion -->

<!-- #region -->
On importe les libs, on fixe la reproductibilité (`set_seed` propage la graine à NumPy **et** PyTorch), et on définit la **palette graphique**. Ce notebook traite des séries *neutres* (pas de notion bon/mauvais) → couleurs neutres de la charte (`primary_1`, `accent_dark`, `lavender`…).
<!-- #endregion -->

```python
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    """Fixe les graines NumPy (et PyTorch si dispo) pour la reproductibilité."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


SEED = 42
set_seed(SEED)

# Palette CHART — séries neutres
PALETTE = {
    "primary_1": "#00798c", "mauvais": "#d1495b", "moyen": "#edae49",
    "accent": "#66a182", "accent_dark": "#2e4057", "lavender": "#9d83b8",
    "dusty_rose": "#b8848e", "beige": "#c9b78b",
}
```

<!-- #region -->
On définit **une fois** les métriques de forecasting (réutilisées dans tout le notebook). Détails et formules en section 10 ; ici on retient surtout **MASE** (comparable entre séries) et **WAPE** (robuste, agrège bien sur N séries).
<!-- #endregion -->

```python
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Indéfinie si y≈0."""
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE (%) — gère mieux les valeurs proches de 0."""
    return float(np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))) * 100)


def mase(y: np.ndarray, yhat: np.ndarray, y_train: np.ndarray, m: int = 1) -> float:
    """Mean Absolute Scaled Error : MAE rapportée à celle du naïf saisonnier
    (calculée sur le train). <1 = mieux que le naïf saisonnier."""
    denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(mae(y, yhat) / denom)


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — robuste à y=0, agrège sur N séries."""
    return float(np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)))
```

<!-- #region -->
Ces fonctions typées sont des **snippets réutilisables** : copiables tels quels dans n'importe quel projet de forecasting.
<!-- #endregion -->

<!-- #region -->
## 1. Quelle méthode choisir ? (matrice de décision 2026)
<!-- #endregion -->

<!-- #region -->
Il n'existe **pas** de méthode universellement meilleure (cf. débat « transformers pour la TS », section 8). Le bon réflexe : partir d'une baseline, puis monter en complexité **seulement si le gain le justifie**.

| Situation | Méthode recommandée | Pourquoi |
|---|---|---|
| Série courte (<200 pts), saisonnière | **ETS** ou **SARIMA** | Interprétables, peu de données nécessaires |
| Calendrier complexe (fêtes, événements) | **Prophet** | Holidays / regressors intégrés |
| Des **centaines/milliers** de séries similaires | **Global ML** (LightGBM/XGBoost via `mlforecast`) | Un modèle apprend la dynamique *cross-séries* |
| Longues séries + covariables | **TFT, N-HiTS, PatchTST** | Le DL capture les interactions complexes |
| Pas de temps pour entraîner | **Foundation model** (Chronos-2, TimesFM, TimeGPT) | Forecast *zero-shot*, aucune phase d'entraînement |
| Changements de régime / ruptures | **Bayesian structural** (BSTS, PyMC) | Modélise explicitement les ruptures |
| Très grande échelle (millions de séries IoT) | **`statsforecast`** (Numba, parallèle) | Scale quasi-linéaire |

**Règle d'or** : toujours rapporter le score **relatif à la baseline naïve** (→ MASE). Un modèle DL avec MASE ≥ 1 ne sert à rien.
<!-- #endregion -->

<!-- #region -->
## 2. Dataset & split temporel
<!-- #endregion -->

<!-- #region -->
On charge *Air Passengers* (nombre mensuel de passagers aériens, 1949-1960 — 144 points) via `statsmodels`. La série présente une **tendance croissante** et une **saisonnalité dont l'amplitude augmente** (→ saison *multiplicative*, important pour ETS).

Le **split est strictement temporel** : les 12 derniers mois en test (jamais de `shuffle` — ce serait une fuite). On prépare aussi `RESULTS` + `log_result`, qui calcule les 5 métriques de chaque modèle et les empile pour le comparatif final.
<!-- #endregion -->

```python
from statsmodels.datasets import get_rdataset

ap = get_rdataset("AirPassengers", "datasets").data
y_all = ap["value"].to_numpy(dtype=float)
idx = pd.date_range("1949-01-01", periods=len(y_all), freq="MS")
ts = pd.Series(y_all, index=idx, name="passengers")

H = 12  # horizon = 12 mois
train, test = ts.iloc[:-H], ts.iloc[-H:]
M = 12  # période saisonnière
print(f"Série : {len(ts)} obs ({ts.index[0]:%Y-%m} → {ts.index[-1]:%Y-%m})")
print(f"Train : {len(train)} | Test : {len(test)} (horizon {H})")

RESULTS: dict[str, dict[str, float]] = {}


def log_result(name: str, yhat: np.ndarray) -> None:
    """Calcule les 5 métriques d'un forecast vs `test` et l'empile dans RESULTS."""
    yv = test.to_numpy()
    RESULTS[name] = {
        "RMSE": rmse(yv, yhat), "MAE": mae(yv, yhat), "sMAPE": smape(yv, yhat),
        "MASE": mase(yv, yhat, train.to_numpy(), M), "WAPE": wape(yv, yhat),
    }


fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train.index, train.values, color=PALETTE["primary_1"], label="Train")
ax.plot(test.index, test.values, color=PALETTE["accent_dark"], lw=2, label="Test")
ax.set_title("Air Passengers — split temporel"); ax.legend()
plt.show()
```

<!-- #region -->
`train` et `test` sont désormais **fixes** et réutilisés par tous les modèles. Toute statistique (moyenne, écart-type, paramètres) doit être estimée sur `train` **seulement**.
<!-- #endregion -->

<!-- #region -->
## 3. Baselines naïves
<!-- #endregion -->

<!-- #region -->
Trois baselines incontournables :

- **Naive** : répète la dernière valeur. Référence des séries sans saison.
- **Seasonal-naive** : répète la dernière saison complète (ici 12 mois). **La** vraie baseline d'une série saisonnière — c'est le dénominateur de MASE.
- **Mean** : moyenne historique (volontairement faible, pour cadrer).

Un modèle digne d'intérêt doit **battre la seasonal-naive** (MASE < 1).
<!-- #endregion -->

```python
def naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Répète la dernière valeur observée."""
    return np.repeat(train.iloc[-1], h)


def seasonal_naive_forecast(train: pd.Series, h: int, m: int = 12) -> np.ndarray:
    """Répète les m dernières valeurs (saison)."""
    last_season = train.iloc[-m:].to_numpy()
    reps = int(np.ceil(h / m))
    return np.tile(last_season, reps)[:h]


def mean_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Moyenne historique."""
    return np.repeat(train.mean(), h)


fc_naive = naive_forecast(train, H)
fc_snaive = seasonal_naive_forecast(train, H, M)
fc_mean = mean_forecast(train, H)
for name, fc in [("Naive", fc_naive), ("SeasonalNaive", fc_snaive), ("Mean", fc_mean)]:
    log_result(name, fc)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train.index[-36:], train.values[-36:], color=PALETTE["primary_1"], label="Train")
ax.plot(test.index, test.values, color=PALETTE["accent_dark"], lw=2, label="Test")
ax.plot(test.index, fc_naive, "--", color=PALETTE["mauvais"], label="Naive")
ax.plot(test.index, fc_snaive, "--", color=PALETTE["accent"], label="Seasonal-naive")
ax.plot(test.index, fc_mean, ":", color=PALETTE["beige"], label="Mean")
ax.set_title("Baselines naïves"); ax.legend()
plt.show()
```

<!-- #region -->
La seasonal-naive (MASE = 1 par construction) suit bien la forme saisonnière mais rate la tendance croissante : tous les modèles ci-dessous cherchent à faire mieux.
<!-- #endregion -->

<!-- #region -->
## 4. Méthodes statistiques
<!-- #endregion -->

<!-- #region -->
### 4.1 ETS (lissage exponentiel)
<!-- #endregion -->

<!-- #region -->
**ETS(Error, Trend, Seasonality)** — *Holt-Winters* en est un cas particulier. Le modèle décompose explicitement la série en **niveau**, **tendance** et **saisonnalité**, mis à jour par lissage exponentiel avec des coefficients $\alpha, \beta, \gamma \in [0, 1]$ :

$$\ell_t = \alpha\,\frac{y_t}{s_{t-m}} + (1-\alpha)(\ell_{t-1}+b_{t-1}), \quad b_t = \beta(\ell_t-\ell_{t-1})+(1-\beta)b_{t-1}, \quad s_t = \gamma\,\frac{y_t}{\ell_t}+(1-\gamma)s_{t-m}$$

Notation : E ∈ {A, M}, T ∈ {N, A, A_d (damped)}, S ∈ {N, A, M}. Ici on prend **saison multiplicative** (`seasonal="mul"`) car l'amplitude croît avec le niveau. La sélection automatique du triplet (par AIC) est faite par `AutoETS` (section 7).
<!-- #endregion -->

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ets = ExponentialSmoothing(
    train, trend="add", seasonal="mul", seasonal_periods=M,
    initialization_method="estimated",
).fit()
fc_ets = np.asarray(ets.forecast(H))
log_result("ETS", fc_ets)
print(f"ETS — RMSE {RESULTS['ETS']['RMSE']:.2f} | MASE {RESULTS['ETS']['MASE']:.3f}")
```

<!-- #region -->
ETS est un **excellent baseline** : souvent difficile à battre sur les séries courtes saisonnières, et très rapide.
<!-- #endregion -->

<!-- #region -->
### 4.2 SARIMA & AutoARIMA
<!-- #endregion -->

<!-- #region -->
**SARIMA**(p,d,q)(P,D,Q)$_s$ combine différenciation, parties AR/MA et leurs analogues saisonniers. La théorie complète (méthodologie Box-Jenkins, lecture ACF/PACF, diagnostic des résidus) est dans `[[TS_ARIMA]]`.

En 2026, on ne choisit quasiment plus les ordres à la main : **`auto_arima`** (pmdarima) explore les combinaisons et sélectionne par AIC. On compare ici un SARIMA fixé manuellement à la version auto.
<!-- #endregion -->

```python
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

sarima = ARIMA(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, M)).fit()
fc_sarima = np.asarray(sarima.forecast(H))
log_result("SARIMA(manuel)", fc_sarima)

auto = pm.auto_arima(train, seasonal=True, m=M, suppress_warnings=True,
                     stepwise=True, error_action="ignore")
fc_auto = np.asarray(auto.predict(n_periods=H))
log_result("AutoARIMA", fc_auto)
print(f"SARIMA manuel (1,1,1)(1,1,1)[12] — MASE {RESULTS['SARIMA(manuel)']['MASE']:.3f}")
print(f"AutoARIMA {auto.order}{auto.seasonal_order} — MASE {RESULTS['AutoARIMA']['MASE']:.3f}")
```

<!-- #region -->
`auto_arima` retient ici un ordre différent du choix manuel et fait mieux : la sélection automatique évite le biais humain sur le choix des ordres.
<!-- #endregion -->

<!-- #region -->
### 4.3 Theta
<!-- #endregion -->

<!-- #region -->
La **méthode Theta** (Assimakopoulos & Nikolopoulos, 2000) décompose la série en deux *Theta lines* (l'une aplatit la courbure, l'autre l'amplifie) puis les recombine. Étonnamment compétitive : régulièrement dans le top 5 des compétitions **M3/M4** malgré sa simplicité. Implémentée nativement dans `statsmodels` (`ThetaModel`) et dans Nixtla (`AutoTheta`).
<!-- #endregion -->

```python
from statsmodels.tsa.forecasting.theta import ThetaModel

theta = ThetaModel(train, period=M).fit()
fc_theta = np.asarray(theta.forecast(H))
log_result("Theta", fc_theta)
print(f"Theta — MASE {RESULTS['Theta']['MASE']:.3f}")
```

<!-- #region -->
À garder en tête comme baseline « gratuite » avant de sortir l'artillerie ML/DL.
<!-- #endregion -->

<!-- #region -->
## 5. Prophet (Meta)
<!-- #endregion -->

<!-- #region -->
**Prophet** est un modèle de **régression additive bayésienne** :

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

où $g(t)$ est la tendance (linéaire ou logistique avec points de rupture), $s(t)$ les saisonnalités (séries de Fourier), $h(t)$ l'effet des jours fériés/événements.

**Forces** : marche *out-of-the-box* du mensuel à l'horaire, robuste aux trous et outliers, holidays par pays intégrés, API très simple. **Faiblesses** : peut sur-lisser les motifs courts, et se fait souvent dépasser par un GBM global ou un foundation model sur les séries longues à dynamique fine.

> ⚠️ **Non exécuté dans ce notebook** : `prophet` n'est pas dans l'environnement (backend `cmdstan` lourd à construire). L'usage tient en : construire un DataFrame `{ds, y}`, instancier `Prophet(yearly_seasonality=True)`, appeler `.fit(df)` puis `.predict(future)` — voir la doc officielle. Prophet reste un **baseline business** pertinent en 2026 quand on veut du rapide et interprétable avec calendrier.
<!-- #endregion -->

<!-- #region -->
## 6. Machine Learning (lag features)
<!-- #endregion -->

<!-- #region -->
Une fois la série transformée en **features décalées** (lags, moyennes glissantes, variables calendaires), n'importe quel régresseur tabulaire s'applique. **XGBoost** et **LightGBM** dominent en 2026.

⚠️ **Piège de fuite** : un lag doit toujours être `shift(k)` avec `k ≥ 1`, et un rolling doit porter sur des valeurs **passées** (`shift(1).rolling(...)`). Ici `make_lag_features` ne garde que des lignes où tous les lags sont définis (`dropna`), ce qui empêche de voir le présent/futur.
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def make_lag_features(s: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 6, 12)) -> pd.DataFrame:
    """Transforme une série en table supervisée (lags + mois). dropna pour éviter
    toute fuite : chaque ligne ne voit que des valeurs passées."""
    df = pd.DataFrame({"y": s})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = df.index.month
    return df.dropna()


feat = make_lag_features(ts)
split_date = train.index[-1]
tr_feat = feat[feat.index <= split_date]
te_feat = feat[feat.index > split_date]
X_tr, y_tr = tr_feat.drop(columns="y"), tr_feat["y"]
X_te, y_te = te_feat.drop(columns="y"), te_feat["y"]

rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
xgr = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       random_state=SEED, n_jobs=1).fit(X_tr, y_tr)
log_result("RF(lags)", rf.predict(X_te))
log_result("XGB(lags)", xgr.predict(X_te))
print(f"RF lags — MASE {RESULTS['RF(lags)']['MASE']:.3f} | "
      f"XGB lags — MASE {RESULTS['XGB(lags)']['MASE']:.3f}")
```

<!-- #region -->
Sur **une seule** série courte, les arbres peinent à extrapoler la tendance (ils ne prédisent jamais hors de l'intervalle vu en entraînement) → ils restent ici derrière ETS. Leur force se révèle sur **beaucoup de séries**.
<!-- #endregion -->

<!-- #region -->
### 6.1 Modèle global (N séries)
<!-- #endregion -->

<!-- #region -->
L'idée **ML 2026** : au lieu d'un modèle par série, on entraîne **un seul** modèle sur l'ensemble empilé de toutes les séries. Il apprend les patterns communs (saisonnalité, formes) et **passe à l'échelle** (millions de séries IoT/retail). C'est exactement ce qu'industrialise `mlforecast` (Nixtla) ; on le fait ici à la main avec XGBoost pour montrer le mécanisme.

On génère un **panel synthétique déterministe** de 20 séries saisonnières hétérogènes (niveau, pente, amplitude, phase variables), on empile leurs features, on entraîne un XGBoost global, puis on prévoit chaque série en **récursif** (chaque prédiction réinjectée comme historique).
<!-- #endregion -->

```python
def make_panel(n_series: int = 20, length: int = 120, m: int = 12,
               seed: int = SEED) -> dict[str, pd.Series]:
    """Panel déterministe de séries saisonnières hétérogènes (démo modèle global)."""
    rng = np.random.default_rng(seed)
    panel: dict[str, pd.Series] = {}
    didx = pd.date_range("2010-01-01", periods=length, freq="MS")
    for i in range(n_series):
        level = rng.uniform(50, 200)
        slope = rng.uniform(0.2, 2.0)
        amp = rng.uniform(10, 60)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 5, length)
        vals = (level + slope * np.arange(length)
                + amp * np.sin(2 * np.pi * np.arange(length) / m + phase) + noise)
        panel[f"s{i}"] = pd.Series(vals, index=didx)
    return panel


def lag_matrix(s: pd.Series, lags: tuple[int, ...]) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = df.index.month
    return df.dropna()


def recursive_forecast(model, history: pd.Series, h: int,
                       lags: tuple[int, ...], freq: str = "MS") -> np.ndarray:
    """Forecast récursif h pas : chaque prédiction devient le nouvel historique."""
    hist = history.copy()
    preds = []
    for _ in range(h):
        next_date = hist.index[-1] + pd.tseries.frequencies.to_offset(freq)
        row = {f"lag_{lag}": hist.iloc[-lag] for lag in lags}
        row["month"] = next_date.month
        x = pd.DataFrame([row])[[f"lag_{lag}" for lag in lags] + ["month"]]
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        hist = pd.concat([hist, pd.Series([yhat], index=[next_date])])
    return np.asarray(preds)


PANEL_LAGS = (1, 2, 3, 12)
panel = make_panel()
rows, panel_train, panel_test = [], {}, {}
for sid, s in panel.items():
    s_tr, s_te = s.iloc[:-H], s.iloc[-H:]
    panel_train[sid], panel_test[sid] = s_tr, s_te
    rows.append(lag_matrix(s_tr, PANEL_LAGS))
big = pd.concat(rows, ignore_index=True)            # toutes les séries empilées
Xg, yg = big.drop(columns="y"), big["y"]
global_model = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                random_state=SEED, n_jobs=1).fit(Xg, yg)

all_true, all_pred = [], []
for sid in panel:
    all_true.append(panel_test[sid].to_numpy())
    all_pred.append(recursive_forecast(global_model, panel_train[sid], H, PANEL_LAGS))
gt, gp = np.concatenate(all_true), np.concatenate(all_pred)
print(f"Modèle global (1 XGB / {len(panel)} séries) — "
      f"WAPE agrégé {wape(gt, gp):.4f} | sMAPE {smape(gt, gp):.2f}%")
```

<!-- #region -->
Un **unique** modèle prévoit correctement 20 séries différentes (WAPE ≈ 4 %). C'est le pattern à privilégier dès qu'on a un grand nombre de séries homogènes — bien plus économe que 20 modèles séparés.
<!-- #endregion -->

<!-- #region -->
## 7. Bench unifié (Nixtla `statsforecast`)
<!-- #endregion -->

<!-- #region -->
L'écosystème **Nixtla** (`statsforecast`, `mlforecast`, `neuralforecast`) impose en 2026 une API unifiée `.fit().predict()` sur un DataFrame **long** `[unique_id, ds, y]`, avec des modèles `Auto*` parallélisés (Numba) et un `evaluate` outillé.

On rejoue ici, en **une seule API**, les modèles statistiques de la section 4 : `AutoARIMA`, `AutoETS`, `AutoTheta`, `SeasonalNaive`.
<!-- #endregion -->

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

sf_train = pd.DataFrame({"unique_id": "air", "ds": train.index, "y": train.to_numpy()})
sf = StatsForecast(
    models=[AutoARIMA(season_length=M), AutoETS(season_length=M),
            AutoTheta(season_length=M), SeasonalNaive(season_length=M)],
    freq="MS", n_jobs=1,
)
sf.fit(sf_train)
sf_fc = sf.predict(h=H)
print("statsforecast — prévisions (head) :")
print(sf_fc.head(3).round(1).to_string(index=False))

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse as uf_rmse, smape as uf_smape

sf_eval_df = sf_fc.merge(
    pd.DataFrame({"unique_id": "air", "ds": test.index, "y": test.to_numpy()}),
    on=["unique_id", "ds"],
)
ev = evaluate(sf_eval_df, metrics=[uf_rmse, uf_smape],
              models=["AutoARIMA", "AutoETS", "AutoTheta", "SeasonalNaive"])
print("\nutilsforecast.evaluate :")
print(ev.round(3).to_string(index=False))
log_result("AutoTheta(SF)", sf_fc["AutoTheta"].to_numpy())
```

<!-- #region -->
La **même API** s'étend à `mlforecast` (modèles ML : LightGBM/XGBoost/sklearn) et `neuralforecast` (30+ modèles DL : PatchTST, TFT, N-HiTS…). C'est le socle recommandé pour industrialiser un bench multi-modèles.
<!-- #endregion -->

<!-- #region -->
## 8. Deep Learning
<!-- #endregion -->

<!-- #region -->
### 8.1 LSTM minimal (PyTorch)
<!-- #endregion -->

<!-- #region -->
Un **LSTM** apprend une dépendance temporelle via des fenêtres glissantes (sliding windows, cf. `[[TS_Generer_Sequence]]`). On **normalise sur le train uniquement** (anti-leak), on entraîne sur des fenêtres `(seq_len) → valeur suivante`, puis on prévoit en **récursif** sur 12 mois avant de dé-normaliser.

> Sur 132 points c'est volontairement *sous-dimensionné* : le DL n'est pertinent qu'avec beaucoup de données / covariables. L'objectif est de montrer une **vraie** boucle d'entraînement exécutée (pas du pseudo-code).
<!-- #endregion -->

```python
import torch
import torch.nn as nn

set_seed(SEED)
L = 12  # longueur de fenêtre

mu, sigma = train.mean(), train.std()
train_n = ((train - mu) / sigma).to_numpy(dtype=np.float32)


def make_windows(series: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Découpe une série en fenêtres glissantes (X) -> valeur suivante (y)."""
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.asarray(X)[..., None], np.asarray(y)  # X: (n, seq_len, 1)


Xw, yw = make_windows(train_n, L)
Xw_t, yw_t = torch.tensor(Xw), torch.tensor(yw)[:, None]


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # dernier timestep


model = LSTMForecaster()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
model.train()
for epoch in range(300):
    opt.zero_grad()
    loss = loss_fn(model(Xw_t), yw_t)
    loss.backward()
    opt.step()

model.eval()
window = train_n[-L:].copy()
preds_n = []
with torch.no_grad():
    for _ in range(H):
        x = torch.tensor(window[None, :, None], dtype=torch.float32)
        yhat = float(model(x).item())
        preds_n.append(yhat)
        window = np.append(window[1:], yhat).astype(np.float32)
fc_lstm = np.asarray(preds_n) * sigma + mu  # dé-normalisation
log_result("LSTM", fc_lstm)
print(f"LSTM (loss finale {loss.item():.4f}) — MASE {RESULTS['LSTM']['MASE']:.3f}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train.index[-36:], train.values[-36:], color=PALETTE["primary_1"], label="Train")
ax.plot(test.index, test.values, color=PALETTE["accent_dark"], lw=2, label="Test")
ax.plot(test.index, fc_lstm, "--", color=PALETTE["lavender"], label="LSTM")
ax.set_title("LSTM PyTorch — forecast récursif 12 mois"); ax.legend()
plt.show()
```

<!-- #region -->
### 8.2 Panorama des architectures DL
<!-- #endregion -->

<!-- #region -->
Les architectures spécialisées TS, à mobiliser via `neuralforecast` ou `darts` (pas à réimplémenter) :

| Modèle | Idée clé | Quand |
|---|---|---|
| **DeepAR** (Amazon) | RNN autorégressif **probabiliste** (quantiles) | Demande, stocks de pièces |
| **N-BEATS / N-HiTS** | Stacks de MLP backcast/forecast ; N-HiTS = multi-rate, plus rapide | Univarié pur, baselines DL solides |
| **TFT** (Google) | Transformer + LSTM + attention quantile, **interprétable** | Multi-horizon avec covariables |
| **TiDE** | MLP encoder-decoder, plus simple que TFT | Baseline DL rapide |
| **PatchTST** | Transformer par *patches* temporels | Champion des benchmarks long-horizon |
| **iTransformer** | Attention *inversée* (sur les variables) | Multivarié |
| **TSMixer** | MLP-Mixer pur (pas d'attention) ; 2-3× moins de mémoire | Forte alternative aux transformers |

**Nuance importante (2026)** : un *position paper* récent conclut que **le débat « transformers vs reste » se solde par un match nul** — pas d'architecture universelle. Les modèles **linéaires** battent encore les transformers sur de nombreux benchmarks, et la communauté (TSLib/THUML) recommande désormais des suites d'évaluation récentes (**GIFT-Eval**, **fev-bench**) plutôt que les benchmarks historiques jugés « plus forcément significatifs ».
<!-- #endregion -->

<!-- #region -->
## 9. Foundation models (zero-shot)
<!-- #endregion -->

<!-- #region -->
La rupture 2024-2026 : des modèles **pré-entraînés sur des milliards de séries** qui prévoient en **zero-shot**, sans réentraînement. Ils sont devenus la **nouvelle baseline** — ils battent souvent un modèle statistique tuné, *out of the box*.

| Modèle | Auteur | Sortie | Statut |
|---|---|---|---|
| **Chronos-2** | Amazon (oct 2025) | encoder-only, 120M | **Apache-2**, univarié + multivarié + covariables, SOTA GIFT-Eval, intégré SageMaker/AutoGluon |
| **TimesFM 2.5** | Google (sept 2025) | 200M, contexte 16k, tête quantile | Open-source, dispo **BigQuery ML** (GA nov 2025) |
| **Moirai 2.0** | Salesforce (août 2025) | decoder-only, quantile, multi-token | Open-source, **#1 MASE** sur GIFT-Eval (modèles non-fuyants), 44 % plus rapide que v1 |
| **TimeGPT** | Nixtla | API propriétaire | Production-ready, SDK Python actif (essai 30 j) |
| **Lag-Llama / Timer-XL** | ServiceNow / THUML | probabiliste / large contexte | Open, plus expérimentaux |

**Workflow zero-shot** (illustration, non exécutée ici car téléchargement de modèle) : on charge le pipeline — p.ex. `ChronosPipeline.from_pretrained("amazon/chronos-2")` — puis on appelle `pipe.predict(context=serie, prediction_length=12)`, qui renvoie directement des **quantiles** (forecast probabiliste natif).

**Quand préférer le classique/ML** : beaucoup de séries homogènes + besoin de contrôler/auditer la dynamique → un **GBM global** (section 6.1) reste souvent meilleur et bien moins coûteux. Et quand l'interprétabilité prime → ETS/SARIMA/Prophet.
<!-- #endregion -->

<!-- #region -->
## 10. Métriques & validation temporelle
<!-- #endregion -->

<!-- #region -->
### 10.1 Métriques
<!-- #endregion -->

<!-- #region -->
| Métrique | Formule | Forces / pièges |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum\|y-\hat y\|$ | Robuste aux outliers ; échelle = série |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y-\hat y)^2}$ | Pénalise les gros écarts |
| **MAPE** | $\frac{100}{n}\sum \frac{\|y-\hat y\|}{\|y\|}$ | Relatif, intuitif. **Explose si $y\approx0$** |
| **sMAPE** | $\frac{100}{n}\sum \frac{2\|y-\hat y\|}{\|y\|+\|\hat y\|}$ | Symétrique, gère mieux $y\to0$ |
| **MASE** | $\dfrac{\text{MAE}}{\text{MAE}_{\text{naïf saisonnier}}}$ | **< 1 = mieux que le naïf**. Comparable entre séries |
| **WAPE** | $\dfrac{\sum\|y-\hat y\|}{\sum\|y\|}$ | Robuste à $y=0$, agrège bien sur N séries |

**Recommandation 2026** : comparer méthodes/datasets → **MASE** et/ou **WAPE** ; reporter au métier → **MAPE** (sur séries non-nulles). Pour les forecasts probabilistes (DeepAR, foundation models) → *quantile loss* / *pinball loss*.
<!-- #endregion -->

<!-- #region -->
### 10.2 Validation
<!-- #endregion -->

<!-- #region -->
- **Single holdout** — simple, suffit pour un POC.
- **Expanding window** (`TimeSeriesSplit`) — chaque fold ajoute du passé au train.
- **Rolling window** — taille de train fixe, fenêtre qui glisse (utile si la dynamique change).
- **Backtesting multi-cutoffs** (Nixtla `cross_validation`) — automatisé et parallélisé.

**Erreurs fatales** : ❌ `train_test_split(shuffle=True)` (leak temporel) · ❌ feature engineering / normalisation sur tout le dataset puis split (leak) · ❌ comparer des modèles sur des cutoffs différents.

On illustre un **backtest expanding-window** (5 folds) sur ETS : on refit le modèle à chaque fold et on mesure le RMSE hors-échantillon.
<!-- #endregion -->

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=H)
fold_rmse = []
fig, ax = plt.subplots(figsize=(11, 4))
for k, (tr_idx, te_idx) in enumerate(tscv.split(ts)):
    s_tr, s_te = ts.iloc[tr_idx], ts.iloc[te_idx]
    m_fit = ExponentialSmoothing(s_tr, trend="add", seasonal="mul",
                                 seasonal_periods=M, initialization_method="estimated").fit()
    pred = np.asarray(m_fit.forecast(len(s_te)))
    fold_rmse.append(rmse(s_te.to_numpy(), pred))
    ax.plot(s_te.index, s_te.values, color=PALETTE["accent_dark"], lw=1.5)
    ax.plot(s_te.index, pred, "--", color=PALETTE["moyen"])
    ax.axvline(s_tr.index[-1], color=PALETTE["beige"], ls=":", alpha=0.6)
ax.set_title("Backtest expanding-window (ETS, 5 folds) — vrais vs prévus")
plt.show()
print(f"RMSE par fold : {[round(x, 1) for x in fold_rmse]}")
print(f"RMSE moyen {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")
```

<!-- #region -->
La dispersion entre folds (± écart-type) est une **mesure d'incertitude** bien plus honnête qu'un score sur un unique holdout.
<!-- #endregion -->

<!-- #region -->
### 10.3 Conformal prediction (2026)
<!-- #endregion -->

<!-- #region -->
Le **shift méthodologique** majeur de 2026 : produire des **intervalles de prédiction avec garantie de couverture distribution-free**, sans hypothèse paramétrique. Principe (*split-conformal*) :

1. Découper le train en *proper-train* + *calibration*.
2. Calculer les **scores de non-conformité** (ici $|y-\hat y|$ sur la calibration).
3. Le **quantile empirique** (corrigé en échantillon fini) de ces scores donne la demi-largeur de l'intervalle pour la couverture visée $1-\alpha$.

⚠️ En TS, l'hypothèse d'**échangeabilité** est violée (dépendance temporelle) → des variantes adaptées existent (ACI, EnbPI, `conformalForecast`). On montre ici la version *split* de base, suffisante pour cadrer l'incertitude.
<!-- #endregion -->

```python
proper_tr, calib = train.iloc[:-H], train.iloc[-H:]
m_cal = ExponentialSmoothing(proper_tr, trend="add", seasonal="mul",
                             seasonal_periods=M, initialization_method="estimated").fit()
calib_pred = np.asarray(m_cal.forecast(H))
residuals = np.abs(calib.to_numpy() - calib_pred)  # scores de non-conformité

alpha = 0.10
n_cal = len(residuals)
q_level = min(1.0, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)  # correction échantillon fini
q_hat = float(np.quantile(residuals, q_level))

lower, upper = fc_ets - q_hat, fc_ets + q_hat
coverage = float(np.mean((test.to_numpy() >= lower) & (test.to_numpy() <= upper)))
print(f"Conformal (90% visé) — q={q_hat:.1f} | couverture empirique {coverage:.0%}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(train.index[-24:], train.values[-24:], color=PALETTE["primary_1"], label="Train")
ax.plot(test.index, test.values, color=PALETTE["accent_dark"], lw=2, label="Test")
ax.plot(test.index, fc_ets, "--", color=PALETTE["moyen"], label="ETS")
ax.fill_between(test.index, lower, upper, color=PALETTE["moyen"], alpha=0.2,
                label="Intervalle conformal 90%")
ax.set_title("Conformal prediction — intervalle distribution-free"); ax.legend()
plt.show()
```

<!-- #region -->
La couverture empirique sur le test est proche de la cible (90 %), sans avoir supposé de loi sur les résidus : c'est tout l'intérêt du conformal.
<!-- #endregion -->

<!-- #region -->
## 11. Comparatif final
<!-- #endregion -->

<!-- #region -->
On regroupe **tous les modèles exécutés** (baselines, statistiques, ML, DL, Nixtla) dans un tableau trié par **MASE**. Le barplot met en évidence la barre **MASE = 1** : tout ce qui est en dessous bat le naïf saisonnier.
<!-- #endregion -->

```python
results_df = (pd.DataFrame(RESULTS).T
              .sort_values("MASE")
              .round({"RMSE": 2, "MAE": 2, "sMAPE": 2, "MASE": 3, "WAPE": 4}))
print("=== COMPARATIF FINAL (trié par MASE) ===")
print(results_df.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
colors = [PALETTE["accent"] if v < 1 else PALETTE["mauvais"] for v in results_df["MASE"]]
ax.barh(results_df.index, results_df["MASE"], color=colors)
ax.axvline(1.0, color=PALETTE["accent_dark"], ls="--", label="MASE = 1 (naïf saisonnier)")
ax.invert_yaxis(); ax.set_xlabel("MASE (plus bas = mieux)")
ax.set_title("Comparatif des méthodes — MASE"); ax.legend()
plt.show()

best = results_df.index[0]
print(f"\nMeilleur modèle (MASE) : {best} = {results_df.loc[best, 'MASE']:.3f}")
print(f"Modèles battant le naïf saisonnier (MASE<1) : "
      f"{(results_df['MASE'] < 1).sum()}/{len(results_df)}")
```

<!-- #region -->
Sur cette série courte et propre, **ETS** domine : la leçon récurrente du forecasting est qu'un bon modèle statistique reste très dur à battre tant qu'on n'a pas *beaucoup* de séries ou de covariables. ML et DL ne prennent l'avantage qu'à l'échelle.
<!-- #endregion -->

<!-- #region -->
## 12. Stack 2026 recommandée
<!-- #endregion -->

<!-- #region -->
| Besoin | Outil |
|---|---|
| POC rapide / un modèle | `statsmodels`, `prophet`, `pmdarima` |
| Bench **statistique** multi-modèles | **`statsforecast`** 2.0.x (Nixtla, Numba) |
| Bench **ML** (lag features, global) | **`mlforecast`** (Nixtla) |
| Bench **DL** (30+ archis) | **`neuralforecast`** 3.x (Nixtla) ou **`darts`** 0.44 (Unit8) |
| **Foundation models** zero-shot | Chronos-2 / TimesFM 2.5 / Moirai 2.0 (open) · TimeGPT (API) |
| Bayésien / structural | PyMC, NumPyro, CmdStanPy |
| Anomalies / streaming | River, STUMPY (matrix profile), Merlion |
| Couverture la plus large | `sktime` (API sklearn-like) |
<!-- #endregion -->

<!-- #region -->
## 13. Pièges & anti-patterns
<!-- #endregion -->

<!-- #region -->
| ❌ Anti-pattern | ✅ Correctif |
|---|---|
| Pas de baseline | Toujours démarrer par naive / seasonal-naive (dénominateur MASE) |
| Comparer sur MSE seul | Rapporter **MASE/WAPE** (relatif, comparable) |
| `train_test_split(shuffle=True)` | Split **temporel** strict (jamais de shuffle) |
| FE / normalisation sur tout le dataset | Stats du **train uniquement** |
| Lag/rolling sans `shift(≥1)` | Garantir que chaque feature ne voit que le passé |
| Arbres pour extrapoler une tendance forte | Détrender d'abord, ou préférer ETS/ARIMA/DL |
| Un modèle par série sur 10 000 séries | **Modèle global** (mlforecast) |
| Forecast ponctuel sans incertitude | Intervalles (quantiles, **conformal**) |
| Choisir un seul horizon | Tester multi-step (récursif vs direct) |
| Croire à une archi DL universelle | Pas de gagnant universel — benchmarker (GIFT-Eval) |
<!-- #endregion -->

<!-- #region -->
## 14. Références
<!-- #endregion -->

<!-- #region -->
**Documentation**

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (fpp3) — https://otexts.com/fpp3/
- Nixtla (statsforecast / mlforecast / neuralforecast) — https://nixtlaverse.nixtla.io/
- darts (Unit8) — https://unit8co.github.io/darts/
- Chronos-2 (Amazon) — https://huggingface.co/amazon/chronos-2
- TimesFM (Google) — https://github.com/google-research/timesfm
- Moirai 2.0 (Salesforce) — https://www.salesforce.com/blog/moirai-2-0/
- GIFT-Eval (benchmark foundation models) — https://huggingface.co/spaces/Salesforce/GIFT-Eval

**Voir aussi**

- `[[TS_Time_Series_Intro]]` — tutoriel débutant
- `[[TS_ARIMA]]` — ARIMA/SARIMA en détail
- `[[TS_Generer_Sequence]]` — fenêtres pour le DL
- `[[TS_Maintenance_Predictive]]` — case study industriel
- `[[DS_Metrics_Reference]]` — métriques au-delà de la TS
<!-- #endregion -->
