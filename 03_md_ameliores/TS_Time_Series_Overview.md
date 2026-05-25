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
# 📈 Time Series — Wiki technique
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki** exhaustif du **forecasting de séries temporelles** : panorama des méthodes (statistiques, ML, DL, foundation models), métriques, validation, librairies 2026.

Complément du notebook **`TS_Time_Series_Intro`** (tutoriel débutant). À utiliser comme **référence** pour choisir une méthode.

Sommaire des familles couvertes :

1. **Statistiques classiques** : ARIMA, SARIMA, ETS, Holt-Winters, Theta.
2. **State-space** : Prophet, structural time series (BSTS).
3. **ML** : régression sur lag features (XGBoost, LightGBM), Nixtla `mlforecast`.
4. **DL** : N-BEATS, N-HiTS, TFT (Temporal Fusion Transformer), TiDE, DeepAR.
5. **Foundation models (2024-2026)** : TimeGPT, Chronos, Moirai, TimesFM — zero-shot forecasting.

Dataset : **Air Passengers** (mutualisé avec `TS_Time_Series_Intro` et `TS_ARIMA`).
<!-- #endregion -->

<!-- #region -->
## 1. Matrice de décision 2026
<!-- #endregion -->

<!-- #region -->
| Situation | Méthode reco | Pourquoi |
|---|---|---|
| Série courte (<200 pts), saisonnière | **SARIMA** ou **ETS** | Interprétables, peu de data nécessaire |
| Série mensuelle/hebdo avec calendrier complexe | **Prophet** | Holidays/regressors intégrés |
| Centaines/milliers de séries similaires | **Global ML** (LightGBM via `mlforecast`) | Apprend la dynamique cross-séries |
| Grosses séries (1k+ pts) avec covariables | **TFT, N-HiTS, DeepAR** | DL capture interactions complexes |
| Pas le temps de fine-tuner | **TimeGPT / Chronos** (zero-shot) | Foundation model, pas d'entraînement |
| Forte non-stationnarité, changements de régime | **Bayesian structural** (BSTS, PyMC) | Modélise explicitement les ruptures |
| Très grande échelle (millions de séries IoT) | **Nixtla `statsforecast` parallélisé** | Numba-accéléré, scale linéaire |
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
try:
    ap = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
    ap.columns = ["passengers"]
except Exception:
    from statsmodels.datasets import get_rdataset
    ap = get_rdataset("AirPassengers", "datasets").data.rename(columns={"value": "passengers"})
    ap.index = pd.date_range("1949-01", periods=len(ap), freq="MS")
    ap = ap[["passengers"]]

# Train/test split temporel (12 derniers mois en test)
train, test = ap.iloc[:-12], ap.iloc[-12:]
print(f"Train: {len(train)} obs | Test: {len(test)} obs")
```

<!-- #region -->
## 2. Statistiques classiques
<!-- #endregion -->

<!-- #region -->
### 2.1 ETS (Exponential Smoothing State Space)
<!-- #endregion -->

<!-- #region -->
**Holt-Winters** est un cas particulier d'ETS. Modélise explicitement niveau, tendance, saisonnalité avec des coefficients de lissage `α, β, γ ∈ [0, 1]`.

Notation **ETS(Error, Trend, Seasonality)** :

- E ∈ {A, M} : erreur additive ou multiplicative
- T ∈ {N, A, A_d, M, M_d} : pas de trend, additive, damped, ...
- S ∈ {N, A, M} : pas de saison, additive, multiplicative

Auto-sélection via AIC : `statsforecast.AutoETS`.
<!-- #endregion -->

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ets = ExponentialSmoothing(
    train["passengers"], trend="add", seasonal="mul", seasonal_periods=12,
).fit()
fcst_ets = ets.forecast(12)
print(f"ETS MAPE: {(abs(test['passengers'] - fcst_ets) / test['passengers']).mean():.3%}")
```

<!-- #region -->
### 2.2 ARIMA / SARIMA
<!-- #endregion -->

<!-- #region -->
**SARIMA**(p,d,q)(P,D,Q,s) — voir notebook **`TS_ARIMA`** pour la théorie complète et la méthodologie Box-Jenkins.

En 2026, on ne fait quasiment plus de selection manuelle des ordres — on utilise **`AutoARIMA`** de `pmdarima` ou `statsforecast` qui parcourt les combinaisons et choisit par AIC.
<!-- #endregion -->

```python
import pmdarima as pm

arima = pm.auto_arima(
    train["passengers"], seasonal=True, m=12,
    suppress_warnings=True, stepwise=True, error_action="ignore",
)
fcst_arima = arima.predict(n_periods=12)
print(f"AutoSARIMA order={arima.order} seasonal_order={arima.seasonal_order}")
print(f"AutoSARIMA MAPE: {(abs(test['passengers'].values - fcst_arima) / test['passengers'].values).mean():.3%}")
```

<!-- #region -->
### 2.3 Theta — le secret bien gardé
<!-- #endregion -->

<!-- #region -->
**Theta method** (Assimakopoulos & Nikolopoulos, 2000) — décompose la série en 2 "Theta lines" puis combine. Étonnamment compétitif sur les compétitions M3/M4 (top 5 régulièrement).

Implémentation : `statsforecast.AutoTheta`.
<!-- #endregion -->

<!-- #region -->
## 3. Prophet (Meta/Facebook)
<!-- #endregion -->

<!-- #region -->
**Prophet** est un modèle de **régression bayésienne additif** : `y(t) = trend(t) + seasonality(t) + holidays(t) + ε`. Conçu pour les séries business avec saisonnalités multiples et calendriers (jours fériés, événements).

**Forces** :

- Marche out-of-the-box sur des données mensuelles à hourly, avec trous.
- API simple, robuste aux outliers.
- Gère naturellement les holidays par pays.

**Faiblesses** :

- Peut sur-lisser les patterns courts (oscillation hebdomadaire fine).
- Moins précis que ML/DL sur séries longues à dynamique complexe.

Encore très utilisé en business 2026 quand on veut un baseline rapide et interprétable.
<!-- #endregion -->

<!-- #region -->
## 4. ML : lag features + GBM
<!-- #endregion -->

<!-- #region -->
Une fois qu'on construit des **lag features** (cf `TS_Time_Series_Intro` section 6), on peut entraîner n'importe quel modèle de régression. **LightGBM** et **XGBoost** sont les plus utilisés en 2026 — `mlforecast` (Nixtla) industrialise ça.

**Pattern Nixtla** :

```python
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

fcst = MLForecast(
    models={"lgbm": LGBMRegressor()},
    freq="MS",
    lags=[1, 2, 3, 6, 12],
    lag_transforms={1: [("rolling_mean", 12)]},
    date_features=["month"],
)
# Format Nixtla : DataFrame long avec colonnes [unique_id, ds, y]
# fcst.fit(df)
# preds = fcst.predict(h=12)
```

**Avantage clé** : Nixtla **gère naturellement N séries** (`unique_id`) → un seul modèle global pour 1k+ séries, qui apprend les patterns communs.
<!-- #endregion -->

<!-- #region -->
## 5. Deep Learning
<!-- #endregion -->

<!-- #region -->
| Modèle | Idée | Quand |
|---|---|---|
| **DeepAR** (Amazon) | RNN autoregressif probabiliste (intervalles de prédiction) | Stocks de pièces, demande |
| **N-BEATS** (ElementAI) | Stack de blocs MLP avec backcast/forecast | Forecasting univarié pur |
| **N-HiTS** | N-BEATS amélioré avec multi-rate sampling | Plus rapide que N-BEATS |
| **TFT** (Google) | Transformer + LSTM + attention quantile | Multi-horizon avec covariables |
| **TiDE** | MLP encoder-decoder, plus simple que TFT | Baseline DL rapide |
| **PatchTST / iTransformer** | Transformer par "patches" temporels | SOTA 2023-2025 sur benchmarks longs |

Bibliothèque de référence 2026 : **`neuralforecast`** (Nixtla) ou **`darts`** (Unit8).
<!-- #endregion -->

<!-- #region -->
## 6. Foundation models (2024-2026)
<!-- #endregion -->

<!-- #region -->
La révolution récente : des modèles **pré-entraînés sur des milliards de séries diverses** qui font du forecasting **zero-shot** (sans réentraîner).

| Modèle | Auteur | Param | Notes |
|---|---|---|---|
| **TimeGPT** | Nixtla | ~7B | Premier foundation model TS, SaaS API |
| **Chronos** | Amazon | T5-base/large | Open weights, basé sur quantization en tokens |
| **Moirai** | Salesforce | ~B | Multi-resolution, transformer |
| **TimesFM** | Google | 200M | Open, surprend par sa concision |
| **Lag-Llama** | open | ~200M | Open, transformer decoder |
| **TimeMoE** | open | MoE | Mixture of experts |

**Workflow zero-shot** :

```python
from chronos import ChronosPipeline
pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
forecast = pipe.predict(context=train_tensor, prediction_length=12)
```

**Quand les utiliser** :

- POC ou tâche où le coût d'annotation/entraînement est élevé.
- Nombreuses séries hétérogènes.
- Combinés à un fine-tuning sur ton domaine pour boost de qualité.

**Quand préférer classique/ML** :

- Beaucoup de séries similaires + besoin de contrôler la dynamique → un GBM global sera souvent meilleur.
- Interprétabilité requise.
<!-- #endregion -->

<!-- #region -->
## 7. Métriques de forecasting
<!-- #endregion -->

<!-- #region -->
| Métrique | Formule | Forces / pièges |
|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | Robuste outliers, échelle = série |
| **MSE / RMSE** | `mean((y - ŷ)²)` | Pénalise les gros écarts |
| **MAPE** | `mean(|y - ŷ| / |y|)` | Relatif, intuitif. **Catastrophe quand `y ≈ 0`** |
| **sMAPE** | `mean(|y - ŷ| / ((|y|+|ŷ|)/2))` | Symétrique, gère mieux `y=0` |
| **MASE** | `MAE / MAE_naive_saisonnier` | <1 = mieux que naïf. Comparable entre séries |
| **wQL** | quantile loss pondérée | Pour distributions/intervalles (DeepAR, NBeatsX) |
| **WAPE** | `Σ|y-ŷ| / Σ|y|` | Robuste à `y=0`, agrège bien sur N séries |

**Recommandation 2026** : pour comparer méthodes/datasets → **MASE** et/ou **WAPE**. Pour reporter à un métier → **MAPE** sur les séries non-zero.
<!-- #endregion -->

<!-- #region -->
## 8. Validation temporelle rigoureuse
<!-- #endregion -->

<!-- #region -->
- **Single holdout** — simple, suffit pour POC.
- **Expanding window CV** (`TimeSeriesSplit`) — chaque fold ajoute du train.
- **Rolling window CV** — taille de train fixe, fenêtre qui glisse.
- **Backtesting** Nixtla — multi-cutoffs automatiques avec parallélisation.

**Erreurs fatales à éviter** :

- ❌ `train_test_split(shuffle=True)` → leak temporel.
- ❌ Feature engineering sur tout le dataset puis split → leak.
- ❌ Normalisation avec stats globales → leak (utiliser stats du train uniquement).
- ❌ Comparer modèles sur des splits différents — toujours fixer les cutoffs.
<!-- #endregion -->

<!-- #region -->
## 9. Stack 2026 recommandée
<!-- #endregion -->

<!-- #region -->
- **POC rapide / un modèle** → `statsmodels`, `prophet`, `pmdarima`.
- **Bench multi-modèles statistiques** → **`statsforecast`** (Nixtla, ARIMA/ETS/Theta/Croston en parallèle).
- **Bench ML** → **`mlforecast`** (Nixtla, LightGBM/XGBoost/sklearn unifié).
- **Bench DL** → **`neuralforecast`** (Nixtla) ou **`darts`**.
- **Foundation models zero-shot** → TimeGPT (SaaS), Chronos (open), TimesFM (open).
- **Bayesian** → PyMC, NumPyro, CmdStanPy.
- **Anomaly detection** → River (streaming), STUMPY (matrix profile), Merlion.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [Forecasting: Principles and Practice — Hyndman](https://otexts.com/fpp3/)
- [Nixtla ecosystem](https://nixtla.github.io/)
- [Darts — Unit8](https://unit8co.github.io/darts/)
- [Chronos — Amazon AWS Labs](https://github.com/amazon-science/chronos-forecasting)
- [TimesFM — Google Research](https://github.com/google-research/timesfm)
- [Prophet — Meta](https://facebook.github.io/prophet/)
- Notebooks liés : `TS_Time_Series_Intro` (débutant), `TS_ARIMA` (ARIMA en détail), `TS_Maintenance_Predictive` (case study).
<!-- #endregion -->
