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
# 📊 ARIMA — Théorie et pratique
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur la **famille ARIMA** — la pierre angulaire du forecasting statistique depuis 50 ans, encore très utilisée en 2026 comme baseline et pour des séries courtes.

Couvre :

1. **AR**, **MA**, **ARMA**, **ARIMA**, **SARIMA**, **SARIMAX** — les variantes.
2. **Méthodologie Box-Jenkins** complète.
3. Lecture **ACF/PACF** pour choisir les ordres.
4. **Diagnostic des résidus** — bruit blanc ou pas.
5. **AutoARIMA** moderne avec `pmdarima` et `statsforecast`.
6. Forecasting + intervalles de confiance.

Dataset : **Air Passengers** (mutualisé avec `TS_Time_Series_Intro` et `TS_Time_Series_Overview`).
<!-- #endregion -->

<!-- #region -->
## 1. La famille ARIMA en un coup d'œil
<!-- #endregion -->

<!-- #region -->
**Notation** : `SARIMA(p, d, q)(P, D, Q, s)`.

- **AR(p)** — AutoRegressive : `y_t = c + φ_1 y_{t-1} + ... + φ_p y_{t-p} + ε_t`
- **MA(q)** — Moving Average : `y_t = c + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}`
- **ARMA(p, q)** = AR(p) + MA(q) — pour séries stationnaires.
- **I(d)** — Integrated : applique `d` différenciations pour rendre stationnaire.
- **ARIMA(p, d, q)** = ARMA sur la série différenciée d fois.
- **SARIMA(p,d,q)(P,D,Q,s)** : ajoute composante saisonnière de période `s`.
- **SARIMAX** : SARIMA + variables exogènes (régresseurs externes).

**Hypothèses** :

- Stationnarité de la série (après différenciations).
- Bruit `ε_t` blanc gaussien.
- Linéarité.

**Forces** :

- Interprétable, bien étudié, intervalles de prédiction propres.
- Peu de data nécessaire (50-200 obs souvent suffit).

**Faiblesses** :

- Mauvais sur séries non-linéaires.
- Mauvais en multi-séries (un modèle par série).
- Pas de covariables sans SARIMAX (et limité).
<!-- #endregion -->

<!-- #region -->
## 2. Méthodologie Box-Jenkins
<!-- #endregion -->

<!-- #region -->
La méthodologie classique en 4 étapes (1970) :

1. **Identification** : visualiser, tester stationnarité (ADF/KPSS), différencier au besoin, lire ACF/PACF pour proposer `(p, q)`.
2. **Estimation** : ajuster le modèle (max likelihood).
3. **Diagnostic** : tester les résidus — sont-ils du bruit blanc ? Sinon, on a oublié un pattern.
4. **Forecasting** : prévoir avec intervalles de confiance.

En 2026, **on remplace les étapes 1-2 par un AutoARIMA** qui parcourt automatiquement les combinaisons. Mais comprendre la méthodologie reste utile pour interpréter et diagnostiquer.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Air Passengers
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
try:
    ap = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
    ap.columns = ["passengers"]
except Exception:
    from statsmodels.datasets import get_rdataset
    ap = get_rdataset("AirPassengers", "datasets").data.rename(columns={"value": "passengers"})
    ap.index = pd.date_range("1949-01", periods=len(ap), freq="MS")
    ap = ap[["passengers"]]

train, test = ap.iloc[:-12], ap.iloc[-12:]
print(f"Train: {len(train)} | Test: {len(test)}")
```

<!-- #region -->
## 3. Identification : ACF / PACF
<!-- #endregion -->

<!-- #region -->
**Règles classiques de lecture** :

| Pattern ACF | Pattern PACF | Modèle |
|---|---|---|
| Décroissance lente | Coupure nette à lag p | **AR(p)** |
| Coupure nette à lag q | Décroissance lente | **MA(q)** |
| Décroissance lente | Décroissance lente | **ARMA(p, q)** |
| ACF saisonnier | PACF saisonnier | Composantes saisonnières (P, Q) |

Pour **Air Passengers** :

- Série non stationnaire (tendance + saison croissante).
- On applique `log` puis `diff(1)` (différenciation simple) puis `diff(12)` (différenciation saisonnière).
- Sur la série transformée, on lit l'ACF/PACF pour identifier `(p, q)` et `(P, Q)`.
<!-- #endregion -->

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

log_diff = np.log(train["passengers"]).diff().diff(12).dropna()
adf_stat, adf_p, *_ = adfuller(log_diff)
print(f"ADF après log+diff1+diff12 : p={adf_p:.4f}  → {'STATIONNAIRE' if adf_p < 0.05 else 'NON'}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(log_diff, lags=24, ax=axes[0], title="ACF (log + diff1 + diff12)")
plot_pacf(log_diff, lags=24, ax=axes[1], title="PACF (log + diff1 + diff12)")
plt.tight_layout()
```

<!-- #region -->
## 4. Estimation manuelle (SARIMA)
<!-- #endregion -->

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ordres proposés à la main : (1,1,1)(1,1,1,12) — un classique pour Air Passengers
model = SARIMAX(
    train["passengers"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit(disp=False)
print(fit.summary().tables[1])
```

<!-- #region -->
## 5. Diagnostic des résidus
<!-- #endregion -->

<!-- #region -->
Si le modèle est bien spécifié, les **résidus** doivent être :

- Centrés sur 0, variance constante.
- Sans autocorrélation (test de **Ljung-Box** : p-value > 0.05).
- Distribués normalement (test de **Jarque-Bera**).

S'ils ne le sont pas, le modèle a manqué un pattern → ajuster `(p, d, q)` ou `(P, D, Q, s)`.
<!-- #endregion -->

```python
fig = fit.plot_diagnostics(figsize=(10, 6))
plt.tight_layout()
```

<!-- #region -->
## 6. Forecasting + intervalles de confiance
<!-- #endregion -->

```python
from sklearn.metrics import mean_absolute_percentage_error

fcst = fit.get_forecast(steps=12)
mean = fcst.predicted_mean
ci = fcst.conf_int(alpha=0.05)  # IC 95%

mape = mean_absolute_percentage_error(test["passengers"], mean)
print(f"SARIMA(1,1,1)(1,1,1,12) MAPE: {mape:.3%}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(train.index, train["passengers"], label="train")
plt.plot(test.index, test["passengers"], label="test (réel)")
plt.plot(mean.index, mean.values, label="forecast", color="red")
plt.fill_between(mean.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=0.2, label="IC 95%")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
```

<!-- #region -->
## 7. AutoARIMA — l'approche moderne
<!-- #endregion -->

<!-- #region -->
Au lieu de chercher `(p, d, q, P, D, Q)` à la main, on utilise un algorithme qui parcourt l'espace et garde le meilleur (AIC, BIC, ou AICc).

Deux implémentations principales en 2026 :

- **`pmdarima.auto_arima`** — classique, basé sur statsmodels.
- **`statsforecast.AutoARIMA`** — Nixtla, beaucoup plus rapide (Numba), supporte le scaling à des milliers de séries en parallèle.
<!-- #endregion -->

```python
import pmdarima as pm

auto = pm.auto_arima(
    train["passengers"],
    seasonal=True, m=12,
    information_criterion="aic",
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
)
print(f"AutoARIMA → order={auto.order}, seasonal_order={auto.seasonal_order}, AIC={auto.aic():.2f}")

fcst_auto = auto.predict(n_periods=12)
mape_auto = mean_absolute_percentage_error(test["passengers"].values, fcst_auto)
print(f"AutoARIMA MAPE: {mape_auto:.3%}")
```

<!-- #region -->
## 8. SARIMAX — avec variables exogènes
<!-- #endregion -->

<!-- #region -->
Quand on a des **covariables** (prix, météo, événements), on les passe via `exog`. Attention :

- Ces covariables doivent être **connues** sur l'horizon de prédiction (ou prédites séparément).
- Si elles ne le sont pas, on tombe sur le problème classique : il faut un modèle pour prédire la covariable, qui peut ajouter du bruit.

```python
# Pseudo-code SARIMAX
# model = SARIMAX(y_train, exog=X_train_exog, order=(1,1,1), seasonal_order=(1,1,1,12))
# fit = model.fit()
# fcst = fit.get_forecast(steps=12, exog=X_future_exog)
```
<!-- #endregion -->

<!-- #region -->
## 9. Quand préférer autre chose qu'ARIMA
<!-- #endregion -->

<!-- #region -->
- **Saisonnalités multiples** (jour + semaine + année) → Prophet, TBATS, DL.
- **Séries non-linéaires** (régimes, ruptures) → modèles state-space, ML.
- **N séries similaires** → ML global ou foundation models (un modèle pour tout).
- **Horizon très long** → ARIMA dégénère vite, DL/foundation models mieux.
- **Covariables nombreuses** → ML / TFT.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [Box & Jenkins (1970) — Time Series Analysis: Forecasting and Control](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method)
- [statsmodels SARIMAX docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [pmdarima — auto_arima](https://alkaline-ml.com/pmdarima/)
- [Hyndman & Athanasopoulos — FPP3, chapitre ARIMA](https://otexts.com/fpp3/arima.html)
- Notebooks liés : `TS_Time_Series_Intro` (bases), `TS_Time_Series_Overview` (panorama 2026).
<!-- #endregion -->
