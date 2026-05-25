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
# ⏰ Time Series — Introduction
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel** pour débutant qui aborde une série temporelle pour la première fois.

Couvre les **fondamentaux** :

1. Qu'est-ce qu'une série temporelle, types de tâches (forecasting, anomaly, classification).
2. **Décomposition** : tendance, saisonnalité, résidus.
3. **Stationnarité** et tests (ADF, KPSS).
4. **Autocorrélation** (ACF / PACF).
5. **Lag features** pour modèles ML standard.
6. **Train/test split** spécifique aux séries (jamais shuffle !).
7. Premier baseline : régression linéaire sur lags.

Dataset : **Air Passengers** (Box & Jenkins, classique).

> Pour une vue **wiki exhaustive** (multiples méthodes), voir `TS_Time_Series_Overview`.
> Pour **ARIMA** en détail, voir `TS_ARIMA`.
> Pour générer des **séquences pour LSTM**, voir `TS_Generer_Sequence`.
> Pour la **maintenance prédictive**, voir `TS_Maintenance_Predictive`.
<!-- #endregion -->

<!-- #region -->
## 1. Qu'est-ce qu'une série temporelle
<!-- #endregion -->

<!-- #region -->
Une **série temporelle** est une séquence ordonnée d'observations indexées par le temps.

```
t :    2020-01   2020-02   2020-03   ...
y :     112       118       132      ...
```

**Particularités** par rapport à un dataset tabulaire :

- **L'ordre compte** — la valeur à `t+1` dépend de celles à `t, t-1, ...`.
- **Autocorrélation** — les valeurs proches dans le temps se ressemblent (sinon : white noise).
- **Non-iid** — on ne peut pas faire un k-fold cross-validation classique.

**Familles de tâches** :

| Tâche | Question | Exemples |
|---|---|---|
| **Forecasting** | Prédire les `h` prochaines valeurs | Ventes, demande énergie, météo |
| **Anomaly detection** | Détecter une rupture / outlier | Fraud, maintenance prédictive |
| **Classification** | Assigner une étiquette à toute la série | ECG normal/anormal, activités IoT |
| **Imputation** | Compléter des trous | Capteurs défaillants |
| **Changepoint detection** | Trouver les ruptures de régime | Régimes économiques |
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Air Passengers — classique de Box & Jenkins (1949-1960, passagers mensuels d'une compagnie aérienne)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
try:
    ap = pd.read_csv(url, parse_dates=["Month"], index_col="Month")
    ap.columns = ["passengers"]
except Exception:
    # Fallback offline (statsmodels embarque ses propres datasets)
    from statsmodels.datasets import get_rdataset
    ap = get_rdataset("AirPassengers", "datasets").data.rename(columns={"value": "passengers"})
    ap.index = pd.date_range("1949-01", periods=len(ap), freq="MS")
    ap = ap[["passengers"]]

print(ap.head())
print(f"\n{len(ap)} observations, du {ap.index.min().date()} au {ap.index.max().date()}")
```

<!-- #region -->
## 2. Visualiser : le premier réflexe
<!-- #endregion -->

```python
ap.plot(figsize=(10, 4), title="Air Passengers — 1949-1960")
plt.ylabel("Passagers (milliers)")
plt.grid(alpha=0.3)
plt.tight_layout()
```

<!-- #region -->
**On voit immédiatement** :

- **Tendance** ascendante (croissance générale).
- **Saisonnalité** annuelle (pic d'été, creux d'hiver).
- **Amplitude croissante** de la saisonnalité → cela suggère une **saisonnalité multiplicative** (cf. log-transform plus tard).
<!-- #endregion -->

<!-- #region -->
## 3. Décomposition tendance / saisonnalité / résidus
<!-- #endregion -->

<!-- #region -->
**Décomposition classique** :

- **Additive** : `y(t) = T(t) + S(t) + R(t)` — quand la saisonnalité est constante en amplitude.
- **Multiplicative** : `y(t) = T(t) · S(t) · R(t)` — quand l'amplitude saisonnière croît avec la tendance.

Une bonne décomposition montre des **résidus** centrés sur 0 et sans pattern apparent.
<!-- #endregion -->

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Multiplicative — adaptée ici car amplitude saison croît avec niveau
decomp = seasonal_decompose(ap["passengers"], model="multiplicative", period=12)
fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
```

<!-- #region -->
## 4. Stationnarité
<!-- #endregion -->

<!-- #region -->
Une série est **stationnaire** si ses propriétés statistiques (moyenne, variance, autocovariance) sont **constantes dans le temps**. La plupart des modèles classiques (ARIMA, VAR) **supposent** la stationnarité.

**Test ADF** (Augmented Dickey-Fuller) : H0 = "présence de racine unité" (non stationnaire).

- `p-value < 0.05` → rejeter H0 → série stationnaire.
- `p-value ≥ 0.05` → ne pas rejeter → potentielle non-stationnarité.

**Test KPSS** : H0 inversée = "série stationnaire". Souvent utilisé en complément du ADF.
<!-- #endregion -->

```python
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_tests(series: pd.Series, name: str = "") -> None:
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    print(f"--- {name} ---")
    print(f"ADF  : stat={adf_stat:.3f}  p={adf_p:.4f}  → {'STATIONNAIRE' if adf_p < 0.05 else 'NON stationnaire'}")
    print(f"KPSS : stat={kpss_stat:.3f}  p={kpss_p:.4f}  → {'STATIONNAIRE' if kpss_p > 0.05 else 'NON stationnaire'}")


stationarity_tests(ap["passengers"], "Original")

# Pour rendre stationnaire : log + différenciation
ap["log"] = np.log(ap["passengers"])
ap["log_diff1"] = ap["log"].diff()
ap["log_diff1_diff12"] = ap["log_diff1"].diff(12)  # diff saisonnière

stationarity_tests(ap["log_diff1_diff12"].dropna(), "log + diff(1) + diff(12)")
```

<!-- #region -->
## 5. ACF et PACF — la signature d'une série
<!-- #endregion -->

<!-- #region -->
- **ACF** (Auto-Correlation Function) : `ρ(k) = Corr(y_t, y_{t-k})`. Combien `y_t` se ressemble-t-il à sa version k pas plus tôt.
- **PACF** (Partial ACF) : la corrélation **directe** entre `y_t` et `y_{t-k}` après avoir enlevé l'effet des lags intermédiaires.

**Lectures classiques** :

| Pattern ACF | Pattern PACF | Modèle suggéré |
|---|---|---|
| Décroissance lente | Coupure nette à lag p | **AR(p)** |
| Coupure nette à lag q | Décroissance lente | **MA(q)** |
| Décroissance lente | Décroissance lente | **ARMA(p, q)** |

Voir notebook `TS_ARIMA` pour exploitation détaillée.
<!-- #endregion -->

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ap["log_diff1_diff12"].dropna(), lags=24, ax=axes[0], title="ACF (différenciée)")
plot_pacf(ap["log_diff1_diff12"].dropna(), lags=24, ax=axes[1], title="PACF (différenciée)")
plt.tight_layout()
```

<!-- #region -->
## 6. Lag features — passer à un modèle ML standard
<!-- #endregion -->

<!-- #region -->
Une fois qu'on a des **lags** comme features, n'importe quel modèle de régression peut prédire la série.

**Features classiques** :

| Feature | Formule | Capture |
|---|---|---|
| `lag_k` | `y(t-k)` | Mémoire courte |
| `rolling_mean_n` | moyenne mobile sur `n` points | Tendance locale |
| `rolling_std_n` | écart-type mobile | Volatilité |
| `month, day_of_week, ...` | extraits du timestamp | Saisonnalités calendaires |
| `diff_k` | `y(t) - y(t-k)` | Changement absolu |
<!-- #endregion -->

```python
df = ap[["passengers"]].copy()
for k in [1, 2, 3, 12]:
    df[f"lag_{k}"] = df["passengers"].shift(k)
df["rolling_mean_12"] = df["passengers"].shift(1).rolling(12).mean()
df["month"] = df.index.month
df = df.dropna()
print(df.head())
```

<!-- #region -->
## 7. Train/test split temporel — JAMAIS shuffle !
<!-- #endregion -->

<!-- #region -->
Pour évaluer un modèle de série temporelle, on **respecte l'ordre temporel** : on entraîne sur le passé, on teste sur le futur. Sinon **leak** massif.

Patterns 2026 :

- **Single split** : 80 % train (passé) / 20 % test (futur récent).
- **Expanding window CV** : `sklearn.model_selection.TimeSeriesSplit` — chaque fold ajoute des données au train, teste sur la fenêtre suivante.
- **Rolling window CV** : fenêtre de train glissante (taille fixe), test sur la suivante. Utile quand la dynamique change avec le temps.
<!-- #endregion -->

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

X = df.drop(columns=["passengers"])
y = df["passengers"]

# Single split temporel — pas de shuffle !
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, preds)
print(f"MAPE baseline LinReg sur lags : {mape:.3%}")

# Expanding window CV
tscv = TimeSeriesSplit(n_splits=3)
for i, (tr, te) in enumerate(tscv.split(X)):
    print(f"  Fold {i}: train {tr.shape[0]} | test {te.shape[0]}")
```

<!-- #region -->
## 8. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
- **Toujours visualiser** la série brute avant tout modèle.
- **Décomposer** pour vérifier tendance, saisonnalité, structure.
- **Tester la stationnarité** (ADF + KPSS) — applique les différenciations nécessaires.
- **Validation temporelle** stricte : jamais `train_test_split(shuffle=True)`.
- **Métriques adaptées** : MAPE (relative), MAE, RMSE. Attention à MAPE quand `y ≈ 0`.
- **Baseline naïf** indispensable : `y(t+1) = y(t)` (persistance) ou `y(t+1) = y(t+1-12)` (saisonnier naïf). Si ton modèle ne bat pas ces baselines, il est inutile.
- **2026 — modern stack** : **Nixtla** (`statsforecast`, `mlforecast`, `neuralforecast`) — couvre des classiques (ARIMA/ETS) jusqu'aux **foundation models** (TimeGPT, Chronos, Moirai) avec une API unifiée. Voir `TS_Time_Series_Overview`.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [Forecasting: Principles and Practice — Hyndman & Athanasopoulos (livre de référence)](https://otexts.com/fpp3/)
- [statsmodels — time series analysis](https://www.statsmodels.org/stable/tsa.html)
- [Nixtla ecosystem (statsforecast, mlforecast, neuralforecast)](https://nixtla.github.io/)
- Notebooks liés : `TS_Time_Series_Overview` (wiki complet), `TS_ARIMA` (ARIMA détaillé), `TS_Generer_Sequence` (LSTM prep), `TS_Maintenance_Predictive` (case study).
<!-- #endregion -->
