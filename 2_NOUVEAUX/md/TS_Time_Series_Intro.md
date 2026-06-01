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
# ⏰ Time Series — Introduction et premiers pas
<!-- #endregion -->

<!-- #region -->
Notebook **tutoriel** pour aborder une série temporelle pour la première fois. On déroule la
boîte à outils fondamentale, du premier coup d'œil à un premier modèle évalué proprement :

1. **Qu'est-ce qu'une série temporelle** et quelles tâches on lui pose.
2. **Visualisation** (line plot + heatmap calendaire).
3. **Décomposition** : tendance, saisonnalité, résidus (additive vs multiplicative).
4. **Stationnarité** et tests (ADF, KPSS).
5. **Différenciation** pour rendre stationnaire.
6. **Autocorrélation** (ACF / PACF) — la signature d'une série.
7. **Lag features** pour passer à un modèle ML standard.
8. **Train/test split temporel** (jamais shuffle !) + baselines.

Dataset : **Air Passengers** (Box & Jenkins, 1949-1960), le classique de l'apprentissage TS.

> Pour la vue **wiki exhaustive** (toutes les méthodes 2026), voir `TS_Time_Series_Overview`.
> Pour **ARIMA** en détail → `TS_ARIMA`. Pour les **séquences LSTM** → `TS_Generer_Sequence`.
> Pour un **case study** maintenance prédictive → `TS_Maintenance_Predictive`.
<!-- #endregion -->

<!-- #region -->
## 1. Qu'est-ce qu'une série temporelle
<!-- #endregion -->

<!-- #region -->
Une **série temporelle** est une séquence ordonnée d'observations indexées par le temps :

```
t :    1949-01   1949-02   1949-03   ...
y :     112       118       132      ...
```

**Particularités** par rapport à un dataset tabulaire classique :

- **L'ordre compte** — la valeur à `t+1` dépend de celles à `t, t-1, …`.
- **Autocorrélation** — les valeurs proches dans le temps se ressemblent (sinon : *white noise*).
- **Non-iid** — on ne peut pas faire un k-fold cross-validation classique (cf. section 8).

**Familles de tâches** :

| Tâche | Question | Exemples |
|---|---|---|
| **Forecasting** | Prédire les `h` prochaines valeurs | Ventes, demande énergie, météo |
| **Anomaly detection** | Détecter une rupture / outlier | Fraude, maintenance prédictive |
| **Classification** | Étiqueter toute la série | ECG normal/anormal, activités IoT |
| **Imputation** | Compléter des trous | Capteurs défaillants |
| **Changepoint detection** | Trouver les ruptures de régime | Régimes économiques |
<!-- #endregion -->

<!-- #region -->
## 2. Setup et dataset
<!-- #endregion -->

<!-- #region -->
On charge le dataset **Air Passengers** de façon robuste : `statsmodels.get_rdataset` (réseau,
mis en cache au premier appel) avec un **fallback synthétique déterministe** si pas de réseau.
On définit aussi une petite palette de couleurs pour des graphiques cohérents.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Palette cohérente : série continue neutre → une seule couleur primaire
PRIMARY = "#00798c"
ACCENT = "#66a182"
MAUVAIS = "#d1495b"


def load_air_passengers() -> pd.Series:
    """Charge le dataset Air Passengers (Box & Jenkins, mensuel 1949-1960).

    Tente `statsmodels.get_rdataset` (réseau, mis en cache) ; à défaut, génère une
    série synthétique déterministe de même structure (offline-safe).

    Returns:
        Série mensuelle indexée par date, nommée "passengers" (144 points).
    """
    try:
        from statsmodels.datasets import get_rdataset
        raw = get_rdataset("AirPassengers", "datasets").data
        s = pd.Series(
            raw["value"].to_numpy(dtype=float),
            index=pd.date_range("1949-01-01", periods=len(raw), freq="MS"),
            name="passengers",
        )
    except Exception:  # fallback offline volontaire
        n = 144
        rng = np.random.default_rng(42)
        t = np.arange(n)
        trend = np.linspace(100, 600, n)
        season = 1.0 + 0.20 * np.sin(2 * np.pi * t / 12)  # saison multiplicative
        noise = rng.normal(1.0, 0.03, n)
        s = pd.Series(
            trend * season * noise,
            index=pd.date_range("1949-01-01", periods=n, freq="MS"),
            name="passengers",
        )
    return s


ts = load_air_passengers()
print(ts.head())
print(f"\n{len(ts)} observations, du {ts.index.min().date()} au {ts.index.max().date()}")
print(f"Range : {ts.min():.0f} - {ts.max():.0f}")
```

<!-- #region -->
On obtient **144 mois** (1949-01 à 1960-12), de ~100 à ~620 passagers : une croissance nette
avec, on va le voir, une forte saisonnalité annuelle.
<!-- #endregion -->

<!-- #region -->
## 3. Visualiser : le premier réflexe
<!-- #endregion -->

<!-- #region -->
Avant tout modèle, **on regarde la série**. Deux vues complémentaires :

- **Line plot** : révèle tendance et saisonnalité d'un coup d'œil.
- **Heatmap calendaire** (année × mois) : rend la structure saisonnière *récurrente* évidente
  (mêmes mois forts/faibles chaque année) et montre la montée d'intensité au fil des années.
<!-- #endregion -->

```python
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def plot_overview(series: pd.Series) -> plt.Figure:
    """Vue d'ensemble : line plot + heatmap calendaire (année × mois)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(series.index, series.values, color=PRIMARY)
    axes[0].set(title="Air Passengers (1949-1960)", ylabel="Passagers (milliers)")
    axes[0].grid(True, alpha=0.3)

    pivot = (
        series.to_frame("y")
        .assign(year=lambda d: d.index.year, month=lambda d: d.index.month)
        .pivot(index="year", columns="month", values="y")
    )
    im = axes[1].imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(MONTHS)
    axes[1].set_yticks(range(len(pivot.index)))
    axes[1].set_yticklabels(pivot.index)
    axes[1].set(title="Heatmap calendaire (année × mois)", ylabel="Année", xlabel="Mois")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    return fig


plot_overview(ts);
```

<!-- #region -->
**On voit immédiatement** :

- **Tendance** ascendante (croissance générale du trafic).
- **Saisonnalité** annuelle : pic estival (juillet-août), creux d'hiver, chaque année.
- **Amplitude croissante** de la saisonnalité avec le niveau → cela suggère une **saisonnalité
  multiplicative** (à confirmer par la décomposition).
<!-- #endregion -->

<!-- #region -->
## 4. Décomposition tendance / saisonnalité / résidus
<!-- #endregion -->

<!-- #region -->
La **décomposition classique** sépare la série en trois composantes :

- **Additive** : $y(t) = T(t) + S(t) + R(t)$ — quand la saisonnalité est d'**amplitude constante**.
- **Multiplicative** : $y(t) = T(t) \cdot S(t) \cdot R(t)$ — quand l'amplitude saisonnière **croît
  avec la tendance**.

Une bonne décomposition laisse des **résidus** centrés et sans structure apparente. On compare
les deux modèles côte à côte pour choisir le bon.
<!-- #endregion -->

```python
from statsmodels.tsa.seasonal import seasonal_decompose

fig, axes = plt.subplots(4, 2, figsize=(14, 10))
for col, model_type in enumerate(["additive", "multiplicative"]):
    res = seasonal_decompose(ts, model=model_type, period=12)
    res.observed.plot(ax=axes[0, col], color=PRIMARY, title=f"{model_type.capitalize()} — Observed")
    res.trend.plot(ax=axes[1, col], color=PRIMARY, title="Trend")
    res.seasonal.plot(ax=axes[2, col], color=ACCENT, title="Seasonal")
    res.resid.plot(ax=axes[3, col], color=MAUVAIS, title="Residuals")
    for r in range(4):
        axes[r, col].grid(True, alpha=0.3)
fig.tight_layout();
```

<!-- #region -->
Sur Air Passengers, le modèle **multiplicatif** donne des **résidus plus homogènes** (variance
stable dans le temps), alors qu'en additif les résidus « respirent » avec la saison — confirmation
que la saisonnalité est bien proportionnelle au niveau. C'est pourquoi on **log-transforme** la
série avant de la modéliser (le log transforme le multiplicatif en additif).
<!-- #endregion -->

<!-- #region -->
## 5. Stationnarité
<!-- #endregion -->

<!-- #region -->
Une série est **stationnaire** si ses propriétés statistiques (moyenne, variance, autocovariance)
sont **constantes dans le temps**. La plupart des modèles classiques (ARIMA, VAR) la **supposent**.

On combine deux tests aux **hypothèses nulles opposées** (piège classique) :

| Test | H0 | Conclusion stationnaire si |
|---|---|---|
| **ADF** (Augmented Dickey-Fuller) | racine unité ⇒ *non*-stationnaire | `p < 0.05` (on rejette H0) |
| **KPSS** | série *stationnaire* | `p > 0.05` (on ne rejette pas H0) |

Les deux d'accord → conclusion solide. En désaccord → série « borderline » (souvent : à différencier).
<!-- #endregion -->

```python
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_tests(series: pd.Series, name: str = "") -> None:
    """Affiche les tests ADF et KPSS (H0 opposées) pour une série.

    ADF  : H0 = non-stationnaire → p < 0.05 ⇒ STATIONNAIRE.
    KPSS : H0 = stationnaire     → p > 0.05 ⇒ STATIONNAIRE.
    """
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS prévient quand p est hors table
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    print(f"--- {name} ---")
    print(f"ADF  : stat={adf_stat:7.3f}  p={adf_p:.4f}  → {'STATIONNAIRE' if adf_p < 0.05 else 'NON stationnaire'}")
    print(f"KPSS : stat={kpss_stat:7.3f}  p={kpss_p:.4f}  → {'STATIONNAIRE' if kpss_p > 0.05 else 'NON stationnaire'}")


stationarity_tests(ts, "Série originale")
```

<!-- #region -->
Les deux tests sont d'accord : la **série originale est non-stationnaire** (ADF `p ≈ 0.99`, KPSS
`p = 0.01`). Logique — il y a une tendance et une saisonnalité marquées. Pour la stationnariser,
on **différencie** (section suivante).
<!-- #endregion -->

<!-- #region -->
## 6. Différenciation
<!-- #endregion -->

<!-- #region -->
**Différencier** = remplacer la série par ses variations, ce qui retire la structure :

- **Ordre 1** : $\nabla y_t = y_t - y_{t-1}$ — retire la **tendance**.
- **Saisonnier (12)** : $\nabla_{12} y_t = y_t - y_{t-12}$ — retire la **saison annuelle**.
- **Combiné** : `diff(1).diff(12)` — retire les deux.

On applique d'abord un **log** pour stabiliser la variance (saisonnalité multiplicative), puis on
re-teste la stationnarité à chaque étape.
<!-- #endregion -->

```python
ts_log = np.log(ts)
ts_diff1 = ts_log.diff().dropna()                 # ordre 1 : enlève la tendance
ts_diff12 = ts_log.diff(12).dropna()              # saisonnier : enlève la saison annuelle
ts_diff_both = ts_log.diff().diff(12).dropna()    # combiné

stationarity_tests(ts_diff1, "log + diff(1)")
stationarity_tests(ts_diff12, "log + diff(12)")
stationarity_tests(ts_diff_both, "log + diff(1) + diff(12)")

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
ts.plot(ax=axes[0, 0], color=PRIMARY, title="Original (non stationnaire)")
ts_diff1.plot(ax=axes[0, 1], color=PRIMARY, title="log + diff(1)")
ts_diff12.plot(ax=axes[1, 0], color=PRIMARY, title="log + diff(12)")
ts_diff_both.plot(ax=axes[1, 1], color=PRIMARY, title="log + diff(1) + diff(12)")
for ax in axes.ravel():
    ax.grid(True, alpha=0.3)
fig.tight_layout();
```

<!-- #region -->
Seul **`log + diff(1) + diff(12)`** rend la série franchement stationnaire (ADF `p ≈ 0.0002` ET
KPSS `p > 0.05` — les deux d'accord). Les différenciations simples ne suffisent pas (ADF
*borderline*). En notation ARIMA, cela correspond à `d = 1` (régulier) et `D = 1` (saisonnier),
ce qui guidera le choix d'un modèle SARIMA dans `TS_ARIMA`.
<!-- #endregion -->

<!-- #region -->
## 7. ACF et PACF — la signature d'une série
<!-- #endregion -->

<!-- #region -->
- **ACF** (Auto-Correlation Function) : $\rho(k) = \mathrm{Corr}(y_t, y_{t-k})$ — à quel point
  `y_t` ressemble à sa version `k` pas plus tôt.
- **PACF** (Partial ACF) : la corrélation **directe** entre `y_t` et `y_{t-k}` une fois retiré
  l'effet des lags intermédiaires.

**Lectures classiques** (utile pour ARIMA) :

| Pattern ACF | Pattern PACF | Modèle suggéré |
|---|---|---|
| Décroissance lente | Coupure nette à lag `p` | **AR(p)** |
| Coupure nette à lag `q` | Décroissance lente | **MA(q)** |
| Décroissance lente | Décroissance lente | **ARMA(p, q)** |
<!-- #endregion -->

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
plot_acf(ts.dropna(), lags=36, ax=axes[0, 0], title="ACF — original")
plot_pacf(ts.dropna(), lags=36, ax=axes[0, 1], title="PACF — original")
plot_acf(ts_diff_both, lags=36, ax=axes[1, 0], title="ACF — log+diff(1)+diff(12)")
plot_pacf(ts_diff_both, lags=36, ax=axes[1, 1], title="PACF — log+diff(1)+diff(12)")
fig.tight_layout();
```

<!-- #region -->
**Lecture** :

- Sur l'**original**, l'ACF décroît très lentement et présente des **pics à lag 12, 24, 36** →
  signature d'une série non-stationnaire à **saisonnalité annuelle**.
- Sur la série **différenciée**, l'ACF retombe vite : la structure a été absorbée. Les pics
  résiduels (notamment lag 12) renseignent sur les ordres MA saisonniers d'un SARIMA.
<!-- #endregion -->

<!-- #region -->
## 8. Lag features — passer à un modèle ML standard
<!-- #endregion -->

<!-- #region -->
Approche **« TS-as-tabular »** : on transforme la série en features (lags, rolling, calendaires,
Fourier) puis on entraîne **n'importe quel régresseur** (LinReg, RandomForest, LightGBM…).

| Feature | Formule | Capture |
|---|---|---|
| `lag_k` | `y(t-k)` | Mémoire courte / saisonnière (`lag_12`) |
| `roll_mean_n` / `roll_std_n` | sur `y` décalée de 1 | Tendance locale / volatilité |
| `month, quarter, dayofweek` | extraits du timestamp | Saisonnalités calendaires |
| `sin12, cos12` | $\sin/\cos(2\pi\,\text{mois}/12)$ | Saisonnalité cyclique (Fourier) |

⚠️ **Piège anti-leak** : les rolling se calculent sur `y.shift(1)`, sinon la fenêtre inclut le
point courant `t` qu'on cherche justement à prédire (fuite du futur).
<!-- #endregion -->

```python
def make_features(
    series: pd.Series,
    lags: tuple[int, ...] = (1, 2, 3, 12),
    roll_windows: tuple[int, ...] = (3, 12),
) -> pd.DataFrame:
    """Transforme une série en features tabulaires (approche TS-as-tabular).

    Génère : lags, rolling mean/std (sur `shift(1)` pour éviter le leak du point
    courant), features calendaires et features de Fourier (saisonnalité cyclique).

    Args:
        series: série temporelle indexée par date.
        lags: décalages `y(t-k)` à créer.
        roll_windows: fenêtres des statistiques mobiles.

    Returns:
        DataFrame des features (colonne cible "y" incluse), NaN de tête non supprimés.
    """
    df = pd.DataFrame({"y": series})
    df = df.assign(
        month=lambda x: x.index.month,
        quarter=lambda x: x.index.quarter,
        year=lambda x: x.index.year,
        dayofweek=lambda x: x.index.dayofweek,
        sin12=lambda x: np.sin(2 * np.pi * x.index.month / 12),  # Fourier
        cos12=lambda x: np.cos(2 * np.pi * x.index.month / 12),
    )
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()  # shift(1) = anti-leak
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std()
    return df


features = make_features(ts).dropna()
print(features.head().round(2))
print(f"\nShape : {features.shape}, NaN restants : {int(features.isna().sum().sum())}")
```

<!-- #region -->
Les premières lignes contiennent des NaN (les lags et rolling n'ont pas assez d'historique) — on
les supprime avec `.dropna()`. Il reste **132 lignes × 15 colonnes**, prêtes pour un régresseur.
<!-- #endregion -->

<!-- #region -->
## 9. Train/test split temporel — JAMAIS shuffle !
<!-- #endregion -->

<!-- #region -->
On évalue en **respectant l'ordre du temps** : entraîner sur le passé, tester sur le futur. Un
`train_test_split(shuffle=True)` ferait fuiter le futur dans le passé → score illusoire.

Stratégies :

- **Single split** : 80 % train (passé) / 20 % test (futur récent).
- **Expanding window CV** (`TimeSeriesSplit`) : chaque fold ajoute du passé au train.
- **Rolling window CV** : fenêtre de train glissante (taille fixe), utile si la dynamique change.

**Toujours commencer par un baseline** (best practice Nixtla 2026) : le **seasonal-naive**
`ŷ(t) = y(t-12)` est redoutable sur les séries saisonnières. Si le modèle ne le bat pas, il est inutile.
<!-- #endregion -->

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error

X = features.drop(columns=["y"])
y = features["y"]

# Split temporel — JAMAIS shuffle
split_idx = int(len(features) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train : {len(X_train)} ({X_train.index.min().date()} → {X_train.index.max().date()})")
print(f"Test  : {len(X_test)}  ({X_test.index.min().date()} → {X_test.index.max().date()})")

# Baseline 1 — seasonal naive : y(t) = y(t-12)
naive_pred = features["lag_12"].iloc[split_idx:]
mape_naive = mean_absolute_percentage_error(y_test, naive_pred)

# Baseline 2 — régression linéaire sur lags
model = LinearRegression()
model.fit(X_train, y_train)
mape_linreg = mean_absolute_percentage_error(y_test, model.predict(X_test))

print(f"\nMAPE seasonal-naive : {mape_naive:.2%}")
print(f"MAPE LinReg (lags)  : {mape_linreg:.2%}")
print(f"→ LinReg {'BAT' if mape_linreg < mape_naive else 'NE BAT PAS'} le baseline naïf")

# Visualisation des folds TimeSeriesSplit (expanding window)
tscv = TimeSeriesSplit(n_splits=5)
fig, ax = plt.subplots(figsize=(10, 4))
for i, (tr_idx, te_idx) in enumerate(tscv.split(features)):
    ax.plot(features.index[tr_idx], np.full(len(tr_idx), i), ".", color=PRIMARY, alpha=0.6)
    ax.plot(features.index[te_idx], np.full(len(te_idx), i), ".", color=MAUVAIS, alpha=0.8)
ax.set_yticks(range(5))
ax.set_yticklabels([f"Fold {i + 1}" for i in range(5)])
ax.set_title("TimeSeriesSplit (bleu = train, rouge = test)")
fig.tight_layout();
```

<!-- #region -->
Le **seasonal-naive** atteint déjà ~9.5 % de MAPE — il faut le battre. La **régression linéaire
sur lags** descend à ~4 % : elle exploite la tendance + la saison + les rolling, donc elle bat
nettement le baseline. La viz `TimeSeriesSplit` montre bien que **chaque fold de test est
strictement postérieur** à son train (expanding window).
<!-- #endregion -->

<!-- #region -->
## 10. Pièges et anti-patterns
<!-- #endregion -->

<!-- #region -->
| ❌ Anti-pattern | ✅ Correctif |
|---|---|
| `train_test_split(shuffle=True)` sur TS | `TimeSeriesSplit` ou split manuel par date |
| `fillna(0)` sur une TS | `interpolate(method="time")` ou `ffill` |
| Ne pas tester la stationnarité | ADF **+** KPSS (H0 opposées) avant ARIMA |
| Conclure « pas saisonnier » à l'œil | Toujours regarder l'ACF / PACF |
| Rolling features sans `shift(1)` | `y.shift(1).rolling(...)` — sinon leak du point courant |
| Oublier les features de Fourier | `sin/cos` pour une saisonnalité de période connue |
| Imputation par moyenne globale | Interpolation **locale** (temporelle) |
| Pas de baseline | Toujours comparer au naïf / seasonal-naive |
<!-- #endregion -->

<!-- #region -->
## 11. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
- **Toujours visualiser** la série brute avant tout modèle.
- **Décomposer** pour identifier tendance / saisonnalité / structure résiduelle.
- **Tester la stationnarité** (ADF + KPSS) et appliquer les différenciations nécessaires.
- **Validation temporelle stricte** : jamais `shuffle=True`.
- **Métriques** : MAPE (relative), MAE, RMSE — et **MASE/WAPE** quand `y` peut s'approcher de 0.
- **Baseline obligatoire** : persistance `ŷ(t)=y(t-1)` ou saisonnier naïf `ŷ(t)=y(t-12)`.

**Stack moderne (2026)** :

- **Nixtla** (`statsforecast`, `mlforecast`, `neuralforecast`) : API unifiée des classiques
  (ARIMA/ETS) aux modèles ML/DL, avec des baselines (Naive, SeasonalNaive) intégrées.
- **Modèles de fondation** pré-entraînés, en *zero-shot* : **TimesFM** (Google), **Chronos-2**
  (Amazon), **Moirai-2 / Moirai-MoE** (Salesforce, multivarié), **TimeGPT** (Nixtla, propriétaire),
  **Lag-Llama**. ⚠️ En 2026 ils sont **compétitifs mais ne dominent pas encore** un modèle
  supervisé bien entraîné : ils se placent entre les classiques et les meilleurs DL. À considérer
  surtout quand on a peu de données ou beaucoup de séries. Détails dans `TS_Time_Series_Overview`.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Forecasting: Principles and Practice — Hyndman & Athanasopoulos (référence libre)](https://otexts.com/fpp3/)
- [statsmodels — Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
- [Nixtla — baseline forecasts](https://www.nixtla.io/blog/baseline-forecasts) & [écosystème Nixtla](https://nixtlaverse.nixtla.io/)
- [The 2026 Time Series Toolkit — Foundation Models (MachineLearningMastery)](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- Notebooks liés : `TS_Time_Series_Overview` (wiki complet), `TS_ARIMA` (ARIMA détaillé),
  `TS_Generer_Sequence` (prep LSTM), `TS_Maintenance_Predictive` (case study).
<!-- #endregion -->
