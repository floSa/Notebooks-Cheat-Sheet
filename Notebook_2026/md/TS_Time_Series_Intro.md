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
Notebook **tutoriel + cheat-sheet** pour aborder une série temporelle de bout en bout. Il se lit
en deux temps :

**Partie 1 — Comprendre et modéliser** (sections 1 à 11) : du premier coup d'œil à un premier
modèle évalué proprement.

1. **Qu'est-ce qu'une série temporelle** et quelles tâches on lui pose.
2. **Visualisation** (line plot + heatmap calendaire).
3. **Décomposition** : tendance, saisonnalité, résidus (additive vs multiplicative).
4. **Extraction de tendance** : `detrend` et régression sur le temps.
5. **Stationnarité** et tests (ADF, KPSS).
6. **Différenciation** pour rendre stationnaire.
7. **Autocorrélation** (ACF / PACF) — la signature d'une série.
8. **Lag features** pour passer à un modèle ML standard.
9. **Train/test split temporel** (jamais shuffle !) + baselines.
10. **Prévision récursive multi-step** (KNN, LinReg, RandomForest).

**Partie 2 — Boîte à outils pandas pour les dates** (sections 12 à 17) : les recettes de
manipulation temporelle indispensables au quotidien (valeurs manquantes, interpolation,
génération de grilles de dates, fusions `merge`/`merge_asof`, déduplication, agrégation pondérée).

Dataset principal : **Air Passengers** (Box & Jenkins, 1949-1960), le classique de l'apprentissage
TS. Les recettes pandas s'appuient sur de petits jeux jouets déterministes (seed = 42).

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
- **Non-iid** — on ne peut pas faire un k-fold cross-validation classique (cf. section 9).

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
MOYEN = "#edae49"


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
## 5. Extraire la tendance : detrend & régression
<!-- #endregion -->

<!-- #region -->
La décomposition n'est pas la seule façon d'isoler la **tendance**. Deux approches directes,
utiles quand on veut juste *retirer* la tendance :

- **`statsmodels.tsa.tsatools.detrend`** : retire une tendance **polynomiale** d'ordre `k` par
  moindres carrés et retourne la série **détendue** (les résidus). La tendance estimée vaut alors
  $T = y - \texttt{detrend}(y)$. Ordre 1 = droite, ordre 2 = parabole.
- **Régression OLS sur le temps** : on régresse $y$ sur une base du temps
  $X = [1, t, t^2, \dots]$ ; la **valeur ajustée** $\hat y = X\hat\beta$ *est* la tendance. La
  pente $\hat\beta_1$ se lit directement en « unités par pas de temps ».

On illustre aussi la variante **log** : log-transformer avant d'estimer la tendance stabilise la
variance d'une série à saisonnalité multiplicative.

> Raccourci viz : `statsmodels.graphics.regressionplots.abline_plot(model_results=res)` trace
> directement la droite de régression à partir d'un résultat OLS.
<!-- #endregion -->

```python
from statsmodels.api import OLS
from statsmodels.tsa.tsatools import detrend


def trend_by_regression(series: pd.Series, order: int = 1) -> tuple[np.ndarray, "OLS"]:
    """Estime la tendance d'une série par régression OLS sur le temps.

    On régresse `y` sur une base polynomiale du temps `t` (ordre `order`). La
    tendance est la valeur ajustée du modèle ; les résidus sont la série détenddue.

    Args:
        series: série temporelle.
        order: degré polynomial (1 = droite, 2 = parabole).

    Returns:
        (valeurs_ajustées, résultat_OLS_ajusté).
    """
    y = series.to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([t ** k for k in range(order + 1)])  # [1, t, t², ...]
    res = OLS(y, X).fit()
    return res.predict(X), res


# detrend() de statsmodels retourne la série SANS sa tendance polynomiale.
notrend_lin = detrend(ts, order=1)      # résidu après retrait d'une droite
notrend_quad = detrend(ts, order=2)     # résidu après retrait d'une parabole
trend_lin = ts - notrend_lin            # tendance linéaire = série - résidu
trend_quad = ts - notrend_quad          # tendance quadratique

# Variante "log" : on stabilise la variance avant d'estimer la tendance.
log_ts = np.log(ts)
log_trend = log_ts - detrend(log_ts, order=1)

fitted_ols, ols_res = trend_by_regression(ts, order=1)
print(f"OLS ordre 1 — intercept={ols_res.params[0]:.1f}, pente={ols_res.params[1]:.2f} passagers/mois")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(ts.index, ts.values, color=PRIMARY, label="série")
axes[0].plot(ts.index, trend_lin.values, color=MAUVAIS, lw=2, label="tendance linéaire (detrend o1)")
axes[0].plot(ts.index, trend_quad.values, color=MOYEN, lw=2, label="tendance quadratique (detrend o2)")
axes[0].set(title="Tendance par detrend (ordre 1 vs 2)", ylabel="Passagers")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(log_ts.index, log_ts.values, color=PRIMARY, label="log(série)")
axes[1].plot(log_ts.index, log_trend.values, color=MAUVAIS, lw=2, label="tendance linéaire sur log")
axes[1].set(title="Tendance sur la série log-transformée", ylabel="log(passagers)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout();
```

<!-- #region -->
La pente OLS (~2.7 passagers/mois) chiffre la croissance moyenne. La tendance **quadratique**
épouse mieux la légère accélération que la droite. Sur le log, la tendance redevient quasi-linéaire
— signe que la croissance est **exponentielle** (multiplicative), ce qui confirme la section 4.
<!-- #endregion -->

<!-- #region -->
## 6. Stationnarité
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
## 7. Différenciation
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
## 8. ACF et PACF — la signature d'une série
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
## 9. Lag features — passer à un modèle ML standard
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
## 10. Train/test split temporel — JAMAIS shuffle !
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
## 11. Prévision récursive multi-step (KNN, LinReg, RandomForest)
<!-- #endregion -->

<!-- #region -->
La section 10 évalue une prédiction **1 pas en avant** (le test dispose des vrais lags). Pour
prévoir un **horizon** de plusieurs pas sans connaître le futur, on utilise la **stratégie
récursive** : on prédit `t+1`, puis on **ré-injecte** cette prédiction comme lag pour prédire
`t+2`, et ainsi de suite (la fenêtre glisse).

$$\hat y_{t+1} = f(y_t, \dots, y_{t-n+1}), \qquad \hat y_{t+2} = f(\hat y_{t+1}, y_t, \dots)$$

⚠️ **Attention** : l'erreur se **propage** (chaque prédiction bruitée nourrit la suivante). On
compare ici trois régresseurs sklearn — **KNN**, **LinReg**, **RandomForest** — sur un horizon de
12 mois, en n'utilisant que les lags autorégressifs.
<!-- #endregion -->

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


def recursive_forecast(history: np.ndarray, model, n_lags: int, horizon: int) -> np.ndarray:
    """Prévision récursive multi-step à partir des seuls lags autorégressifs.

    On entraîne `model` à prédire `y(t)` depuis `[y(t-1), …, y(t-n_lags)]`, puis on
    prédit pas à pas : chaque prédiction est ré-injectée comme lag pour la suivante
    (la fenêtre glisse). C'est la stratégie "recursive" du forecasting multi-step.

    Args:
        history: valeurs observées (1D), ordonnées dans le temps.
        model: régresseur sklearn (doit exposer `fit`/`predict`).
        n_lags: nombre de retards utilisés comme features.
        horizon: nombre de pas à prédire.

    Returns:
        Tableau des `horizon` valeurs prédites.
    """
    X, target = [], []
    for i in range(n_lags, len(history)):
        X.append(history[i - n_lags:i][::-1])   # [lag1, lag2, …, lagN]
        target.append(history[i])
    model.fit(np.asarray(X), np.asarray(target))

    window = list(history[-n_lags:])
    preds = []
    for _ in range(horizon):
        x = np.asarray(window[::-1]).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        window = window[1:] + [yhat]
    return np.asarray(preds)


HORIZON, N_LAGS = 12, 12
history = ts.to_numpy(dtype=float)
train_hist, true_future = history[:-HORIZON], history[-HORIZON:]
future_index = ts.index[-HORIZON:]

models = {
    "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
    "LinReg": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
}
forecasts = {}
for name, mdl in models.items():
    preds = recursive_forecast(train_hist, mdl, n_lags=N_LAGS, horizon=HORIZON)
    forecasts[name] = preds
    mape = mean_absolute_percentage_error(true_future, preds)
    print(f"{name:14s} MAPE 12 mois : {mape:.2%}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(ts.index, history, color=PRIMARY, label="observé")
palette = [MAUVAIS, MOYEN, ACCENT]
for (name, preds), c in zip(forecasts.items(), palette):
    ax.plot(future_index, preds, "--o", color=c, ms=4, label=f"prévision {name}")
ax.axvline(future_index[0], color="grey", ls=":")
ax.set(title="Prévision récursive multi-step (12 mois)", ylabel="Passagers")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout();
```

<!-- #region -->
La **régression linéaire** s'en sort le mieux ici (~3 % de MAPE) : la dynamique d'Air Passengers
est très régulière, ce que le modèle linéaire extrapole bien. Le **KNN** souffre le plus (il ne
peut pas extrapoler au-delà des valeurs vues → il « plafonne »), la **RandomForest** est entre les
deux (même limite d'extrapolation, mais plus robuste au bruit). C'est le rappel classique :
**les modèles à base d'arbres / de voisins n'extrapolent pas** une tendance, contrairement aux
modèles linéaires.
<!-- #endregion -->

<!-- #region -->
## 12. Valeurs manquantes sur l'axe des dates
<!-- #endregion -->

<!-- #region -->
On passe à la **boîte à outils pandas** : les manipulations temporelles du quotidien. Premier
réflexe sur un nouveau jeu de données — **visualiser où sont les trous**. La librairie
**`missingno`** trace une matrice de complétude (une ligne par timestamp, une colonne par
variable, blanc = valeur manquante) : on repère d'un coup d'œil les capteurs défaillants et les
périodes de panne.
<!-- #endregion -->

```python
import missingno as msno

rng = np.random.default_rng(42)
dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
df_missing = pd.DataFrame({"date": dates}).set_index("date")
for i in range(8):
    col = f"capteur_{i + 1}"
    df_missing[col] = rng.random(len(df_missing))
    holes = rng.choice(len(df_missing), size=int(len(df_missing) * 0.2), replace=False)
    df_missing.iloc[holes, df_missing.columns.get_loc(col)] = np.nan

print(f"Taux de NaN par capteur :\n{(df_missing.isna().mean() * 100).round(1)}")

fig, ax = plt.subplots(figsize=(12, 5))
msno.matrix(df_missing, ax=ax, sparkline=False, color=(0.0, 0.47, 0.55))
ax.set_title("Matrice de complétude (blanc = valeur manquante)")
fig.tight_layout();
```

<!-- #region -->
Chaque capteur a ~20 % de trous (générés aléatoirement ici). Sur de vraies données, les bandes
blanches **alignées** entre colonnes trahiraient une panne système (toutes les variables manquent
en même temps), tandis que des trous **indépendants** pointent vers des capteurs individuels.
<!-- #endregion -->

<!-- #region -->
## 13. Compléter les dates manquantes & interpoler
<!-- #endregion -->

<!-- #region -->
Quand des **dates entières manquent** (échantillonnage irrégulier), on procède en deux temps :

1. **Densifier la grille** : `resample("D").asfreq()` insère les jours absents avec des `NaN`.
2. **Interpoler** : `interpolate(method="time")` remplit les `NaN` en **pondérant par l'écart réel
   entre timestamps** (plus correct que `method="linear"` qui ignore l'espacement des dates).
<!-- #endregion -->

```python
gaps = pd.DataFrame(
    {"date": ["2016-01-01", "2016-01-03", "2016-01-07", "2016-01-11"], "value": [10, 20, 40, 0]}
)
gaps["date"] = pd.to_datetime(gaps["date"])
gaps = gaps.set_index("date")
print("Avant (dates espacées irrégulièrement) :")
print(gaps)

# Étape 1 — densifier la grille au jour : asfreq insère les jours absents (NaN).
filled = gaps.resample("D").asfreq()
# Étape 2 — interpoler ; method="time" pondère par l'écart réel entre timestamps.
filled["value_interp"] = filled["value"].interpolate(method="time")
print("\nAprès resample('D') + interpolate('time') :")
print(filled)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filled.index, filled["value_interp"], "-", color=ACCENT, label="interpolé (time)")
ax.plot(gaps.index, gaps["value"], "o", color=PRIMARY, ms=9, label="observé")
ax.set(title="Complétion + interpolation temporelle", ylabel="value")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout();
```

<!-- #region -->
Les jours absents (02, 04, 05, 06…) ont été créés puis remplis par une droite reliant les points
observés. Pour une série à tendance/saison forte, préférer `interpolate(method="spline")` ou une
imputation par modèle ; `method="time"` reste le défaut sûr et local.
<!-- #endregion -->

<!-- #region -->
## 14. Générer des grilles de dates (fréquences & fenêtres)
<!-- #endregion -->

<!-- #region -->
`pd.date_range` génère une grille régulière selon un **alias de fréquence** : `"D"` (jour), `"MS"`
(début de mois), `"W-MON"` (chaque lundi), `"h"` (heure)… Très utile pour fabriquer un index cible
propre, puis y aligner ses données.

Premier exemple : 20 observations hebdomadaires ancrées sur le lundi, puis insertion d'une date
**hors-grille** et complétion des semaines manquantes. ⚠️ `DataFrame.append` a été **supprimé** de
pandas 2.x → on utilise **`pd.concat`**.
<!-- #endregion -->

```python
rng = np.random.default_rng(42)

# 20 observations hebdomadaires, ancrées sur le lundi (freq="W-MON").
weekly = pd.DataFrame(
    {
        "Dates": pd.date_range(start="2023-07-31", periods=20, freq="W-MON"),
        "Value": rng.integers(1, 100, size=20),
    }
)
print("Grille hebdomadaire (W-MON) :")
print(weekly.head())

# Insérer une date hors-grille puis compléter les semaines manquantes (concat, PAS append).
base = weekly.head(5).copy()
extra = pd.DataFrame([{"Dates": pd.Timestamp("2023-06-05"), "Value": 5}])
base = pd.concat([base, extra], ignore_index=True)
new_weeks = pd.date_range(start="2023-06-12", end="2023-07-24", freq="W-MON")
gaps_weeks = pd.DataFrame({"Dates": new_weeks, "Value": np.nan})
base = pd.concat([base, gaps_weeks], ignore_index=True).sort_values("Dates").reset_index(drop=True)
print(f"\nAprès insertion + complétion ({len(base)} lignes) :")
print(base)
```

<!-- #region -->
Second exemple : représenter une **fenêtre** (du lundi au dimanche) sous forme de chaîne
`"AAAA-MM-JJ/AAAA-MM-JJ"`, puis re-parser le **début de fenêtre** en `datetime` (indispensable pour
re-trier ou fusionner ensuite).
<!-- #endregion -->

```python
periode = 10
debut_semaine = pd.date_range(start="2019-09-16", periods=periode, freq="W-MON")
fin_semaine = debut_semaine + pd.offsets.Week(weekday=6)  # le dimanche suivant

windows = pd.DataFrame(
    {
        "semaine": debut_semaine.strftime("%Y-%m-%d") + "/" + fin_semaine.strftime("%Y-%m-%d"),
        "Value": rng.integers(0, 9, size=periode),
    }
)
print("Colonne fenêtre 'AAAA-MM-JJ/AAAA-MM-JJ' :")
print(windows.head(3))

# Récupérer le début de fenêtre comme vraie date (utile pour re-trier / merger).
windows["debut"] = pd.to_datetime(windows["semaine"].str.split("/").str[0])
print("\nDébut de fenêtre reparsé en datetime :")
print(windows[["semaine", "debut"]].head(3))
```

<!-- #region -->
La chaîne de fenêtre est lisible pour un humain mais **inutilisable telle quelle** pour des calculs
temporels : on garde la colonne `debut` (datetime) comme clé de tri/jointure.
<!-- #endregion -->

<!-- #region -->
## 15. Fusionner deux séries : merge & merge_asof
<!-- #endregion -->

<!-- #region -->
Deux capteurs n'échantillonnent presque jamais aux **mêmes instants**. Deux façons de les
rapprocher :

- **`pd.merge`** : jointure sur clé **exacte**. Avec `how="outer"`, les dates non communes sont
  conservées et complétées par `NaN`.
- **`pd.merge_asof`** : jointure sur la date **la plus proche** (`direction="nearest"` /
  `"backward"` / `"forward"`). Idéal pour aligner deux séries désynchronisées sans perdre de lignes.

On part de deux DataFrames aux dates légèrement décalées.
<!-- #endregion -->

```python
df1 = pd.DataFrame(
    {"Date": ["2022-01-01", "2022-01-08", "2022-01-15", "2022-01-22"], "Valeur1": [10, 20, 30, 40]}
)
df1["Date"] = pd.to_datetime(df1["Date"])
df1 = df1.sort_values("Date")

df2 = pd.DataFrame(
    {"Date": ["2022-01-01", "2022-01-08", "2022-01-16", "2022-01-23"], "Valeur2": [100, 200, 300, 400]}
)
df2["Date"] = pd.to_datetime(df2["Date"])
df2 = df2.sort_values("Date")
print("df1 :\n", df1, "\n\ndf2 :\n", df2, sep="")
```

<!-- #region -->
`df1` et `df2` coïncident les deux premières semaines puis divergent (15 vs 16, 22 vs 23). On
compare les deux jointures sur ce décalage.
<!-- #endregion -->

```python
# merge classique : jointure sur clé EXACTE (outer garde toutes les dates).
merged = pd.merge(df1, df2, on="Date", how="outer")
print("pd.merge (exact, outer) — les dates non communes deviennent NaN :")
print(merged)

# merge_asof : jointure sur la date la PLUS PROCHE (clé approximative, séries non alignées).
asof = pd.merge_asof(df1, df2, on="Date", direction="nearest")
print("\npd.merge_asof (nearest) — chaque ligne de df1 reçoit le df2 le plus proche dans le temps :")
print(asof)
```

<!-- #region -->
Le `merge` exact crée **6 lignes** (les dates 15, 16, 22, 23 ne s'apparient pas → `NaN`). Le
`merge_asof` garde les **4 lignes** de `df1` et leur associe la mesure de `df2` la plus proche :
aucune ligne perdue, aucun `NaN`. C'est l'outil de référence pour **aligner des capteurs
asynchrones**. (`merge_asof` exige des clés **triées**.)
<!-- #endregion -->

<!-- #region -->
## 16. Dédupliquer des timestamps (moyenne des doublons)
<!-- #endregion -->

<!-- #region -->
Plusieurs mesures **au même instant** (capteurs redondants, logs concurrents) ? On agrège par
timestamp. `groupby(timestamp).mean()` renvoie **une ligne par instant**, valeurs moyennées (on
pourrait aussi prendre `.median()`, `.first()`, `.max()` selon le besoin).
<!-- #endregion -->

```python
dup = pd.DataFrame(
    {
        "Date": ["2023-08-01 12:30:00", "2023-08-01 12:30:00", "2023-08-01 13:00:00", "2023-08-01 13:00:00"],
        "Valeur1": [10, 20, 15, 25],
        "Valeur2": [5, 10, 7, 12],
    }
)
dup["Date"] = pd.to_datetime(dup["Date"])
print("Avec doublons de timestamp :\n", dup, sep="")

# groupby sur le timestamp + moyenne : 1 ligne par instant, valeurs moyennées.
deduped = dup.groupby("Date", as_index=False).mean()
print("\nAprès dédup (moyenne des doublons) :\n", deduped, sep="")
```

<!-- #region -->
Les deux paires de doublons (12:30 et 13:00) sont fusionnées : `Valeur1` → moyenne (15, 20),
`Valeur2` → moyenne (7.5, 9.5). La série a maintenant un index temporel **unique**, prérequis de
`asfreq`, `resample` et `merge_asof`.
<!-- #endregion -->

<!-- #region -->
## 17. Nettoyage temporel : recettes avancées
<!-- #endregion -->

<!-- #region -->
Trois recettes plus pointues, fréquentes sur des **flux d'événements** horodatés (IoT, traces GPS).

**(a) Agrégation horaire pondérée par la durée.** Chaque mesure « couvre » l'intervalle jusqu'à la
suivante : sa durée $\Delta t$. Pour agréger à l'heure sans biaiser, on pondère chaque valeur par
sa **part de l'heure** $\;w_i = \Delta t_i / \sum_{\text{heure}} \Delta t\;$ avant de sommer. (On
utilise `resample("h")` — minuscule ; `"H"` est **déprécié** depuis pandas 2.2.)
<!-- #endregion -->

```python
events = pd.DataFrame(
    {
        "time": [
            "2023-07-19 16:32:08", "2023-07-19 16:32:28", "2023-07-19 16:32:50",
            "2023-07-19 17:21:08", "2023-07-19 17:21:25", "2023-07-19 17:44:03",
            "2023-07-19 17:45:10", "2023-07-19 17:46:20", "2023-07-19 17:47:20",
        ],
        "vitesse": [1.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 0.0],
    }
)
events["time"] = pd.to_datetime(events["time"])

# Durée (h) que "couvre" chaque mesure = écart jusqu'à la mesure suivante.
events["dt_h"] = events["time"].diff().dt.total_seconds().div(3600).shift(-1)
events = events[events["vitesse"] > 0].copy()                      # garder les mesures actives
events["dt_sum_h"] = events.groupby(events["time"].dt.hour)["dt_h"].transform("sum")
events["poids"] = events["dt_h"] / events["dt_sum_h"]              # part de l'heure couverte
events["vitesse_ponderee"] = events["poids"] * events["vitesse"]   # moyenne pondérée par la durée

hourly = events.resample("h", on="time").agg(vitesse_moy=("vitesse_ponderee", "sum"))
print("Vitesse horaire pondérée par la durée de chaque mesure :")
print(hourly)
```

<!-- #region -->
**(b) Boucher les longs trous par une ligne « heure tronquée ».** Quand l'écart entre deux mesures
dépasse 1 h, on injecte une ligne à `value = 0`, datée à l'**heure pleine** qui suit le dernier
point — utile pour matérialiser une période d'inactivité avant un `resample`.
<!-- #endregion -->

```python
series_gap = pd.DataFrame(
    {
        "time": [
            "2023-07-19 16:32:08", "2023-07-19 16:32:28", "2023-07-19 16:32:50",
            "2023-07-19 17:33:10", "2023-07-19 19:33:28", "2023-07-19 20:33:00",
            "2023-07-19 20:33:50",
        ],
        "value": [1, 2, 3, 6, 1, 2, 1],
    }
)
series_gap["time"] = pd.to_datetime(series_gap["time"])

# Repérer les trous > 1h et y injecter une ligne à value=0, datée à l'heure pleine suivante.
series_gap["gap_h"] = series_gap["time"].shift(-1).sub(series_gap["time"]).dt.total_seconds().div(3600)
injected = series_gap[series_gap["gap_h"] > 1].copy()
injected["time"] = (injected["time"] + pd.offsets.Hour(1)).dt.floor("h")
injected["value"] = 0
patched = (
    pd.concat([series_gap, injected])
    .drop(columns="gap_h")
    .sort_values("time")
    .reset_index(drop=True)
)
print("Lignes 'heure tronquée' injectées sur les trous > 1h :")
print(patched)
```

<!-- #region -->
**(c) Ne garder que le premier `NaN` de chaque trou.** Sur une série criblée de `NaN` consécutifs,
on veut souvent **un seul marqueur** par trou (puis supprimer le reste). On numérote les segments
consécutifs `NaN`/non-`NaN` via une somme cumulée de changements, puis on ne marque que le premier
`NaN` de chaque segment.
<!-- #endregion -->

```python
runs = pd.DataFrame(
    {
        "time": pd.date_range("2023-07-19 16:32:00", periods=6, freq="20s"),
        "value": [np.nan, 1.0, np.nan, np.nan, np.nan, 3.0],
    }
)
print("Avant (plusieurs NaN consécutifs) :\n", runs.to_string(index=False), sep="")

# Numéroter les segments consécutifs (NaN / non-NaN), puis dans chaque segment de
# NaN ne marquer que le premier (à 0) — les autres restent NaN pour être supprimés.
runs["segment"] = (runs["value"].notnull() != runs["value"].notnull().shift()).cumsum()
for _, grp in runs.groupby("segment"):
    if grp["value"].isnull().any():
        first_nan = grp.index[grp["value"].isnull()].min()
        runs.at[first_nan, "value"] = 0
runs = runs.drop(columns="segment")
print("\nAprès (1 seul marqueur 0 par trou) :\n", runs.to_string(index=False), sep="")
```

<!-- #region -->
Le trou de 3 `NaN` (indices 2-3-4) ne conserve qu'un `0` à l'indice 2 ; les deux autres restent
`NaN` et seront éliminés par un `dropna()`. L'astuce de la **somme cumulée de changements**
(`(x != x.shift()).cumsum()`) est le motif standard pour identifier des **plages consécutives** en
pandas.
<!-- #endregion -->

<!-- #region -->
## 18. Pièges et anti-patterns
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
| Attendre un arbre/KNN qu'il **extrapole** une tendance | Modèle linéaire, ou différencier/détrendre avant |
| `DataFrame.append` (supprimé en pandas 2.x) | `pd.concat([...], ignore_index=True)` |
| `resample("H")` (déprécié) | `resample("h")` (minuscule) |
| `merge_asof` sur clés non triées | Trier les deux DataFrames sur la clé d'abord |
<!-- #endregion -->

<!-- #region -->
## 19. Bonnes pratiques 2026
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
## 20. Sources
<!-- #endregion -->

<!-- #region -->
- [Forecasting: Principles and Practice — Hyndman & Athanasopoulos (référence libre)](https://otexts.com/fpp3/)
- [statsmodels — Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)
- [pandas — Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [missingno — visualisation des valeurs manquantes](https://github.com/ResidentMario/missingno)
- [Nixtla — baseline forecasts](https://www.nixtla.io/blog/baseline-forecasts) & [écosystème Nixtla](https://nixtlaverse.nixtla.io/)
- [The 2026 Time Series Toolkit — Foundation Models (MachineLearningMastery)](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- Notebooks liés : `TS_Time_Series_Overview` (wiki complet), `TS_ARIMA` (ARIMA détaillé),
  `TS_Generer_Sequence` (prep LSTM), `TS_Maintenance_Predictive` (case study).
<!-- #endregion -->
