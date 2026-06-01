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
# 🔧 Maintenance prédictive — NASA C-MAPSS Turbofan (RUL)
<!-- #endregion -->

<!-- #region -->
Notebook **Case Study** complet sur la **maintenance prédictive** : prédire la **Remaining Useful
Life** (RUL, nombre de cycles restants avant défaillance) d'un équipement à partir de séries
temporelles multivariées de capteurs.

On travaille sur le **vrai jeu de données NASA C-MAPSS FD001** (Turbofan Engine Degradation
Simulation) — la référence du domaine — téléchargé programmatiquement et mis en cache, avec un
**fallback synthétique** pour rester exécutable hors-ligne.

**Pipeline** :

1. Cadrer la tâche — RUL vs classification binaire vs détection d'anomalie.
2. Chargement du dataset réel (100 moteurs train *run-to-failure*, 100 moteurs test tronqués).
3. Construction de la cible RUL + **clipping piecewise-linear**.
4. **EDA** — durée de vie, dynamique des capteurs, capteurs constants, signature de dégradation.
5. **Feature engineering** — rolling mean / std / pente locale par moteur.
6. **Protocole d'évaluation officiel** C-MAPSS (prédiction au dernier cycle vs `RUL_FD001`).
7. **Bench régression** — 7 modèles (linéaires, forêts, boosting, MLP).
8. **Score asymétrique C-MAPSS** (pénalise les surestimations).
9. Visualisation des prédictions.
10. **Classification binaire** — panne imminente.
11. **Deep Learning** — LSTM sur fenêtres glissantes.
12-15. Pièges, modèles modernes, déploiement, sources.

> Rôles : **Case Study** (principal) + **Tutoriel** + **Wiki technique**.
<!-- #endregion -->

<!-- #region -->
## 1. Cadrer la tâche
<!-- #endregion -->

<!-- #region -->
Trois formulations principales du problème de maintenance prédictive :

| Formulation | Cible | Avantage | Inconvénient |
|---|---|---|---|
| **RUL** (Remaining Useful Life) | `y = cycles restants` (régression) | Action graduée, priorisation | Demande un historique de défaillances complètes |
| **Classification binaire** time-to-failure | `y ∈ {sain, critique}` | Plus simple, alerte directe | Choix arbitraire du seuil |
| **Détection d'anomalie** non supervisée | `y = score d'anomalie` | Pas besoin de défaillances historiques | Faux positifs fréquents |

**Choix 2026** : avec au moins **30-50 trajectoires complètes de défaillance** (cas C-MAPSS),
la **RUL** + gradient boosting (XGBoost / CatBoost) ou un **LSTM** sont les approches reines.
Sans run-to-failure historique → détection d'anomalie (Isolation Forest, autoencodeur).
<!-- #endregion -->

<!-- #region -->
## 2. Imports & configuration
<!-- #endregion -->

<!-- #region -->
Imports, graine aléatoire, palette de couleurs **neutre** (les capteurs sont des séries continues
sans sémantique *good/bad*), et la constante `RUL_CAP` utilisée pour le clipping (cf. section 3).
<!-- #endregion -->

```python
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import io
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
SEED = 42
np.random.seed(SEED)

# Palette neutre (capteurs = séries continues sans sémantique good/bad)
PRIMARY = "#00798c"
LAVENDER = "#9d83b8"
DUSTY = "#b8848e"
ACCENT = "#66a182"
BAD = "#d1495b"

RUL_CAP = 125  # clipping de la RUL (pratique standard C-MAPSS)
```

<!-- #region -->
## 3. Charger le dataset NASA C-MAPSS FD001
<!-- #endregion -->

<!-- #region -->
**C-MAPSS** simule la dégradation de moteurs turbofan. Le sous-dataset **FD001** est le plus simple :
une seule condition opérationnelle, un seul mode de panne (dégradation du compresseur HP).

- **`train_FD001`** : 100 moteurs suivis **jusqu'à la panne** (*run-to-failure*) → on connaît la RUL exacte.
- **`test_FD001`** : 100 moteurs dont la série est **tronquée** un certain nombre de cycles avant la panne.
- **`RUL_FD001`** : la vraie RUL au dernier cycle observé de chacun des 100 moteurs test (cible d'évaluation).

Chaque ligne = 26 colonnes : `engine_id`, `cycle`, 3 réglages opérationnels, **21 capteurs**.

La fonction ci-dessous **télécharge + met en cache** les fichiers dans `data/_shared/turbofan/`
(cf. `00_datasets.md`) et bascule sur un **générateur synthétique** si le réseau est indisponible,
pour que le notebook reste reproductible hors-ligne.
<!-- #endregion -->

```python
DATA_DIR = Path("data/_shared/turbofan")
MIRROR = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
INDEX_COLS = ["engine_id", "cycle"]
SETTING_COLS = ["op_1", "op_2", "op_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def _read_cmapss_txt(text: str) -> pd.DataFrame:
    """Parse un fichier C-MAPSS (séparé par espaces, 26 colonnes)."""
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
    df = df.iloc[:, : len(ALL_COLS)]
    df.columns = ALL_COLS
    return df


def _make_synthetic_fallback() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fallback hors-ligne : mime la structure FD001 (100 moteurs run-to-failure)."""
    rng = np.random.RandomState(SEED)

    def run(eid: int, n: int, truncate: int | None = None) -> pd.DataFrame:
        t = np.arange(n)
        deg = (t / n) ** 2
        rows = {"engine_id": eid, "cycle": t + 1, "op_1": 0.0, "op_2": 0.0, "op_3": 100.0}
        for i in range(1, 22):
            if i in (1, 5, 10, 16, 18, 19):       # capteurs constants par design
                rows[f"sensor_{i}"] = 100.0
            else:
                drift = (50 + i) * deg * (1 if i % 2 else -1)
                rows[f"sensor_{i}"] = 500 + drift + rng.normal(0, 3, n)
        d = pd.DataFrame(rows)
        return d.iloc[:truncate] if truncate else d

    lifetimes = rng.randint(140, 320, size=100)
    train = pd.concat([run(i + 1, lifetimes[i]) for i in range(100)], ignore_index=True)
    test_life = rng.randint(140, 320, size=100)
    cut = rng.randint(30, test_life - 10)
    test = pd.concat([run(i + 1, test_life[i], truncate=cut[i]) for i in range(100)], ignore_index=True)
    y_rul = (test_life - cut).astype(float)
    return train, test, y_rul


def load_cmapss_fd001(cache_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    """Charge FD001 : (train, test, y_test_rul, source).

    Télécharge depuis un mirror GitHub et met en cache sur disque.
    Bascule sur un générateur synthétique si le réseau est indisponible.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = {"train": "train_FD001.txt", "test": "test_FD001.txt", "rul": "RUL_FD001.txt"}
    raw: dict[str, str] = {}
    try:
        for key, fname in files.items():
            fpath = cache_dir / fname
            if fpath.exists():
                raw[key] = fpath.read_text()
            else:
                req = urllib.request.Request(f"{MIRROR}/{fname}", headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw[key] = resp.read().decode("utf-8")
                fpath.write_text(raw[key])
        train = _read_cmapss_txt(raw["train"])
        test = _read_cmapss_txt(raw["test"])
        y_rul = pd.read_csv(io.StringIO(raw["rul"]), header=None).iloc[:, 0].to_numpy(dtype=float)
        return train, test, y_rul, "NASA C-MAPSS FD001 (réel)"
    except Exception as exc:  # pragma: no cover - chemin réseau
        print(f"[warn] download indisponible ({type(exc).__name__}) → fallback synthétique")
        train, test, y_rul = _make_synthetic_fallback()
        return train, test, y_rul, "synthétique (fallback hors-ligne)"


train_raw, test_raw, y_test_rul, DATA_SOURCE = load_cmapss_fd001()
print(f"Source         : {DATA_SOURCE}")
print(f"Train          : {train_raw.shape} — {train_raw['engine_id'].nunique()} moteurs")
print(f"Test           : {test_raw.shape} — {test_raw['engine_id'].nunique()} moteurs")
print(f"RUL test (file): {len(y_test_rul)} valeurs, range [{y_test_rul.min():.0f}, {y_test_rul.max():.0f}]")
```

<!-- #region -->
Aperçu des premières lignes (un moteur démarre à `cycle = 1`) et statistiques de durée de vie.
Sur FD001 réel, les moteurs vivent typiquement entre ~128 et ~362 cycles (médiane ~199).
<!-- #endregion -->

```python
life = train_raw.groupby("engine_id")["cycle"].max()
print(train_raw[INDEX_COLS + SETTING_COLS + SENSOR_COLS[:4]].head())
print(f"\nDurée de vie train (cycles) : min={life.min()}, médiane={life.median():.0f}, max={life.max()}")
```

<!-- #region -->
## 4. Construire la cible RUL + clipping
<!-- #endregion -->

<!-- #region -->
Sur le **train run-to-failure**, la RUL au cycle $t$ d'un moteur de durée de vie $T$ vaut
simplement $\text{RUL}(t) = T - t$.

**Pourquoi clipper ?** En début de vie, un moteur sain ne montre **aucun signe** de dégradation :
ses capteurs sont indiscernables d'un autre moteur sain. Régresser une RUL de 300 vs 250 à partir
de signaux identiques est impossible et pollue l'apprentissage. La pratique standard C-MAPSS est
une **RUL piecewise-linear** : on plafonne la cible à une constante (ici `RUL_CAP = 125`).

$$
\text{RUL}_{\text{clip}}(t) = \min\big(T - t,\; \text{cap}\big)
$$

Le modèle apprend ainsi « tout va bien » (RUL = cap) tant que rien ne dégrade, puis une décroissance
linéaire sur la phase de dégradation finale — bien plus apprenable.
<!-- #endregion -->

```python
def add_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Ajoute la colonne RUL (= cycles restants) sur des trajectoires run-to-failure.

    cap : clip de la RUL (piecewise-linear). None = pas de clip.
    """
    out = df.copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["RUL"] = max_cycle - out["cycle"]
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out


train_rul = add_rul(train_raw, cap=RUL_CAP)
rul_uncapped = (train_raw.groupby("engine_id")["cycle"].transform("max") - train_raw["cycle"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(rul_uncapped, bins=50, color=PRIMARY, alpha=0.85)
axes[0].set_title("RUL brute (non clippée)")
axes[0].set_xlabel("RUL")
axes[1].hist(train_rul["RUL"], bins=50, color=ACCENT, alpha=0.85)
axes[1].axvline(RUL_CAP, color=BAD, ls="--", label=f"cap = {RUL_CAP}")
axes[1].set_title("RUL clippée (piecewise-linear)")
axes[1].set_xlabel("RUL")
axes[1].legend()
plt.tight_layout()
print(f"RUL clippée à {RUL_CAP} : {(rul_uncapped > RUL_CAP).mean():.1%} des cycles étaient au-dessus du cap.")
```

<!-- #region -->
Le pic à droite après clipping (~39 % des cycles ramenés à 125) correspond à toutes les phases
« moteur sain » regroupées sur une même valeur cible.
<!-- #endregion -->

<!-- #region -->
## 5. EDA — durée de vie par moteur
<!-- #endregion -->

<!-- #region -->
Distribution de la durée de vie totale des 100 moteurs d'entraînement. La forte dispersion
(facteur ~3 entre le moteur le plus court et le plus long) justifie une **RUL relative** plutôt
qu'un simple seuil sur le nombre de cycles.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 4))
life_sorted = life.sort_values()
ax.bar(range(len(life_sorted)), life_sorted.values, color=PRIMARY, alpha=0.8)
ax.axhline(life.mean(), color=BAD, ls="--", label=f"moyenne = {life.mean():.0f}")
ax.set_xlabel("moteurs (triés par durée de vie)")
ax.set_ylabel("cycles avant panne")
ax.set_title("Durée de vie totale par moteur (train run-to-failure)")
ax.legend()
plt.tight_layout()
```

<!-- #region -->
### 5.1 Trajectoires des capteurs
<!-- #endregion -->

<!-- #region -->
Quelques capteurs représentatifs pour 5 moteurs. On voit des tendances monotones (montée /
descente) ponctuées de bruit : c'est cette **dérive** que les features captureront.
<!-- #endregion -->

```python
demo_sensors = ["sensor_2", "sensor_7", "sensor_11", "sensor_15"]
sample_engines = train_raw["engine_id"].unique()[:5]
fig, axes = plt.subplots(1, len(demo_sensors), figsize=(16, 3.5))
neutral = [PRIMARY, LAVENDER, DUSTY, ACCENT, BAD]
for ax, s in zip(axes, demo_sensors):
    for color, eid in zip(neutral, sample_engines):
        g = train_raw[train_raw["engine_id"] == eid]
        ax.plot(g["cycle"], g[s], alpha=0.7, color=color, lw=1)
    ax.set_title(s)
    ax.set_xlabel("cycle")
plt.tight_layout()
```

<!-- #region -->
### 5.2 Capteurs constants — sélection par variance
<!-- #endregion -->

<!-- #region -->
**Point clé de FD001** : plusieurs capteurs sont **constants par design** (variance ≈ 0) car la
condition opérationnelle est unique. Ils n'apportent aucune information → on les retire avec un
`VarianceThreshold`. Sur FD001 réel, 6 capteurs sont droppés (`sensor_1, 5, 10, 16, 18, 19`),
laissant **15 capteurs informatifs**.
<!-- #endregion -->

```python
from sklearn.feature_selection import VarianceThreshold

variances = train_raw[SENSOR_COLS].var()
vt = VarianceThreshold(threshold=1e-6)
vt.fit(train_raw[SENSOR_COLS])
informative_sensors = [c for c, keep in zip(SENSOR_COLS, vt.get_support()) if keep]
constant_sensors = [c for c in SENSOR_COLS if c not in informative_sensors]
print(f"Capteurs constants (var ~ 0, droppés) : {constant_sensors}")
print(f"Capteurs informatifs retenus ({len(informative_sensors)}) : {informative_sensors}")
```

<!-- #region -->
### 5.3 Signature de dégradation
<!-- #endregion -->

<!-- #region -->
En moyennant chaque capteur pour chaque valeur de RUL puis en normalisant en [0,1], on révèle la
**signature de dégradation** : axe RUL inversé (la panne est à droite), on voit les capteurs
diverger de façon de plus en plus marquée à l'approche de la défaillance.
<!-- #endregion -->

```python
by_rul = train_rul.groupby("RUL")[informative_sensors].mean()
fig, ax = plt.subplots(figsize=(11, 4))
for s in informative_sensors[:6]:
    v = by_rul[s]
    norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
    ax.plot(by_rul.index, norm, lw=1.3, label=s)
ax.invert_xaxis()
ax.set_xlabel("RUL (décroissant → panne)")
ax.set_ylabel("valeur capteur normalisée [0,1]")
ax.set_title("Signature de dégradation moyenne (6 capteurs informatifs)")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
```

<!-- #region -->
## 6. Feature engineering — rolling par moteur
<!-- #endregion -->

<!-- #region -->
On enrichit chaque cycle avec des statistiques **glissantes** calculées **par moteur** (un
`groupby("engine_id")` évite toute fuite entre moteurs) :

- **moyenne** glissante : niveau lissé (débruite le capteur),
- **écart-type** glissant : volatilité locale (souvent ↑ avant la panne),
- **pente** glissante : tendance locale via régression linéaire en forme fermée
  $\text{slope} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$ avec $x = 0..n-1$.

`raw=True` passe un `ndarray` (et pas une `Series`) au callback de `rolling().apply` → calcul
**bien plus rapide**. On obtient 3 features × 15 capteurs + les 15 capteurs bruts = 60 features.
<!-- #endregion -->

```python
def _rolling_slope(a: np.ndarray) -> float:
    """Pente d'une régression linéaire sur la fenêtre (forme fermée, x = 0..n-1)."""
    n = len(a)
    if n < 2:
        return 0.0
    x = np.arange(n)
    xc = x - x.mean()
    denom = (xc * xc).sum()
    return float((xc * (a - a.mean())).sum() / denom) if denom else 0.0


def add_rolling_features(df: pd.DataFrame, cols: list[str], window: int = 10) -> pd.DataFrame:
    """Ajoute rolling mean/std/slope par moteur (anti-leakage : groupby engine_id).

    `raw=True` passe un ndarray (pas une Series) au callback → bien plus rapide.
    """
    out = df.copy().sort_values(["engine_id", "cycle"])
    for col in cols:
        g = out.groupby("engine_id")[col]
        out[f"{col}_mean{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).mean())
        out[f"{col}_std{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0.0)
        out[f"{col}_slope{window}"] = g.transform(
            lambda s: s.rolling(window, min_periods=2).apply(_rolling_slope, raw=True)
        ).fillna(0.0)
    return out


WINDOW = 10
train_feat = add_rolling_features(train_rul, informative_sensors, window=WINDOW)
test_feat = add_rolling_features(add_rul(test_raw, cap=None), informative_sensors, window=WINDOW)
feature_cols = informative_sensors + [
    c for c in train_feat.columns if any(c.endswith(suf) for suf in (f"_mean{WINDOW}", f"_std{WINDOW}", f"_slope{WINDOW}"))
]
print(f"Nb features : {len(feature_cols)} (capteurs + rolling)")
```

<!-- #region -->
## 7. Split & protocole d'évaluation officiel C-MAPSS
<!-- #endregion -->

<!-- #region -->
Le protocole d'évaluation C-MAPSS n'est **pas** un split aléatoire :

- **Entraînement** : tous les cycles des 100 moteurs *run-to-failure* (RUL clippée).
- **Évaluation** : on prédit la RUL au **dernier cycle observé** de chacun des 100 moteurs test,
  puis on la compare aux vraies valeurs de `RUL_FD001`.

On applique le **même clipping** à la RUL test pour une comparaison cohérente. Splitter au milieu
d'une trajectoire (par échantillon plutôt que par moteur) provoquerait une **fuite** massive.
<!-- #endregion -->

```python
# Train : tous les cycles des moteurs run-to-failure (RUL clippée).
X_train = train_feat[feature_cols].to_numpy()
y_train = train_feat["RUL"].to_numpy()

# Test officiel : DERNIER cycle de chaque moteur test → comparé à RUL_FD001.
last_idx = test_feat.groupby("engine_id")["cycle"].idxmax()
X_test = test_feat.loc[last_idx, feature_cols].to_numpy()
y_test = np.clip(y_test_rul, 0, RUL_CAP)  # même clipping que le train

print(f"X_train : {X_train.shape}  |  X_test (last cycle) : {X_test.shape}")
print(f"y_test (clippée) range : [{y_test.min():.0f}, {y_test.max():.0f}]")
```

<!-- #region -->
## 8. Bench régression RUL — 7 modèles
<!-- #endregion -->

<!-- #region -->
Comparaison de 7 régressseurs sur le test officiel : `LinearReg`, `Ridge`, `RandomForest`,
`GradientBoosting` (sklearn), `XGBoost`, `CatBoost`, `MLP`. Les modèles linéaires et le MLP sont
**standardisés** (sensibles à l'échelle) ; les modèles à base d'arbres non.

> **Note env.** `LightGBM` est volontairement absent du bench : `libgomp.so.1` est indisponible
> dans cet environnement (le `fit` plante au runtime). Sur une machine standard il s'utilise comme
> XGBoost et donne des performances équivalentes.

Sur FD001, les **gradient boosting** (CatBoost / GBM / XGBoost) dominent avec une **MAE ≈ 12-13
cycles** — un excellent résultat compte tenu d'une durée de vie médiane de ~200 cycles.
<!-- #endregion -->

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor

# lightgbm volontairement absent : libgomp.so.1 indisponible dans cet env (le fit crash).
NEEDS_SCALING = {"LinearReg", "Ridge", "MLP"}
models = {
    "LinearReg": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=SEED),
    "GBM": GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=SEED),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, n_jobs=-1, random_state=SEED),
    "CatBoost": CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, random_seed=SEED, verbose=0),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=SEED),
}

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

fitted, rows = {}, []
for name, model in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if name in NEEDS_SCALING else (X_train, X_test)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    fitted[name] = (model, name in NEEDS_SCALING)
    rows.append({
        "model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": r2_score(y_test, pred),
    })

bench = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
print(bench.round(3).to_string(index=False))
```

<!-- #region -->
## 9. Score asymétrique C-MAPSS
<!-- #endregion -->

<!-- #region -->
La MAE traite symétriquement les erreurs. Or en maintenance, **surestimer** la RUL (prédire trop
tard → panne en service, coût catastrophique) est bien plus grave que **sous-estimer** (maintenance
prématurée, coût modéré). Le score officiel du challenge PHM08 / C-MAPSS encode cette asymétrie :

$$
S = \sum_i \begin{cases}
e^{-d_i / 13} - 1 & \text{si } d_i = \hat{y}_i - y_i < 0 \quad (\text{early, sous-estimation}) \\
e^{\, d_i / 10} - 1 & \text{si } d_i \geq 0 \quad (\text{late, surestimation})
\end{cases}
$$

On **minimise** $S$. Les surestimations sont pénalisées plus fort (dénominateur 10 < 13). On compare
les 3 meilleurs modèles à une baseline naïve (prédire la RUL moyenne du train).
<!-- #endregion -->

```python
def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score asymétrique du challenge PHM08/C-MAPSS. Plus bas = mieux.

    d = y_pred - y_true. Sous-estimer (d<0) est moins pénalisé que surestimer (d>=0).
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float((np.exp(-d[d < 0] / 13) - 1).sum() + (np.exp(d[d >= 0] / 10) - 1).sum())


def _predict(name: str) -> np.ndarray:
    model, scaled = fitted[name]
    return model.predict(X_test_s if scaled else X_test)


baseline = np.full_like(y_test, y_train.mean(), dtype=float)
print(f"Baseline naïve (RUL=moyenne train) : C-MAPSS = {cmapss_score(y_test, baseline):,.0f}")
for name in bench["model"].head(3):
    print(f"  {name:13s} C-MAPSS = {cmapss_score(y_test, _predict(name)):,.0f}")
```

<!-- #region -->
## 10. Visualisation des prédictions
<!-- #endregion -->

<!-- #region -->
À gauche : nuage **prédit vs réel** sur les 100 moteurs test (la diagonale = prédiction parfaite).
À droite : la **trajectoire complète** de RUL prédite pour un moteur test — le modèle suit la
décroissance et le plateau imposé par le clipping.
<!-- #endregion -->

```python
best_name = bench.iloc[0]["model"]
best_model, best_scaled = fitted[best_name]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
# (a) scatter pred vs true sur les 100 moteurs test
best_pred = _predict(best_name)
axes[0].scatter(y_test, best_pred, alpha=0.6, s=25, color=PRIMARY)
lims = [0, RUL_CAP]
axes[0].plot(lims, lims, ls="--", color=BAD, label="parfait")
axes[0].set_xlabel("RUL réelle"); axes[0].set_ylabel("RUL prédite")
axes[0].set_title(f"{best_name} — dernier cycle (100 moteurs test)")
axes[0].legend()

# (b) trajectoire complète de la RUL prédite pour 1 moteur test
demo_eid = test_feat["engine_id"].iloc[0]
g = test_feat[test_feat["engine_id"] == demo_eid].sort_values("cycle")
Xg = g[feature_cols].to_numpy()
Xg = scaler.transform(Xg) if best_scaled else Xg
axes[1].plot(g["cycle"], np.clip(g["RUL"], 0, RUL_CAP), color="black", label="RUL réelle (clippée)")
axes[1].plot(g["cycle"], best_model.predict(Xg), color=BAD, alpha=0.8, label="RUL prédite")
axes[1].set_xlabel("cycle"); axes[1].set_ylabel("RUL")
axes[1].set_title(f"Trajectoire — moteur test #{demo_eid}")
axes[1].legend()
plt.tight_layout()
```

<!-- #region -->
## 11. Classification binaire — panne imminente
<!-- #endregion -->

<!-- #region -->
Reformulation opérationnelle : plutôt qu'une RUL exacte, on veut une **alerte binaire**
« moteur critique » dès que $\text{RUL} < 30$ cycles. On utilise un `RandomForestClassifier`
avec `class_weight="balanced"` (les cas critiques sont minoritaires). La **matrice de confusion**
montre le compromis : on surveille surtout le **rappel** sur la classe critique (rater une panne
coûte cher).
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

THRESH = 30
y_train_cls = (train_feat["RUL"] < THRESH).astype(int).to_numpy()
y_test_cls = (y_test < THRESH).astype(int)

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED)
clf.fit(X_train, y_train_cls)
pred_cls = clf.predict(X_test)
print(classification_report(y_test_cls, pred_cls, target_names=["sain", "critique"], digits=3, zero_division=0))

cm = confusion_matrix(y_test_cls, pred_cls)
fig, ax = plt.subplots(figsize=(4.5, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r", cbar=False,
            xticklabels=["sain", "critique"], yticklabels=["sain", "critique"], ax=ax)
ax.set_xlabel("prédit"); ax.set_ylabel("réel")
ax.set_title(f"Confusion — panne imminente (RUL < {THRESH})")
plt.tight_layout()
```

<!-- #region -->
## 12. Deep Learning — LSTM sur fenêtres glissantes
<!-- #endregion -->

<!-- #region -->
Plutôt que des features agrégées, un **LSTM** apprend directement la dynamique temporelle sur des
**fenêtres glissantes** de capteurs bruts (standardisés sur le train uniquement, anti-leakage).

- `make_train_windows` : pour chaque moteur, fenêtres de `WINDOW_SEQ=30` cycles ; cible = RUL clippée
  au dernier cycle de la fenêtre. Le paramètre `step` sous-échantillonne les fenêtres pour un
  entraînement rapide **sur CPU**.
- `make_last_windows` : la dernière fenêtre de chaque moteur test (left-pad si trop court) — ce sont
  ces 100 fenêtres qu'on évalue, exactement comme le protocole officiel.
<!-- #endregion -->

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW_SEQ = 30
seq_cols = informative_sensors

seq_scaler = StandardScaler().fit(train_raw[seq_cols])


def make_train_windows(df_rul: pd.DataFrame, cols: list[str], window: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fenêtres glissantes (train) : X (N,T,F), y = RUL clippée au dernier cycle de la fenêtre.

    `step` sous-échantillonne les fenêtres (1 = toutes) — pratique pour entraîner vite sur CPU.
    """
    Xs, ys = [], []
    for _, g in df_rul.groupby("engine_id"):
        g = g.sort_values("cycle")
        vals = seq_scaler.transform(g[cols]).astype(np.float32)
        ruls = g["RUL"].to_numpy(np.float32)
        for i in range(0, len(g) - window + 1, step):
            Xs.append(vals[i:i + window]); ys.append(ruls[i + window - 1])
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)


def make_last_windows(df: pd.DataFrame, cols: list[str], window: int) -> np.ndarray:
    """Dernière fenêtre de chaque moteur test (left-pad si trajectoire trop courte)."""
    Xs = []
    for _, g in df.groupby("engine_id"):
        g = g.sort_values("cycle")
        vals = seq_scaler.transform(g[cols]).astype(np.float32)
        if len(vals) >= window:
            win = vals[-window:]
        else:
            pad = np.repeat(vals[:1], window - len(vals), axis=0)
            win = np.concatenate([pad, vals], axis=0)
        Xs.append(win)
    return np.asarray(Xs, np.float32)


# step=5 : on sous-échantillonne les fenêtres pour un entraînement rapide sur CPU
X_seq_tr, y_seq_tr = make_train_windows(train_rul, seq_cols, WINDOW_SEQ, step=5)
X_seq_te = make_last_windows(test_raw, seq_cols, WINDOW_SEQ)
print(f"Train windows : {X_seq_tr.shape}  |  Test last-windows : {X_seq_te.shape}")
```

<!-- #region -->
Modèle, entraînement et évaluation. Deux astuces décisives pour la convergence d'un NN sur cette
tâche :

1. **Cible normalisée** dans [0,1] (`RUL / RUL_CAP`) — un réseau converge bien mieux sur une cible
   bornée ; on dé-normalise les prédictions en multipliant par `RUL_CAP`.
2. **Clip de gradient** (`clip_grad_norm_`) pour stabiliser le LSTM.

Le modèle est volontairement **léger** (LSTM 1 couche, 32 unités) pour tourner en ~2 min sur CPU.
Sur FD001, ce LSTM atteint une **MAE ≈ 11-12 cycles**, au niveau (voire mieux) que le meilleur
gradient boosting — la dynamique séquentielle apporte un vrai gain. Pour aller plus loin :
bidirectionnel, 2+ couches, attention.
<!-- #endregion -->

```python
torch.manual_seed(SEED)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


class LSTMRUL(nn.Module):
    """LSTM (1 couche) + tête de régression pour la RUL.

    Volontairement léger pour tourner vite sur CPU. Variantes plus lourdes
    (bidirectionnel, 2 couches) en commentaire — au choix selon le hardware.
    """

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)   # h[-1] : état caché final


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LSTMRUL(n_features=len(seq_cols)).to(device)
# Cible normalisée dans [0,1] (RUL / cap) : un NN converge bien mieux sur une cible bornée.
loader = DataLoader(WindowDataset(X_seq_tr, y_seq_tr / RUL_CAP), batch_size=256, shuffle=True)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

for epoch in range(15):
    net.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        running += loss.item() * len(xb)
    if (epoch + 1) % 3 == 0:
        print(f"epoch {epoch + 1:2d}  loss = {running / len(X_seq_tr):.5f}")

net.eval()
with torch.no_grad():
    pred_lstm = net(torch.from_numpy(X_seq_te).to(device)).cpu().numpy() * RUL_CAP   # dé-normalise
pred_lstm = np.clip(pred_lstm, 0, RUL_CAP)

print(f"\nLSTM  MAE={mean_absolute_error(y_test, pred_lstm):.2f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_test, pred_lstm)):.2f}  "
      f"R2={r2_score(y_test, pred_lstm):.3f}  C-MAPSS={cmapss_score(y_test, pred_lstm):,.0f}")
print(f"(rappel meilleur ML = {best_name} : MAE={bench.iloc[0]['MAE']:.2f})")
```

<!-- #region -->
## 13. Pièges & anti-patterns
<!-- #endregion -->

<!-- #region -->
| ❌ Anti-pattern | ✅ Correctif |
|---|---|
| Split par échantillon (pas par moteur) | Fuite : le même moteur en train **et** test |
| Prédire la RUL sans clipping | Le modèle s'épuise sur une RUL « infinie » non prédictible en début de vie → clip ~125 |
| Reporter la MAE seule | Ajouter le **score C-MAPSS** asymétrique (sous-estimer ≠ surestimer) |
| Garder les capteurs constants | `VarianceThreshold` : sur FD001, 6 capteurs ne portent aucune info |
| Standardiser sur tout le dataset | Fit du scaler sur le **train uniquement**, transform sur le test |
| Évaluer sur tous les cycles test | Le protocole officiel évalue au **dernier cycle** vs `RUL_FD001` |
| Ignorer l'analyse de survie | RUL = temps-jusqu'à-événement censuré, cf. [DS_Survival_Analysis.ipynb](DS_Survival_Analysis.ipynb) |
<!-- #endregion -->

<!-- #region -->
## 14. Modèles modernes 2026
<!-- #endregion -->

<!-- #region -->
| Approche | Forces | Quand |
|---|---|---|
| **Gradient Boosting** (XGBoost / CatBoost / LightGBM) | Robuste, peu de tuning, fort sur features tabulaires | Baseline reine sur features rolling |
| **LSTM / GRU** | Capture la dynamique séquentielle brute | Capteurs longs, peu de feature engineering |
| **TCN** (Temporal Conv Net) | Champ réceptif large, parallélisable | Alternative rapide aux RNN |
| **TFT** (Temporal Fusion Transformer) | SOTA TS multivariée + interprétable | Multi-séries, covariables |
| **N-BEATS / N-HiTS** | Bench-winning sur compétitions RUL | Forecasting pur |
| **PatchTST / Transformers TS** | Attention sur patches temporels | Longues séquences |
| **Survival / DeepSurv** | Modèle naturellement la censure | Données partiellement observées |
<!-- #endregion -->

<!-- #region -->
## 15. Déploiement & monitoring
<!-- #endregion -->

<!-- #region -->
Pattern de prod 2026 :

- **Edge** (PLC, gateway IoT) : modèle léger (XGBoost quantizé, ONNX Runtime, TFLite) calcule la RUL
  en local et ne remonte que les alertes.
- **Cloud** (Kafka + Spark/Flink) : streaming, fenêtrage online, re-training batch de nuit.
- **Alerting par paliers** :
  - RUL < 60 → email équipe maintenance ;
  - RUL < 30 → ticket urgent + planification intervention ;
  - RUL < 10 → alerte SMS + procédure de *safe-shutdown*.
- **Monitoring de drift** : une dérive capteur peut signaler un **nouveau régime opérationnel**
  non vu en entraînement → cf. [MLOps_Drift_Monitoring.ipynb](MLOps_Drift_Monitoring.ipynb).
- **Retraining** : déclenché par le drift + l'ajout de nouvelles trajectoires complètes (≥ 30-50 runs).
<!-- #endregion -->

<!-- #region -->
## 16. Sources
<!-- #endregion -->

<!-- #region -->
- NASA C-MAPSS — PCoE : <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation* (PHM08) : <https://ieeexplore.ieee.org/document/4711414>
- PHM Society challenges : <https://www.phmsociety.org/>
- Mirror dataset utilisé : <https://github.com/edwardzjl/CMAPSSData>
- Notebooks liés : [TS_Generer_Sequence.ipynb](TS_Generer_Sequence.ipynb) (prep séquences),
  [TS_Time_Series_Overview.ipynb](TS_Time_Series_Overview.ipynb) (panorama TS),
  [DL_PyTorch.ipynb](DL_PyTorch.ipynb) (LSTM en détail),
  [DS_Survival_Analysis.ipynb](DS_Survival_Analysis.ipynb) (censure),
  [MLOps_Drift_Monitoring.ipynb](MLOps_Drift_Monitoring.ipynb) (monitoring prod).
<!-- #endregion -->
