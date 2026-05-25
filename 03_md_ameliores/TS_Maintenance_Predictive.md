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
# 🔧 Maintenance prédictive — Case study complet
<!-- #endregion -->

<!-- #region -->
Notebook **Case Study** complet sur la **maintenance prédictive** : prédire l'instant ou la probabilité de défaillance d'un équipement (turbine, pompe, moteur) à partir de séries temporelles de capteurs.

**Pipeline complet** :

1. Cadrage — RUL (Remaining Useful Life) vs détection d'anomalie vs classification binaire.
2. **EDA** multi-niveaux sur signaux capteurs : distribution du RUL, max RUL/cycle par moteur, dynamique des capteurs.
3. **Feature engineering** : sélection par variance, rolling means / stds / pentes locales (trends).
4. **Bench régression RUL** : LinearReg, RandomForest, GradientBoosting, AdaBoost, MLP, XGBoost, LightGBM, CatBoost.
5. **Classification binaire** time-to-failure (sain vs critique).
6. **Deep Learning** — LSTM séquence-to-target sur fenêtres glissantes.
7. **Évaluation** : MAE, RMSE, R² + score asymétrique **C-MAPSS**.
8. **Déploiement** edge / cloud.

Dataset : **NASA C-MAPSS Turbofan** (cf `00_datasets.md` — `data/_shared/turbofan/README.md`). Pour cette démo nous utilisons un **synthétique** qui mime sa structure : 20 moteurs run jusqu'à défaillance, 3 capteurs dégradent, ajout de bruit.
<!-- #endregion -->

<!-- #region -->
## 1. Cadrer la tâche
<!-- #endregion -->

<!-- #region -->
Trois formulations principales :

| Formulation | Cible | Avantage | Inconvénient |
|---|---|---|---|
| **RUL** (Remaining Useful Life) | `y = cycles restants avant défaillance` (régression) | Action graduée, priorisation | Demande historique de défaillances |
| **Classification binaire** time-to-failure | `y ∈ {sain, défaillance dans N jours}` | Plus simple, alerte directe | Choix arbitraire du seuil N |
| **Anomaly detection** non supervisé | `y = score d'anomalie` | Pas besoin de défaillances historiques | Faux positifs fréquents |

**Choix 2026** : si tu as au moins **30-50 cycles complets de défaillance**, **RUL** + XGBoost/LightGBM est l'approche reine. Sinon → anomaly detection (Isolation Forest, autoencoder).
<!-- #endregion -->

<!-- #region -->
## 2. Données synthétiques (proxy NASA Turbofan)
<!-- #endregion -->

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

rng = np.random.RandomState(42)


def generate_engine_run(engine_id: int, n_cycles: int) -> pd.DataFrame:
    """Génère le run d'un moteur : 3 capteurs qui dégradent + bruit + 2 labels classif.

    - sensor_1 (pression) : monte avec dégradation
    - sensor_2 (vibration) : descend avec dégradation + oscillation cyclique
    - sensor_3 (température) : monte fortement vers la fin
    - rul : cycles restants
    - label1 : 1 si rul < 30 (critique)
    - label2 : 1 si rul < 60 (warning)
    """
    t = np.arange(n_cycles)
    degradation = (t / n_cycles) ** 2
    s1 = 100 + 50 * degradation + rng.normal(0, 5, n_cycles)
    s2 = 800 + 30 * np.sin(t / 5) - 80 * degradation + rng.normal(0, 8, n_cycles)
    s3 = 20 + 5 * np.cos(t / 7) + 15 * degradation + rng.normal(0, 1, n_cycles)
    rul = n_cycles - 1 - t
    return pd.DataFrame({
        "engine_id": engine_id, "cycle": t,
        "sensor_1": s1, "sensor_2": s2, "sensor_3": s3,
        "rul": rul,
        "label1": (rul < 30).astype(int),
        "label2": (rul < 60).astype(int),
    })


engines = pd.concat([generate_engine_run(i, n_cycles=rng.randint(120, 250)) for i in range(20)])
print(f"N engines : {engines['engine_id'].nunique()}")
print(f"Total cycles : {len(engines)}")
print(f"Distribution labels : label1 (critique) = {engines['label1'].mean():.2%}, label2 (warning) = {engines['label2'].mean():.2%}")
print(engines.head())
```

<!-- #region -->
## 3. EDA — distribution du RUL et dynamique des capteurs
<!-- #endregion -->

<!-- #region -->
### 3.1 Distribution du RUL max par moteur
<!-- #endregion -->

```python
# Max RUL par moteur = durée de vie totale
rul_max_per_engine = engines.groupby("engine_id")["rul"].max().sort_values()

fig, ax = plt.subplots(figsize=(11, 4))
rul_max_per_engine.plot(kind="bar", ax=ax, color="steelblue", alpha=0.8)
ax.axhline(rul_max_per_engine.mean(), color="red", linestyle="--", label=f"Moyenne = {rul_max_per_engine.mean():.0f}")
ax.axhline(rul_max_per_engine.median(), color="darkblue", linestyle="-", label=f"Médiane = {rul_max_per_engine.median():.0f}")
ax.set_xlabel("engine_id")
ax.set_ylabel("RUL max (= durée de vie)")
ax.set_title("Durée de vie totale par moteur")
ax.legend()
plt.tight_layout()
```

<!-- #region -->
### 3.2 Trajectoires des capteurs vs RUL
<!-- #endregion -->

```python
# 5 moteurs aléatoires, capteurs vs cycle
sample_engines = engines["engine_id"].unique()[:5]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, sensor in zip(axes, ["sensor_1", "sensor_2", "sensor_3"]):
    for eng_id in sample_engines:
        g = engines[engines["engine_id"] == eng_id]
        ax.plot(g["cycle"], g[sensor], alpha=0.5, label=f"engine {eng_id}")
    ax.set_title(f"{sensor} vs cycle (5 moteurs)")
    ax.set_xlabel("cycle"); ax.set_ylabel(sensor)
    ax.grid(alpha=0.3)
plt.tight_layout()
```

<!-- #region -->
### 3.3 Capteurs moyennés par RUL (vue agrégée)
<!-- #endregion -->

```python
# Moyenne des capteurs pour chaque valeur de RUL — révèle la dégradation typique
sensor_means_by_rul = engines.groupby("rul")[["sensor_1", "sensor_2", "sensor_3"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(11, 4))
for col in ["sensor_1", "sensor_2", "sensor_3"]:
    # Normalisation min-max pour comparaison visuelle
    vals = sensor_means_by_rul[col]
    norm = (vals - vals.min()) / (vals.max() - vals.min())
    ax.plot(sensor_means_by_rul["rul"], norm, label=col)
ax.invert_xaxis()      # RUL décroissant : plus à droite = plus proche de la défaillance
ax.set_xlabel("RUL (décroissant →)")
ax.set_ylabel("Valeur capteur normalisée [0,1]")
ax.set_title("Dynamique moyenne des capteurs vs RUL — la dégradation s'accélère vers la fin")
ax.legend()
plt.tight_layout()
```

<!-- #region -->
## 4. Feature engineering
<!-- #endregion -->

<!-- #region -->
### 4.1 Sélection par variance
<!-- #endregion -->

<!-- #region -->
Les capteurs **constants** (variance ≈ 0) sont du bruit ou inutiles → drop. Sur NASA C-MAPSS, plusieurs capteurs sont constants par design (settings opérationnels).
<!-- #endregion -->

```python
from sklearn.feature_selection import VarianceThreshold

sensor_cols = ["sensor_1", "sensor_2", "sensor_3"]
variances = engines[sensor_cols].var()
print("Variance par capteur :")
print(variances.round(2))

vt = VarianceThreshold(threshold=0.1)
vt.fit(engines[sensor_cols])
kept_features = [c for c, k in zip(sensor_cols, vt.get_support()) if k]
print(f"Features conservées (var > 0.1) : {kept_features}")
```

<!-- #region -->
### 4.2 Features rolling : means, stds, pentes
<!-- #endregion -->

```python
def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str], window: int = 10) -> pd.DataFrame:
    """Ajoute rolling features (mean, std, min, max, slope) par moteur."""
    out = df.copy().sort_values(["engine_id", "cycle"])

    for col in sensor_cols:
        g = out.groupby("engine_id")[col]
        out[f"{col}_mean_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).mean())
        out[f"{col}_std_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0)
        out[f"{col}_min_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).min())
        out[f"{col}_max_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).max())

        # Pente locale via régression linéaire sur la fenêtre
        def _slope(s: pd.Series) -> float:
            if len(s) < 2:
                return 0.0
            x = np.arange(len(s))
            return float(np.polyfit(x, s.values, 1)[0])

        out[f"{col}_slope_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=2).apply(_slope, raw=False)).fillna(0)

    return out


engines_feat = add_rolling_features(engines, sensor_cols, window=10)
print(f"Nouvelles features : {[c for c in engines_feat.columns if c not in engines.columns][:8]}")
print(f"Total features : {len(engines_feat.columns)}")
```

<!-- #region -->
### 4.3 PCA pour réduction de dim (3 stratégies)
<!-- #endregion -->

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

feat_cols = [c for c in engines_feat.columns if c not in ["engine_id", "cycle", "rul", "label1", "label2"]]

# Stratégie 1 : PCA globale sur tout
X_all = StandardScaler().fit_transform(engines_feat[feat_cols])
pca = PCA(n_components=4)
proj = pca.fit_transform(X_all)
print(f"PCA globale : variance expliquée = {pca.explained_variance_ratio_.round(3).tolist()} (cumul = {pca.explained_variance_ratio_.cumsum()[-1]:.3f})")

# Visualisation 2D coloriée par RUL
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(proj[:, 0], proj[:, 1], c=engines_feat["rul"], cmap="viridis_r", s=4, alpha=0.5)
plt.colorbar(sc, ax=ax, label="RUL (bleu=fin de vie)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA globale — trajectoire de dégradation visible")
plt.tight_layout()
```

<!-- #region -->
## 5. Bench régression RUL — 6 algorithmes
<!-- #endregion -->

```python
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# Split PAR MOTEUR (jamais splitter au milieu d'un cycle)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(gss.split(engines_feat, groups=engines_feat["engine_id"]))
train, test = engines_feat.iloc[tr_idx], engines_feat.iloc[te_idx]
print(f"Train : {train['engine_id'].nunique()} moteurs ({len(train):,} cycles)")
print(f"Test  : {test['engine_id'].nunique()} moteurs ({len(test):,} cycles)")

X_train, y_train = train[feat_cols], train["rul"]
X_test, y_test = test[feat_cols], test["rul"]

models = {
    "LinearReg":   LinearRegression(),
    "Ridge":       Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12),
    "GBM":         GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    "AdaBoost":    AdaBoostRegressor(n_estimators=100, random_state=42),
    "XGBoost":     xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
    "LightGBM":    lgb.LGBMRegressor(n_estimators=200, max_depth=-1, num_leaves=31, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
    "MLP":         MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
}

results = []
for name, model in models.items():
    # Pour les modèles linéaires/NN : scaling
    if name in ("LinearReg", "Ridge", "MLP"):
        sc = StandardScaler().fit(X_train)
        Xtr_s, Xte_s = sc.transform(X_train), sc.transform(X_test)
    else:
        Xtr_s, Xte_s = X_train.values, X_test.values

    model.fit(Xtr_s, y_train)
    preds = model.predict(Xte_s)
    results.append({
        "model": name,
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    })

bench_df = pd.DataFrame(results).sort_values("MAE")
print(bench_df.to_string(index=False))
```

<!-- #region -->
## 6. Score asymétrique C-MAPSS
<!-- #endregion -->

<!-- #region -->
En maintenance prédictive, **sous-estimer la RUL** (prédire fin de vie trop tôt) coûte une maintenance prématurée. **Surestimer** (prédire trop tard) coûte une panne en service (coût catastrophique).

Le score officiel C-MAPSS pénalise les sous-estimations moins que les surestimations :

$$
S = \sum_i \begin{cases} e^{-d_i / 13} - 1 & \text{si } d_i = \hat{y}_i - y_i < 0 \text{ (early)} \\ e^{d_i / 10} - 1 & \text{si } d_i \geq 0 \text{ (late)} \end{cases}
$$

On veut **minimiser** ce score (asymétrique, exponentiel).
<!-- #endregion -->

```python
def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score asymétrique du challenge NASA C-MAPSS. Plus bas = mieux."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    early = np.exp(-d[d < 0] / 13) - 1
    late = np.exp(d[d >= 0] / 10) - 1
    return float(early.sum() + late.sum())


# Bench avec score C-MAPSS sur les 3 meilleurs
top3 = bench_df.head(3)["model"].tolist()
print("Top-3 modèles par MAE — comparaison C-MAPSS :")
for name in top3:
    model = models[name]
    if name in ("LinearReg", "Ridge", "MLP"):
        sc = StandardScaler().fit(X_train)
        preds = model.predict(sc.transform(X_test))
    else:
        preds = model.predict(X_test.values)
    print(f"  {name:15s}  C-MAPSS = {cmapss_score(y_test.values, preds):.1f}")
```

<!-- #region -->
## 7. Classification binaire — failure imminente (label1)
<!-- #endregion -->

```python
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
clf.fit(X_train, train["label1"])
preds_clf = clf.predict(X_test)
proba_clf = clf.predict_proba(X_test)[:, 1]

print(classification_report(test["label1"], preds_clf, target_names=["sain", "critique"], digits=3))

# Confusion matrix visualisée
cm = confusion_matrix(test["label1"], preds_clf)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["sain", "critique"],
            yticklabels=["sain", "critique"], ax=ax)
ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
ax.set_title("Confusion matrix — failure imminente")
plt.tight_layout()
```

<!-- #region -->
## 8. Deep Learning — LSTM sur fenêtres glissantes
<!-- #endregion -->

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW_SIZE = 20    # 20 derniers cycles en input


def make_windows_per_engine(df: pd.DataFrame, feat_cols: list[str], window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère fenêtres glissantes par moteur. Renvoie X (3D), Y (RUL au last cycle), ids."""
    Xs, Ys, ids = [], [], []
    for eng_id, g in df.groupby("engine_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        vals = g[feat_cols].values
        ruls = g["rul"].values
        n = len(g) - window + 1
        for i in range(n):
            Xs.append(vals[i:i + window])
            Ys.append(ruls[i + window - 1])
            ids.append(eng_id)
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32), np.array(ids)


# Standardiser les features sur train uniquement (pas de leak)
sensor_feat_cols = ["sensor_1", "sensor_2", "sensor_3"]
sc_seq = StandardScaler().fit(train[sensor_feat_cols])
train_seq = train.copy(); train_seq[sensor_feat_cols] = sc_seq.transform(train[sensor_feat_cols])
test_seq = test.copy(); test_seq[sensor_feat_cols] = sc_seq.transform(test[sensor_feat_cols])

X_tr_seq, y_tr_seq, _ = make_windows_per_engine(train_seq, sensor_feat_cols, WINDOW_SIZE)
X_te_seq, y_te_seq, _ = make_windows_per_engine(test_seq, sensor_feat_cols, WINDOW_SIZE)
print(f"Train windows: X={X_tr_seq.shape}  Y={y_tr_seq.shape}")
print(f"Test  windows: X={X_te_seq.shape}  Y={y_te_seq.shape}")
```

```python
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMRUL(nn.Module):
    """LSTM bi-directionnel + tête de régression pour RUL."""

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=2,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, n_features)
        _, (h, _) = self.lstm(x)
        # h : (num_layers * 2, B, hidden) — concat dernières couches forward + backward
        last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(last).squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRUL(n_features=X_tr_seq.shape[-1], hidden=32, dropout=0.2).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"LSTM RUL : {n_params:,} paramètres")

loader_tr = DataLoader(WindowDataset(X_tr_seq, y_tr_seq), batch_size=64, shuffle=True)
loader_te = DataLoader(WindowDataset(X_te_seq, y_te_seq), batch_size=128, shuffle=False)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()
```

```python
losses = []
for epoch in range(15):
    model.train()
    total = 0; n = 0
    for xb, yb in loader_tr:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * xb.size(0); n += xb.size(0)
    losses.append(total / n)
    if (epoch + 1) % 3 == 0:
        print(f"Ep {epoch+1:2d}  loss = {losses[-1]:.4f}")

# Eval
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in loader_te:
        all_preds.append(model(xb.to(device)).cpu().numpy())
preds_lstm = np.concatenate(all_preds)

mae_lstm = mean_absolute_error(y_te_seq, preds_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_te_seq, preds_lstm))
r2_lstm = r2_score(y_te_seq, preds_lstm)
cmapss_lstm = cmapss_score(y_te_seq, preds_lstm)
print(f"\nLSTM RUL : MAE={mae_lstm:.2f}  RMSE={rmse_lstm:.2f}  R²={r2_lstm:.3f}  C-MAPSS={cmapss_lstm:.1f}")
```

<!-- #region -->
## 9. Visualisation des prédictions vs réel
<!-- #endregion -->

```python
# Prendre les meilleurs prédictions (par exemple LightGBM) et comparer au LSTM
best_name = bench_df.iloc[0]["model"]
best_model = models[best_name]
if best_name in ("LinearReg", "Ridge", "MLP"):
    sc = StandardScaler().fit(X_train)
    best_preds = best_model.predict(sc.transform(X_test))
else:
    best_preds = best_model.predict(X_test.values)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Plot 1 : best ML vs réel (sample 500 points)
axes[0].scatter(y_test[:500], best_preds[:500], alpha=0.3, s=10)
axes[0].plot([0, y_test.max()], [0, y_test.max()], "r--", label="perfect")
axes[0].set_xlabel("RUL réel"); axes[0].set_ylabel("RUL prédit")
axes[0].set_title(f"{best_name} — pred vs true")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Plot 2 : LSTM vs réel
axes[1].scatter(y_te_seq[:500], preds_lstm[:500], alpha=0.3, s=10, color="orange")
axes[1].plot([0, y_te_seq.max()], [0, y_te_seq.max()], "r--", label="perfect")
axes[1].set_xlabel("RUL réel"); axes[1].set_ylabel("RUL prédit")
axes[1].set_title("LSTM RUL — pred vs true")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
```

<!-- #region -->
## 10. Déploiement / Alerting
<!-- #endregion -->

<!-- #region -->
Pattern de prod 2026 :

- **Edge** (PLC, gateway IoT) : modèle léger (XGBoost quantizé, TFLite, ONNX Runtime) calcule la RUL en local et envoie uniquement les alertes.
- **Cloud** (Kafka + Spark/Flink) : streaming, fenêtrage online, batch ML training nuit. Voir notebooks `DE_Kafka_Streaming` et `MLOps_Pipelines_Airflow`.
- **Alerting** par paliers :
  - RUL < 60 cycles → email équipe maintenance.
  - RUL < 30 cycles → ticket urgent + planification intervention.
  - RUL < 10 cycles → alerte SMS + procédure de safe-shutdown.
- **Monitoring** : drift detection — la dérive du capteur peut indiquer un nouveau régime opérationnel non vu en train. Voir `MLOps_Drift_Monitoring`.
- **Retraining** : déclenché par drift + nouveaux cycles complets ajoutés à l'historique (cycle ≥ 30-50 nouveaux runs).
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [NASA C-MAPSS dataset — PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Saxena et al. (2008) — Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://ieeexplore.ieee.org/document/4711414)
- [PHM Society challenges](https://www.phmsociety.org/)
- Notebooks liés : `TS_Generer_Sequence` (prep séquences), `TS_Time_Series_Overview` (panorama TS), `DL_PyTorch` (LSTM en détail), `MLOps_Drift_Monitoring` (monitoring prod).
<!-- #endregion -->
