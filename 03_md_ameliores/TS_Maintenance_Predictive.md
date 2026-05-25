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
# 🔧 Maintenance prédictive — Case study
<!-- #endregion -->

<!-- #region -->
Notebook **Case Study** complet sur la **maintenance prédictive** : prédire l'instant ou la probabilité de défaillance d'un équipement (turbine, pompe, moteur) à partir de séries temporelles de capteurs.

Pipeline complet :

1. Cadrage — RUL (Remaining Useful Life) vs détection d'anomalie.
2. **EDA** sur signaux multi-capteurs.
3. **Preprocessing** : imputation, outliers, normalisation, feature engineering.
4. **PCA** pour réduction de dim (3 stratégies : par cycle, global, par capteur).
5. **Modèles** : régression RUL (XGBoost, LSTM), classification time-to-failure.
6. **Évaluation** spécifique : score asymétrique (sous-estimer la défaillance coûte plus cher).
7. **Déploiement** edge / cloud, alerting.

Dataset : **NASA Turbofan C-MAPSS** (jeu de données de référence pour la maintenance prédictive). Download via `data/_shared/turbofan/README.md` (cf `00_datasets.md`). Pour cette démo, on utilise un dataset **synthétique** qui mime la structure.
<!-- #endregion -->

<!-- #region -->
## 1. Cadrer la tâche
<!-- #endregion -->

<!-- #region -->
Trois formulations principales :

| Formulation | Cible | Avantage | Inconvénient |
|---|---|---|---|
| **RUL** (Remaining Useful Life) | `y = cycles restants avant défaillance` (régression) | Action graduée, priorisation | Demande historique de défaillances |
| **Classification time-to-failure** | `y ∈ {sain, défaillance dans N jours}` | Plus simple, alerte binaire | Choix arbitraire du seuil N |
| **Anomaly detection** | `y = score d'anomalie` (non supervisé) | Pas besoin de défaillances historiques | Faux positifs fréquents |

**Choix 2026** : si tu as au moins **30-50 cycles complets de défaillance**, **RUL** + XGBoost ou LSTM est l'approche reine. Sinon → anomaly detection (isolation forest, autoencoder) sur l'état normal.
<!-- #endregion -->

<!-- #region -->
## 2. Données synthétiques (proxy NASA Turbofan)
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)

def generate_engine_run(engine_id: int, n_cycles: int) -> pd.DataFrame:
    """Génère le run d'un moteur : 3 capteurs qui dégradent + bruit, du cycle 0 à la défaillance."""
    t = np.arange(n_cycles)
    # Dégradation accélérée vers la fin (signature classique)
    degradation = (t / n_cycles) ** 2
    s1 = 100 + 50 * degradation + rng.normal(0, 5, n_cycles)        # ex: pression
    s2 = 800 + 30 * np.sin(t / 5) - 80 * degradation + rng.normal(0, 8, n_cycles)  # ex: vibration
    s3 = 20 + 5 * np.cos(t / 7) + 15 * degradation + rng.normal(0, 1, n_cycles)    # ex: température
    rul = n_cycles - 1 - t
    return pd.DataFrame({
        "engine_id": engine_id, "cycle": t,
        "sensor_1": s1, "sensor_2": s2, "sensor_3": s3,
        "rul": rul,
    })


engines = pd.concat([generate_engine_run(i, n_cycles=rng.randint(120, 250)) for i in range(20)])
print(engines.head())
print(f"\nN engines : {engines['engine_id'].nunique()}, total cycles : {len(engines)}")
```

<!-- #region -->
## 3. EDA — capteurs vs RUL
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, sensor in zip(axes, ["sensor_1", "sensor_2", "sensor_3"]):
    for eng_id in engines["engine_id"].unique()[:5]:
        g = engines[engines["engine_id"] == eng_id]
        ax.plot(g["cycle"], g[sensor], alpha=0.5)
    ax.set_title(f"{sensor} (5 moteurs)")
    ax.set_xlabel("cycle")
    ax.grid(alpha=0.3)
plt.tight_layout()
```

<!-- #region -->
**Constat** :

- Les capteurs montrent une **dégradation systématique** vers la fin (cycles élevés).
- Du bruit qui masque le signal sur les cycles intermédiaires.
- Les amplitudes/échelles diffèrent → **normalisation obligatoire**.
<!-- #endregion -->

<!-- #region -->
## 4. Feature engineering
<!-- #endregion -->

<!-- #region -->
Au lieu d'utiliser les capteurs bruts, on calcule des **agrégations sur fenêtre glissante** par moteur :

- moyenne, écart-type, min, max, range sur les `W` derniers cycles
- pente locale (régression linéaire sur la fenêtre)
- nombre de dépassements de seuil
<!-- #endregion -->

```python
def add_rolling_features(df: pd.DataFrame, sensor_cols: list[str], window: int = 10) -> pd.DataFrame:
    """Ajoute des features rolling par moteur."""
    out = df.copy().sort_values(["engine_id", "cycle"])
    for col in sensor_cols:
        g = out.groupby("engine_id")[col]
        out[f"{col}_mean_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).mean())
        out[f"{col}_std_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0)
        out[f"{col}_min_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).min())
        out[f"{col}_max_{window}"] = g.transform(lambda s: s.rolling(window, min_periods=1).max())
    return out


engines_feat = add_rolling_features(engines, ["sensor_1", "sensor_2", "sensor_3"], window=10)
print(engines_feat.columns.tolist())
print(engines_feat.head())
```

<!-- #region -->
## 5. PCA — 3 stratégies à connaître
<!-- #endregion -->

<!-- #region -->
Sur des datasets avec **beaucoup de capteurs** (NASA C-MAPSS = 21 capteurs), la PCA aide à :

- Réduire la dimensionnalité pour un modèle simple.
- Détecter visuellement les phases (sain vs dégradation).
- Identifier les capteurs porteurs d'information vs constants.

**3 stratégies différentes** :

| Stratégie | Description | Quand |
|---|---|---|
| **PCA globale** | Fit PCA sur tous les cycles de tous moteurs | Vue d'ensemble, détection de classes |
| **PCA par moteur** | Fit PCA par moteur | Si forte hétérogénéité entre unités |
| **PCA par phase** | Fit PCA séparément sur "sain" vs "dégradation" | Pour caractériser chaque phase |
<!-- #endregion -->

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sensor_cols = ["sensor_1", "sensor_2", "sensor_3"]
scaler = StandardScaler()
X = scaler.fit_transform(engines[sensor_cols])
pca = PCA(n_components=2)
proj = pca.fit_transform(X)

plt.figure(figsize=(8, 5))
sc = plt.scatter(proj[:, 0], proj[:, 1], c=engines["rul"], cmap="viridis_r", s=5, alpha=0.5)
plt.colorbar(sc, label="RUL (bleu=fin de vie)")
plt.title("PCA globale colorée par RUL")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.tight_layout()
```

<!-- #region -->
## 6. Modèle RUL — XGBoost sur features rolling
<!-- #endregion -->

```python
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Train/test split PAR MOTEUR (jamais splitter au milieu d'un cycle)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(gss.split(engines_feat, groups=engines_feat["engine_id"]))
train, test = engines_feat.iloc[tr_idx], engines_feat.iloc[te_idx]

feature_cols = [c for c in engines_feat.columns if c not in ["engine_id", "cycle", "rul"]]
X_train, y_train = train[feature_cols], train["rul"]
X_test, y_test = test[feature_cols], test["rul"]

model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print(f"MAE RUL : {mae:.2f} cycles")
```

<!-- #region -->
## 7. Score asymétrique — coût business
<!-- #endregion -->

<!-- #region -->
En maintenance prédictive, **sous-estimer la RUL** (prédire fin de vie trop tôt) coûte une maintenance prématurée (coût visible). **Surestimer** (prédire trop tard) coûte une panne en service (coût catastrophique).

Le score officiel C-MAPSS pénalise les sous-estimations moins que les surestimations :

$$
S = \sum_i \begin{cases} e^{-d_i / 13} - 1 & \text{si } d_i = \hat{y}_i - y_i < 0 \text{ (early)} \\ e^{d_i / 10} - 1 & \text{si } d_i \geq 0 \text{ (late)} \end{cases}
$$

On veut **minimiser** ce score (asymétrique, exponentiel).
<!-- #endregion -->

```python
def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score asymétrique du challenge NASA C-MAPSS. Plus bas = mieux."""
    d = y_pred - y_true
    early_penalty = np.exp(-d[d < 0] / 13) - 1
    late_penalty = np.exp(d[d >= 0] / 10) - 1
    return float(early_penalty.sum() + late_penalty.sum())


print(f"C-MAPSS score : {cmapss_score(y_test.values, preds):.1f}")
```

<!-- #region -->
## 8. Variante DL — LSTM sur séquences
<!-- #endregion -->

<!-- #region -->
Plutôt qu'agréger en features rolling, on peut donner directement les **séquences** (cycles passés) à un LSTM. Voir `TS_Generer_Sequence` pour la prep, et le notebook DL framework de ton choix (`DL_PyTorch`, `DL_TensorFlow`, ...) pour l'architecture.

Pseudo-archi :

```python
# class LSTMRUL(nn.Module):
#     def __init__(self, n_features, hidden=64):
#         super().__init__()
#         self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=2, dropout=0.2)
#         self.head = nn.Linear(hidden, 1)
#     def forward(self, x):  # x : (B, T, n_features)
#         _, (h, _) = self.lstm(x)
#         return self.head(h[-1]).squeeze(-1)
```
<!-- #endregion -->

<!-- #region -->
## 9. Déploiement / Alerting
<!-- #endregion -->

<!-- #region -->
Pattern de prod 2026 :

- **Edge** (PLC, gateway IoT) : modèle léger (XGBoost quantizé, TFLite) qui calcule la RUL en local et envoie uniquement les alertes.
- **Cloud** (Kafka + Spark/Flink) : streaming, fenêtrage online, batch ML training nuit.
- **Alerting** : seuil sur RUL (< 50 cycles = avertissement, < 20 cycles = critique).
- **Monitoring** : drift detection — la dérive du capteur peut indiquer un nouveau régime opérationnel non vu en train.
- **Retraining** : déclenché par drift + nouveaux cycles complets ajoutés à l'historique.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [NASA C-MAPSS dataset — PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Saxena et al. (2008) — Damage Propagation Modeling for Aircraft Engine](https://ieeexplore.ieee.org/document/4711414)
- [PHM Society challenges](https://www.phmsociety.org/)
- Notebooks liés : `TS_Generer_Sequence` (prep LSTM), `TS_Time_Series_Overview` (panorama).
<!-- #endregion -->
