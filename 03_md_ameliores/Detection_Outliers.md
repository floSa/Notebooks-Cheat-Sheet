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
# 🚨 Détection d'outliers
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur la **détection d'outliers / anomalies**.

Couvre les **familles principales** avec un decision tree clair :

1. **Statistiques univariées** — Z-score, IQR, MAD.
2. **Modèles de densité** — Elliptic Envelope, GMM.
3. **Modèles de distance** — LOF (Local Outlier Factor).
4. **Modèles d'ensemble / forest** — Isolation Forest, Extended Isolation Forest.
5. **Modèles à noyaux** — One-Class SVM.
6. **Deep Learning** — autoencoders, autoencoder variationnel.
7. **Pour séries temporelles** — STUMPY (matrix profile), River (streaming).
8. **Decision tree** pour choisir.

> Pour la détection d'anomalies **dans des séries temporelles** (capteurs IoT, time series industrielles), voir aussi `TS_Maintenance_Predictive`.

Dataset : synthétique avec outliers injectés.
<!-- #endregion -->

<!-- #region -->
## 1. Cadrer : 3 contextes différents
<!-- #endregion -->

<!-- #region -->
| Contexte | Approche reco |
|---|---|
| **Cleaning de données** (drop avant modèle) | IQR / Z-score / Isolation Forest |
| **Anomaly detection métier** (fraud, intrusion, défaut machine) | Isolation Forest, autoencoder, ou supervisé si labels |
| **Novelty detection** (détecter ce qui ne ressemble pas au train normal) | One-Class SVM, autoencoder |

**Distinction clé** :

- **Outlier detection** : il y a des anomalies dans le train, on veut les identifier (non supervisé sur data contaminée).
- **Novelty detection** : le train est "propre", on détecte les nouvelles données qui s'en éloignent.
<!-- #endregion -->

<!-- #region -->
## 2. Statistiques univariées
<!-- #endregion -->

<!-- #region -->
### 2.1 Z-score (suppose normalité)
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

rng = np.random.RandomState(42)
# 200 points normaux + 10 outliers
x = np.concatenate([rng.normal(0, 1, 200), rng.uniform(5, 8, 10)])

# Z-score : |z| > 3 = outlier (règle empirique pour normale)
z = (x - x.mean()) / x.std()
outliers_z = np.abs(z) > 3
print(f"Z-score : {outliers_z.sum()} outliers détectés")
```

<!-- #region -->
### 2.2 IQR — Inter-Quartile Range (robuste)
<!-- #endregion -->

<!-- #region -->
Plus robuste que le Z-score (qui est sensible aux outliers eux-mêmes via la moyenne et l'écart-type).

```
Q1, Q3 = quantiles(x, [0.25, 0.75])
IQR = Q3 - Q1
outlier ⇔ x < Q1 - 1.5·IQR  OR  x > Q3 + 1.5·IQR
```

Le facteur **1.5** est la convention (Tukey, 1977). On peut le durcir (3) pour ne flagger que les "far outliers".
<!-- #endregion -->

```python
q1, q3 = np.percentile(x, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers_iqr = (x < lower) | (x > upper)
print(f"IQR : {outliers_iqr.sum()} outliers détectés (borne [{lower:.2f}, {upper:.2f}])")
```

<!-- #region -->
### 2.3 MAD — Median Absolute Deviation (ultra robuste)
<!-- #endregion -->

```python
median = np.median(x)
mad = np.median(np.abs(x - median))
# Constante 1.4826 fait du MAD un estimateur cohérent de sigma pour gaussien
modified_z = 0.6745 * (x - median) / mad  # ≈ |z| > 3.5 = outlier
outliers_mad = np.abs(modified_z) > 3.5
print(f"MAD : {outliers_mad.sum()} outliers détectés")
```

<!-- #region -->
## 3. Modèles de densité — Elliptic Envelope
<!-- #endregion -->

<!-- #region -->
**Idée** : ajuste une **Gaussienne robuste** (Minimum Covariance Determinant) sur la data. Les points hors d'une ellipse de confiance sont des outliers.

**Suppose une distribution gaussienne multivariée**. Marche bien sur des features quasi-gaussiennes.
<!-- #endregion -->

```python
from sklearn.covariance import EllipticEnvelope

# 2D pour la viz
X2d = np.column_stack([
    np.concatenate([rng.normal(0, 1, 200), rng.uniform(3, 6, 10)]),
    np.concatenate([rng.normal(0, 1, 200), rng.uniform(3, 6, 10)]),
])
true_labels = np.concatenate([np.zeros(200), np.ones(10)])

ee = EllipticEnvelope(contamination=0.05, random_state=42)
pred_ee = ee.fit_predict(X2d)  # -1 = outlier, +1 = inlier
print(f"EllipticEnvelope : {(pred_ee == -1).sum()} flagged")
```

<!-- #region -->
## 4. LOF — Local Outlier Factor
<!-- #endregion -->

<!-- #region -->
**Idée** : compare la densité locale d'un point à celle de ses **k plus proches voisins**. Si elle est nettement plus basse → outlier.

**Force** : marche sur des datasets avec **plusieurs clusters de densités différentes** (un point peut être "normal" dans un cluster sparse mais "outlier" dans un cluster dense).

**Faiblesse** : ne se généralise pas à de nouvelles données par défaut. Utiliser `novelty=True` si besoin.
<!-- #endregion -->

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
pred_lof = lof.fit_predict(X2d)
print(f"LOF : {(pred_lof == -1).sum()} flagged")
```

<!-- #region -->
## 5. Isolation Forest — le go-to 2026
<!-- #endregion -->

<!-- #region -->
**Idée** : **arbres aléatoires** qui isolent un point en splittant aléatoirement. Les outliers sont **isolés en peu de splits** (chemin court depuis la racine).

**Forces** :

- Pas d'hypothèse sur la distribution.
- Scalable à grand `N` et grand `d` (O(N log N)).
- Marche bien high-dim (sans malédiction de la dim qui plombe LOF).
- Score d'anomalie interprétable (longueur de chemin moyen).
- Supporte le **novelty detection** (`fit` sur normal puis `predict` sur new).

C'est le **baseline universel 2026** pour la détection d'anomalies tabulaires.
<!-- #endregion -->

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
pred_iso = iso.fit_predict(X2d)
scores_iso = iso.decision_function(X2d)  # plus c'est bas, plus c'est anomal
print(f"Isolation Forest : {(pred_iso == -1).sum()} flagged")
```

<!-- #region -->
### 5.1 Extended Isolation Forest (EIF)
<!-- #endregion -->

<!-- #region -->
Variante (Hariri 2019) qui utilise des **splits obliques** au lieu d'axis-aligned. Évite les artefacts du IF classique sur les bordures des clusters. Disponible via `pip install eif` ou implémenté dans `pyod`.
<!-- #endregion -->

<!-- #region -->
## 6. One-Class SVM
<!-- #endregion -->

<!-- #region -->
**Idée** : SVM qui apprend une **frontière englobant la majorité des données** (noyau RBF souvent). Les points dehors = anomalies.

**Forces** : bon en novelty detection sur des données complexes / non-linéaires.

**Faiblesses** : **lent** sur grand `N` (O(N²) à entraîner). Sensible aux hyperparamètres (`nu`, `gamma`).
<!-- #endregion -->

```python
from sklearn.svm import OneClassSVM

ocs = OneClassSVM(nu=0.05, gamma="scale", kernel="rbf")
pred_ocs = ocs.fit_predict(X2d)
print(f"One-Class SVM : {(pred_ocs == -1).sum()} flagged")
```

<!-- #region -->
## 7. Deep Learning — Autoencoders
<!-- #endregion -->

<!-- #region -->
**Idée** : un **autoencoder** apprend à reconstruire les données normales. Les outliers ont une **erreur de reconstruction élevée**.

```python
# Pseudo-code
# import torch.nn as nn
# class AE(nn.Module):
#     def __init__(self, d_in, d_latent=4):
#         super().__init__()
#         self.enc = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, d_latent))
#         self.dec = nn.Sequential(nn.Linear(d_latent, 16), nn.ReLU(), nn.Linear(16, d_in))
#     def forward(self, x): return self.dec(self.enc(x))
#
# # train sur data normale, MSE loss
# # à l'inference : score = ||x - AE(x)||² ; seuil sur ce score
```

**Quand le faire** : data complexe (images, signaux), beaucoup de data normale, capacité à entraîner un DL.

**Variantes 2026** :

- **VAE** (Variational AE) : score probabiliste.
- **AE convolutional** : pour images.
- **Diffusion-based anomaly detection** (état de l'art 2024-2026 sur l'image).
<!-- #endregion -->

<!-- #region -->
## 8. Pour séries temporelles
<!-- #endregion -->

<!-- #region -->
- **STUMPY** — matrix profile, détecte anomalies et motifs récurrents.
- **River** — anomaly detection en **streaming** (HalfSpaceTrees, OneClassSVM online).
- **Merlion** (Salesforce) — librairie complète anomaly detection TS.
- **Darts** — `anomaly_detectors` (autoencoder, threshold sur forecast residuals).

Approche classique : entraîner un modèle de forecasting, calculer les **résidus**, flagger ceux > seuil (souvent défini par bootstrap des erreurs sur le train).
<!-- #endregion -->

<!-- #region -->
## 9. Decision tree — choisir la méthode
<!-- #endregion -->

<!-- #region -->
```
Est-ce 1 seule variable ?
├─ Oui → IQR (robuste) ou MAD (très robuste)
└─ Non — multivariée
    │
    Donnes-tu un label ? (sup vs non-sup)
    ├─ Oui → ML supervisé classique (RF, XGB, ...) sur la classification anomalie/normal
    └─ Non
        │
        Distribution ~ gaussienne ?
        ├─ Oui → Elliptic Envelope (rapide, simple)
        └─ Non
            │
            Densité des clusters varie ?
            ├─ Oui → LOF (capture la densité locale)
            └─ Non
                │
                Beaucoup de data / high-dim ?
                ├─ Oui → Isolation Forest (le go-to)
                └─ Non — non-linéarité complexe ?
                    ├─ Oui → One-Class SVM ou Autoencoder
                    └─ Non → IsoForest reste le meilleur compromis
```

**Conseil 2026** : **Isolation Forest** comme baseline systématique. Si insuffisant : autoencoder. Si série temporelle : matrix profile / Merlion.
<!-- #endregion -->

<!-- #region -->
## 10. Évaluation et seuil
<!-- #endregion -->

<!-- #region -->
- **Si labels dispo** : Precision, Recall, F1 sur les outliers ; PR-AUC est plus informative que ROC-AUC en cas d'extrême déséquilibre.
- **Si pas de labels** : utiliser un **stress test métier** — montrer les top-N anomalies à un expert qui valide.
- **Seuil** : `contamination` dans sklearn fixe la proportion d'outliers attendue. Réglage par cross-val si labels.
- **Toujours examiner visuellement** les top anomalies — souvent on découvre que certaines sont en fait des erreurs de saisie, ce qui est utile.
<!-- #endregion -->

<!-- #region -->
## 11. Lib panoramique : PyOD
<!-- #endregion -->

<!-- #region -->
**PyOD** (Python Outlier Detection) — implémente **40+ algos** d'anomaly detection avec une API unifiée (`fit / predict / decision_function`). Inclut deep (AutoEncoder, VAE, GAN), classique (KNN, LOF, ABOD, CBLOF, ...), et ensembles.

```python
# from pyod.models.iforest import IForest
# from pyod.models.lof import LOF
# from pyod.models.autoencoder import AutoEncoder
# model = IForest(contamination=0.05).fit(X)
# scores = model.decision_function(X)
```
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [sklearn — Outlier and Novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [PyOD — Python Outlier Detection](https://pyod.readthedocs.io/)
- [STUMPY — matrix profile](https://stumpy.readthedocs.io/)
- [Merlion — Salesforce TS anomaly detection](https://github.com/salesforce/Merlion)
- Hariri et al. (2019) — Extended Isolation Forest
- Notebooks liés : `EDA_Stats_Analyse_Desc_Visual`, `TS_Maintenance_Predictive`.
<!-- #endregion -->
