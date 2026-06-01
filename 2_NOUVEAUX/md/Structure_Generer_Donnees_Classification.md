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
# 🎲 Générer des données synthétiques pour ML
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** : générer des **datasets synthétiques** pour benchmarker, débugger, illustrer.

Cas d'usage typiques :

- Tester un algo sans data réelle.
- Illustrer un concept pédagogique (biais/variance, déséquilibre, outliers).
- Augmenter un dataset (data augmentation).
- Préserver la confidentialité (synthetic data anonymisée).

Couverture :

1. `sklearn.datasets.make_*` — le toolkit de référence.
2. **Distribution gaussiennes** et leur paramétrage.
3. **Datasets 2D classiques** : circles, moons, blobs.
4. **make_classification** — multiclass paramétrable.
5. **make_regression** — pour la régression.
6. **Imbalance** artificiel pour tests.
7. **Données mixtes** (num + cat) via faker / SDV.
8. **Synthetic Data Vaults (SDV)** — modèles génératifs pour cas réels.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
```

<!-- #region -->
## 1. sklearn.datasets.make_* — vue d'ensemble
<!-- #endregion -->

<!-- #region -->
sklearn fournit nativement des **générateurs de datasets** pour la classification, la régression, le clustering :

| Fonction | Utilisation |
|---|---|
| `make_classification` | Multi-classe paramétrable (n_features, n_informative, ...) |
| `make_regression` | Régression linéaire bruitée |
| `make_blobs` | Clusters gaussiens isotropes |
| `make_moons` | 2 demi-lunes (non-linéaire) |
| `make_circles` | 2 cercles concentriques (non-linéaire) |
| `make_gaussian_quantiles` | Classes par quantiles d'une gaussienne |
| `make_multilabel_classification` | Multi-label |
| `make_swiss_roll` | Variété 3D non-linéaire (pour manifold learning) |
<!-- #endregion -->

<!-- #region -->
## 2. Datasets 2D — visualiser pour comprendre
<!-- #endregion -->

```python
from sklearn.datasets import (
    make_blobs, make_moons, make_circles,
    make_gaussian_quantiles, make_classification,
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Blobs : clusters gaussiens
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=15)
axes[0, 0].set_title("make_blobs(centers=4) — clusters convexes")

# Moons : 2 demi-lunes (non-linéaire)
X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
axes[0, 1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
axes[0, 1].set_title("make_moons — non-linéaire (KNN/NN > LR)")

# Circles : cercles concentriques
X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)
axes[0, 2].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
axes[0, 2].set_title("make_circles — non-linéaire radiale")

# Gaussian quantiles
X, y = make_gaussian_quantiles(n_samples=300, n_classes=3, random_state=42)
axes[1, 0].scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=15)
axes[1, 0].set_title("make_gaussian_quantiles(n=3) — classes par quantile")

# Classification 2D (n_informative=2)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            n_clusters_per_class=2, random_state=42)
axes[1, 1].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
axes[1, 1].set_title("make_classification(n_features=2)")

# Imbalanced (1:9 ratio)
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                            weights=[0.9, 0.1], random_state=42)
axes[1, 2].scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
axes[1, 2].set_title("make_classification déséquilibré 90/10")

plt.tight_layout()
```

<!-- #region -->
## 3. make_classification — paramétrage détaillé
<!-- #endregion -->

<!-- #region -->
```python
make_classification(
    n_samples=100,         # nombre d'observations
    n_features=20,         # nombre total de features
    n_informative=2,       # features qui portent vraiment l'info
    n_redundant=2,         # combinaison linéaire des informatives
    n_repeated=0,          # duplications
    n_classes=2,           # nombre de classes
    n_clusters_per_class=2,
    weights=None,          # déséquilibre des classes : [0.9, 0.1]
    flip_y=0.01,           # % d'erreurs de label (bruit cible)
    class_sep=1.0,         # plus grand = classes plus séparables
    hypercube=True,
    shift=0.0, scale=1.0,
    shuffle=True, random_state=42,
)
```

Permet de **construire à la demande** un dataset avec les caractéristiques voulues pour tester un algo.
<!-- #endregion -->

```python
# Exemple : dataset bruité, déséquilibré, classes peu séparables
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=5, n_redundant=5,
    n_classes=3, weights=[0.7, 0.2, 0.1],
    class_sep=0.8, flip_y=0.05, random_state=42,
)
df_synth = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
df_synth["target"] = y
print(df_synth["target"].value_counts(normalize=True).round(3))
print(df_synth.head())
```

<!-- #region -->
## 4. make_regression — régression bruitée
<!-- #endregion -->

```python
from sklearn.datasets import make_regression

X, y, coef = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,         # features qui contribuent réellement
    noise=10.0,              # bruit gaussien sur la cible
    coef=True,               # renvoie les vrais coefs
    random_state=42,
)
print(f"Vrais coefs : {coef.round(2)}")  # 5 non-nuls (les informatifs), 5 nuls
```

<!-- #region -->
## 5. Imbalance contrôlée pour bench
<!-- #endregion -->

```python
def imbalance_dataset(n=1000, ratio=0.05, n_features=10, random_state=42):
    """Dataset binaire avec `ratio` positifs (1) et 1-ratio négatifs (0)."""
    X, y = make_classification(
        n_samples=n, n_features=n_features,
        n_informative=n_features // 2,
        n_classes=2, weights=[1 - ratio, ratio],
        flip_y=0.01, random_state=random_state,
    )
    return X, y


X, y = imbalance_dataset(n=2000, ratio=0.05)
print(f"Positifs : {y.sum()} / {len(y)} ({y.mean():.1%})")
```

<!-- #region -->
## 6. Données mixtes (num + cat) — Faker
<!-- #endregion -->

<!-- #region -->
Pour des données qui ressemblent à du tabulaire réel (noms, adresses, dates), utiliser **`faker`** :

```python
# pip install faker
"""
from faker import Faker
import pandas as pd

fake = Faker("fr_FR")  # locale française
fake.seed_instance(42)

df = pd.DataFrame([{
    "name": fake.name(),
    "email": fake.email(),
    "company": fake.company(),
    "address": fake.address(),
    "phone": fake.phone_number(),
    "birthdate": fake.date_of_birth(minimum_age=18, maximum_age=80),
    "job": fake.job(),
    "salary": fake.random_int(20000, 100000),
} for _ in range(100)])

print(df.head())
"""
```
<!-- #endregion -->

<!-- #region -->
## 7. SDV (Synthetic Data Vault) — modèles génératifs
<!-- #endregion -->

<!-- #region -->
Pour générer des données qui **conservent les corrélations** d'un dataset réel (utile pour partager sans révéler la data) :

- **`sdv`** (Synthetic Data Vault) — Tabular / Time series / Relational generators (CTGAN, TVAE, GaussianCopula).
- **`ydata-synthetic`** (anciennement ydata-quality) — GAN-based.
- **`gretel.ai`** — SaaS managé.

```python
"""
from sdv.tabular import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_df)
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_df)
synth_df = synthesizer.sample(num_rows=1000)
"""
```

Important : **toujours évaluer** la similarité statistique + la confidentialité (membership inference attack).
<!-- #endregion -->

<!-- #region -->
## 8. Création de DataFrame depuis JSON ou structure custom
<!-- #endregion -->

```python
import json

# JSON nested → DataFrame
json_str = """
[
    {"id": 1, "name": "Alice", "scores": {"math": 90, "english": 85}},
    {"id": 2, "name": "Bob", "scores": {"math": 75, "english": 92}}
]
"""
data = json.loads(json_str)
df = pd.json_normalize(data, sep="_")
print(df)

# JSON Lines (1 obj par ligne — format streaming)
# df = pd.read_json("file.jsonl", lines=True)
```

<!-- #region -->
## 9. Quand utiliser quoi
<!-- #endregion -->

<!-- #region -->
| Besoin | Outil |
|---|---|
| Tester un algo, illustrer un concept 2D | `make_blobs/moons/circles` |
| Bench classifieur avec contrôle des params | `make_classification` |
| Bench régression avec ground truth | `make_regression` |
| Tester gestion de déséquilibre | `make_classification(weights=...)` |
| Simuler structure tabulaire métier (PII) | `faker` |
| Augmenter / partager data sans révéler | `SDV` (GaussianCopula, CTGAN) |
| Stress test pipeline data | combine `faker` + boucles |
| Time series synthétiques | voir `TS_Generer_Sequence` |
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [sklearn — Dataset generators](https://scikit-learn.org/stable/datasets/sample_generators.html)
- [Faker — docs](https://faker.readthedocs.io/)
- [SDV — Synthetic Data Vault](https://sdv.dev/)
- Notebooks liés : `TS_Generer_Sequence`, `NLP_Classification_Smote`, `ML_Regression_Classification_Multiple`.
<!-- #endregion -->
