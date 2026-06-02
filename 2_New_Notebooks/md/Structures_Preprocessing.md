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
# 🧹 Preprocessing — Cheat-sheet + Tutoriel
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** sur le **preprocessing** des données pour ML. Concentre les recettes utiles avec **sklearn Pipeline** comme fil rouge.

> Ce notebook **absorbe** l'ancien `Structures_Preprocessing_Function_Utiles` (utilitaires micro qui n'avaient pas de raison d'être seuls).

Couverture :

1. **Vue d'ensemble** : Pipeline + ColumnTransformer (pattern).
2. **Missing data** : imputation (voir aussi `Test_donnees_manquante_modeles`).
3. **Encoding catégoriel** : OneHot, Ordinal, Target, embeddings.
4. **Scaling** : Standard, MinMax, Robust, Power transforms.
5. **Discrétisation** (KBinsDiscretizer).
6. **Interactions / Polynomiales**.
7. **Réduction de dim** : PCA, LOF outliers, SelectKBest.
8. **Utilitaires Python génériques** (anciens Function_Utiles).
9. **Sauvegarde** des transformations.

Dataset : **Titanic** (mutualisé).
<!-- #endregion -->

<!-- #region -->
## 1. Le pattern fondamental — Pipeline + ColumnTransformer
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

titanic = sns.load_dataset("titanic").dropna(subset=["embarked"])
X = titanic[["age", "fare", "sibsp", "parch", "pclass", "sex", "embarked"]]
y = titanic["survived"]

num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "embarked"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ]), cat_cols),
])

pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(X, y)
print(f"Score : {pipe.score(X, y):.4f}")
```

<!-- #region -->
## 2. Missing data (rappel)
<!-- #endregion -->

<!-- #region -->
| Stratégie | Quand |
|---|---|
| `SimpleImputer(strategy="median")` | Numérique robuste |
| `SimpleImputer(strategy="most_frequent")` | Catégoriel |
| `SimpleImputer(strategy="constant", fill_value=-1)` | Marquer comme "spécial" |
| `KNNImputer(n_neighbors=5)` | Capture corrélations |
| `IterativeImputer` (MICE) | Sophistiqué, lent |
| **Ne pas imputer + XGBoost** | Souvent meilleur, voir `Test_donnees_manquante_modeles` |

Toujours ajouter un **indicateur de manquance** quand la manquance est informative.
<!-- #endregion -->

<!-- #region -->
## 3. Encoding catégoriel
<!-- #endregion -->

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

# OneHot — standard pour modèles linéaires / NN
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
ohe.fit(X[cat_cols])
print(ohe.get_feature_names_out())

# Ordinal — pour arbres, ou catégories ordonnées
oe = OrdinalEncoder(categories=[[3,2,1], ["male","female"], ["S","C","Q"]])
oe.fit(X[cat_cols])

# Target Encoder (sklearn 1.3+) — pour haute cardinalité
# encode chaque cat par la moyenne du target sur ce groupe (avec CV pour anti-leak)
te = TargetEncoder(cv=5, random_state=42, smooth="auto")
X_te = te.fit_transform(X[cat_cols], y)
print(f"TargetEncoder shape : {X_te.shape}")
```

<!-- #region -->
## 4. Scaling
<!-- #endregion -->

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# Comparaison
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

age = X["age"].dropna().to_numpy().reshape(-1, 1)

scalers = {
    "raw": age.flatten(),
    "Standard (mean=0, std=1)": StandardScaler().fit_transform(age).flatten(),
    "MinMax [0,1]": MinMaxScaler().fit_transform(age).flatten(),
    "Robust (median/IQR)": RobustScaler().fit_transform(age).flatten(),
    "Power Yeo-Johnson (~gaussien)": PowerTransformer(method="yeo-johnson").fit_transform(age).flatten(),
}

fig, axes = plt.subplots(1, len(scalers), figsize=(20, 3), sharey=False)
for ax, (name, vals) in zip(axes, scalers.items()):
    ax.hist(vals, bins=30, color="steelblue")
    ax.set_title(name, fontsize=10)
plt.tight_layout()
```

<!-- #region -->
**Choix** :

- `StandardScaler` : standard pour LogReg, SVM, NN, KNN. Suppose distribution ~gaussienne.
- `MinMaxScaler` : si on veut bornes [0,1] (images, NN avec sigmoid).
- `RobustScaler` : robuste aux outliers (utilise median + IQR).
- `PowerTransformer` : si distribution très skewed (sale gauche/droite), force vers gaussienne. Yeo-Johnson gère les valeurs négatives (Box-Cox demande positif).
- **Pas besoin** pour les arbres / GBM (XGBoost / LightGBM / CatBoost).
<!-- #endregion -->

<!-- #region -->
## 5. Discrétisation
<!-- #endregion -->

```python
from sklearn.preprocessing import KBinsDiscretizer

# Transforme une variable continue en bins (utile pour exposer la non-linéarité à un modèle linéaire)
disc = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
age_binned = disc.fit_transform(age)
print(f"Bins : {sorted(set(age_binned.flatten().astype(int)))}")
```

<!-- #region -->
## 6. Interactions et polynomiales
<!-- #endregion -->

```python
from sklearn.preprocessing import PolynomialFeatures

# x1, x2 → 1, x1, x2, x1², x1·x2, x2²
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
X_poly = poly.fit_transform(np.array([[1, 2], [3, 4]]))
print(f"Avant : (2, 2)  Après poly d=2 : {X_poly.shape}")
print(f"Noms : {poly.get_feature_names_out()}")
```

<!-- #region -->
## 7. PCA & SelectKBest dans la pipeline
<!-- #endregion -->

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Pipeline complète avec PCA + sélection
pipe_full = Pipeline([
    ("prep", preprocessor),
    ("pca", PCA(n_components=5)),
    ("sel", SelectKBest(score_func=mutual_info_classif, k=3)),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe_full.fit(X, y)
print(f"Score pipeline complète : {pipe_full.score(X, y):.4f}")
```

<!-- #region -->
## 8. Utilitaires Python génériques (ex-Function_Utiles)
<!-- #endregion -->

<!-- #region -->
Petites fonctions utiles régulièrement en preprocessing data :
<!-- #endregion -->

```python
def variable_name_to_string(var) -> str:
    """Récupère le nom d'une variable (utile pour print debugging)."""
    # Hack via globals
    for name, value in globals().items():
        if value is var:
            return name
    return "unknown"


def find_in_nested(d: dict, key_path: list[str], default=None):
    """Recherche dans un dict de dicts via une suite de clés."""
    for k in key_path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def flatten_list(nested: list) -> list:
    """Aplatit une liste de listes (1 niveau)."""
    return [item for sublist in nested for item in sublist]


def safe_int(val, default: int = 0) -> int:
    """Conversion int qui ne crash pas."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# Tests
nested_dict = {"a": {"b": {"c": 42}}}
print(find_in_nested(nested_dict, ["a", "b", "c"]))      # 42
print(find_in_nested(nested_dict, ["a", "z"], default="N/A"))  # N/A
print(flatten_list([[1,2], [3], [4,5,6]]))                # [1,2,3,4,5,6]
print(safe_int("123"), safe_int("abc", default=-1))       # 123, -1
```

<!-- #region -->
## 9. Sauvegarde des transformations
<!-- #endregion -->

```python
import joblib
import tempfile
from pathlib import Path

# Pickle le pipeline entier (transformer + classifier)
ckpt = Path(tempfile.gettempdir()) / "demo_pipeline.joblib"
joblib.dump(pipe, ckpt)

# Recharge et utilise
loaded = joblib.load(ckpt)
print(f"Score model rechargé : {loaded.score(X, y):.4f}")
```

<!-- #region -->
**Important** : la version de sklearn/python utilisée pour `save` doit matcher au `load`. Pour la prod, préférer **ONNX** ou **mlflow models** qui découplent.
<!-- #endregion -->

<!-- #region -->
## 10. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
- **Toujours encapsuler** dans Pipeline (jamais de `fit_transform` séparé sur train/test).
- **ColumnTransformer** dès qu'on a des features mixtes.
- **Pas de scaling pour les arbres** — gain nul, complexité ajoutée.
- **Target encoding** : utiliser la version sklearn 1.3+ avec `cv` pour éviter le leak.
- **Imputer dans la pipeline** : `fit_transform` train, `transform` test (sinon stats du test fuitent).
- Pour des pipelines complexes : **`sklearn.set_config(transform_output="pandas")`** pour conserver les noms de colonnes à la sortie.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [sklearn — Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [sklearn — Compose](https://scikit-learn.org/stable/modules/compose.html)
- [Effective ML for Tabular Data — Coqueret et al.](https://link.springer.com/book/10.1007/978-3-031-04123-9)
- Notebooks liés : `Structures_DataFrame`, `Test_donnees_manquante_modeles`, `ML_Regression_Classification_CV_GridSearch`.
<!-- #endregion -->
