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
# ❓ Données manquantes — Stratégies et impact
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Expérience comparée** sur la **gestion des données manquantes** (NaN) en ML : panorama des stratégies, bench expérimental de l'impact sur la perf du modèle.

Couvre :

1. **Typologie** : MCAR / MAR / MNAR — comprendre **pourquoi** les NaN sont là.
2. **Approches simples** : drop, mean/median, mode, forward/backward fill.
3. **Approches modèle** : KNNImputer, IterativeImputer (MICE), missForest.
4. **Approches "ignore"** : XGBoost / LightGBM / CatBoost gèrent nativement les NaN.
5. **Indicateur de manquance** comme feature.
6. **Bench expérimental** sur dataset Titanic.
7. **Bonnes pratiques** 2026.

Dataset : **Titanic** (qui contient naturellement beaucoup de NaN dans `age`, `cabin`, `deck`).
<!-- #endregion -->

<!-- #region -->
## 1. Typologie des manquances
<!-- #endregion -->

<!-- #region -->
| Type | Définition | Exemple |
|---|---|---|
| **MCAR** (Missing Completely At Random) | Indépendant de la valeur de la donnée et des autres variables | Erreur transmission aléatoire |
| **MAR** (Missing At Random) | Dépend d'autres variables observées | Hommes plus enclins à ne pas répondre à une enquête santé |
| **MNAR** (Missing Not At Random) | Dépend de la valeur manquante elle-même | Salaires très élevés non renseignés (biais social) |

**Importance** : la stratégie d'imputation a un sens statistique **seulement si MAR ou MCAR**. En MNAR, toute imputation introduit un biais. Solution = modélisation conjointe ou collecte additionnelle.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

df = sns.load_dataset("titanic")
print(f"Shape : {df.shape}")
print("\nMissing par colonne :")
print(df.isna().sum().sort_values(ascending=False).head(7))
```

```python
# Heatmap des NaN — visualiser les patterns
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isna().T, cbar=False, cmap="binary", ax=ax)
ax.set_title("Pattern des NaN (Titanic)")
plt.tight_layout()
```

<!-- #region -->
## 2. Approches simples
<!-- #endregion -->

<!-- #region -->
### 2.1 Drop
<!-- #endregion -->

<!-- #region -->
- `dropna()` au niveau **observation** : on perd des lignes. Si NaN concentré sur une colonne très utilisée → catastrophique.
- `dropna(axis=1)` au niveau **colonne** : on perd la variable. Acceptable si > 70 % NaN et faible info.

**À utiliser** : pour les colonnes avec > 80 % de NaN sans info récupérable (ex: `deck` dans Titanic).
<!-- #endregion -->

```python
print(f"Avant : {len(df)} lignes")
print(f"Drop lignes ANY NaN : {len(df.dropna())} (perte massive)")
print(f"Drop col deck (77% NaN) : {df.drop(columns=['deck']).shape}")
```

<!-- #region -->
### 2.2 Constants : mean / median / mode
<!-- #endregion -->

<!-- #region -->
- **Median** pour numérique → robuste aux outliers.
- **Mean** si distribution normale.
- **Mode** pour catégorielle.
- **Valeur constante** (`-1`, `"Unknown"`) → permet au modèle d'apprendre "c'est manquant" comme info.

**Limite** : sous-estime la variance, ignore les corrélations.
<!-- #endregion -->

```python
from sklearn.impute import SimpleImputer

# Numérique : médiane
imp_num = SimpleImputer(strategy="median")
df_imputed = df.copy()
df_imputed[["age"]] = imp_num.fit_transform(df[["age"]])
print(f"Median imputed age : {df_imputed['age'].mean():.2f} (vs raw {df['age'].dropna().mean():.2f})")

# Catégorielle : mode
imp_cat = SimpleImputer(strategy="most_frequent")
df_imputed[["embarked"]] = imp_cat.fit_transform(df[["embarked"]])
print(f"Mode imputed embarked : {df_imputed['embarked'].value_counts().to_dict()}")
```

<!-- #region -->
## 3. Approches modèle
<!-- #endregion -->

<!-- #region -->
### 3.1 KNNImputer
<!-- #endregion -->

<!-- #region -->
**Idée** : pour chaque ligne avec NaN, trouver les `k` voisins (sur les features non manquantes), imputer par moyenne pondérée.

**Forces** : capture les corrélations entre variables.
**Faiblesses** : sensible aux échelles (toujours standardiser avant), lent sur grand N.
<!-- #endregion -->

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

num_cols = ["age", "fare", "pclass", "sibsp", "parch"]
X = df[num_cols].copy()
X_scaled = StandardScaler().fit_transform(X.fillna(X.median()))  # init pour scaler

# Imputer sur les vraies données
knn_imp = KNNImputer(n_neighbors=5)
X_knn = knn_imp.fit_transform(X)
print(f"KNN imputed age mean : {pd.DataFrame(X_knn, columns=num_cols)['age'].mean():.2f}")
```

<!-- #region -->
### 3.2 IterativeImputer (MICE)
<!-- #endregion -->

<!-- #region -->
**MICE** (Multivariate Imputation by Chained Equations) : pour chaque variable avec NaN, entraîne un **modèle** (par défaut BayesianRidge) qui la prédit à partir des autres. Itère plusieurs fois jusqu'à convergence.

**Forces** : flexible (n'importe quel estimateur), gère les types mixtes.
**Faiblesses** : lent, peut diverger.
<!-- #endregion -->

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

mice = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False)
X_mice = mice.fit_transform(X)
print(f"MICE imputed age mean : {pd.DataFrame(X_mice, columns=num_cols)['age'].mean():.2f}")
```

<!-- #region -->
### 3.3 missForest
<!-- #endregion -->

<!-- #region -->
**missForest** = IterativeImputer avec **RandomForest** comme estimateur. Souvent **le plus performant** sur données tabulaires mixtes.

Disponible via `missingpy` (lib séparée) ou en passant `estimator=RandomForestRegressor()` à `IterativeImputer`.
<!-- #endregion -->

<!-- #region -->
## 4. Approches "ignore" — laisser le modèle gérer
<!-- #endregion -->

<!-- #region -->
**XGBoost / LightGBM / CatBoost** gèrent les NaN nativement : à chaque split, ils apprennent vers quel côté envoyer les NaN. Souvent **plus performant** que n'importe quelle imputation, surtout en MAR/MNAR.

> Quand on a un GBM en tête, **ne pas imputer** est souvent la meilleure stratégie. Tester systématiquement les deux.

```python
# import xgboost as xgb
# model = xgb.XGBClassifier(missing=np.nan)  # explicite
# model.fit(X_with_nans, y)  # marche directement
```
<!-- #endregion -->

<!-- #region -->
## 5. Indicateur de manquance
<!-- #endregion -->

<!-- #region -->
**Idée** : créer une **colonne booléenne** `was_missing` pour chaque variable, en complément de l'imputation. Le modèle peut alors apprendre à exploiter le **fait** d'être manquant (info MNAR utile).

**Particulièrement utile** quand la manquance est elle-même informative (ex: revenu manquant = revenu très élevé).
<!-- #endregion -->

```python
from sklearn.impute import SimpleImputer

X_aug = X.copy()
for col in X.columns:
    if X[col].isna().any():
        X_aug[f"{col}_missing"] = X[col].isna().astype(int)
X_aug[X.columns] = SimpleImputer(strategy="median").fit_transform(X_aug[X.columns])
print(f"Avec indicateurs : {X_aug.columns.tolist()}")
```

<!-- #region -->
## 6. Bench expérimental
<!-- #endregion -->

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Préparer Titanic propre
df_b = sns.load_dataset("titanic").dropna(subset=["embarked"])
X = df_b[["age", "fare", "pclass", "sibsp", "parch"]]  # contient NaN dans age
y = df_b["survived"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# 1. Drop ligne (= subset sans NaN)
mask = X.notna().all(axis=1)
results["drop_lignes"] = cross_val_score(
    Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]),
    X[mask], y[mask], cv=cv, scoring="roc_auc",
).mean()

# 2. Median imputation + LR
results["median+LR"] = cross_val_score(
    Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
              ("lr", LogisticRegression(max_iter=1000))]),
    X, y, cv=cv, scoring="roc_auc",
).mean()

# 3. KNN imputation + LR
results["knn+LR"] = cross_val_score(
    Pipeline([("imp", KNNImputer(n_neighbors=5)), ("sc", StandardScaler()),
              ("lr", LogisticRegression(max_iter=1000))]),
    X, y, cv=cv, scoring="roc_auc",
).mean()

# 4. MICE + LR
results["mice+LR"] = cross_val_score(
    Pipeline([("imp", IterativeImputer(max_iter=10, random_state=42)),
              ("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]),
    X, y, cv=cv, scoring="roc_auc",
).mean()

# 5. XGBoost natif (pas d'imputation)
results["xgb_native"] = cross_val_score(
    xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0),
    X, y, cv=cv, scoring="roc_auc",
).mean()

print("\n--- Bench AUC (5-fold CV) ---")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} {v:.4f}")
```

<!-- #region -->
## 7. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
1. **Comprendre** d'abord pourquoi les NaN sont là (MCAR/MAR/MNAR) — discussion avec le métier.
2. **Visualiser** le pattern de NaN (heatmap) — détecter les clusters de manquance.
3. **Ajouter des indicateurs** `was_missing` — l'info de manquance est souvent prédictive.
4. **Pour un GBM** : commencer **sans imputer** (laisser le modèle gérer).
5. **Pour un modèle linéaire / NN** : tester median, KNN, MICE en bench CV.
6. **Toujours imputer dans une Pipeline** sklearn (`fit_transform` sur train, `transform` sur test) — sinon leak.
7. Pour les **séries temporelles** : `ffill` / `bfill` / interpolation linéaire / spline / saisonnière (`pandas.DataFrame.interpolate`).
8. **Ne jamais imputer le target** — toujours dropper les obs sans label.
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- [sklearn — Imputation](https://scikit-learn.org/stable/modules/impute.html)
- [Van Buuren — Flexible Imputation of Missing Data (livre)](https://stefvanbuuren.name/fimd/)
- [missingpy](https://github.com/epsilon-machine/missingpy) (missForest)
- [Datawig (AWS) — Deep Learning imputation](https://github.com/awslabs/datawig)
- Notebooks liés : `Structures_Preprocessing`, `ML_Regression_Classification_CV_GridSearch`.
<!-- #endregion -->
