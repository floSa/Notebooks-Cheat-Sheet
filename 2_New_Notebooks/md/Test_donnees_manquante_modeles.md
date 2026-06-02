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
Notebook **Tutoriel + Wiki + Expérience comparée** sur la **gestion des données manquantes** (NaN) en ML : panorama des stratégies, maths de la typologie, et **bench expérimental** de l'impact sur la performance du modèle.

Couvre :

1. **Typologie** : MCAR / MAR / MNAR — comprendre **pourquoi** les NaN sont là.
2. **Approches simples** : drop, mean/median, mode, valeur constante.
3. **Approches modèle** : `KNNImputer`, `IterativeImputer` (MICE), missForest.
4. **Approches « ignore »** : XGBoost / CatBoost gèrent **nativement** les NaN.
5. **Indicateur de manquance** comme feature.
6. **Expérience** : impact du **% de NaN** sur la perf (modèle seul vs imputation).
7. **Bench** comparatif des stratégies (AUC, CV).
8. **Bonnes pratiques** 2026.

Dataset : **Titanic** (`seaborn.load_dataset`), qui contient naturellement beaucoup de NaN dans `age`, `cabin`, `deck`.
<!-- #endregion -->

<!-- #region -->
## 1. Typologie des manquances
<!-- #endregion -->

<!-- #region -->
Avant toute imputation, il faut comprendre le **mécanisme** qui génère les NaN (formalisme de Rubin). Soit $X$ la matrice de données et $M$ le masque binaire des manquances ($M_{ij}=1$ si $X_{ij}$ est manquant).

| Type | Définition formelle | Exemple |
|---|---|---|
| **MCAR** (Missing Completely At Random) | $P(M \mid X_{obs}, X_{mis}) = P(M)$ — indépendant de tout | Erreur de transmission aléatoire |
| **MAR** (Missing At Random) | $P(M \mid X_{obs}, X_{mis}) = P(M \mid X_{obs})$ — dépend du **observé** | Hommes moins enclins à répondre à une enquête santé |
| **MNAR** (Missing Not At Random) | dépend de la valeur **manquante** elle-même | Salaires très élevés non renseignés (biais social) |

**Importance** : une imputation a un sens statistique **seulement sous MCAR ou MAR**. Sous **MNAR**, toute imputation introduit un biais — solution = modélisation conjointe (du mécanisme de manquance) ou collecte additionnelle.
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

<!-- #region -->
Visualiser le **pattern** des NaN (et pas seulement leur compte) révèle les **clusters** de manquance — utile pour diagnostiquer MAR (manquances corrélées entre colonnes).
<!-- #endregion -->

```python
# Heatmap des NaN — visualiser les patterns
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(df.isna().T, cbar=False, cmap="binary", ax=ax)
ax.set_title("Pattern des NaN (Titanic)")
plt.tight_layout()
plt.show()
```

<!-- #region -->
## 2. Approches simples
<!-- #endregion -->

<!-- #region -->
### 2.1 Drop
<!-- #endregion -->

<!-- #region -->
- `dropna()` au niveau **observation** : on perd des lignes. Si NaN concentré sur une colonne très utilisée → catastrophique (ici, dropper toute ligne avec un NaN fait passer de 891 à 182 lignes).
- `dropna(axis=1)` au niveau **colonne** : on perd la variable. Acceptable si > 70 % NaN et faible info.

**À utiliser** : pour les colonnes avec > 80 % de NaN sans info récupérable (ex : `deck` dans Titanic, 77 % NaN).
<!-- #endregion -->

```python
print(f"Avant : {len(df)} lignes")
print(f"Drop lignes ANY NaN : {len(df.dropna())} (perte massive)")
print(f"Drop col deck (77% NaN) : {df.drop(columns=['deck']).shape}")
```

<!-- #region -->
### 2.2 Constantes : mean / median / mode
<!-- #endregion -->

<!-- #region -->
- **Median** pour numérique → robuste aux outliers.
- **Mean** si distribution ~normale.
- **Mode** (`most_frequent`) pour catégorielle.
- **Valeur constante** (`-1`, `"Unknown"`) → permet au modèle d'apprendre « c'est manquant » comme info.

**Limite** : sous-estime la variance, ignore les corrélations entre variables.
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
**Idée** : pour chaque ligne avec NaN, trouver les `k` plus proches voisins (sur les features non manquantes, distance euclidienne tronquée aux dimensions observées), imputer par moyenne (pondérée) de leurs valeurs.

**Forces** : capture les corrélations entre variables.
**Faiblesses** : sensible aux échelles (toujours **standardiser** avant), coût $O(n^2)$ — lent sur grand $n$.
<!-- #endregion -->

```python
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

num_cols = ["age", "fare", "pclass", "sibsp", "parch"]
X = df[num_cols].copy()
X_scaled = StandardScaler().fit_transform(X.fillna(X.median()))  # init pour scaler

knn_imp = KNNImputer(n_neighbors=5)
X_knn = knn_imp.fit_transform(X)
print(f"KNN imputed age mean : {pd.DataFrame(X_knn, columns=num_cols)['age'].mean():.2f}")
```

<!-- #region -->
### 3.2 IterativeImputer (MICE)
<!-- #endregion -->

<!-- #region -->
**MICE** (Multivariate Imputation by Chained Equations) : pour chaque variable $X_j$ avec NaN, on entraîne un **modèle** $X_j \sim f(X_{-j})$ (par défaut `BayesianRidge`) qui la prédit à partir des autres. On itère sur toutes les colonnes plusieurs passes jusqu'à (quasi) convergence.

**Forces** : flexible (n'importe quel estimateur), gère les types mixtes, restitue la structure multivariée.
**Faiblesses** : plus lent, peut diverger si fort multicolinéarité.
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
**missForest** = `IterativeImputer` dont l'estimateur par variable est une **RandomForest** (au lieu de `BayesianRidge`). Souvent **le plus performant** sur données tabulaires mixtes et non linéaires, car il capture interactions et non-linéarités sans hypothèse de forme.

Disponible nativement via sklearn en passant `estimator=RandomForestRegressor()` à `IterativeImputer` (la lib historique `missingpy` est non maintenue).
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor

# missForest = MICE dont l'estimateur par variable est une RandomForest.
miss_forest = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    max_iter=10,
    random_state=42,
)
X_rf = miss_forest.fit_transform(X)
print(f"missForest imputed age mean : {pd.DataFrame(X_rf, columns=num_cols)['age'].mean():.2f}")
```

<!-- #region -->
## 4. Approches « ignore » — laisser le modèle gérer
<!-- #endregion -->

<!-- #region -->
**XGBoost / LightGBM / CatBoost** gèrent les NaN **nativement** : à chaque split d'arbre, l'algorithme apprend la **direction par défaut** (gauche/droite) vers laquelle envoyer les valeurs manquantes, en maximisant le gain. Aucune imputation n'est nécessaire — et c'est souvent **plus performant** que n'importe quelle imputation, surtout en MAR/MNAR (le « manquant » devient un signal exploité par l'arbre).

> Avec un GBM, **ne pas imputer** est fréquemment la meilleure stratégie. À tester systématiquement contre l'imputation.

Ci-dessous, on entraîne directement sur des features contenant des NaN (`age`), **sans imputation**, et on mesure l'AUC en validation croisée.
<!-- #endregion -->

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

df_nat = sns.load_dataset("titanic").dropna(subset=["embarked"])
X_nat = df_nat[["age", "fare", "pclass", "sibsp", "parch"]]  # 'age' contient des NaN bruts
y_nat = df_nat["survived"]
cv_nat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Aucune imputation : on passe directement les NaN au modèle.
# (n_jobs/thread_count=1 ici pour rester reproductible et éviter une
#  sur-souscription de threads OpenMP ; mettre -1 sur une machine multicœur saine.)
auc_xgb = cross_val_score(
    XGBClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=1, verbosity=0),
    X_nat, y_nat, cv=cv_nat, scoring="roc_auc",
).mean()
auc_cat = cross_val_score(
    CatBoostClassifier(iterations=200, depth=4, random_state=42, verbose=0, thread_count=1),
    X_nat, y_nat, cv=cv_nat, scoring="roc_auc",
).mean()
print(f"AUC XGBoost (NaN natif, sans imputation)  : {auc_xgb:.4f}")
print(f"AUC CatBoost (NaN natif, sans imputation) : {auc_cat:.4f}")
```

<!-- #region -->
Les deux GBM atteignent une AUC compétitive **sans la moindre imputation** : le coût d'ingénierie d'un pipeline d'imputation n'est pas toujours justifié quand le modèle final est à base d'arbres.
<!-- #endregion -->

<!-- #region -->
## 5. Indicateur de manquance
<!-- #endregion -->

<!-- #region -->
**Idée** : créer une **colonne booléenne** `<col>_missing` pour chaque variable concernée, **en complément** de l'imputation. Le modèle peut alors apprendre à exploiter le **fait** d'être manquant (signal MNAR utile), tout en disposant d'une valeur imputée plausible.

**Particulièrement utile** quand la manquance est elle-même informative (ex : revenu manquant ⇔ revenu très élevé). C'est exactement ce que fait `SimpleImputer(add_indicator=True)`.
<!-- #endregion -->

```python
X_aug = X.copy()
for col in X.columns:
    if X[col].isna().any():
        X_aug[f"{col}_missing"] = X[col].isna().astype(int)
X_aug[X.columns] = SimpleImputer(strategy="median").fit_transform(X_aug[X.columns])
print(f"Avec indicateurs : {X_aug.columns.tolist()}")
```

<!-- #region -->
## 6. Expérience — impact du % de données manquantes
<!-- #endregion -->

<!-- #region -->
**Question centrale** (l'expérience d'origine, ici modernisée et corrigée) : quand le **taux de NaN augmente**, comment se dégrade la performance, et **l'imputation aide-t-elle** ?

Protocole reproductible :

1. Dataset **Titanic**, cible binaire `survived` (vraie classification — on corrige le montage initial Boston+Classifier, conceptuellement invalide car Boston est de la régression).
2. Pour chaque taux $p \in \{0, 5, \dots, 30\}\,\%$, on injecte des NaN **MCAR** au taux $p$ sur toutes les features (graine fixée).
3. Validation croisée stratifiée 5-fold ; l'imputeur est **fit sur le train uniquement** puis appliqué au test (**pas de fuite**).
4. On compare, pour chaque famille de modèle, **modèle + imputation A** vs **B**.

Première figure : **RandomForest** (qui ne sait pas gérer les NaN nativement) — `SimpleImputer(mean)` vs `KNNImputer(3)`.
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Données : Titanic, cible binaire valide.
df_exp = sns.load_dataset("titanic").dropna(subset=["embarked"]).reset_index(drop=True)
feat_cols = ["age", "fare", "pclass", "sibsp", "parch"]
X_exp = df_exp[feat_cols].copy()
y_exp = df_exp["survived"].to_numpy()


def inject_missing(frame: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Injecte des NaN MCAR au taux `rate` (reproductible via `seed`)."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for col in out.columns:
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan
    return out


def imp_mean(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    """Imputation par la moyenne, fit sur le train uniquement."""
    imp = SimpleImputer(strategy="mean")
    return imp.fit_transform(X_tr), imp.transform(X_te)


def imp_knn(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    """Imputation KNN, fit sur le train uniquement."""
    imp = KNNImputer(n_neighbors=3)
    return imp.fit_transform(X_tr), imp.transform(X_te)


def sweep(make_model, impute, percentages: list[int], n_splits: int = 5) -> list[float]:
    """Accuracy moyenne (CV) en fonction du % de NaN injectés.

    `impute(X_tr, X_te)` renvoie (X_tr_clean, X_te_clean) ; `impute=None`
    => on laisse les NaN au modèle (GBM natif).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []
    for p in percentages:
        fold_acc = []
        for k, (tr, te) in enumerate(cv.split(X_exp, y_exp)):
            X_tr = inject_missing(X_exp.iloc[tr], p / 100, seed=1000 + k)
            X_te = inject_missing(X_exp.iloc[te], p / 100, seed=2000 + k)
            y_tr, y_te = y_exp[tr], y_exp[te]
            if impute is not None:
                X_tr, X_te = impute(X_tr, X_te)
            model = make_model()
            model.fit(X_tr, y_tr)
            fold_acc.append(accuracy_score(y_te, model.predict(X_te)))
        scores.append(float(np.mean(fold_acc)))
    return scores


percentages_missing = list(range(0, 31, 5))

rf_mean = sweep(lambda: RandomForestClassifier(random_state=42), imp_mean, percentages_missing)
rf_knn = sweep(lambda: RandomForestClassifier(random_state=42), imp_knn, percentages_missing)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(percentages_missing, rf_mean, marker="o", label="RF + SimpleImputer(mean)")
ax.plot(percentages_missing, rf_knn, marker="s", label="RF + KNNImputer(3)")
ax.set_xlabel("Pourcentage de données manquantes (MCAR)")
ax.set_ylabel("Accuracy moyenne (5-fold CV)")
ax.set_title("RandomForest : impact du % de NaN selon l'imputation")
ax.legend()
plt.tight_layout()
plt.show()
```

<!-- #region -->
La performance se dégrade quand $p$ croît (l'information disparaît), et l'écart `mean` vs `KNN` reste faible ici car Titanic est petit et peu multivarié — sur des données plus corrélées, le KNN/MICE creuse l'écart.

Deuxième figure : **gradient boosting** (l'expérience d'origine avec XGBoost) — **NaN natif** (sans imputation) vs `KNNImputer(3)` + XGBoost.
<!-- #endregion -->

```python
xgb_native = sweep(
    lambda: XGBClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=1, verbosity=0),
    None,  # NaN natif : pas d'imputation
    percentages_missing,
)
xgb_knn = sweep(
    lambda: XGBClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=1, verbosity=0),
    imp_knn,
    percentages_missing,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(percentages_missing, xgb_native, marker="o", label="XGBoost (NaN natif)")
ax.plot(percentages_missing, xgb_knn, marker="s", label="KNNImputer(3) + XGBoost")
ax.set_xlabel("Pourcentage de données manquantes (MCAR)")
ax.set_ylabel("Accuracy moyenne (5-fold CV)")
ax.set_title("Gradient boosting : NaN natif vs imputation KNN")
ax.legend()
plt.tight_layout()
plt.show()
```

<!-- #region -->
**Conclusion de l'expérience** : pour un GBM, laisser les NaN au modèle (NaN natif) tient la route face à une imputation KNN, sans le coût et le risque de fuite d'un pipeline d'imputation. Pour un modèle non tolérant aux NaN (RandomForest, modèle linéaire), une imputation **fit sur le train** est obligatoire, et son choix (mean / KNN / MICE) se valide en CV.
<!-- #endregion -->

<!-- #region -->
## 7. Bench expérimental des stratégies
<!-- #endregion -->

<!-- #region -->
Comparaison directe, en AUC 5-fold, des grandes familles de stratégies sur les NaN **naturels** de Titanic (sans injection artificielle) : drop, imputation simple/KNN/MICE devant une régression logistique, et XGBoost natif.
<!-- #endregion -->

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

df_b = sns.load_dataset("titanic").dropna(subset=["embarked"])
Xb = df_b[["age", "fare", "pclass", "sibsp", "parch"]]
yb = df_b["survived"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# 1. Drop ligne (= subset sans NaN)
mask = Xb.notna().all(axis=1)
results["drop_lignes"] = cross_val_score(
    Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]),
    Xb[mask], yb[mask], cv=cv, scoring="roc_auc",
).mean()

# 2. Median imputation + LR
results["median+LR"] = cross_val_score(
    Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
              ("lr", LogisticRegression(max_iter=1000))]),
    Xb, yb, cv=cv, scoring="roc_auc",
).mean()

# 3. KNN imputation + LR
results["knn+LR"] = cross_val_score(
    Pipeline([("imp", KNNImputer(n_neighbors=5)), ("sc", StandardScaler()),
              ("lr", LogisticRegression(max_iter=1000))]),
    Xb, yb, cv=cv, scoring="roc_auc",
).mean()

# 4. MICE + LR
results["mice+LR"] = cross_val_score(
    Pipeline([("imp", IterativeImputer(max_iter=10, random_state=42)),
              ("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]),
    Xb, yb, cv=cv, scoring="roc_auc",
).mean()

# 5. XGBoost natif (pas d'imputation)
results["xgb_native"] = cross_val_score(
    xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=1, verbosity=0),
    Xb, yb, cv=cv, scoring="roc_auc",
).mean()

print("\n--- Bench AUC (5-fold CV) ---")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} {v:.4f}")
```

<!-- #region -->
## 8. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
1. **Comprendre** d'abord pourquoi les NaN sont là (MCAR/MAR/MNAR) — discussion avec le métier.
2. **Visualiser** le pattern de NaN (heatmap) — détecter les clusters de manquance.
3. **Ajouter des indicateurs** `was_missing` — l'info de manquance est souvent prédictive.
4. **Pour un GBM** (XGBoost / CatBoost) : commencer **sans imputer** (laisser le modèle gérer).
5. **Pour un modèle linéaire / NN** : tester median, KNN, MICE en bench CV.
6. **Toujours imputer dans une Pipeline** sklearn (`fit` sur train, `transform` sur test) — sinon **fuite de données**.
7. Pour les **séries temporelles** : `ffill` / `bfill` / interpolation linéaire / spline / saisonnière (`pandas.DataFrame.interpolate`).
8. **Ne jamais imputer le target** — toujours dropper les observations sans label.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [sklearn — Imputation](https://scikit-learn.org/stable/modules/impute.html)
- [Van Buuren — Flexible Imputation of Missing Data (livre)](https://stefvanbuuren.name/fimd/)
- [missingpy](https://github.com/epsilon-machine/missingpy) (missForest historique)
- [Datawig (AWS) — Deep Learning imputation](https://github.com/awslabs/datawig)
- Notebooks liés : `Structures_Preprocessing`, `ML_Regression_Classification_CV_GridSearch`.
<!-- #endregion -->
