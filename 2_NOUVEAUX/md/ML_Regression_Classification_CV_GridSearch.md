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
# 🏆 ML — Bench complet (CV + GridSearch + Hyperopt)
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki technique** : référence exhaustive pour bencher rapidement les algos classiques sklearn sur un problème donné, avec **cross-validation propre** et **recherche d'hyperparamètres** (GridSearch, RandomizedSearch, Optuna).

Sections :

1. **Cross-validation** — KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit.
2. **GridSearch vs RandomSearch vs HalvingSearch vs Optuna** — quoi choisir.
3. **Pipeline + ColumnTransformer** — la bonne pratique sklearn.
4. **Bench régression** : Ridge, Lasso, SVR, DecisionTree, RandomForest, GBM, AdaBoost.
5. **Bench classification** : LogReg, SVC, KNN, NB, DT, RF, GBM, AdaBoost, MLP.
6. **Clustering** (bonus) : KMeans, DBSCAN, HDBSCAN.
7. **Optimisation moderne avec Optuna**.
8. **Calibration** des probabilités.

Datasets : **California Housing** (regr) + **Titanic** (classif), mutualisés.
<!-- #endregion -->

<!-- #region -->
## 1. Cross-validation : choisir le bon split
<!-- #endregion -->

<!-- #region -->
| Type | Quand |
|---|---|
| **KFold** | Régression / classif équilibrée |
| **StratifiedKFold** | Classification (préserve la proportion de classes par fold) |
| **GroupKFold** | Quand les obs sont groupées (multiple obs par patient, par utilisateur, ...) |
| **TimeSeriesSplit** | Séries temporelles — pas de shuffle ! |
| **RepeatedKFold** | Pour réduire la variance de l'estimation |
| **LeaveOneOut** | Datasets tout petits |
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

data = fetch_california_housing(as_frame=True)
X_reg, y_reg = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

titanic = sns.load_dataset("titanic").dropna(subset=["age", "embarked"])
X_cls = pd.get_dummies(titanic[["pclass","age","sibsp","parch","fare","sex","embarked"]], drop_first=True)
y_cls = titanic["survived"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)

print(f"Regression : {X_tr.shape}  Classif : {Xc_tr.shape}")
```

<!-- #region -->
## 2. Pipeline + ColumnTransformer : LA bonne pratique
<!-- #endregion -->

<!-- #region -->
**Toujours** encapsuler preprocessing + modèle dans une `Pipeline`. Sinon :

- Risque de **leak** quand on fait fit sur le train ET le test (ex: StandardScaler fit sur tout).
- Difficulté à sérialiser pour la prod (un seul `.pkl`).
- Difficulté à grid-searcher les hyperparamètres du preprocessing.

Pour des features **mixtes** (num + cat), utiliser `ColumnTransformer`.
<!-- #endregion -->

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["pclass", "sex", "embarked"]

# Recharger Titanic propre pour avoir les vraies colonnes
titanic2 = sns.load_dataset("titanic").dropna(subset=["embarked"])
Xt = titanic2[num_cols + cat_cols]
yt = titanic2["survived"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, Xt, yt, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"LogReg CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")
```

<!-- #region -->
## 3. Bench régression — N modèles d'un coup
<!-- #endregion -->

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import make_scorer, mean_squared_error

def bench_regression(X, y, models: dict, cv=5) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("sc", StandardScaler()), ("m", model)])
        rmse_scores = -cross_val_score(pipe, X, y, cv=cv,
                                       scoring="neg_root_mean_squared_error", n_jobs=-1)
        rows.append({"model": name, "rmse_mean": rmse_scores.mean(), "rmse_std": rmse_scores.std()})
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


regression_models = {
    "Ridge":         Ridge(alpha=1.0),
    "Lasso":         Lasso(alpha=0.01),
    "ElasticNet":    ElasticNet(alpha=0.01, l1_ratio=0.5),
    "DecisionTree":  DecisionTreeRegressor(max_depth=8, random_state=42),
    "RandomForest":  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GBM":           GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost":      AdaBoostRegressor(n_estimators=100, random_state=42),
}
print(bench_regression(X_tr, y_tr, regression_models, cv=3).to_string(index=False))
```

<!-- #region -->
## 4. Bench classification
<!-- #endregion -->

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

def bench_classif(X, y, models: dict, cv=5) -> pd.DataFrame:
    rows = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model in models.items():
        pipe = Pipeline([("sc", StandardScaler()), ("m", model)])
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "auc_mean": scores.mean(), "auc_std": scores.std()})
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)


classif_models = {
    "LogReg":        LogisticRegression(max_iter=1000),
    "KNN":           KNeighborsClassifier(n_neighbors=5),
    "NB":            GaussianNB(),
    "SVC_rbf":       SVC(kernel="rbf", probability=True, random_state=42),
    "DecisionTree":  DecisionTreeClassifier(max_depth=8, random_state=42),
    "RandomForest":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GBM":           GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost":      AdaBoostClassifier(n_estimators=100, random_state=42),
    "MLP":           MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
}
print(bench_classif(Xc_tr, yc_tr, classif_models, cv=3).to_string(index=False))
```

<!-- #region -->
## 5. GridSearch vs RandomSearch vs HalvingSearch
<!-- #endregion -->

<!-- #region -->
| Méthode | Idée | Quand |
|---|---|---|
| **GridSearchCV** | Toutes les combinaisons | Petite grille (< 100 combos) |
| **RandomizedSearchCV** | N tirages aléatoires dans des distributions | Grande grille, budget fixe |
| **HalvingGridSearchCV** / **HalvingRandomSearchCV** | Successive halving — élimine les mauvais sur peu de data | Grande grille, gain de temps |
| **BayesSearchCV** (`scikit-optimize`) | Optimisation bayésienne | Coût d'éval élevé |
| **Optuna** | TPE / CMA-ES, pruners, distribué | **Standard 2026** pour la recherche sérieuse |
<!-- #endregion -->

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, loguniform

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# RandomSearch — plus efficace que Grid si dimensions élevées
rs = RandomizedSearchCV(
    rf,
    param_distributions={
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 15),
        "min_samples_leaf": randint(1, 10),
    },
    n_iter=20, cv=3, scoring="roc_auc", n_jobs=-1, random_state=42,
)
rs.fit(Xc_tr.fillna(Xc_tr.median(numeric_only=True)), yc_tr)
print(f"RandomSearch best AUC = {rs.best_score_:.4f}")
print(f"Best params : {rs.best_params_}")
```

<!-- #region -->
## 6. Optuna — l'approche moderne
<!-- #endregion -->

<!-- #region -->
**Optuna** propose un `study.optimize(objective, n_trials=N)` avec :

- **Sampler** intelligent (TPE par défaut — bayésien efficient).
- **Pruner** — abandon précoce des trials sans potentiel.
- **Visualisation** : importance des hyperparams, plot 2D, plot contour.
- **Distribué** facile via `study_name + storage="postgresql://..."`.
<!-- #endregion -->

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    n_est = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    model = RandomForestClassifier(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=min_leaf, random_state=42, n_jobs=-1,
    )
    pipe = Pipeline([("sc", StandardScaler()), ("m", model)])
    scores = cross_val_score(
        pipe,
        Xc_tr.fillna(Xc_tr.median(numeric_only=True)),
        yc_tr, cv=3, scoring="roc_auc", n_jobs=-1,
    )
    return scores.mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, show_progress_bar=False)
print(f"Optuna best AUC = {study.best_value:.4f}")
print(f"Best params      : {study.best_params}")
```

<!-- #region -->
## 7. Clustering (bonus rapide)
<!-- #endregion -->

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X_num = X_tr.values[:2000]   # subset pour aller vite
X_num_sc = StandardScaler().fit_transform(X_num)

for k in [3, 5, 8]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_num_sc)
    sil = silhouette_score(X_num_sc, km.labels_)
    print(f"KMeans k={k}  silhouette={sil:.3f}")

# DBSCAN — auto-détection du nombre de clusters + bruit
db = DBSCAN(eps=0.5, min_samples=10).fit(X_num_sc)
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"DBSCAN  n_clusters={n_clusters}, bruit={(db.labels_ == -1).sum()}")
```

<!-- #region -->
## 8. Calibration des probabilités
<!-- #endregion -->

<!-- #region -->
Beaucoup de modèles donnent des `predict_proba` **mal calibrés** (la sortie 0.8 ne signifie pas vraiment "80 % de chance"). Important pour :

- Métiers où le score est interprété comme une proba (médical, finance, scoring crédit).
- Combinaisons de modèles (ensembling probabiliste).
- Évaluation par log-loss / Brier score.

**Méthodes** :

- **Platt scaling** (sigmoid) : ajuste une régression logistique sur les scores → meilleure si data large + monotonie quasi-sigmoidale.
- **Isotonic regression** : non-paramétrique, plus flexible, demande plus de data.

```python
# from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# cal = CalibratedClassifierCV(model, method="isotonic", cv=5).fit(X_train, y_train)
# proba_cal = cal.predict_proba(X_test)[:, 1]
```
<!-- #endregion -->

<!-- #region -->
## 9. Pièges classiques
<!-- #endregion -->

<!-- #region -->
- ❌ `fit_transform(X)` sur train+test → leak (utiliser `Pipeline`).
- ❌ Choisir le modèle sur le test set → biais d'optimisme (utiliser un val set ou CV).
- ❌ `shuffle=True` sur séries temporelles → catastrophe.
- ❌ Comparer des modèles sur des splits différents → variance arbitraire.
- ❌ Oublier `random_state` → résultats non reproductibles.
- ✅ Toujours rapporter `mean ± std` d'une CV, pas un seul score.
- ✅ Conserver le **test set vierge** jusqu'à la fin pour le score final.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [scikit-learn — Model selection](https://scikit-learn.org/stable/model_selection.html)
- [Optuna — docs](https://optuna.readthedocs.io/)
- [INRIA SKLearn MOOC](https://inria.github.io/scikit-learn-mooc/)
- Notebooks liés : `ML_Regression_Classification_Multiple`, `ML_Bagging_Boosting`, `ML_Optimisation_de_Modèles`, `ML_Explication_Feature_Importance_Selection`.
<!-- #endregion -->
