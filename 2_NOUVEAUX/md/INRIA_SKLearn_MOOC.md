---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Tour complet de scikit-learn — d'après le MOOC INRIA
<!-- #endregion -->

<!-- #region -->
Ce notebook est une refonte **2026** de notes prises sur le [MOOC INRIA *Machine learning in Python with scikit-learn*](https://inria.github.io/scikit-learn-mooc/). L'original était un brouillon Colab fourre-tout ; on en garde le périmètre — un vrai **tour de scikit-learn** — mais remis en ordre, corrigé (API obsolètes, variables non définies) et **augmenté**.

Rôles : **Wiki technique + Tutoriel + Cheat-sheet**. Code valable pour **scikit-learn 1.8** (déc. 2025).

**Sommaire**

1. Préambule & configuration
2. Jeux de données fil rouge
3. L'API scikit-learn (Pipeline, ColumnTransformer)
4. Validation & cross-validation
5. Modèles ensemblistes (bagging, boosting, nested CV)
6. Sélection de caractéristiques
7. Interprétation des modèles
8. Métriques
9. Encodage des variables catégorielles
10. Nouveautés scikit-learn 1.8 (2026)
11. Récapitulatif
<!-- #endregion -->

<!-- #region -->
## 1. Préambule & configuration
<!-- #endregion -->

<!-- #region -->
Une leçon centrale du brouillon original : **tout importer en tête** et **définir avant d'utiliser**. On centralise ici imports, graine aléatoire et palette de couleurs. `set_config(transform_output="pandas")` fait renvoyer des DataFrames (et non des tableaux numpy) en sortie de chaque transformer — sorties bien plus lisibles.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

RANDOM_STATE = 0

# Palette du repo (cohérence visuelle entre notebooks)
PALETTE = ["#00798c", "#d1495b", "#edae49", "#66a182",
           "#2e4057", "#9d83b8", "#b8848e", "#c9b78b"]
PRIMARY_1, MAUVAIS, MOYEN, ACCENT, ACCENT_DARK, LAVENDER, DUSTY_ROSE, BEIGE = PALETTE

# Sortie pandas par défaut pour tous les transformers
sklearn.set_config(transform_output="pandas")

print("scikit-learn", sklearn.__version__)
print("numpy", np.__version__, "| pandas", pd.__version__)
```

<!-- #region -->
## 2. Jeux de données fil rouge
<!-- #endregion -->

<!-- #region -->
On utilise trois jeux mutualisés, chacun pour un usage précis :

| Dataset | Chargement | Usage dans ce notebook |
|---|---|---|
| **California Housing** | `fetch_california_housing` | Régression : ensembles, interprétation, importance, PDP |
| **make_classification** | `make_classification` | Sélection de caractéristiques (features informatives contrôlées) |
| **Titanic** | `seaborn.load_dataset` | Encodage catégoriel, classification binaire, métriques, seuil |

Tout est chargé programmatiquement — rien à committer.
<!-- #endregion -->

```python
from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target = target * 100  # cible exprimée en milliers de dollars (k$)
print(data.shape, "| features:", list(data.columns))
data.head(3)
```

<!-- #region -->
Le jeu **California Housing** : 20 640 districts, 8 caractéristiques numériques (revenu médian `MedInc`, âge des logements, nombre moyen de pièces, position géographique…). La cible est la valeur médiane des logements (ici en k$).
<!-- #endregion -->

```python
from sklearn.datasets import make_classification

X_clf, y_clf = make_classification(
    n_samples=5000, n_features=20, n_informative=3, n_redundant=2,
    n_repeated=0, n_classes=3, n_clusters_per_class=1, random_state=RANDOM_STATE,
)
X_clf.shape, np.unique(y_clf)
```

<!-- #region -->
Le **Titanic** : on garde 4 colonnes numériques et 3 catégorielles, cible binaire `survived`. C'est le support des sections encodage / métriques / seuil de décision (il remplace l'ancien chargement via Google Drive, inutilisable hors Colab).
<!-- #endregion -->

```python
titanic = sns.load_dataset("titanic").dropna(subset=["embarked"])
CAT_NUM = ["age", "fare", "sibsp", "parch"]
CAT_CAT = ["pclass", "sex", "embarked"]
X_cat = titanic[CAT_NUM + CAT_CAT].copy()
y_cat = titanic["survived"].copy()
print("Titanic", X_cat.shape, "| taux de survie:", round(y_cat.mean(), 3))
X_cat.head(3)
```

<!-- #region -->
## 3. L'API scikit-learn
<!-- #endregion -->

<!-- #region -->
Toute la cohérence de scikit-learn tient en trois interfaces :

| Interface | Méthode clé | Exemple |
|---|---|---|
| **Estimator** | `fit(X, y)` | tout objet apprenant |
| **Transformer** | `transform(X)` / `fit_transform` | `StandardScaler`, `OneHotEncoder` |
| **Predictor** | `predict(X)`, `predict_proba`, `score` | `Ridge`, `LogisticRegression` |

C'est ce contrat uniforme qui rend les objets **composables** (Pipeline) et **interchangeables** (on remplace un modèle par un autre sans changer le reste du code).
<!-- #endregion -->

<!-- #region -->
### 3.1 Pipeline
<!-- #endregion -->

<!-- #region -->
La leçon n°1 du MOOC : **encapsuler le prétraitement et le modèle dans une `Pipeline`**. Un seul objet qu'on `fit`, qu'on passe à la cross-validation et qu'on tune — et surtout : le prétraitement est ré-appris **sur chaque fold d'entraînement**, ce qui évite les fuites de données (cf §6.2).
<!-- #endregion -->

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=RANDOM_STATE, test_size=0.25)

pipe_reg = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
pipe_reg.fit(data_train, target_train)
print("R2 test:", round(pipe_reg.score(data_test, target_test), 3))
```

<!-- #region -->
### 3.2 ColumnTransformer
<!-- #endregion -->

<!-- #region -->
Pour des données mixtes (numérique + catégoriel), `ColumnTransformer` applique un prétraitement **différent par groupe de colonnes** : imputation médiane + standardisation sur le numérique, imputation du mode + one-hot sur le catégoriel. On l'isole dans une fonction réutilisée dans tout le notebook.
<!-- #endregion -->

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def build_titanic_preprocessor() -> ColumnTransformer:
    """ColumnTransformer : imputation+scaling sur le numérique, imputation+OneHot sur le catégoriel."""
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, CAT_NUM),
                              ("cat", cat_pipe, CAT_CAT)])


pipe_cat = Pipeline([("prep", build_titanic_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
pipe_cat.fit(X_cat, y_cat)
print("steps:", [name for name, _ in pipe_cat.steps])
print("accuracy train:", round(pipe_cat.score(X_cat, y_cat), 3))
```

<!-- #region -->
`OneHotEncoder(sparse_output=False)` est nécessaire ici car `transform_output="pandas"` (§1) ne supporte pas les matrices creuses.
<!-- #endregion -->

<!-- #region -->
## 4. Validation & cross-validation
<!-- #endregion -->

<!-- #region -->
Évaluer un modèle sur ses données d'entraînement surestime sa performance. La **validation croisée** découpe les données en plis (*folds*), entraîne sur les uns et évalue sur l'autre, puis fait la moyenne.

| Splitter | Quand l'utiliser |
|---|---|
| `KFold` | Régression, données i.i.d. |
| `StratifiedKFold` | Classification (préserve les proportions de classes) |
| `GroupKFold` | Échantillons groupés (patients, utilisateurs) |
| `TimeSeriesSplit` | Séries temporelles (pas de fuite du futur) |

Règle d'or : toujours rapporter **moyenne ± écart-type**, jamais un seul pli.
<!-- #endregion -->

```python
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold


def evaluate_cv(model, X, y, scoring, cv=5) -> dict:
    """Renvoie un dict {metric: 'mean ± std'} via cross_validate."""
    res = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    out = {}
    for key in res:
        if key.startswith("test_"):
            out[key] = f"{res[key].mean():.3f} ± {res[key].std():.3f}"
    out["fit_time"] = f"{res['fit_time'].mean():.3f}s"
    return out


evaluate_cv(pipe_reg, data, target, scoring="r2")
```

<!-- #region -->
### 4.1 Validation curve
<!-- #endregion -->

<!-- #region -->
La **courbe de validation** trace la performance en fonction d'**un hyperparamètre**. Elle révèle le compromis biais/variance : à faible complexité, train et validation sont bas (sous-apprentissage) ; à forte complexité, le train monte mais la validation décroche (sur-apprentissage). On échantillonne les données pour garder un temps de calcul raisonnable.
<!-- #endregion -->

```python
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.tree import DecisionTreeRegressor

# sous-échantillon pour rester rapide
sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(data), 4000, replace=False)
data_sub, target_sub = data.iloc[sub_idx], target.iloc[sub_idx]

fig, ax = plt.subplots(figsize=(7, 4))
ValidationCurveDisplay.from_estimator(
    DecisionTreeRegressor(random_state=RANDOM_STATE), data_sub, target_sub,
    param_name="max_depth", param_range=[1, 3, 5, 8, 12, 20],
    scoring="neg_mean_absolute_error", cv=5, n_jobs=-1, ax=ax)
_ = ax.set_title("Validation curve — DecisionTree (max_depth)")
```

<!-- #region -->
### 4.2 Learning curve
<!-- #endregion -->

<!-- #region -->
La **courbe d'apprentissage** trace la performance en fonction de la **taille du jeu d'entraînement** :

- train et validation convergent vers un score bas → **biais élevé** (sous-apprentissage) : augmenter la complexité.
- écart persistant train ≫ validation → **variance élevée** (sur-apprentissage) : régulariser ou ajouter des données.
<!-- #endregion -->

```python
from sklearn.model_selection import LearningCurveDisplay
from sklearn.ensemble import HistGradientBoostingRegressor

fig, ax = plt.subplots(figsize=(7, 4))
LearningCurveDisplay.from_estimator(
    HistGradientBoostingRegressor(random_state=RANDOM_STATE), data_sub, target_sub,
    train_sizes=np.linspace(0.1, 1.0, 5), scoring="neg_mean_absolute_error",
    cv=5, n_jobs=-1, ax=ax)
_ = ax.set_title("Learning curve — HistGBDT")
```

<!-- #region -->
## 5. Modèles ensemblistes
<!-- #endregion -->

<!-- #region -->
Un ensemble combine plusieurs modèles faibles en un modèle fort. Deux grandes familles :

- **Bagging** (bootstrap aggregating) : entraîne des modèles **en parallèle** sur des sous-échantillons bootstrap, puis moyenne. Réduit surtout la **variance**. Ex. : Random Forest.
- **Boosting** : entraîne des modèles **séquentiellement**, chacun corrigeant les erreurs du précédent. Réduit surtout le **biais**. Ex. : AdaBoost, Gradient Boosting.
<!-- #endregion -->

<!-- #region -->
### 5.1 Arbre de décision (baseline)
<!-- #endregion -->

<!-- #region -->
Un arbre seul est interprétable mais sur-apprend facilement : c'est la **brique de base** des ensembles, rarement utilisé seul. On mesure son erreur absolue moyenne (MAE) comme référence.
<!-- #endregion -->

```python
from sklearn.metrics import mean_absolute_error

tree = DecisionTreeRegressor(random_state=RANDOM_STATE)
tree.fit(data_train, target_train)
print("MAE test:", round(mean_absolute_error(target_test, tree.predict(data_test)), 2), "k$")
```

<!-- #region -->
### 5.2 Bagging
<!-- #endregion -->

<!-- #region -->
`BaggingRegressor` entraîne plusieurs arbres sur des échantillons bootstrap et moyenne leurs prédictions. **Note API (sklearn 1.8)** : le paramètre se nomme `estimator=` (l'ancien `base_estimator=` du MOOC a été supprimé en 1.4).
<!-- #endregion -->

```python
from sklearn.ensemble import BaggingRegressor

bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
    n_estimators=20, n_jobs=-1, random_state=RANDOM_STATE)
bagging.fit(data_train, target_train)
print("MAE test:", round(mean_absolute_error(target_test, bagging.predict(data_test)), 2), "k$")
```

<!-- #region -->
On tune le bagging par **recherche aléatoire** (`RandomizedSearchCV`) : on échantillonne `n_iter` combinaisons d'hyperparamètres au lieu de tester toute la grille. Le préfixe `estimator__max_depth` cible un paramètre de l'arbre interne.
<!-- #endregion -->

```python
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "n_estimators": randint(10, 30),
    "max_samples": [0.5, 0.8, 1.0],
    "max_features": [0.5, 0.8, 1.0],
    "estimator__max_depth": randint(3, 10),
}
search = RandomizedSearchCV(
    BaggingRegressor(estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
                     n_jobs=-1, random_state=RANDOM_STATE),
    param_dist, n_iter=10, scoring="neg_mean_absolute_error",
    random_state=RANDOM_STATE, n_jobs=-1)
search.fit(data_train, target_train)
print("best params:", search.best_params_)
print("MAE test:", round(mean_absolute_error(target_test, search.predict(data_test)), 2), "k$")
```

<!-- #region -->
### 5.3 Random Forest
<!-- #endregion -->

<!-- #region -->
La **forêt aléatoire** est un bagging d'arbres **décorrélés** (chaque split ne considère qu'un sous-ensemble aléatoire de features). Contrairement à AdaBoost, augmenter `n_estimators` ne provoque **pas** de sur-apprentissage : la courbe se stabilise. La forêt est donc une baseline robuste qui demande peu de réglage.
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor

fig, ax = plt.subplots(figsize=(7, 4))
ValidationCurveDisplay.from_estimator(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    data_sub, target_sub, param_name="n_estimators",
    param_range=[1, 5, 10, 20, 50], scoring="neg_mean_absolute_error",
    cv=3, n_jobs=-1, ax=ax)
_ = ax.set_title("Validation curve — RandomForest (n_estimators)")
```

<!-- #region -->
### 5.4 Boosting
<!-- #endregion -->

<!-- #region -->
Le boosting construit les arbres **séquentiellement**. Le **gradient boosting** ajuste chaque nouvel arbre sur les **résidus** (le gradient de la perte) du modèle courant : $F_{m}(x) = F_{m-1}(x) + \nu\, h_m(x)$, où $h_m$ approxime $-\partial L / \partial F$ et $\nu$ est le taux d'apprentissage. AdaBoost, lui, repondère les exemples mal prédits — et **sur-apprend** quand `n_estimators` devient grand (à comparer avec la forêt ci-dessus).
<!-- #endregion -->

```python
from sklearn.ensemble import AdaBoostRegressor

ada_range = np.unique(np.logspace(0, 1.8, num=10).astype(int))
fig, ax = plt.subplots(figsize=(7, 4))
ValidationCurveDisplay.from_estimator(
    AdaBoostRegressor(random_state=RANDOM_STATE), data_sub, target_sub,
    param_name="n_estimators", param_range=ada_range,
    scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, ax=ax)
_ = ax.set_title("Validation curve — AdaBoost (n_estimators)")
```

<!-- #region -->
### 5.5 HistGradientBoosting (recommandé en 2026)
<!-- #endregion -->

<!-- #region -->
`HistGradientBoostingRegressor` discrétise chaque feature en **histogrammes** (256 *bins* par défaut), ce qui réduit drastiquement le nombre de splits à évaluer. Résultat : beaucoup plus rapide que `GradientBoostingRegressor` sur les gros jeux (>10 000 échantillons), à performance comparable, avec support natif des catégorielles et des valeurs manquantes. **C'est le choix par défaut moderne pour le tabulaire** (équivalent LightGBM dans scikit-learn). Le tableau ci-dessous compare temps et MAE.
<!-- #endregion -->

```python
from sklearn.ensemble import GradientBoostingRegressor

rows = []
for name, model in [
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    ("HistGradientBoosting", HistGradientBoostingRegressor(max_iter=100, random_state=RANDOM_STATE)),
]:
    res = cross_validate(model, data, target, scoring="neg_mean_absolute_error",
                         cv=5, n_jobs=-1)
    rows.append({"modèle": name,
                 "MAE": round(-res["test_score"].mean(), 2),
                 "fit_time_s": round(res["fit_time"].mean(), 2)})
comp_boost = pd.DataFrame(rows)
comp_boost
```

<!-- #region -->
### 5.6 Nested cross-validation
<!-- #endregion -->

<!-- #region -->
Tuner les hyperparamètres **et** estimer la performance sur les mêmes plis biaise l'estimation (optimisme). La **validation croisée imbriquée** sépare les deux : une CV **interne** (`GridSearchCV`) choisit les hyperparamètres, une CV **externe** (`cross_validate`) estime la performance généralisée. On inspecte les meilleurs paramètres par pli externe.
<!-- #endregion -->

```python
from sklearn.model_selection import GridSearchCV

hist_grid = {"max_depth": [3, 8], "learning_rate": [0.1, 0.5]}
inner_search = GridSearchCV(
    HistGradientBoostingRegressor(max_iter=200, early_stopping=True,
                                  random_state=RANDOM_STATE),
    hist_grid, scoring="neg_mean_absolute_error", n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
nested = cross_validate(inner_search, data, target, cv=outer_cv,
                        scoring="r2", return_estimator=True, n_jobs=-1)
print(f"R2 nested: {nested['test_score'].mean():.3f} ± {nested['test_score'].std():.3f}")
for i, est in enumerate(nested["estimator"]):
    print(f"  fold {i}: best={est.best_params_}")
```

<!-- #region -->
On agrège les scores de la CV **interne** pour visualiser quelles combinaisons d'hyperparamètres sont robustes d'un pli externe à l'autre. Des combinaisons proches indiquent que le choix exact importe peu.
<!-- #endregion -->

```python
index_cols = [f"param_{n}" for n in hist_grid]
inner_frames = []
for cv_idx, est in enumerate(nested["estimator"]):
    df_in = pd.DataFrame(est.cv_results_)[index_cols + ["mean_test_score"]]
    df_in = df_in.set_index(index_cols).rename(columns={"mean_test_score": f"CV{cv_idx}"})
    inner_frames.append(df_in)
inner_cv_results = pd.concat(inner_frames, axis=1).T

fig, ax = plt.subplots(figsize=(8, 4))
inner_cv_results.plot.box(vert=False, ax=ax,
                          color={"whiskers": "black", "medians": "black", "caps": "black"})
ax.set_title("Nested CV — scores internes par combinaison d'hyperparamètres")
_ = ax.set_xlabel("neg MAE")
```

<!-- #region -->
## 6. Sélection de caractéristiques
<!-- #endregion -->

<!-- #region -->
Trois familles de méthodes :

| Famille | Principe | Exemple sklearn |
|---|---|---|
| **Filtre** | score statistique feature↔cible, indépendant du modèle | `SelectKBest(f_classif)` |
| **Embedded** | la sélection découle du modèle (coefs, importances) | `SelectFromModel` |
| **Wrapper** | on entraîne/évalue des sous-ensembles de features | `RFECV` |

Objectif principal : **gagner du temps** et réduire le bruit — les modèles modernes (GBM) ignorent déjà assez bien les features non informatives.
<!-- #endregion -->

<!-- #region -->
### 6.1 Filtre univarié — SelectKBest
<!-- #endregion -->

<!-- #region -->
`SelectKBest(f_classif, k=5)` garde les 5 features au plus fort score F de l'ANOVA. Rapide, mais **univarié** : il ignore les interactions entre features. On le place **dans une pipeline** pour qu'il soit ré-appris à chaque fold.
<!-- #endregion -->

```python
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier

# sous-échantillon classification pour les parties lentes
Xc = X_clf[:2000]
yc = y_clf[:2000]

clf_baseline = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE)
pipe_kbest = make_pipeline(SelectKBest(score_func=f_classif, k=5),
                           RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                                  random_state=RANDOM_STATE))
print("baseline   :", evaluate_cv(clf_baseline, Xc, yc, scoring="accuracy", cv=5))
print("k=5 (filter):", evaluate_cv(pipe_kbest, Xc, yc, scoring="accuracy", cv=5))
```

<!-- #region -->
**Le piège du *data leak***. Sélectionner les features sur **toutes** les données *avant* la cross-validation laisse fuiter l'information du test dans l'entraînement. On le démontre sur du **bruit pur** avec une cible aléatoire : la vraie accuracy doit être ~0,5. La sélection hors-CV gonfle artificiellement le score ; la même sélection encapsulée dans une pipeline reste honnête.
<!-- #endregion -->

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as _LogReg

# Cas d'école : QUE du bruit + cible aléatoire -> la vraie accuracy doit être ~0.5.
rng = np.random.RandomState(RANDOM_STATE)
X_noise = rng.randn(200, 5000)          # 200 échantillons, 5000 features de bruit
y_noise = rng.randint(0, 2, size=200)   # cible sans aucun lien avec X

# MAUVAIS : sélection des "meilleures" features sur TOUTES les données avant la CV
selector_leak = SelectKBest(score_func=f_classif, k=20)
X_noise_leak = selector_leak.fit_transform(X_noise, y_noise)
score_leak = cross_val_score(_LogReg(max_iter=1000), X_noise_leak, y_noise, cv=5).mean()
# BON : sélection encapsulée dans la pipeline (refaite sur chaque fold d'entraînement)
pipe_clean = make_pipeline(SelectKBest(f_classif, k=20), _LogReg(max_iter=1000))
score_clean = cross_val_score(pipe_clean, X_noise, y_noise, cv=5).mean()
print(f"score 'leak' (sélection hors CV)  : {score_leak:.3f}  <- gonflé artificiellement")
print(f"score propre (sélection dans pipe): {score_clean:.3f}  <- ~chance, honnête")
```

<!-- #region -->
### 6.2 Embedded — SelectFromModel
<!-- #endregion -->

<!-- #region -->
`SelectFromModel` utilise les `feature_importances_` (ou coefficients) d'un modèle pour seuiller les features. On compare, via box plot, les scores avec et sans sélection.
<!-- #endregion -->

```python
pipe_sfm = make_pipeline(
    SelectFromModel(RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                           random_state=RANDOM_STATE)),
    RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE))
res_base = pd.DataFrame(cross_validate(clf_baseline, Xc, yc, cv=5, n_jobs=-1))
res_sfm = pd.DataFrame(cross_validate(pipe_sfm, Xc, yc, cv=5, n_jobs=-1))
cmp_sfm = pd.concat([res_base, res_sfm], axis=1,
                    keys=["sans sélection", "SelectFromModel"]).swaplevel(axis="columns")
fig, ax = plt.subplots(figsize=(7, 4))
cmp_sfm["test_score"].plot.box(vert=False, ax=ax,
                               color={"whiskers": "black", "medians": "black", "caps": "black"})
ax.set_xlabel("accuracy")
_ = ax.set_title("SelectFromModel vs sans sélection")
```

<!-- #region -->
### 6.3 Wrapper — RFECV
<!-- #endregion -->

<!-- #region -->
La **Recursive Feature Elimination** retire itérativement la feature la moins importante ; `RFECV` choisit automatiquement le nombre optimal de features par cross-validation. **Note API (sklearn 1.8)** : on lit la courbe via `rfecv.cv_results_["mean_test_score"]` (l'attribut `grid_scores_` du MOOC a été supprimé en 1.2).
<!-- #endregion -->

```python
rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE),
    step=1, cv=StratifiedKFold(3), scoring="accuracy", min_features_to_select=3, n_jobs=-1)
rfecv.fit(Xc, yc)
print("nb features optimal:", rfecv.n_features_)
mean_scores = rfecv.cv_results_["mean_test_score"]
n_feats = range(rfecv.min_features_to_select,
                rfecv.min_features_to_select + len(mean_scores))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(n_feats), mean_scores, marker="o", color=PRIMARY_1)
ax.set_xlabel("Nombre de caractéristiques")
ax.set_ylabel("Accuracy (CV)")
_ = ax.set_title("RFECV — score vs nombre de features")
```

<!-- #region -->
## 7. Interprétation des modèles
<!-- #endregion -->

<!-- #region -->
Quatre angles complémentaires : les **coefficients** (modèles linéaires), l'**importance par impureté** (arbres), l'**importance par permutation** (agnostique) et les **dépendances partielles** (effet marginal).
<!-- #endregion -->

<!-- #region -->
### 7.1 Coefficients des modèles linéaires
<!-- #endregion -->

<!-- #region -->
Dans un modèle linéaire, la cible est une combinaison linéaire des features ; chaque coefficient mesure l'effet d'une feature **toutes choses égales par ailleurs** (dépendance conditionnelle). **Attention** : comparer des coefficients n'a de sens que si les features sont à la **même échelle** — on affiche côte à côte les coefficients de Ridge et l'écart-type des features pour le montrer.
<!-- #endregion -->

```python
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.model_selection import RepeatedKFold

ridge = RidgeCV()
ridge.fit(data_train, target_train)
coefs_ridge = pd.DataFrame(ridge.coef_, columns=["Coef"], index=data.columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
coefs_ridge.plot.barh(ax=axes[0], color=PRIMARY_1, legend=False)
axes[0].set_title("Coefficients Ridge (données brutes)")
axes[0].axvline(0, color=".5")
data_train.std().plot.barh(ax=axes[1], color=BEIGE)
axes[1].set_title("Écart-type des features")
print("R2 test:", round(ridge.score(data_test, target_test), 3))
```

<!-- #region -->
Le **Lasso** (pénalité L1) pousse certains coefficients à exactement zéro : il fait donc une **sélection de features**. En inspectant les coefficients sur les plis d'une `RepeatedKFold`, on évalue leur **stabilité** : des coefficients très variables d'un pli à l'autre (features corrélées) ne doivent pas être sur-interprétés.
<!-- #endregion -->

```python
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.015, random_state=RANDOM_STATE))
cv_lasso = cross_validate(lasso, data_train, target_train,
                          cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE),
                          return_estimator=True, n_jobs=-1)
coefs_cv = pd.DataFrame([est[-1].coef_ for est in cv_lasso["estimator"]],
                        columns=data.columns)
fig, ax = plt.subplots(figsize=(9, 4))
sns.boxplot(data=coefs_cv, orient="h", color=ACCENT, saturation=0.6, ax=ax)
ax.axvline(0, color=".5")
ax.set_title("Lasso — stabilité des coefficients sur les folds CV")
_ = ax.set_xlabel("Importance du coefficient")
```

<!-- #region -->
### 7.2 Importance par impureté (MDI)
<!-- #endregion -->

<!-- #region -->
Pour les arbres, l'importance d'une feature = réduction totale (normalisée) du critère d'impureté qu'elle apporte sur l'ensemble des splits. **Biais connu** : la MDI surestime les features à **haute cardinalité** et continues. À recouper avec la permutation (§7.3).
<!-- #endregion -->

```python
rf_reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
rf_reg.fit(data_train, target_train)
imp = pd.Series(rf_reg.feature_importances_, index=data.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
imp.plot.barh(ax=ax, color=PRIMARY_1)
_ = ax.set_title("Importance par impureté (MDI) — RandomForest")
```

<!-- #region -->
### 7.3 Importance par permutation
<!-- #endregion -->

<!-- #region -->
Principe **agnostique au modèle** : on permute aléatoirement les valeurs d'une feature et on mesure la **chute de score**. Plus la chute est grande, plus la feature est importante. On en donne une implémentation pédagogique (clés cohérentes), puis on la compare à `sklearn.inspection.permutation_importance`. Limite : sur features corrélées, la permutation crée des combinaisons irréalistes.
<!-- #endregion -->

```python
from sklearn.inspection import permutation_importance


def permutation_importance_maison(model, X, y, n_repeats: int = 5) -> dict:
    """Importance par permutation, version pédagogique (clés cohérentes).

    Pour chaque colonne : on permute ses valeurs et on mesure la chute de score.
    Retourne {'importances_mean', 'importances_std'}.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    baseline = model.score(X, y)
    means, stds = [], []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(baseline - model.score(Xp, y))
        means.append(np.mean(drops))
        stds.append(np.std(drops))
    return {"importances_mean": np.array(means), "importances_std": np.array(stds)}
```

<!-- #region -->
On applique les deux versions sur un sous-échantillon du test et on vérifie qu'elles concordent (corrélation des moyennes proche de 1).
<!-- #endregion -->

```python
Xte_sub = data_test.iloc[:2000]
yte_sub = target_test.iloc[:2000]
home = permutation_importance_maison(rf_reg, Xte_sub, yte_sub, n_repeats=5)
sk = permutation_importance(rf_reg, Xte_sub, yte_sub, n_repeats=5,
                            random_state=RANDOM_STATE, n_jobs=-1)
order = np.argsort(sk.importances_mean)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(np.array(data.columns)[order], sk.importances_mean[order],
        xerr=sk.importances_std[order], color=ACCENT)
ax.set_title("Permutation importance (sklearn.inspection)")
print("maison vs sklearn (corrélation des moyennes):",
      round(np.corrcoef(home["importances_mean"], sk.importances_mean)[0, 1], 3))
```

<!-- #region -->
### 7.4 Partial Dependence & ICE
<!-- #endregion -->

<!-- #region -->
Le **Partial Dependence Plot (PDP)** montre l'effet **marginal moyen** d'une feature sur la prédiction ; les courbes **ICE** (Individual Conditional Expectation) montrent cet effet échantillon par échantillon — utiles pour repérer des effets hétérogènes que la moyenne masque.
<!-- #endregion -->

```python
from sklearn.inspection import PartialDependenceDisplay

hist_reg = HistGradientBoostingRegressor(max_iter=100, random_state=RANDOM_STATE)
hist_reg.fit(data_train, target_train)
fig, ax = plt.subplots(figsize=(11, 4))
PartialDependenceDisplay.from_estimator(
    hist_reg, data_train, features=["MedInc", "AveRooms"],
    kind="both", subsample=200, random_state=RANDOM_STATE, ax=ax)
_ = fig.suptitle("Partial Dependence + ICE — HistGBDT")
```

<!-- #region -->
## 8. Métriques
<!-- #endregion -->

<!-- #region -->
L'erreur classique du débutant : utiliser l'accuracy partout. Le choix de la métrique dépend de la tâche et de la distribution.

| Tâche | Métrique recommandée |
|---|---|
| Régression | MAE, RMSE, R² |
| Classif binaire équilibrée | Accuracy, F1, ROC-AUC |
| Classif déséquilibrée | F1, PR-AUC, MCC |
| Probabilités calibrées | Log-loss, Brier, courbe de calibration |
| Multi-classe | Macro/Weighted F1 |

On passe les métriques via `scoring=` (liste des noms disponibles : `sklearn.metrics.get_scorer_names()`).
<!-- #endregion -->

```python
multi = cross_validate(hist_reg, data, target,
                       scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
                       cv=5, n_jobs=-1)
metrics_reg = pd.DataFrame({
    "MAE": [-multi["test_neg_mean_absolute_error"].mean()],
    "RMSE": [-multi["test_neg_root_mean_squared_error"].mean()],
    "R2": [multi["test_r2"].mean()],
}).round(3)
metrics_reg
```

<!-- #region -->
Pour la classification, on visualise trois diagnostics complémentaires sur le Titanic : la **courbe ROC** (sensibilité vs taux de faux positifs), la **courbe Précision-Rappel** (plus informative en déséquilibre) et la **courbe de calibration** (les probabilités prédites correspondent-elles aux fréquences observées ?).
<!-- #endregion -->

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
    X_cat, y_cat, random_state=RANDOM_STATE, stratify=y_cat)
clf_titanic = Pipeline([("prep", build_titanic_preprocessor()),
                        ("clf", LogisticRegression(max_iter=1000))])
clf_titanic.fit(Xtr_t, ytr_t)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
RocCurveDisplay.from_estimator(clf_titanic, Xte_t, yte_t, ax=axes[0], color=PRIMARY_1)
axes[0].set_title("ROC")
PrecisionRecallDisplay.from_estimator(clf_titanic, Xte_t, yte_t, ax=axes[1], color=ACCENT)
axes[1].set_title("Precision-Recall")
CalibrationDisplay.from_estimator(clf_titanic, Xte_t, yte_t, ax=axes[2], n_bins=8)
_ = axes[2].set_title("Courbe de calibration")
```

<!-- #region -->
## 9. Encodage des variables catégorielles
<!-- #endregion -->

<!-- #region -->
Le brouillon original contenait une erreur fréquente — voici la règle **correcte** :

| Encodeur | Quand l'utiliser |
|---|---|
| `OneHotEncoder` | variables **nominales** (sans ordre) de cardinalité faible ; modèles linéaires / NN |
| `OrdinalEncoder` | variables **ordinales** (avec ordre) ; ou pour les arbres qui s'en accommodent |
| `TargetEncoder` (1.3+) | **haute cardinalité** ; encode par la moyenne de la cible, avec CV interne anti-fuite |

On compare les trois sur les colonnes catégorielles du Titanic.
<!-- #endregion -->

```python
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

X_cat_only = titanic[CAT_CAT].copy()


def encoder_pipeline(encoder) -> Pipeline:
    """Pipeline : imputation catégorielle -> encoder -> LogisticRegression."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", encoder),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


encoders = {
    "OneHot (nominal)": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "Ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "Target": TargetEncoder(random_state=RANDOM_STATE),
}
rows = []
for name, enc in encoders.items():
    sc = cross_val_score(encoder_pipeline(enc), X_cat_only, y_cat, cv=5).mean()
    rows.append({"encodage": name, "accuracy": round(sc, 3)})
enc_cmp = pd.DataFrame(rows)
enc_cmp
```

<!-- #region -->
## 10. Nouveautés scikit-learn 1.8 (2026)
<!-- #endregion -->

<!-- #region -->
scikit-learn **1.8.0** (10 décembre 2025) supporte Python 3.11 à 3.14, y compris le CPython *free-threaded* (sans GIL). Trois ajouts récents méritent d'être connus.
<!-- #endregion -->

<!-- #region -->
### 10.1 TunedThresholdClassifierCV
<!-- #endregion -->

<!-- #region -->
Par défaut un classifieur tranche à une probabilité de 0,5. `TunedThresholdClassifierCV` **post-optimise ce seuil** par cross-validation selon une métrique métier (ici F1) — précieux en cas de classes déséquilibrées ou de coûts asymétriques.
<!-- #endregion -->

```python
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import f1_score

base_clf = Pipeline([("prep", build_titanic_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr_t, ytr_t)
tuned = TunedThresholdClassifierCV(
    Pipeline([("prep", build_titanic_preprocessor()),
              ("clf", LogisticRegression(max_iter=1000))]),
    scoring="f1", cv=5).fit(Xtr_t, ytr_t)
print("seuil par défaut (0.5)  F1:", round(f1_score(yte_t, base_clf.predict(Xte_t)), 3))
print(f"seuil optimisé ({tuned.best_threshold_:.2f}) F1:",
      round(f1_score(yte_t, tuned.predict(Xte_t)), 3))
```

<!-- #region -->
### 10.2 FrozenEstimator
<!-- #endregion -->

<!-- #region -->
`FrozenEstimator` **gèle** un estimateur déjà entraîné : un méta-estimateur (ici `FixedThresholdClassifier` à seuil fixe) peut l'utiliser **sans le ré-entraîner**. Utile pour fixer un seuil après coup, ou pour le *stacking* avec des modèles pré-entraînés.
<!-- #endregion -->

```python
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import FixedThresholdClassifier

frozen = FixedThresholdClassifier(FrozenEstimator(base_clf), threshold=0.3)
frozen.fit(Xtr_t, ytr_t)  # ne réentraîne PAS base_clf (gelé)
print("FixedThreshold(0.3) F1:", round(f1_score(yte_t, frozen.predict(Xte_t)), 3))
```

<!-- #region -->
### 10.3 Metadata routing
<!-- #endregion -->

<!-- #region -->
Le **metadata routing** (activé via `enable_metadata_routing`) permet d'acheminer proprement des métadonnées — comme `sample_weight` ou `groups` — à travers les méta-estimateurs (Pipeline, `cross_validate`…). On déclare l'intention avec `.set_fit_request(sample_weight=True)`, puis on passe la donnée via `params=`.
<!-- #endregion -->

```python
from sklearn import config_context

with config_context(enable_metadata_routing=True):
    weighted_clf = HistGradientBoostingRegressor(
        max_iter=50, random_state=RANDOM_STATE).set_fit_request(sample_weight=True)
    sw = np.ones(len(data_train))
    sw[target_train > target_train.median()] = 2.0  # pondère les logements chers
    routed = cross_validate(weighted_clf, data_train, target_train,
                            params={"sample_weight": sw}, cv=3, scoring="r2", n_jobs=-1)
print(f"R2 (sample_weight routé via cross_validate): {routed['test_score'].mean():.3f}")
```

<!-- #region -->
## 11. Récapitulatif
<!-- #endregion -->

<!-- #region -->
À retenir de ce tour de scikit-learn :

- **Pipeline partout** : encapsuler prétraitement + modèle évite les fuites et simplifie le tuning.
- **CV rigoureuse** : moyenne ± écart-type, splitter adapté, validation imbriquée pour ne pas biaiser l'estimation.
- **HistGradientBoosting** est le défaut tabulaire moderne ; RandomForest reste une baseline robuste.
- **Interpréter avec plusieurs angles** : coefficients (à features standardisées), impureté (biaisée), permutation (agnostique), PDP/ICE.
- **Choisir la métrique** selon la tâche ; calibrer et tuner le seuil si besoin.

Notebooks frères de ce projet pour approfondir :

- `ML_Regression_Classification_CV_GridSearch` — bench et tuning détaillés.
- `ML_Bagging_Boosting` — ensembles modernes (XGBoost, LightGBM).
- `ML_Explication_Feature_Importance_Selection` — SHAP, RFECV, Boruta.
- `ML_Optimisation_de_Modèles` — Optuna.
- `ML_MLFlow_Bench` — tracking et MLOps.

Référence : [MOOC INRIA scikit-learn](https://inria.github.io/scikit-learn-mooc/).
<!-- #endregion -->
