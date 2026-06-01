---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="lbMiM4mqoHbD" -->
# A compléter
<!-- #endregion -->

<!-- #region id="n7WQMBXNq_5a" -->
# Modèles ensembliste
<!-- #endregion -->

<!-- #region id="YcrQT-SHsRtp" -->
On prendra le jeu de données de logements californiens et le diviser en un jeu d'entraînement et un jeu de test.
<!-- #endregion -->

```python id="WLneZ3dupqk_"
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1640684490311, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="Q7KKvlfEpyZV" outputId="d722dcc1-f3fa-4b9e-c935-badd3c84797a"
data.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 266, "status": "ok", "timestamp": 1640684492734, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="xc7BPOMSp74k" outputId="76b65880-4905-488b-a337-030d363dfb41"
target.head()
```

```python id="GPeDZjOQqCGA"
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)
```

```python id="_LH9nS0QJiai"
test du dataframe , avec la target qui change dans le cadre de la boubles suivant les liste [DBScan , Neighbor Joining , UPGMA sans que ce lui ci dner perdure dans le temps]
```

<!-- #region id="Yjrulzu6tJVS" -->
##Random Forest
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 630, "status": "ok", "timestamp": 1640684684824, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="c64OMP4preDh" outputId="b6d143fd-5295-49d3-9d54-bb6aa01f1060"
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=3)
forest.fit(data_train, target_train)
target_predicted = forest.predict(data_test)
print(f"Mean absolute error: "
      f"{mean_absolute_error(target_test, target_predicted):.3f} grams")
```

```python id="VSx48EfUca-s"
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
train_scores, test_scores = validation_curve(
    forest, data_train, target_train,
    param_name="n_estimators", param_range=param_range,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_errors, test_errors = -train_scores, -test_scores
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"elapsed": 649, "status": "ok", "timestamp": 1640685105677, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="5O8hJxric2aY" outputId="473327ba-4c84-4931-e128-57ccf4a2be23"
plt.errorbar(param_range, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training score",
             alpha=0.7)
plt.errorbar(param_range, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Cross-validation score",
             alpha=0.7)

plt.legend()
plt.ylabel("Mean absolute error in k$\n(smaller is better)")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for RandomForest regressor")
```

<!-- #region id="m1RKvPm6dM0S" -->
Contrairement au régresseur AdaBoost, nous pouvons constater que l'augmentation du nombre d'arbres dans la forêt augmente la performance de généralisation (en diminuant l'erreur absolue moyenne) de la forêt aléatoire. En fait, une forêt aléatoire a moins de chance de souffrir de surajustement qu'AdaBoost lorsque le nombre d'estimateurs augmente.
<!-- #endregion -->

<!-- #region id="b06_OChgepUC" -->
#### Cross Validation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 93461, "status": "ok", "timestamp": 1640685650668, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="Ac9oy7oAesKc" outputId="3040c190-7931-44b9-bd4c-3f0bce902a80"
random_forest = RandomForestRegressor(n_estimators=200, n_jobs=2)
cv_results_rf = cross_validate(
    random_forest, data, target, scoring="neg_mean_absolute_error",
    n_jobs=2,
)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1640685650669, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="J7roud-aevUI" outputId="7abb3f2f-f23d-41c0-dc30-dd60b38a58b4"
print("Random Forest")
print(f"Mean absolute error via cross-validation: "
      f"{-cv_results_rf['test_score'].mean():.3f} +/- "
      f"{cv_results_rf['test_score'].std():.3f} k$")
print(f"Average fit time: "
      f"{cv_results_rf['fit_time'].mean():.3f} seconds")
print(f"Average score time: "
      f"{cv_results_rf['score_time'].mean():.3f} seconds")
```

<!-- #region id="WrwCoUQJesxy" -->

<!-- #endregion -->

```python id="g6D6Agp6tBLJ"
tree_predictions = []
for tree in forest.estimators_:
    tree_predictions.append(tree.predict(data_test))

forest_predictions = forest.predict(data_test)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 239, "status": "ok", "timestamp": 1640638571445, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="uJgBK42jrOsY" outputId="48ac8ae2-9bca-4e7e-ad28-b90f22b6b7e1"
target_predicted = search.predict(data_test)
print(f"Erreur absolue moyenne après réglage du bagging regressorc:\n"
      f"{mean_absolute_error(target_test, target_predicted):.2f} k$")
```

<!-- #region id="KqOyTgVJakoe" -->
## Bagging
<!-- #endregion -->

```python id="JrwUrOEwqPpV"
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1984, "status": "ok", "timestamp": 1640684503181, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="kbeUHFC9qItJ" outputId="2dd7eae8-07cd-4adc-abee-a15ee9451a3b"
tree = DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator=tree, n_jobs=2)
bagging.fit(data_train, target_train)
target_predicted = bagging.predict(data_test)
print(f"Mean absolute error sur le bagging regressor:\n"
      f"{mean_absolute_error(target_test, target_predicted):.2f} k$")
```

<!-- #region id="QSmZseibatuR" -->
### Recherche de Paramètres Optimum
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 260, "status": "ok", "timestamp": 1640684505103, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="LsP3cDlSqScW" outputId="209166f8-756f-447f-e73b-4f19a02509d8"
for param in bagging.get_params().keys():
    print(param)
```

```python id="VYCOwCeyq8-g"
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
```

```python id="-XTCioqFqhfB"
param_grid = {
    "n_estimators": randint(10, 30),
    "max_samples": [0.5, 0.8, 1.0],
    "max_features": [0.5, 0.8, 1.0],
    "base_estimator__max_depth": randint(3, 10),
}
search = RandomizedSearchCV(
    bagging, param_grid, n_iter=20, scoring="neg_mean_absolute_error"
)
_ = search.fit(data_train, target_train)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 253, "status": "ok", "timestamp": 1640638544899, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="DZV0J_XgrFbN" outputId="3e875dc5-25b6-476c-dd16-fbbf0a46ed55"
import pandas as pd

columns = [f"param_{name}" for name in param_grid.keys()]
columns += ["mean_test_score", "std_test_score", "rank_test_score"]
cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results[columns].sort_values(by="rank_test_score")
cv_results["mean_test_score"] = -cv_results["mean_test_score"]
cv_results.head()
```

<!-- #region id="7huE0h-rr4j1" -->
On constate que le prédicteur fourni par le bagging égresseur ne nécessite pas beaucoup d'être tuné avec des hyperparamètres par rapport à un arbre de décision simple.
<!-- #endregion -->

<!-- #region id="3PPpY9p2aXOf" -->
## Boosting
<!-- #endregion -->

```python id="MEZ07UZvcBQG"
### Ada Boost
```

```python id="vma6dIyebqGr"
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import validation_curve

adaboost = AdaBoostRegressor()
param_range = np.unique(np.logspace(0, 1.8, num=30).astype(int))
train_scores, test_scores = validation_curve(
    adaboost, data_train, target_train,
    param_name="n_estimators", param_range=param_range,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_errors, test_errors = -train_scores, -test_scores
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"elapsed": 542, "status": "ok", "timestamp": 1640684920903, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="6diNL8X1cH06" outputId="04b56307-0ed8-4ab2-d1cf-d8f4734e3a66"
import matplotlib.pyplot as plt

plt.errorbar(param_range, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training score",
             alpha=0.7)
plt.errorbar(param_range, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Cross-validation score",
             alpha=0.7)

plt.legend()
plt.ylabel("Mean absolute error in k$\n(smaller is better)")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for AdaBoost regressor")
```

<!-- #region id="LO_4L5d5df5c" -->
### Gradient-boosting decision tree (GBDT)
<!-- #endregion -->

```python id="HExO8mU4d8RG"
from sklearn.model_selection import cross_validate
```

```python id="ReBgyp3vcJIy"
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting = GradientBoostingRegressor(n_estimators=200)
cv_results_gbdt = cross_validate(
    gradient_boosting, data, target, scoring="neg_mean_absolute_error",
    n_jobs=2,
)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1640685395071, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="wVJ5ZR8Cd5Ky" outputId="fe5d59d0-800d-445f-8dd5-9881ffacd1f6"
print("Gradient Boosting Decision Tree")
print(f"Mean absolute error via cross-validation: "
      f"{-cv_results_gbdt['test_score'].mean():.3f} +/- "
      f"{cv_results_gbdt['test_score'].std():.3f} k$")
print(f"Average fit time: "
      f"{cv_results_gbdt['fit_time'].mean():.3f} seconds")
print(f"Average score time: "
      f"{cv_results_gbdt['score_time'].mean():.3f} seconds")
```

```python id="4W2VKAKleC90"
gbdt = GradientBoostingRegressor(max_depth=5, learning_rate=0.5)
from sklearn.model_selection import validation_curve

param_range = [1, 2, 5, 10, 20, 50, 100]
gbdt_train_scores, gbdt_test_scores = validation_curve(
    gbdt,
    data_train,
    target_train,
    param_name="n_estimators",
    param_range=param_range,
    scoring="neg_mean_absolute_error",
    n_jobs=2,
)
gbdt_train_errors, gbdt_test_errors = -gbdt_train_scores, -gbdt_test_scores
```

```python colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1640685809669, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="dCjVDSb0fkXN" outputId="2a1ed52c-1c9b-4f9d-e3d7-087f468bdbbc"
import matplotlib.pyplot as plt

plt.errorbar(
    param_range,
    gbdt_train_errors.mean(axis=1),
    yerr=gbdt_train_errors.std(axis=1),
    label="Training",
)
plt.errorbar(
    param_range,
    gbdt_test_errors.mean(axis=1),
    yerr=gbdt_test_errors.std(axis=1),
    label="Cross-validation",
)

plt.legend()
plt.ylabel("Mean absolute error in k$\n(smaller is better)")
plt.xlabel("# estimators")
_ = plt.title("Validation curve for GBDT regressor")
```

<!-- #region id="QdYieVYXf4mf" -->
###Speeding-up gradient-boosting
<!-- #endregion -->

<!-- #region id="JjmxVNcJgLuZ" -->
Présentation d'une version modifiée du gradient boosting qui utilise un nombre réduit de splits lors de la construction des différents arbres. Cet algorithme est appelé "histogramme gradient boosting" dans scikit-learn.

Nous avons mentionné précédemment que la forêt aléatoire est un algorithme efficace puisque chaque arbre de l'ensemble peut être ajusté en même temps de manière indépendante. Par conséquent, l'algorithme évolue efficacement avec le nombre de cœurs (paralellisation) et le nombre d'échantillons.

L'algorithme de gradient-boosting est un algorithme séquentiel. Il faut que les arbres N-1 aient été ajustés pour pouvoir ajuster l'arbre à l'étape N. Par conséquent, l'algorithme est assez coûteux en termes de calcul. La partie la plus coûteuse de cet algorithme est la recherche de la meilleure division de l'arbre, qui est une approche par force brute : toutes les divisions possibles sont évaluées et la meilleure est choisie. Nous avons expliqué ce processus dans le cahier "tree in depth", auquel vous pouvez vous référer.

Pour accélérer l'algorithme de gradient-boosting, on peut réduire le nombre de divisions à évaluer. En conséquence, les performances de généralisation d'un tel arbre seraient réduites. Cependant, puisque nous combinons plusieurs arbres dans un gradient-boosting, nous pouvons ajouter plus d'estimateurs pour surmonter ce problème.

Nous allons réaliser une implémentation naïve de cet algorithme en utilisant des blocs de construction de Scikit-learn. Tout d'abord, nous allons charger l'ensemble de données sur le logement en Californie.
<!-- #endregion -->

```python id="ppMuxtpGfk4e"
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor

gradient_boosting = GradientBoostingRegressor(n_estimators=200)
cv_results_gbdt = cross_validate(
    gradient_boosting, data, target, scoring="neg_mean_absolute_error",
    n_jobs=2
)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1640686190609, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="apfsdrb6g7tu" outputId="ac9940e5-3866-4505-a5d6-77bb48eb2850"
print("Gradient Boosting Decision Tree")
print(f"Mean absolute error via cross-validation: "
      f"{-cv_results_gbdt['test_score'].mean():.3f} +/- "
      f"{cv_results_gbdt['test_score'].std():.3f} k$")
print(f"Average fit time: "
      f"{cv_results_gbdt['fit_time'].mean():.3f} seconds")
print(f"Average score time: "
      f"{cv_results_gbdt['score_time'].mean():.3f} seconds")
```

<!-- #region id="Uj9dgFIXhTl2" -->
Une façon d'accélérer le gradient boosting est de réduire le nombre de split prises en compte dans la construction de l'arbre. Une façon de le faire est de bin les données avant de les soumettre au boosting de gradient. Un transformateur appelé KBinsDiscretizer effectue une telle transformation. Ainsi, nous pouvons canaliser ce prétraitement avec le boosting du gradient.

Nous pouvons d'abord démontrer la transformation effectuée par le KBinsDiscretizer.
<!-- #endregion -->

<!-- #region id="R9LZbTeohlNx" -->
#### Bin
<!-- #endregion -->

```python id="HZrPVXgWhm-E"
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(
    n_bins=256, encode="ordinal", strategy="quantile")
data_trans = discretizer.fit_transform(data)
```

<!-- #region id="Y_TjvPeIh3Aq" -->
Les plus petits bins seront supprimés.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 257, "status": "ok", "timestamp": 1640686342047, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="r1n-yDSgg9WW" outputId="ba3d8e98-e89f-4e16-d4da-0a64c2ab6656"
data_trans
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 298, "status": "ok", "timestamp": 1640686421030, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="kcXXmWLqhsyS" outputId="f7ddb4f1-8027-444e-f86c-8d6bd6f1ba9a"
[len(np.unique(col)) for col in data_trans.T]
```

<!-- #region id="Orpm9jUEiRB4" -->
Après cette transformation, nous voyons que nous avons au maximum 256 valeurs uniques par caractéristiques. Maintenant, nous allons utiliser ce transformeur pour discrétiser les données avant d'entraîner le gradient boosting régresseur  .
<!-- #endregion -->

```python id="ZIhyDH7IiABy"
from sklearn.pipeline import make_pipeline

gradient_boosting = make_pipeline(
    discretizer, GradientBoostingRegressor(n_estimators=200))
cv_results_gbdt = cross_validate(
    gradient_boosting, data, target, scoring="neg_mean_absolute_error",
    n_jobs=2,
)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1640686591471, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="qvPIp3wiijop" outputId="2d8330a9-f01c-4214-b86a-0cae4e062d16"
print("Gradient Boosting Decision Tree with KBinsDiscretizer")
print(f"Mean absolute error via cross-validation: "
      f"{-cv_results_gbdt['test_score'].mean():.3f} +/- "
      f"{cv_results_gbdt['test_score'].std():.3f} k$")
print(f"Average fit time: "
      f"{cv_results_gbdt['fit_time'].mean():.3f} seconds")
print(f"Average score time: "
      f"{cv_results_gbdt['score_time'].mean():.3f} seconds")
```

<!-- #region id="tdAuJt23iqPZ" -->
* Ici, nous constatons que le temps d'entrainement a été considérablement réduit mais que les performances de généralisation du modèle sont identiques.  
* Scikit-learn fournit des classes spécifiques qui sont encore plus optimisées pour les grands ensembles de données, appelées **HistGradientBoostingClassifier** et **HistGradientBoostingRegressor**.  
* Chaque caractéristique des données de l'ensemble de données est d'abord classée en calculant des histogrammes, qui sont ensuite utilisés pour évaluer les divisions potentielles.  
* Le nombre de divisions à évaluer est alors beaucoup plus faible. Cet algorithme devient beaucoup plus efficace que le gradient boosting lorsque le jeu de données contient plus de 10 000 échantillons.

Nous donnons ci-dessous un exemple pour un grand ensemble de données et nous comparons les temps de calcul avec l'expérience de la section précédente.

<!-- #endregion -->

```python id="mxbPA93tikev"
from sklearn.ensemble import HistGradientBoostingRegressor

histogram_gradient_boosting = HistGradientBoostingRegressor(
    max_iter=200, random_state=0)
cv_results_hgbdt = cross_validate(
    histogram_gradient_boosting, data, target,
    scoring="neg_mean_absolute_error", n_jobs=2,
)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1640686752876, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="pVc0-0WgjO6E" outputId="3d3e820f-b09e-42c8-f2fe-bac6950dd137"
print("Histogram Gradient Boosting Decision Tree")
print(f"Mean absolute error via cross-validation: "
      f"{-cv_results_hgbdt['test_score'].mean():.3f} +/- "
      f"{cv_results_hgbdt['test_score'].std():.3f} k$")
print(f"Average fit time: "
      f"{cv_results_hgbdt['fit_time'].mean():.3f} seconds")
print(f"Average score time: "
      f"{cv_results_hgbdt['score_time'].mean():.3f} seconds")
```

<!-- #region id="NwwJwQzajs4B" -->
#### Avec la Cross Validation
<!-- #endregion -->

```python id="aJ2hyCUzjQTl"
from sklearn.ensemble import HistGradientBoostingRegressor

hist_gbdt = HistGradientBoostingRegressor(
    max_iter=1000, early_stopping=True, random_state=0)
```

```python id="eMH8svFnjwt8"
from sklearn.model_selection import GridSearchCV

params = {
    "max_depth": [3, 8],
    "max_leaf_nodes": [15, 31],
    "learning_rate": [0.1, 1],
}

search = GridSearchCV(hist_gbdt, params)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 276, "status": "ok", "timestamp": 1640686952111, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="YlQh2rYAjzCo" outputId="71d2a86f-f9f5-4860-8a65-9ac02cc15727"
search
```

<!-- #region id="Wiv-bvwalO8R" -->
* **validation croisée** 5 fois
<!-- #endregion -->

```python id="4xzi9s-3kBt4"
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
results = cross_validate(
    search, data, target, cv=cv, return_estimator=True, n_jobs=2)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1640687348325, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="BSS4EGxIlIFS" outputId="e737cd6b-e83a-45b9-9709-5fb7538e0cf0"
print(f"R2 score with cross-validation:\n"
      f"{results['test_score'].mean():.3f} +/- "
      f"{results['test_score'].std():.3f}")
```

<!-- #region id="b9rI2Cnwln5N" -->
Examinons ensuite l'entrée de l'estimateur des résultats et vérifiez les meilleures valeurs des paramètres. Vérifiez également le nombre d'arbres utilisés par le modèle.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 271, "status": "ok", "timestamp": 1640687400331, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="DhMbE3QmlhQd" outputId="fa92d368-bad4-4cd8-ee4c-02f336b42b8a"
for estimator in results["estimator"]:
    print(estimator.best_params_)
    print(f"# trees: {estimator.best_estimator_.n_iter_}")
```

<!-- #region id="kEUcMYr6mDEV" -->
Nous observons que les paramètres varient.  
L'intuitionà à avoir est que les résultats du CV interne sont très proches pour un certain ensemble de paramètres.

L'idée est d'alors inspectez les résultats de la CV interne pour chaque estimateur du CV externe.
Ensuite agréger le score moyen du test pour chaque combinaison de paramètres et faire un box plot de ces scores.
<!-- #endregion -->

```python id="bG9H-KqglvIs"
import pandas as pd

index_columns = [f"param_{name}" for name in params.keys()]
columns = index_columns + ["mean_test_score"]

inner_cv_results = []
for cv_idx, estimator in enumerate(results["estimator"]):
    search_cv_results = pd.DataFrame(estimator.cv_results_)
    search_cv_results = search_cv_results[columns].set_index(index_columns)
    search_cv_results = search_cv_results.rename(
        columns={"mean_test_score": f"CV {cv_idx}"})
    inner_cv_results.append(search_cv_results)
inner_cv_results = pd.concat(inner_cv_results, axis=1).T
```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"elapsed": 302, "status": "ok", "timestamp": 1640687610922, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="0n2VGF3KmiWa" outputId="8d833896-36b0-40e4-86be-13fba2d40a79"
inner_cv_results
```

```python colab={"base_uri": "https://localhost:8080/", "height": 310} executionInfo={"elapsed": 706, "status": "ok", "timestamp": 1640687661278, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="zDmWDeKZmfwh" outputId="2cef4d49-e97e-4ba4-8f48-2df2a1ea9d71"
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
inner_cv_results.plot.box(vert=False, color=color)
plt.xlabel("R2 score")
plt.ylabel("Parameters")
_ = plt.title("Inner CV results with parameters\n"
              "(max_depth, max_leaf_nodes, learning_rate)")
```

<!-- #region id="yv0RkpHWmzmx" -->
Nous voyons que les 4 premiers ensembles de paramètres classés sont très proches. Nous pourrions choisir n'importe laquelle de ces 4 combinaisons. Cela coïncide avec les résultats que nous observons en inspectant les meilleurs paramètres du CV externe.
<!-- #endregion -->

<!-- #region id="Bapy0VWjoBxD" -->
#Feature selection
<!-- #endregion -->

<!-- #region id="UD_tjgtBQo6j" -->
## Classification
<!-- #endregion -->

<!-- #region id="-r2ytFZXx2bH" -->
### Réduire le temps d'exécussion
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 319, "status": "ok", "timestamp": 1641465134806, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="nU_zdc9p63fL" outputId="30669ce6-39b4-455c-a24f-361278985c2c"
from sklearn.datasets import make_classification

data, target = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=0,
)
data.shape
```

<!-- #region id="VVk1zIgthq6M" -->
* forêt aléatoire sans sélection de caractéristiques
<!-- #endregion -->

```python id="auuk4ZLogn0F"
from sklearn.ensemble import RandomForestClassifier

model_without_selection = RandomForestClassifier(n_jobs=2)
```

<!-- #region id="sbF-ULBOh7Te" -->
* Forêt aléatoire avec  une étape de sélection des caractéristiques pour entraîner ce classifieur. La sélection des caractéristiques est basée sur un test univarié (valeur F de l'ANOVA) entre chaque caractéristique et la cible que nous voulons prédire. Les caractéristiques ayant les deux scores les plus significatifs sont sélectionnées.
<!-- #endregion -->

```python id="75Pw6H7bgn26"
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import make_pipeline


model_with_selection = make_pipeline(
    SelectKBest(score_func=f_classif, k=5), # 5 features selectionnées
    RandomForestClassifier(n_jobs=2),
)
```

<!-- #region id="V1aQ6YU9iPr0" -->
* Apprentissage des Modèles
<!-- #endregion -->

```python id="LelgiEIygn5a"
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results_without_selection = cross_validate(model_without_selection, data,
                                              target)
cv_results_without_selection = pd.DataFrame(cv_results_without_selection)
```

```python id="8YIUNsidgn76"
cv_results_with_selection = cross_validate(
    model_with_selection, data, target, return_estimator=True)
cv_results_with_selection = pd.DataFrame(cv_results_with_selection)
```

```python id="kpXBUJZNgn-_"
cv_results = pd.concat(
    [cv_results_without_selection, cv_results_with_selection],
    axis=1,
    keys=["Without feature selection", "With feature selection"],
)
# swap the level of the multi-index of the columns
cv_results = cv_results.swaplevel(axis="columns")
```

<!-- #region id="p2XEndh5iofK" -->
*  Temps d'entrainement de chaque pipeline.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"elapsed": 322, "status": "ok", "timestamp": 1641465172351, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="4Qf9OwMOidNi" outputId="7e9f82a8-87d7-4032-906e-2eb002f0eb13"
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
cv_results["fit_time"].plot.box(color=color, vert=False)
plt.xlabel("Elapsed time (s)")
_ = plt.title("Time to fit the model")
```

<!-- #region id="UqV5qUYdjBZm" -->
*  Temps de prédication de chaque pipeline.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"elapsed": 308, "status": "ok", "timestamp": 1641465176670, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="OsIuBvoTidQz" outputId="20a8c546-24bb-4c02-e25c-342e995244df"
cv_results["score_time"].plot.box(color=color, vert=False)
plt.xlabel("Elapsed time (s)")
_ = plt.title("Time to make prediction")
```

<!-- #region id="F4jRv-1BjjO3" -->
* **Caractérsitiques qui ont été sélectionnées pendant la validation croisée.**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 310, "status": "ok", "timestamp": 1641465191504, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="UH4KqW6CidUU" outputId="6f522b2c-4216-4a93-ddf2-b869b9048464"
import numpy as np

for idx, pipeline in enumerate(cv_results_with_selection["estimator"]):
    print(
        f"Fold #{idx} - features selected are: "
        f"{np.argsort(pipeline[0].scores_)[-5:]}"
    )
```

<!-- #region id="mrdGZTg4kcmi" -->
* **Ici l'objectif de la feature selection et d'obtenir un gain de temps surtout à l'apprentissage. Pour une spec sur les performance, les modèles  excluent nativement les caractéristiques non informatives.**
<!-- #endregion -->

<!-- #region id="CzjDKe7Vluuq" -->
### Performance des modèles
<!-- #endregion -->

<!-- #region id="ZsG7fdHl9msz" -->
#### **Sans jeu de Test**
<!-- #endregion -->

<!-- #region id="eaADT6zr86UA" -->
* Modèle de référence
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4592, "status": "ok", "timestamp": 1641465223011, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="8-oG8em4kcHw" outputId="19b0bc57-4a34-45d4-f4a4-8bc2ec101c0e"
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# solution
model = RandomForestClassifier()
test_score = cross_val_score(model, data, target, n_jobs=3)
print(f"The mean accuracy is: {test_score.mean():.3f}")
```

<!-- #region id="rujcZnAX8-mb" -->
* Modèle avec 5 features selectionnées
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3593, "status": "ok", "timestamp": 1641465265996, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="QmidWLznkcNf" outputId="ffd3e0b2-0fd1-4917-f9d0-ebc1c7ca20f6"
# solution
from sklearn.feature_selection import SelectKBest, f_classif

# solution
feature_selector = SelectKBest(score_func=f_classif, k=5)
data_subset = feature_selector.fit_transform(data, target)
test_score = cross_val_score(model, data_subset, target)
print(f"The mean accuracy is: {test_score.mean():.3f}")
```

<!-- #region id="4N8SnSWT9FKi" -->
#### **Avec jeu de Test**
<!-- #endregion -->

<!-- #region id="ILOa_frq9wmq" -->
* Modèle de référence
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3598, "status": "ok", "timestamp": 1641465347578, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="0Seddg6Ev3Dw" outputId="34ebac57-e114-4ba3-c637-4bba8f466208"
from sklearn.pipeline import make_pipeline

# solution
model = RandomForestClassifier()
pipe = make_pipeline(feature_selector, model)
test_score = cross_val_score(pipe, data, target)
print(f"The mean accuracy is: {test_score.mean():.3f}")
```

<!-- #region id="uL5tbDgo9yuP" -->
* Modèle avec feature selection
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 864, "status": "ok", "timestamp": 1641465296478, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="eG8dvoZ_kcQf" outputId="c8a3772c-008d-4767-84fa-66be621bc5a3"
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)
feature_selector.fit(data_train, target_train)
data_train_subset = feature_selector.transform(data_train)
data_test_subset = feature_selector.transform(data_test)
model.fit(data_train_subset, target_train)
test_score = model.score(data_test_subset, target_test)
print(f"The mean accuracy is: {test_score:.3f}")
```

<!-- #region id="xDzCZ9HGwxpX" -->
#### Comparaison avec visualisation
<!-- #endregion -->

<!-- #region id="ct3UrGHlz_05" -->
* Modèle de référence
<!-- #endregion -->

```python id="IXMR0vuZv3J0"
from sklearn.ensemble import RandomForestClassifier

model_without_selection = RandomForestClassifier()

import pandas as pd
from sklearn.model_selection import cross_validate

cv_results_without_selection = cross_validate(
    model_without_selection, data, target, cv=3)
cv_results_without_selection = pd.DataFrame(cv_results_without_selection)
```

<!-- #region id="hp4W46r20Cia" -->
* Modèle avec features selection
<!-- #endregion -->

```python id="-OFVeP7hzoOo"
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel

feature_selector = SelectFromModel(RandomForestClassifier())
model_with_selection = make_pipeline(
    feature_selector, RandomForestClassifier())

cv_results_with_selection = cross_validate(model_with_selection, data, target,
                                           cv=5)
cv_results_with_selection = pd.DataFrame(cv_results_with_selection)
```

<!-- #region id="zh6KMIl00IXB" -->
* Comparaison des résultats
<!-- #endregion -->

```python id="SzcuRkzb0LV1"
cv_results = pd.concat(
    [cv_results_without_selection, cv_results_with_selection],
    axis=1,
    keys=["Without feature selection", "With feature selection"],
).swaplevel(axis="columns")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 352} executionInfo={"elapsed": 330, "status": "ok", "timestamp": 1641465566966, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="4Wn93FGszoRk" outputId="bf039094-3596-4ed5-daa9-a9230ee42a18"
import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
cv_results["test_score"].plot.box(color=color, vert=False)
plt.xlabel("Accuracy")
_ = plt.title("Limites de l'utilisation d'une forêt aléatoire pour la sélection des caractéristiques")
```

<!-- #region id="Pljtu9VN1OPF" -->
### Recursive feature elimination with cross-validation
<!-- #endregion -->

```python id="MRQs9xl7zoTq"
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
```

```python id="f0IDLdyrzoXa"
from sklearn.ensemble import RandomForestClassifier
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 65316, "status": "ok", "timestamp": 1641465718373, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="onVXwsLk1f0f" outputId="b0624cc5-40c4-4b62-c5bf-1bbca22849c9"
min_features_to_select = 3 # Nombre minimum de caractéristiques à prendre en compte
model = RandomForestClassifier()

rfecv = RFECV(
    estimator=model,
    step=1,
    cv=StratifiedKFold(3),
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(data, target)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1641465718375, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="zRAiDKWl1f3w" outputId="de3cefdd-7dc6-40f8-8dc2-0e950b028b00"
print("Optimal number of features : %d" % rfecv.n_features_)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 336} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1641465718375, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="5v3T8rof1f6Q" outputId="1f503f73-7241-4057-af3d-500021458a21"
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_,
)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1997, "status": "ok", "timestamp": 1641466575836, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="I6WX-KcEBpH4" outputId="cf9ce626-2203-4a8b-cbfb-863248683708"
model = RandomForestClassifier()
model.fit(data, target)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} executionInfo={"elapsed": 310, "status": "ok", "timestamp": 1641467047455, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="3089jnr_BfYp" outputId="93267338-a08e-499b-ccc1-d39015e1acd6"
importances = model.feature_importances_

indices = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
#_ = ax.set_yticklabels(np.array(data.columns)[indices])
_ = ax.set_yticklabels(indices)
```

<!-- #region id="1aEDpfdOEOnC" -->
L'importance correspond à combien cette caractéristique est utilisée dans chaque arbre de la forêt. Formellement, elle est calculée comme la réduction totale (normalisée) du critère apportée par cette caractéristique.
<!-- #endregion -->

<!-- #region id="-bU5MYftQvpY" -->
## Regression
<!-- #endregion -->

<!-- #region id="RhtHvloDQj_0" -->
###1. Modèle linéaire

- Dans les modèles linéaires, la valeur cible est modélisée comme une combinaison linéaire des caractéristiques.
- Les coefficients représentent la relation entre la caractéristique donnée
et la cible, en supposant que toutes les autres caractéristiques restent constantes (dépendance conditionnelle).

<!-- #endregion -->

```python id="N5V2SxpsA2uL"
from sklearn.datasets import fetch_california_housing
import pandas as pd

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1641906466500, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="q5kVXo0p1f9g" outputId="15c74ddb-30c1-4944-bd67-fb87b62d8e83"
X.shape
```

```python id="9Mpc6dl9Rax3"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=29)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 903} id="kYwNOC0fRqCH" outputId="5ab62172-1d8b-4699-b236-2f8f30b8d958" executionInfo={"status": "ok", "timestamp": 1641906503692, "user_tz": -60, "elapsed": 37197, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}}
import seaborn as sns

train_dataset = X_train.copy()
train_dataset.insert(0, "MedHouseVal", y_train)
_ = sns.pairplot(
    train_dataset[['MedHouseVal', 'Latitude', 'AveRooms', 'AveBedrms', 'MedInc']],
    kind='reg', diag_kind='kde', plot_kws={'scatter_kws': {'alpha': 0.1}})
```

<!-- #region id="zhCLsBX1fM-A" -->
#### Sans mise à l'échelle
<!-- #endregion -->

```python id="FUmWh2BQRqIx" colab={"base_uri": "https://localhost:8080/", "height": 484} executionInfo={"status": "ok", "timestamp": 1641907723365, "user_tz": -60, "elapsed": 621, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="4ba805e8-7092-44ac-aae4-5ec537e3e13c"
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

model = RidgeCV()
model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

coefs = pd.DataFrame(
   model.coef_,
   columns=['Coefficients'], index=X_train.columns
)
print(coefs)
coefs.plot(kind='barh', figsize=(12, 4))
plt.title('Ridge model')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="f-l6AmlCUyoF" executionInfo={"status": "ok", "timestamp": 1641907715211, "user_tz": -60, "elapsed": 338, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="f2ab5f94-393a-4131-949a-23d16d8733be"
X_train.std(axis=0).plot(kind='barh', figsize=(12, 4))
plt.title('Features std. dev.')
plt.subplots_adjust(left=.3)
plt.xlim((0, 100))
```

<!-- #region id="XbFi1EkUfa_i" -->
##### Autres Méthode
<!-- #endregion -->

```python id="ghOIeK6LRqLY" colab={"base_uri": "https://localhost:8080/", "height": 413} executionInfo={"status": "ok", "timestamp": 1641907342877, "user_tz": -60, "elapsed": 304, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="87bb30a7-3b0d-4f61-d1ec-8d09a3cd46bd"
# define the model
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# fit the model
model.fit(X_train, y_train)
faeatures = list(X_train.columns)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in zip(faeatures,importance):
	print('Feature: %s, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
```

<!-- #region id="2HPdej6RfU6U" -->
#### Avec mise à l'échelle
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 373} id="e2Cf2Qy_fgSE" executionInfo={"status": "ok", "timestamp": 1641910826974, "user_tz": -60, "elapsed": 580, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="2c406296-52b3-4429-8e1e-73ffe02b9c0a"
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

model = make_pipeline(StandardScaler(), Lasso(alpha=.015))

model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

coefs = pd.DataFrame(
   model[1].coef_,
   columns=['Coefficients'], index=X_train.columns
)

coefs.plot(kind='barh', figsize=(12, 5))
plt.title('Ridge model')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)
```

<!-- #region id="oP8eaN1vVMg6" -->
#### Cross validation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="R_BJlllpSch5" executionInfo={"status": "ok", "timestamp": 1641910829393, "user_tz": -60, "elapsed": 1236, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="80449d37-59c1-4ad1-8fca-2154b15ece02"
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

cv_model = cross_validate(
   model, X_train, y_train, cv=RepeatedKFold(n_splits=5, n_repeats=5),
   return_estimator=True, n_jobs=2
)

# si model est un pipeline : model = make_pipeline(StandardScaler(), RidgeCV())
coefs = pd.DataFrame(
   [model[1].coef_
    for model in cv_model['estimator']],
   columns=X_train.columns
)
"""# Sans pipeline
coefs = pd.DataFrame(  [model.coef_   for model in cv_model['estimator']], columns=X_train.columns )"""

plt.figure(figsize=(12, 5))
sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5)
plt.axvline(x=0, color='.5')
plt.xlabel('Coefficient importance')
plt.title('Coefficient importance and its variability')
plt.subplots_adjust(left=.3)
```

<!-- #region id="-INTc2SRbPQ_" -->
Chaque coefficient semble assez stable, ce qui signifie que les différents modèles de la régression, accordent presque le même poids à la même caractéristique.
<!-- #endregion -->

<!-- #region id="TNS-M7vNYcB8" -->
###Modèles linéaires à coefficients dispersés (sparses) (Lasso)

Il est important de garder à l'esprit que les associations extraites dépendent du modèle. Pour illustrer ce point, nous considérons un modèle Lasso, qui effectue une sélection de caractéristiques avec une pénalité L1. Ajustons un modèle Lasso avec un fort paramètre de régularisation alpha
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 373} id="6QWBemK8YbBo" executionInfo={"status": "ok", "timestamp": 1641908855718, "user_tz": -60, "elapsed": 646, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="fee18cb6-d222-424d-f36c-cb8da1e77a54"
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

model = make_pipeline(StandardScaler(), Lasso(alpha=.015))

model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

coefs = pd.DataFrame(
   model[1].coef_,
   columns=['Coefficients'], index=X_train.columns
)

coefs.plot(kind='barh', figsize=(12, 5))
plt.title('Ridge model')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)
```

<!-- #region id="eUyeWMCSaRj3" -->
Le graphique ci-dessus nous renseigne sur les dépendances entre une caractéristique spécifique et la cible lorsque toutes les autres caractéristiques restent constantes, c'est-à-dire les dépendances conditionnelles.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 350} id="zPWXfKsxYbEK" executionInfo={"status": "ok", "timestamp": 1641909437233, "user_tz": -60, "elapsed": 1173, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="576695f6-573d-4de1-a69b-23ec58fc6ccb"
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

cv_model = cross_validate(
   model, X_train, y_train, cv=RepeatedKFold(n_splits=5, n_repeats=5),
   return_estimator=True, n_jobs=2
)
coefs = pd.DataFrame(
   [model[1].coef_
    for model in cv_model['estimator']],
   columns=X_train.columns
)
plt.figure(figsize=(12, 5))
sns.boxplot(data=coefs, orient='h', color='cyan', saturation=0.5)
plt.axvline(x=0, color='.5')
plt.xlabel('Coefficient importance')
plt.title('Coefficient importance and its variability')
plt.subplots_adjust(left=.3)
```

<!-- #region id="p1_3L9iqa--Z" -->
Les caractéristiques ayant forte variabilité et non nuls. Étant donné qu'ils sont fortement corrélés, le modèle peut choisir l'un d'entre eux pour bien prédire. Ce choix est un peu arbitraire, et ne doit pas être sur-interprété.
<!-- #endregion -->

<!-- #region id="8kFly5f2dniJ" -->
* Les coefficients doivent être mis à l'échelle dans la même unité de mesure pour retrouver l'importance des caractéristiques et pour les comparer.  

* Les coefficients des modèles linéaires multivariés représentent la dépendance entre une caractéristique donnée et la cible, conditionnellement aux autres caractéristiques.

* Les caractéristiques corrélées peuvent induire des instabilités dans les coefficients des modèles linéaires et leurs effets ne peuvent pas être bien séparés.

L'inspection des coefficients à travers les folds de la validation croisée? donne une idée de leur stabilité.

<!-- #endregion -->

<!-- #region id="UCuGofzahyCM" -->
### RandomForest
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 357} id="FTo9PMIjYbGY" executionInfo={"status": "ok", "timestamp": 1641911347197, "user_tz": -60, "elapsed": 12172, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="9085b8ca-b543-4c0f-b72d-f93becd4a33a"
from sklearn.ensemble import RandomForestRegressor
import numpy as np

model = RandomForestRegressor()

model.fit(X_train, y_train)

print(f'model score on training data: {model.score(X_train, y_train)}')
print(f'model score on testing data: {model.score(X_test, y_test)}')

importances = model.feature_importances_

indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(12,5))
ax.barh(range(len(importances)), importances[indices])
ax.set_yticks(range(len(importances)))
_ = ax.set_yticklabels(np.array(X_train.columns)[indices])
```

<!-- #region id="6JvRBq2EjviF" -->
### Importance des caractéristiques par permutation

On permute une caractéristique afin de voir si le modèle change sa prédiction. Ainsi, le changement dans la prédiction correspondra à l'importance de la caractéristique.  

il est important d'exécuter plusieurs fois le programme et d'inspecter la moyenne et l'écart-type de l'importance des caractéristiques.
<!-- #endregion -->

<!-- #region id="4MErPzFAk6-m" -->
#### Sur unique caractéristique
<!-- #endregion -->

```python id="kRSXGVaJYbI2"
def get_score_after_permutation(model, X, y, curr_feat):
    """ retourne le score du modèle quand curr_feat est permuté """

    X_permuted = X.copy()
    col_idx = list(X.columns).index(curr_feat)
    # permute one column
    X_permuted.iloc[:, col_idx] = np.random.permutation(
        X_permuted[curr_feat].values)

    permuted_score = model.score(X_permuted, y)
    return permuted_score


def get_feature_importance(model, X, y, curr_feat):
    """ comparer le score lorsque curr_feat est permuté """

    baseline_score_train = model.score(X, y)
    permuted_score_train = get_score_after_permutation(model, X, y, curr_feat)

    # feature importance is the difference between the two scores
    feature_importance = baseline_score_train - permuted_score_train
    return feature_importance
```

```python colab={"base_uri": "https://localhost:8080/"} id="H54rFw2RSclA" executionInfo={"status": "ok", "timestamp": 1641911899705, "user_tz": -60, "elapsed": 10309, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="d2e0029d-9c2f-424e-890b-9e782a978cd5"
curr_feat = 'MedInc'
n_repeats = 10

list_feature_importance = []
for n_round in range(n_repeats):
    list_feature_importance.append(
        get_feature_importance(model, X_train, y_train, curr_feat))

print(
    f'l\'importance de la feature "{curr_feat}" sur l\'entrainement est '
    f'{np.mean(list_feature_importance):.3} '
    f'+/- {np.std(list_feature_importance):.3}')
```

<!-- #region id="f7CPO6SAlBPR" -->
#### Sur Toutes les caractéristique
<!-- #endregion -->

```python id="tR-gCDoTlAmN"
def permutation_importance(model, X, y, n_repeats=10):
    """Calculer le score d'importance pour chaque caractéristique."""

    importances = []
    for curr_feat in X.columns:
        list_feature_importance = []
        for n_round in range(n_repeats):
            list_feature_importance.append(
                get_feature_importance(model, X, y, curr_feat))

        importances.append(list_feature_importance)

    return {'importances_moyenne': np.mean(importances, axis=1),
            'importances_std': np.std(importances, axis=1),
            'importances': importances}

# Cette fonction pourrait être accessible directement depuis sklearn
# from sklearn.inspection import permutation_importance
```

```python id="Er4j9vkIlAo8"
def plot_importantes_features(perm_importance_result, feat_name):
    """ bar plot the feature importance """

    fig, ax = plt.subplots()

    indices = perm_importance_result['importances_mean'].argsort()
    plt.barh(range(len(indices)),
             perm_importance_result['importances_mean'][indices],
             xerr=perm_importance_result['importances_std'][indices])

    ax.set_yticks(range(len(indices)))
    _ = ax.set_yticklabels(feat_name[indices])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="WOncnrpnlArc" executionInfo={"status": "ok", "timestamp": 1641912135373, "user_tz": -60, "elapsed": 75830, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="ff12edc6-9b6a-42aa-997f-daf625239264"
perm_importance_result_train = permutation_importance(
    model, X_train, y_train, n_repeats=10)

plot_importantes_features(perm_importance_result_train, X_train.columns)
```

<!-- #region id="_h9KxXCvmVyh" -->
* Pour les features corrélées, la permutation peut donner un échantillon non réaliste.

* Il n'est pas certains de garder ces résultats entre les données d'entrainement ou de test.

* Notez que la suppression d'une colonne et l'ajustement d'un nouveau modèle ne permettent pas d'analyser l'importance de la feature pour un modèle spécifique, car le modèle sera à nouveaux ajusté.
<!-- #endregion -->

```python id="TzJu9_kalAtt"

```

```python id="JbbxD_QilAw9"

```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 19038, "status": "ok", "timestamp": 1641454741756, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="ZL1EVkXEmuv4" outputId="5e26c1cf-3fe2-4c1f-a9c4-d70889076e7d"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 668, "status": "ok", "timestamp": 1641454742959, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="N9vsDjIrrORA" outputId="8f129f25-77cc-4755-81e5-3137d3636211"
import pandas as pd

adult_census = pd.read_csv(path+"adult-census-numeric-all.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]
adult_census.head()
```

```python id="5MKbDwS-xy-s"

```

```python id="IRV57CV_xzBh"

```

<!-- #region id="4AUyyYkNWuhY" -->
# Ici l'idée est d'encoder les variable qualitatives :
- Une solution pour les deux types de variables qualitatives:
 - Ordinales :  OneHotEncoder
 - Nominales : Dummy

<!-- #endregion -->

```python id="8mqMipzyranZ"
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 334, "status": "ok", "timestamp": 1641455066968, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="1eD729G6WGGS" outputId="f412b3cd-bd10-4481-b27f-09369a4d8969"
data2
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 408, "status": "ok", "timestamp": 1641455073784, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="6Hs9QvW0WHpa" outputId="8af77ee1-4513-4823-84e9-8c6d77818acc"
target2
```

```python colab={"base_uri": "https://localhost:8080/"} id="2qsjmfDjxAw-" executionInfo={"status": "ok", "timestamp": 1643357974329, "user_tz": -60, "elapsed": 24843, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="148acc1f-2ed9-4b28-e304-9c58c756ac53"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 393} executionInfo={"elapsed": 545, "status": "ok", "timestamp": 1643357999313, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="y8Hab0UwWLcD" outputId="e8ac8fd0-38e8-4a0d-dd6b-09268ea5c006"
import pandas as pd

adult_census = pd.read_csv(path+"adult.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]
print(adult_census.shape)
adult_census.head()
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1643358004141, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}, "user_tz": -60} id="1PhF9HhDdQoc" outputId="65d84a0c-288f-4a48-c05a-082eea5f6415"
adult_census.education.value_counts()
```

```python id="JJXqEAd3d_tD" colab={"base_uri": "https://localhost:8080/", "height": 224} executionInfo={"status": "ok", "timestamp": 1643358079960, "user_tz": -60, "elapsed": 550, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="246c5823-850c-44c8-b9ca-32aceedcd106"
dult_census_all = pd.read_csv(path+"adult-census-numeric-all.csv")
data, target = adult_census_all.drop(columns="class"), adult_census_all["class"]
print(adult_census_all.shape)
adult_census_all.head()
```

```python id="sVXByN_ixkJD"

```
