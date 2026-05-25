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
# 🌲 Bagging & Boosting
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur les **méthodes d'ensemble** : Bagging, Random Forest, Extra-Trees, AdaBoost, GBM, et la trinité moderne **XGBoost / LightGBM / CatBoost**.

Sections :

1. **Pourquoi ensembler** — biais/variance, intuition.
2. **Bootstrap** : la base mathématique du bagging.
3. **Bagging** classique sklearn.
4. **Random Forest** — bagging + random feature subsampling.
5. **Extra-Trees** — randomisation supplémentaire.
6. **Boosting** — AdaBoost, principe.
7. **Gradient Boosting Machines** (GBM) — maths.
8. **XGBoost, LightGBM, CatBoost** — la trinité 2026.
9. **Stacking** — meta-ensembling.

Dataset : **California Housing** (mutualisé).
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi ensembler
<!-- #endregion -->

<!-- #region -->
Reprenons la décomposition biais-variance d'un modèle :

$$
\text{Erreur} = \text{Biais}^2 + \text{Variance} + \text{Bruit}
$$

Un **arbre de décision profond** a :

- Biais faible (il peut s'adapter à n'importe quelle forme).
- Variance énorme (un nouvel échantillon donne un arbre très différent).

**Idée du Bagging** : entraîner `M` arbres sur des bootstraps différents et **moyenner**. Si les arbres sont décorrélés, la variance est divisée par `M` (théorème central limite). Le biais reste le même.

**Idée du Boosting** : entraîner `M` modèles **faibles** (arbres peu profonds, high bias) **séquentiellement**, chaque modèle corrigeant les erreurs du précédent. Cela **réduit le biais** (et la variance dans une moindre mesure).
<!-- #endregion -->

<!-- #region -->
## 2. Bootstrap
<!-- #endregion -->

<!-- #region -->
Un **bootstrap sample** de taille `n` est un échantillon **avec remise** de `n` éléments dans le dataset original (qui contient lui-même `n` éléments).

Conséquence : ~63 % des observations originales se retrouvent dans un bootstrap (le reste est dupliqué, et ~37 % sont **hors-sac** — Out-Of-Bag, OOB). Le OOB sert d'**estimateur intégré** de l'erreur de généralisation pour les bagging-based models (`oob_score=True` dans `RandomForestClassifier`).
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


def rmse(model, X, y) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))
```

<!-- #region -->
## 3. Bagging classique
<!-- #endregion -->

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bag = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=None),
    n_estimators=50,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,        # estime l'erreur sur les obs OOB de chaque arbre
    random_state=42,
    n_jobs=-1,
).fit(X_tr, y_tr)
print(f"Bagging RMSE test = {rmse(bag, X_te, y_te):.4f}")
print(f"Bagging OOB R²    = {bag.oob_score_:.4f}")
```

<!-- #region -->
## 4. Random Forest
<!-- #endregion -->

<!-- #region -->
**Random Forest** = Bagging + à chaque split, ne considérer qu'un **sous-ensemble aléatoire des features** (`max_features = √d` pour classif, `d/3` pour régression).

Cette double randomisation (bootstrap + features) **décorrèle davantage les arbres** → réduit plus efficacement la variance que le bagging seul.

**Hyperparamètres-clés** :

- `n_estimators` : plus = mieux jusqu'à un plateau (typique 100-500).
- `max_depth` / `min_samples_leaf` : contrôlent la complexité par arbre.
- `max_features` : taille du subset aléatoire à chaque split (par défaut `sqrt`/`1/3`).
- `n_jobs=-1` : parallélisation native.
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
print(f"Random Forest RMSE = {rmse(rf, X_te, y_te):.4f}")
print(f"Extra-Trees   RMSE = {rmse(et, X_te, y_te):.4f}")
```

<!-- #region -->
**Extra-Trees** (Extremely Randomized Trees) : encore plus aléatoire — au lieu de chercher le meilleur seuil pour chaque feature, on tire `m` seuils au hasard et on choisit le meilleur. Souvent **plus rapide et légèrement meilleur** que RF sur problèmes bruyants.
<!-- #endregion -->

<!-- #region -->
## 5. Boosting — AdaBoost (le grand-père)
<!-- #endregion -->

<!-- #region -->
**AdaBoost** (Adaptive Boosting, Freund & Schapire 1995) :

1. Entraîne un classifieur faible sur les données pondérées.
2. Calcule son taux d'erreur `ε`.
3. Augmente le **poids** des exemples mal classés, baisse celui des bien classés.
4. Pondère le modèle par `α = log((1-ε)/ε) / 2`.
5. Répète. Prédiction finale = somme pondérée des `M` classifieurs.

Limites : sensible aux outliers (qui finissent par dominer les poids), supplanté par GBM.
<!-- #endregion -->

```python
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print(f"AdaBoost RMSE = {rmse(ada, X_te, y_te):.4f}")
```

<!-- #region -->
## 6. Gradient Boosting (GBM)
<!-- #endregion -->

<!-- #region -->
**Idée plus générale** que AdaBoost : à chaque itération, le nouveau modèle est entraîné pour **prédire le gradient négatif de la loss** par rapport aux prédictions courantes (= les **résidus** pour la loss MSE).

Formellement, on construit une suite :

$$
F_{m+1}(x) = F_m(x) + \nu \cdot h_m(x), \quad h_m \approx -\nabla L(y, F_m(x))
$$

où `ν` est le **learning rate** (shrinkage), et `h_m` un weak learner (arbre).

**Hyperparamètres-clés** :

- `n_estimators` (M) : nombre d'arbres. Plus = plus précis mais risque overfit.
- `learning_rate` (ν) : 0.01-0.1 typique. Plus petit = plus précis mais plus lent.
- `max_depth` : profondeur des arbres (typique 3-8 pour GBM).
- `subsample` (< 1) : stochastic gradient boosting (SGB), réduit la variance.
<!-- #endregion -->

```python
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=42,
).fit(X_tr, y_tr)
print(f"GBM sklearn RMSE = {rmse(gbm, X_te, y_te):.4f}")
```

<!-- #region -->
## 7. La trinité 2026 — XGBoost / LightGBM / CatBoost
<!-- #endregion -->

<!-- #region -->
Ces 3 implémentations **dépassent largement** le GBM sklearn en performance et fonctionnalités. Différences :

| Aspect | **XGBoost** | **LightGBM** | **CatBoost** |
|---|---|---|---|
| Auteur | DMLC (2014) | Microsoft (2016) | Yandex (2017) |
| Vitesse training | bonne | **très rapide** (leaf-wise) | bonne |
| Vitesse inference | bonne | **très rapide** | bonne |
| Catégorielles natives | non (encoding requis) | partiel | **✅ natif (ordered target encoding)** |
| GPU | ✅ | ✅ | ✅ |
| Régularisation | L1+L2 | L1+L2 | L2 |
| Overfit-resistance | Bon | Sensible | **Très bon** (ordered boosting) |
| Quand l'utiliser | Standard, communauté énorme | Très large data, beaucoup de features | Beaucoup de catégorielles |

**Pratique 2026** : démarrer avec **LightGBM** (rapide, paire bien avec Optuna). Si beaucoup de catégorielles non-encodées : **CatBoost**.
<!-- #endregion -->

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

xgb_model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist",     # binarisation rapide (par défaut sur récent)
    random_state=42, n_jobs=-1,
).fit(X_tr, y_tr)
print(f"XGBoost   RMSE = {rmse(xgb_model, X_te, y_te):.4f}")

lgb_model = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
).fit(X_tr, y_tr)
print(f"LightGBM  RMSE = {rmse(lgb_model, X_te, y_te):.4f}")

cb_model = CatBoostRegressor(
    iterations=300, learning_rate=0.05, depth=6,
    random_state=42, verbose=False,
).fit(X_tr, y_tr)
print(f"CatBoost  RMSE = {rmse(cb_model, X_te, y_te):.4f}")
```

<!-- #region -->
### 7.1 Early stopping
<!-- #endregion -->

<!-- #region -->
Tous supportent l'early stopping : on fixe `n_estimators` très grand et on s'arrête quand la métrique sur un val set stagne pendant `early_stopping_rounds` itérations.

```python
# LightGBM
# lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

# XGBoost
# xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
```
<!-- #endregion -->

<!-- #region -->
### 7.2 Importance des features
<!-- #endregion -->

```python
import matplotlib.pyplot as plt

imp = pd.DataFrame({
    "feature": X.columns,
    "xgb": xgb_model.feature_importances_,
    "lgb": lgb_model.feature_importances_,
}).sort_values("xgb", ascending=False)
print(imp)
```

<!-- #region -->
## 8. Stacking (meta-ensembling)
<!-- #endregion -->

<!-- #region -->
**Idée** : combiner les prédictions de plusieurs modèles diversifiés avec un **méta-modèle** (souvent linéaire). Plus puissant que la moyenne brute (`VotingRegressor`) car le méta-modèle apprend **comment** combiner.

`sklearn.ensemble.StackingRegressor` / `StackingClassifier` implémentent ça avec une CV-out-of-fold pour éviter le leak.

```python
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import RidgeCV
# stack = StackingRegressor(
#     estimators=[("rf", rf), ("lgb", lgb_model), ("cb", cb_model)],
#     final_estimator=RidgeCV(),
#     cv=5, n_jobs=-1,
# )
# stack.fit(X_tr, y_tr)
```

**Bénéfice** : typiquement +0.5 à +2 % de perf vs le meilleur modèle individuel. À condition que les modèles soient **diversifiés** (différentes familles).

**Coût** : multiplication du temps de training par N.
<!-- #endregion -->

<!-- #region -->
## 9. Conseils pratiques 2026
<!-- #endregion -->

<!-- #region -->
- **Toujours benchmarker LightGBM** en premier sur tabulaire — souvent dans le top 3 sans tuning.
- **Optuna** + early stopping = la combo gagnante en 2026 (voir `ML_Optimisation_de_Modèles`).
- **CatBoost** si beaucoup de catégorielles à fortes cardinalités (target encoding contrôlé).
- **GPU** vaut le coup à partir de ~1M observations + 100 features (sinon, CPU avec `n_jobs=-1` est très rapide).
- **Random Forest** reste un excellent baseline robuste et sans tuning.
- **Stacking** = dernier 1-2 % de perf si tu as le budget.
- Pour l'**interprétabilité** : SHAP, voir `ML_Explication_Feature_Importance_Selection`.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [XGBoost docs](https://xgboost.readthedocs.io/)
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [CatBoost docs](https://catboost.ai/docs/)
- [Friedman (1999) — Greedy Function Approximation: A Gradient Boosting Machine](https://jerryfriedman.su.domains/)
- [Breiman (2001) — Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
- Notebooks liés : `ML_Regression_Classification_CV_GridSearch`, `ML_Optimisation_de_Modèles`, `ML_Explication_Feature_Importance_Selection`.
<!-- #endregion -->
