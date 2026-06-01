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
# 🧮 Régression & Classification — Vue d'ensemble
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel** pédagogique qui introduit la régression et la classification supervisée à partir des concepts fondamentaux (biais/variance) jusqu'aux modèles ensemble.

Public visé : débutant/intermédiaire qui veut comprendre **les intuitions** derrière les algos avant d'aller plus loin dans la pratique (voir `ML_Regression_Classification_CV_GridSearch` pour la pratique exhaustive).

Couvre :

1. **Cadre supervisé** — `(X, y)`, train/test split, métriques.
2. **Biais / Variance** — le concept clé qui guide tout.
3. **KNN** — exemple intuitif pour visualiser biais/variance.
4. **Régression linéaire / logistique** — modèles paramétriques de base.
5. **Arbres** — non-paramétriques, interprétables.
6. **Bootstrap & ensembles** — Bagging, Random Forest, Boosting (lien vers `ML_Bagging_Boosting`).
7. **Métriques d'évaluation** — quoi mesurer.

Datasets : **California Housing** (régression) + **Titanic** (classification), mutualisés.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre supervisé
<!-- #endregion -->

<!-- #region -->
On dispose de `n` exemples `(x_i, y_i)` avec `x ∈ ℝ^d` (features) et `y ∈ ℝ` (régression) ou `y ∈ {0, 1, ..., K-1}` (classification).

**But** : apprendre une fonction `f̂(x)` qui prédit `y` à partir de `x` sur des nouvelles données non vues.

**Étapes universelles** :

1. Split train/test (stratifié si classification déséquilibrée).
2. Preprocessing (encoding catégorielles, scaling si nécessaire, NaN).
3. Choix de la classe de modèles et entraînement.
4. **Validation** par cross-validation pour éviter l'overfit au test set.
5. Test final unique.
6. (Si prod) deployment + monitoring.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
sns.set_theme(style="whitegrid")

# Régression
data = fetch_california_housing(as_frame=True)
X_reg, y_reg = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
print(f"Régression : {X_tr.shape} features, {len(y_tr)} train")

# Classification (Titanic)
titanic = sns.load_dataset("titanic")
titanic = titanic.dropna(subset=["age", "embarked"])
X_cls = pd.get_dummies(titanic[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]], drop_first=True)
y_cls = titanic["survived"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
print(f"Classification : {Xc_tr.shape} features, {len(yc_tr)} train (positifs {yc_tr.mean():.1%})")
```

<!-- #region -->
## 2. Biais / Variance — le concept fondateur
<!-- #endregion -->

<!-- #region -->
**Décomposition de l'erreur de généralisation** d'un modèle de régression :

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{Biais}^2} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Bruit irréductible}}
$$

- **Biais élevé** = sous-apprentissage. Le modèle est trop simple pour capter la structure. Ex: régression linéaire sur une fonction sinusoïdale.
- **Variance élevée** = sur-apprentissage. Le modèle change beaucoup avec un autre échantillon d'apprentissage. Ex: arbre de décision profond sans régularisation.
- **Bruit** = erreur irréductible, dépend de la qualité des features et des données.

**Trade-off** : augmenter la complexité du modèle baisse le biais MAIS augmente la variance. On cherche le **sweet spot** qui minimise la somme.

Outils pour gérer ce trade-off :

- **Régularisation** (Ridge, Lasso, dropout) — réduit la variance au prix d'un peu de biais.
- **Ensembles** (Bagging, Boosting) — réduisent la variance et/ou le biais en agrégeant des modèles diverses.
- **Cross-validation** — estime le risque de généralisation et permet de choisir la complexité.
<!-- #endregion -->

<!-- #region -->
## 3. KNN — pour visualiser biais/variance
<!-- #endregion -->

<!-- #region -->
**K-Nearest Neighbors** : `f̂(x) = moyenne (ou vote) des K plus proches voisins de x dans le train`.

- **K=1** : variance maximale (épouse parfaitement le train, généralise mal).
- **K=N** : biais maximal (renvoie la moyenne globale, pas de différentiation).
- **K optimal** : trouvé par CV.

Idéal pour illustrer le trade-off : on **voit** le modèle changer de comportement quand K varie.
<!-- #endregion -->

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# KNN exige du scaling (distances sensibles aux échelles)
scores = []
for k in [1, 3, 5, 10, 25, 50, 100]:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    rmse_train = np.sqrt(mean_squared_error(y_tr, pipe.predict(X_tr)))
    rmse_test = np.sqrt(mean_squared_error(y_te, pipe.predict(X_te)))
    scores.append({"k": k, "train_rmse": rmse_train, "test_rmse": rmse_test})

scores_df = pd.DataFrame(scores)
print(scores_df.to_string(index=False))

# k=1 : train ≈ 0 (overfit), test élevé. k grand : les deux convergent vers une erreur élevée (underfit).
```

<!-- #region -->
## 4. Régression linéaire / logistique
<!-- #endregion -->

<!-- #region -->
### 4.1 Linéaire (OLS)
<!-- #endregion -->

<!-- #region -->
Minimise la somme des erreurs quadratiques : `argmin_β Σᵢ (yᵢ - xᵢᵀβ)²`. Solution analytique `β̂ = (XᵀX)⁻¹ Xᵀy`.

**Variantes régularisées** :

- **Ridge** (L2) : `+ λ‖β‖²` — réduit tous les coefs vers 0, stable même si features colinéaires.
- **Lasso** (L1) : `+ λ‖β‖₁` — induit la **sparsité** (certains coefs deviennent exactement 0). Sélection de variables intégrée.
- **ElasticNet** : combinaison des deux.
<!-- #endregion -->

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

for name, model in [("OLS", LinearRegression()),
                    ("Ridge α=1", Ridge(alpha=1.0)),
                    ("Lasso α=0.1", Lasso(alpha=0.1))]:
    m = Pipeline([("scaler", StandardScaler()), ("m", model)]).fit(X_tr, y_tr)
    rmse = np.sqrt(mean_squared_error(y_te, m.predict(X_te)))
    n_nonzero = np.sum(np.abs(m.named_steps["m"].coef_) > 1e-6)
    print(f"{name:15s} RMSE={rmse:.4f}  features non-nulles={n_nonzero}")
```

<!-- #region -->
### 4.2 Logistique
<!-- #endregion -->

<!-- #region -->
Pour la classification binaire : modélise `P(y=1|x) = σ(xᵀβ)` avec `σ` la sigmoïde. Optimisation par max log-likelihood.

Étendue au multi-classe via softmax (multinomial logistic).
<!-- #endregion -->

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logit = LogisticRegression(max_iter=1000).fit(Xc_tr, yc_tr)
preds = logit.predict(Xc_te)
proba = logit.predict_proba(Xc_te)[:, 1]
print(f"Acc={accuracy_score(yc_te, preds):.3f}  F1={f1_score(yc_te, preds):.3f}  AUC={roc_auc_score(yc_te, proba):.3f}")
```

<!-- #region -->
## 5. Arbres de décision
<!-- #endregion -->

<!-- #region -->
Un **arbre** partitionne l'espace en régions rectangulaires, chaque feuille prédisant la valeur moyenne (régression) ou la classe majoritaire (classif).

**Critères de split** :

- Régression : variance, MSE.
- Classification : entropie de Shannon, indice de Gini.

**Forces** : interprétable (graphique), non-linéaire, capte interactions, pas besoin de scaling.
**Faiblesses** : très haute variance — un arbre seul **overfit massivement**. D'où les ensembles.
<!-- #endregion -->

```python
from sklearn.tree import DecisionTreeRegressor

for depth in [3, 5, 10, None]:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_tr, y_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_tr, dt.predict(X_tr)))
    rmse_te = np.sqrt(mean_squared_error(y_te, dt.predict(X_te)))
    print(f"max_depth={str(depth):5s}  train={rmse_tr:.4f}  test={rmse_te:.4f}")
```

<!-- #region -->
## 6. Bootstrap & Ensembles (introduction)
<!-- #endregion -->

<!-- #region -->
Pour réduire la variance d'un modèle high-variance (arbre profond) : **ensembler N modèles** entraînés sur des sous-échantillons différents et **moyenner** leurs prédictions.

**Bagging** = Bootstrap Aggregating : N modèles entraînés indépendamment sur bootstrap samples.

**Random Forest** = bagging + sub-sélection aléatoire des features à chaque split.

**Boosting** = N modèles entraînés **séquentiellement**, chaque modèle corrigeant les erreurs du précédent. Réduit le **biais** (à l'inverse du bagging qui réduit la variance).

Voir notebook dédié **`ML_Bagging_Boosting`** pour les détails.
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)

print(f"RF  : RMSE={np.sqrt(mean_squared_error(y_te, rf.predict(X_te))):.4f}")
print(f"GBM : RMSE={np.sqrt(mean_squared_error(y_te, gb.predict(X_te))):.4f}")
```

<!-- #region -->
## 7. Métriques (sélection guidée)
<!-- #endregion -->

<!-- #region -->
### 7.1 Régression
<!-- #endregion -->

<!-- #region -->
| Métrique | Formule | Quand |
|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | Robuste aux outliers, échelle = y |
| **MSE / RMSE** | `mean((y - ŷ)²)` | Pénalise les gros écarts |
| **MAPE** | `mean(|y - ŷ| / |y|)` | Relatif, métier-friendly. Échoue si `y ≈ 0` |
| **R²** | `1 - SSres / SStot` | Proportion de variance expliquée, comparable cross-tâches |
| **Quantile loss** (pinball) | `max((y-ŷ)·τ, (ŷ-y)·(1-τ))` | Quand on veut un quantile (intervalles) |
<!-- #endregion -->

<!-- #region -->
### 7.2 Classification
<!-- #endregion -->

<!-- #region -->
| Métrique | Quand |
|---|---|
| Accuracy | Classes équilibrées, coût FP=FN |
| F1 / Precision / Recall | Classes déséquilibrées |
| ROC-AUC | Classement binaire, classes équilibrées-ish |
| PR-AUC | Classement binaire très déséquilibré |
| Log-loss | Métrique propre pour probabilités calibrées |
| MCC | Robuste à n'importe quel déséquilibre |
| Cohen's Kappa | Agreement vs annotateur |

Pour la **classification déséquilibrée**, voir `NLP_Classification_Smote` (les leçons s'appliquent au tabulaire aussi).
<!-- #endregion -->

<!-- #region -->
## 8. Prochaines étapes
<!-- #endregion -->

<!-- #region -->
- **`ML_Regression_Classification_CV_GridSearch`** — bench complet de N algos avec CV propre.
- **`ML_Bagging_Boosting`** — bagging, RF, gradient boosting, XGBoost, LightGBM, CatBoost.
- **`ML_Optimisation_de_Modèles`** — Optuna pour la recherche d'hyperparamètres.
- **`ML_Explication_Feature_Importance_Selection`** — SHAP, LIME, feature selection (RFECV, Boruta).
- **`ML_MLFlow_Bench`** — versioning et MLOps.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [The Elements of Statistical Learning — Hastie, Tibshirani, Friedman (livre de référence)](https://hastie.su.domains/ElemStatLearn/)
- [scikit-learn — User Guide](https://scikit-learn.org/stable/user_guide.html)
- [INRIA Scikit-Learn MOOC](https://inria.github.io/scikit-learn-mooc/) — voir notebook `INRIA_SKLearn_MOOC`.
<!-- #endregion -->
