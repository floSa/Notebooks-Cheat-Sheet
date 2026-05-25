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
# 🔍 Interprétabilité & Feature Selection
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki** sur deux sujets liés : **comprendre** ce qui rentre dans la prédiction d'un modèle, et **sélectionner** les features qui apportent vraiment.

Couvre :

1. **Pourquoi l'interprétabilité** — debugging, compliance, fairness.
2. **Feature Importance "built-in"** (RandomForest, XGBoost, etc.).
3. **Permutation Importance** — la méthode model-agnostique propre.
4. **SHAP** — Shapley values, le standard 2026 (TreeExplainer, KernelExplainer, DeepExplainer).
5. **LIME** — explications locales par approximation linéaire.
6. **Partial Dependence Plots (PDP) & ICE**.
7. **Feature Selection** : filter (univariate), wrapper (RFE/RFECV), embedded (Lasso, model-based), Boruta.
8. **Display API de sklearn** — toutes les courbes en 1 ligne.

Dataset : **California Housing** + **Titanic** (mutualisés).
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi l'interprétabilité (en 2026)
<!-- #endregion -->

<!-- #region -->
- **Debugging** : ce modèle marche-t-il pour les bonnes raisons ou exploite-t-il un proxy douteux (ex: code postal → ethnie) ?
- **Compliance** : RGPD article 22 (décision automatisée), AI Act 2024 (UE).
- **Trust** : utilisateur final métier qui doit valider une recommandation.
- **Découverte** scientifique : quelles features sont causalement / corrélativement importantes.
- **Fairness** : détecter des biais cachés (sexe, race, âge) influençant les décisions.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
sns.set_theme(style="whitegrid")

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
print(f"RF R² test = {rf.score(X_te, y_te):.4f}")
```

<!-- #region -->
## 2. Feature Importance "built-in"
<!-- #endregion -->

<!-- #region -->
La plupart des modèles tree-based exposent `feature_importances_` :

- **RandomForest / Extra-Trees / GBM** : Mean Decrease in Impurity (Gini ou MSE). Rapide mais **biaisée vers les features à grande cardinalité** (numériques continues, catégorielles à beaucoup de modalités).
- **XGBoost / LightGBM** : `gain` (gain total en split sur cette feature), `weight` (nombre de splits), `cover` (samples affectés). Privilégier **gain** en général.
<!-- #endregion -->

```python
imp_builtin = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Built-in (MDI) importance :")
print(imp_builtin.round(3))
```

<!-- #region -->
## 3. Permutation Importance — model-agnostic & propre
<!-- #endregion -->

<!-- #region -->
**Idée** : pour chaque feature, on **permute** ses valeurs sur le test set, on mesure la baisse de performance. Une feature importante → permuter casse beaucoup la perf.

**Avantages** :

- **Model-agnostic** (marche pour n'importe quel modèle, même boîte noire).
- **Non biaisée** vers la cardinalité.
- Évalue l'importance sur **données non vues**.

**Limites** :

- Lent (N permutations × N features).
- Pour des features **corrélées**, l'importance est partagée → utiliser des permutations conditionnelles ou SHAP.
<!-- #endregion -->

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
imp_perm = pd.DataFrame({
    "mean": perm.importances_mean,
    "std": perm.importances_std,
}, index=X.columns).sort_values("mean", ascending=False)
print("Permutation importance :")
print(imp_perm.round(4))
```

<!-- #region -->
## 4. SHAP — Shapley values, le standard 2026
<!-- #endregion -->

<!-- #region -->
**SHAP** (SHapley Additive exPlanations, Lundberg 2017) attribue à chaque feature **sa contribution moyenne** à la prédiction, en s'appuyant sur la théorie des jeux coopératifs.

Propriétés mathématiquement uniques :

- **Local accuracy** : `Σ φᵢ + φ₀ = f(x)`.
- **Missingness** : feature absente → contribution 0.
- **Consistency** : si un modèle dépend plus d'une feature, son SHAP augmente.

**3 explainers principaux** :

| Explainer | Modèles | Vitesse |
|---|---|---|
| **TreeExplainer** | sklearn trees, XGBoost, LightGBM, CatBoost | **Très rapide** (algo exact polynomial) |
| **KernelExplainer** | N'importe lequel (model-agnostic) | Lent (LIME-like sampling) |
| **DeepExplainer** | TensorFlow, PyTorch | Moyen, GPU possible |
| **LinearExplainer** | Linéaires | Trivial et exact |

**Visualisations** :

- **Summary plot** : top features + impact de chaque observation.
- **Waterfall plot** : décomposition d'**une** prédiction.
- **Dependence plot** : effet d'une feature en fonction de sa valeur.
- **Force plot** : équivalent waterfall horizontal interactif.
<!-- #endregion -->

```python
import shap

# TreeExplainer : algo polynomial spécifique aux arbres
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_te.iloc[:200])  # subset pour aller vite
print(f"SHAP values shape : {shap_values.shape}  (obs, features)")

# Importance globale = moyenne |SHAP|
shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
print("\nSHAP importance globale :")
print(shap_imp.round(3))
```

<!-- #region -->
### 4.1 Summary plot et waterfall
<!-- #endregion -->

```python
import matplotlib.pyplot as plt

# Summary plot — vue d'ensemble : importance + direction de l'effet par feature
shap.summary_plot(shap_values, X_te.iloc[:200], show=False)
plt.tight_layout()
plt.show()
```

```python
# Waterfall plot — décomposition d'UNE prédiction individuelle
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_te.iloc[0].values,
        feature_names=X.columns.tolist(),
    ),
    show=False,
)
plt.tight_layout()
plt.show()
```

<!-- #region -->
## 5. LIME — Local Interpretable Model-agnostic Explanations
<!-- #endregion -->

<!-- #region -->
**Idée** : autour d'une observation à expliquer, on **perturbe** ses features, on observe les prédictions, et on ajuste un **modèle linéaire localement** (régression linéaire avec poids gaussiens centrés sur l'obs).

Les **coefficients linéaires** donnent l'explication.

**Quand l'utiliser plutôt que SHAP** :

- Modèles non-tree, très lents en KernelSHAP.
- Texte (LIME a un mode tokenisé).
- Image (LIME peut perturber des superpixels).

**Limites** :

- Moins théoriquement fondé que SHAP.
- Non-déterministe (sampling).
- Sensible aux hyperparamètres (`num_samples`, kernel width).
<!-- #endregion -->

<!-- #region -->
## 6. Partial Dependence Plots & ICE
<!-- #endregion -->

<!-- #region -->
**PDP** : effet **marginal moyen** d'une feature sur la prédiction, en moyennant les autres features.
**ICE** : la même chose mais **par observation** (pas moyenné) — révèle l'hétérogénéité.

Très utile pour montrer le **sens** de la relation feature → output (monotone ? bell shape ? marche d'escalier ?).
<!-- #endregion -->

```python
from sklearn.inspection import PartialDependenceDisplay

disp = PartialDependenceDisplay.from_estimator(
    rf, X_te, features=["MedInc", "HouseAge", "AveRooms"],
    kind="average",
    grid_resolution=20,
)
disp.figure_.set_size_inches(15, 4)
disp.figure_.tight_layout()
```

<!-- #region -->
## 7. Feature Selection — 3 familles
<!-- #endregion -->

<!-- #region -->
### 7.1 Filter (univariate)
<!-- #endregion -->

<!-- #region -->
Calcule une **statistique par feature** sans modèle, garde les top-k.

- `SelectKBest(score_func=f_classif | mutual_info_classif | chi2)` (sklearn).
- Rapide mais ignore les **interactions** entre features.
<!-- #endregion -->

```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression

sel_k = SelectKBest(score_func=mutual_info_regression, k=4).fit(X_tr, y_tr)
top_features = X.columns[sel_k.get_support()].tolist()
print(f"Top 4 features (MI) : {top_features}")
```

<!-- #region -->
### 7.2 Wrapper — RFE / RFECV
<!-- #endregion -->

<!-- #region -->
**Recursive Feature Elimination** : entraîne un modèle, retire la feature la moins importante, recommence. Avec CV (`RFECV`) on choisit automatiquement le nombre optimal de features.

Coût : N entrainements. À faire sur sous-échantillon si gros dataset.
<!-- #endregion -->

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    step=1, cv=3, scoring="r2", n_jobs=-1, min_features_to_select=2,
).fit(X_tr.iloc[:2000], y_tr.iloc[:2000])
print(f"RFECV : n_features optimal = {rfecv.n_features_}")
print(f"Sélectionnées : {X.columns[rfecv.support_].tolist()}")
```

<!-- #region -->
### 7.3 Embedded — Lasso, SelectFromModel
<!-- #endregion -->

<!-- #region -->
La sélection est **intégrée** à l'entraînement du modèle :

- **Lasso (L1)** : coefs deviennent exactement 0 → sélection.
- **Tree-based + SelectFromModel** : seuil sur `feature_importances_`.
<!-- #endregion -->

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Lasso : sélection auto via L1
lasso = LassoCV(cv=3, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
n_kept = (np.abs(lasso.coef_) > 1e-6).sum()
print(f"LassoCV : α={lasso.alpha_:.4f}  features non-nulles = {n_kept}/{X.shape[1]}")

# SelectFromModel sur RF
sfm = SelectFromModel(rf, prefit=True, threshold="median").transform(X_tr)
print(f"SelectFromModel (median) : {sfm.shape[1]} features gardées")
```

<!-- #region -->
### 7.4 Boruta
<!-- #endregion -->

<!-- #region -->
**Boruta** (Kursa 2010) compare l'importance de chaque feature à des **shadow features** (versions permutées des originales). Une feature est gardée seulement si son importance est statistiquement supérieure aux meilleures shadow features.

Plus rigoureux que SelectFromModel (significativité statistique), plus lent. Dispo via `boruta_py`.

```python
# from boruta import BorutaPy
# boruta = BorutaPy(rf, n_estimators="auto", random_state=42).fit(X_tr.values, y_tr.values)
# kept = X.columns[boruta.support_].tolist()
```
<!-- #endregion -->

<!-- #region -->
## 8. Display API sklearn — courbes en 1 ligne
<!-- #endregion -->

<!-- #region -->
Depuis sklearn 1.0+, les **Display classes** standardisent la viz d'éval :

- `RocCurveDisplay`, `PrecisionRecallDisplay`
- `ConfusionMatrixDisplay`
- `CalibrationDisplay`
- `LearningCurveDisplay`, `ValidationCurveDisplay`
- `DecisionBoundaryDisplay`
- `PartialDependenceDisplay` (vu plus haut)

```python
# from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
# ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
# RocCurveDisplay.from_estimator(clf, X_test, y_test)
```
<!-- #endregion -->

<!-- #region -->
## 9. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### Interprétabilité
<!-- #endregion -->

<!-- #region -->
- **Permutation Importance** > built-in MDI (biais cardinalité).
- **SHAP TreeExplainer** sur arbres → exact et rapide, l'aller-retour standard 2026.
- Pour **expliquer une décision unique** au métier : SHAP waterfall ou force plot.
- Pour **expliquer le comportement global** : SHAP summary plot + dependence plots sur les top features.
- **Fairness** : disaggregate les métriques par sous-groupes (sex, race, age) via `fairlearn` ou `aif360`.
<!-- #endregion -->

<!-- #region -->
### Feature Selection
<!-- #endregion -->

<!-- #region -->
- Pour réduire massivement (> 50 features) : **filter + wrapper**.
- Pour fine-tune : **Lasso** ou **Boruta**.
- Toujours **comparer** la perf avant/après — parfois supprimer dégrade légèrement (info redondante mais utile).
- **Ne pas sélectionner avant un GBM** : XGBoost/LightGBM gèrent très bien les features inutiles (importance ≈ 0). Sélectionner avant un modèle linéaire ou un NN simple.
- Toujours splitter avant la sélection (sinon leak).
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [SHAP — docs officielles](https://shap.readthedocs.io/)
- [Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
- [Interpretable ML book — Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)
- [sklearn — Inspection module](https://scikit-learn.org/stable/inspection.html)
- [LIME](https://github.com/marcotcr/lime) · [Boruta-py](https://github.com/scikit-learn-contrib/boruta_py)
- [Fairlearn](https://fairlearn.org/) · [AIF360](https://aif360.mybluemix.net/)
- Notebooks liés : `ML_Bagging_Boosting`, `ML_Regression_Classification_CV_GridSearch`.
<!-- #endregion -->
