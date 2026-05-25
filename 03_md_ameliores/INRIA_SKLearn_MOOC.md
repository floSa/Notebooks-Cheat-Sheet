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
# 🎓 Synthèse INRIA Scikit-Learn MOOC
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki** synthétisant les concepts du **MOOC INRIA / Scikit-Learn** (Loïc Estève et al.) — référence pédagogique francophone pour le ML avec scikit-learn.

> Ce notebook est une **synthèse + actualisation 2026** des notes prises lors du MOOC INRIA. Pour la version complète, voir [https://inria.github.io/scikit-learn-mooc/](https://inria.github.io/scikit-learn-mooc/).

Couvre les modules-clés :

1. Pipeline et ColumnTransformer.
2. Cross-validation et tuning.
3. Familles de modèles (linéaire, SVM, arbres, ensembles).
4. **Métriques** et leur choix.
5. **Sélection de modèle** et hyperparamètres.
6. **Feature engineering** (categorical encoding, scaling, interactions).
7. **Validation curve / Learning curve** : diagnostiquer biais/variance.
<!-- #endregion -->

<!-- #region -->
## 1. Pipeline + ColumnTransformer
<!-- #endregion -->

<!-- #region -->
La leçon n°1 du MOOC : **encapsuler tout dans une Pipeline**. Sinon leak ou debug impossible.

Pour des features mixtes (num + cat), `ColumnTransformer` applique des prep différents par groupe de colonnes.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

titanic = sns.load_dataset("titanic").dropna(subset=["embarked"])
X = titanic[["age", "fare", "sibsp", "parch", "pclass", "sex", "embarked"]]
y = titanic["survived"]

num = ["age", "fare", "sibsp", "parch"]
cat = ["pclass", "sex", "embarked"]
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
print("Pipeline construite :", pipe.steps)
```

<!-- #region -->
## 2. Cross-validation rigoureuse
<!-- #endregion -->

<!-- #region -->
- **KFold** vs **StratifiedKFold** vs **GroupKFold** vs **TimeSeriesSplit** : choisir selon la nature des données.
- `cross_val_score` pour un seul score, `cross_validate` pour plusieurs metrics + temps.
- Toujours `random_state` + `shuffle=True` (sauf TS).
- Toujours rapporter **mean ± std** (jamais un seul fold).
<!-- #endregion -->

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "roc_auc", "f1"], n_jobs=-1)
print({k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")})
```

<!-- #region -->
## 3. Familles de modèles — décision en 2026
<!-- #endregion -->

<!-- #region -->
| Famille | Représentants | Forces | Quand |
|---|---|---|---|
| **Linéaire** | LogReg, Ridge, Lasso, ElasticNet | Vitesse, interprétation, calibration | Baseline, données larges, déséquilibré |
| **SVM** | SVC, SVR | Frontières complexes via kernels | Petit-moyen, non-linéaire |
| **KNN** | KNeighbors* | Non-paramétrique, intuitif | Mémoriel, petit/moyen |
| **Arbres** | DecisionTree | Interprétable, non-linéaire | Rarement seul, base d'ensembles |
| **Ensemble bagging** | RandomForest, ExtraTrees | Robuste, peu de tuning | Baseline solide |
| **Ensemble boosting** | XGBoost, LightGBM, CatBoost | **SOTA tabulaire 2026** | Production tabulaire |
| **MLP** | sklearn ou DL | Pour data complexe | Quand GBM ne suffit pas |

> Pour **>95 % des problèmes ML tabulaires** en 2026 : **LightGBM ou XGBoost**, baseline LogReg en comparaison.
<!-- #endregion -->

<!-- #region -->
## 4. Métriques — bien choisir
<!-- #endregion -->

<!-- #region -->
Erreur classique du débutant : **accuracy partout**. Le MOOC INRIA insiste sur le choix de la métrique en fonction de la tâche et de la distribution.

| Tâche | Métrique reco |
|---|---|
| Classif binaire équilibrée | Accuracy, F1 |
| Classif déséquilibrée | F1, PR-AUC, MCC (voir `NLP_Classification_Smote`) |
| Régression | MAE, RMSE, R² |
| Probabilités calibrées | Log-loss, Brier score |
| Multi-classe | Macro/Weighted F1, Cohen's Kappa |
| Ranking / Top-k | NDCG, MAP, Top-k accuracy |
<!-- #endregion -->

<!-- #region -->
## 5. Sélection de modèle et tuning
<!-- #endregion -->

<!-- #region -->
Workflow standard :

1. **Train/Val/Test** ou **Train/Test + CV** sur train.
2. **Baseline** simple (LogReg, DummyClassifier) pour calibrer.
3. **Bench** de N familles avec CV.
4. **Tuning** du top 1-2 avec GridSearch/RandomSearch/Optuna.
5. **Test final** sur le test set vierge.

Voir notebooks **`ML_Regression_Classification_CV_GridSearch`** et **`ML_Optimisation_de_Modèles`** pour le détail.
<!-- #endregion -->

<!-- #region -->
## 6. Feature engineering — astuces du MOOC
<!-- #endregion -->

<!-- #region -->
### 6.1 Encoding catégoriel
<!-- #endregion -->

<!-- #region -->
| Encoder | Quand |
|---|---|
| `OneHotEncoder` | Cardinalité faible (<20), modèles linéaires ou NN |
| `OrdinalEncoder` | Pour arbres (qui s'en accommodent), ou cat ordonnée |
| `TargetEncoder` (sklearn 1.3+) | Haute cardinalité, attention au leak (utiliser CV target encoding) |
| Embeddings (DL) | Très haute cardinalité (millions d'IDs) |
<!-- #endregion -->

<!-- #region -->
### 6.2 Scaling
<!-- #endregion -->

<!-- #region -->
- **StandardScaler** : standard pour LogReg, SVM, NN, KNN.
- **MinMaxScaler** : quand on veut bornes [0,1] (images, NN avec sigmoide).
- **RobustScaler** : avec outliers (utilise médiane + IQR).
- **PowerTransformer** (Yeo-Johnson / Box-Cox) : si distribution très skewed.
- **Pas besoin** pour les arbres / GBM.
<!-- #endregion -->

<!-- #region -->
### 6.3 Features dérivées
<!-- #endregion -->

<!-- #region -->
- **Interactions** : `PolynomialFeatures(degree=2, interaction_only=True)` pour LogReg.
- **Binning** : `KBinsDiscretizer` pour exposer des non-linéarités à un modèle linéaire.
- **Features de date** : year, month, day_of_week, is_weekend → pour TS et séries calendaires.
<!-- #endregion -->

<!-- #region -->
## 7. Diagnostic — Learning et Validation Curves
<!-- #endregion -->

<!-- #region -->
**Learning curve** : performance vs **taille du train** → diagnostic :

- Si train et val convergent à un score bas → **biais élevé** (underfit), augmenter la complexité.
- Si train >> val → **variance élevée** (overfit), régulariser ou plus de data.

**Validation curve** : performance vs **hyperparamètre** → choisir la valeur du sweet spot.
<!-- #endregion -->

```python
from sklearn.model_selection import LearningCurveDisplay, ValidationCurveDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# Learning curve sur GBM
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
LearningCurveDisplay.from_estimator(
    Pipeline([("prep", prep), ("clf", GradientBoostingClassifier(random_state=42))]),
    X, y, ax=axes[0], scoring="roc_auc", cv=3, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
)
axes[0].set_title("Learning curve — GBM")

ValidationCurveDisplay.from_estimator(
    Pipeline([("prep", prep), ("clf", GradientBoostingClassifier(random_state=42))]),
    X, y, ax=axes[1], scoring="roc_auc", cv=3, n_jobs=-1,
    param_name="clf__n_estimators",
    param_range=[10, 50, 100, 200, 500],
)
axes[1].set_title("Validation curve — n_estimators")
plt.tight_layout()
```

<!-- #region -->
## 8. À aller plus loin
<!-- #endregion -->

<!-- #region -->
- [Le MOOC complet (français + anglais)](https://inria.github.io/scikit-learn-mooc/)
- Notebooks frères de ce projet :
  - `ML_Regression_Classification_Multiple` — intuitions biais/variance.
  - `ML_Regression_Classification_CV_GridSearch` — bench complet.
  - `ML_Bagging_Boosting` — ensembles modernes.
  - `ML_Optimisation_de_Modèles` — Optuna.
  - `ML_Explication_Feature_Importance_Selection` — SHAP, RFECV, Boruta.
  - `ML_MLFlow_Bench` — tracking et MLOps.
<!-- #endregion -->
