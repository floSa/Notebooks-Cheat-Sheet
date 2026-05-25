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
# 🎯 Optimisation d'hyperparamètres
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur l'**optimisation d'hyperparamètres** — le passage obligé pour passer d'un modèle baseline à un modèle compétitif.

Couvre :

1. **Vocabulaire** : hyperparamètres vs paramètres, distributions, espaces.
2. **GridSearch / RandomSearch** (sklearn) — rappel.
3. **Halving Search** — successive halving, gain de temps.
4. **Bayesian Optimization** — TPE, GP-based (BoTorch / scikit-optimize).
5. **Optuna** — le standard 2026 (samplers, pruners, distribué).
6. **Multi-framework** : sklearn / XGBoost / LightGBM / CatBoost / PyTorch / Keras.
7. **Multi-objective** (compromis perf vs vitesse vs taille).
8. **Pièges et bonnes pratiques**.

Datasets : **California Housing** (régression).
<!-- #endregion -->

<!-- #region -->
## 1. Hyperparamètres vs paramètres
<!-- #endregion -->

<!-- #region -->
- **Paramètres** : appris par l'optimisation (coefs régression, poids réseau, splits arbre).
- **Hyperparamètres** : fixés **avant** training (`learning_rate`, `max_depth`, `n_estimators`, `dropout`, ...).

L'**optimisation d'hyperparamètres** (HPO) cherche le tuple qui maximise une métrique de validation.

**Types de variables HPO** :

- **Continues** : `learning_rate ∈ [1e-4, 1e-1]` (souvent log-scale).
- **Entières** : `n_estimators ∈ [50, 500]`.
- **Catégorielles** : `kernel ∈ {linear, rbf, poly}`.
- **Conditionnelles** : `if kernel='rbf': gamma ∈ [...]`. Native dans Optuna.
<!-- #endregion -->

<!-- #region -->
## 2. Espaces et distributions
<!-- #endregion -->

<!-- #region -->
| Distribution | Quand |
|---|---|
| **Uniform** | Valeur sans préférence à priori |
| **LogUniform** | Quand l'échelle "compte plus que la valeur" (learning_rate, regularization) |
| **Normal / TruncNormal** | Quand on connaît une zone probable |
| **Categorical** | Choix discret sans ordre |
| **IntUniform / IntLogUniform** | Entiers, parfois log (n_estimators) |

**Règle pratique** : un hyperparamètre comme `learning_rate` se cherche en **log** (entre 1e-4 et 1e-1 ≠ entre 0.0001 et 0.1 pour un sampler uniforme — log distribue uniformément les ordres de grandeur).
<!-- #endregion -->

<!-- #region -->
## 3. GridSearch vs RandomSearch vs HalvingSearch
<!-- #endregion -->

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# 1. GridSearch (toutes les combos) — limité ici à 8 pour aller vite
grid = GridSearchCV(
    rf, param_grid={"n_estimators": [50, 100], "max_depth": [5, 10, 15, None]},
    cv=3, scoring="r2", n_jobs=-1,
)
grid.fit(X_tr.iloc[:2000], y_tr.iloc[:2000])
print(f"Grid    best score = {grid.best_score_:.4f}  params={grid.best_params_}")

# 2. RandomSearch (échantillonnage)
rand = RandomizedSearchCV(
    rf,
    param_distributions={"n_estimators": randint(50, 200), "max_depth": randint(3, 20)},
    n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42,
)
rand.fit(X_tr.iloc[:2000], y_tr.iloc[:2000])
print(f"Random  best score = {rand.best_score_:.4f}  params={rand.best_params_}")

# 3. HalvingRandomSearch — élimine les mauvais candidats sur peu de data
halving = HalvingRandomSearchCV(
    rf,
    param_distributions={"n_estimators": randint(50, 300), "max_depth": randint(3, 20)},
    n_candidates=20, cv=3, scoring="r2", n_jobs=-1, random_state=42, factor=3,
)
halving.fit(X_tr.iloc[:2000], y_tr.iloc[:2000])
print(f"Halving best score = {halving.best_score_:.4f}  params={halving.best_params_}")
```

<!-- #region -->
## 4. Bayesian Optimization
<!-- #endregion -->

<!-- #region -->
**Idée** : maintenir un **modèle de substitution** (proxy) de la fonction objectif et choisir le prochain point à évaluer en maximisant une **fonction d'acquisition** (Expected Improvement, Upper Confidence Bound) qui équilibre **exploration** et **exploitation**.

Deux variantes :

- **Gaussian Process** (`scikit-optimize`, `BoTorch`) — sensitive au scaling, lourd au-delà de quelques centaines de trials.
- **TPE** (Tree-structured Parzen Estimator, Bergstra 2011) — utilisé par Hyperopt et Optuna. Scale mieux, supporte les variables conditionnelles.
<!-- #endregion -->

<!-- #region -->
## 5. Optuna — le standard 2026
<!-- #endregion -->

<!-- #region -->
**Optuna** est devenu le standard de fait pour la HPO en 2026 :

- **Define-by-run API** : on définit l'espace dans l'objective Python (variables conditionnelles natives).
- **Samplers** : TPESampler (par défaut), CmaEsSampler (CMA-ES), GPSampler, GridSampler, BruteForceSampler, NSGAIISampler (multi-objectif).
- **Pruners** : MedianPruner, SuccessiveHalvingPruner, HyperbandPruner — abandonnent les trials sans potentiel.
- **Distribué** facile : `study_name + storage="postgresql://..."` ou `redis://`.
- **Visualisations** : importance, parallel coordinate, contour plot, slice plot.
<!-- #endregion -->

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


def objective(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 50, 500),
        "learning_rate":  trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":     trial.suggest_int("num_leaves", 15, 255),
        "max_depth":      trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":      trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":      trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":     trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    scores = cross_val_score(model, X_tr, y_tr, cv=3, scoring="r2", n_jobs=-1)
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=15, show_progress_bar=False)
print(f"Best R²    = {study.best_value:.4f}")
print(f"Best params = {study.best_params}")
```

<!-- #region -->
### 5.1 Pruner — abandonner les mauvais trials tôt
<!-- #endregion -->

<!-- #region -->
Pour le DL ou les modèles itératifs (GBM, NN, ...), on peut reporter une métrique intermédiaire et laisser Optuna couper les essais qui démarrent mal :

```python
def objective(trial):
    model = ...  # build
    for epoch in range(n_epochs):
        train_one_epoch(model)
        val_score = evaluate(model)
        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_score


study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
```

Le **MedianPruner** abandonne les trials sous la médiane des trials précédents à la même étape. Le **HyperbandPruner** est plus agressif.
<!-- #endregion -->

<!-- #region -->
### 5.2 Visualisations
<!-- #endregion -->

```python
# Importance des hyperparamètres (via fANOVA)
try:
    importance = optuna.importance.get_param_importances(study)
    print("Param importance :")
    for k, v in importance.items():
        print(f"  {k:20s} {v:.3f}")
except Exception as e:
    print(f"Importance skip : {e}")
```

<!-- #region -->
## 6. Spécifique par framework
<!-- #endregion -->

<!-- #region -->
| Framework | Astuces HPO 2026 |
|---|---|
| **sklearn** | Pipeline + Optuna direct ou `BayesSearchCV` |
| **XGBoost** | `xgb.cv` natif + Optuna ; `tree_method='hist'`; early_stopping |
| **LightGBM** | `lgb.cv` natif ; Optuna est best-in-class (intégration tutoriels officiels) |
| **CatBoost** | `cv` natif ou Optuna ; moins de tuning utile (defaults solides) |
| **PyTorch** | Optuna + Ray Tune ; PyTorch Lightning trainer hooks |
| **Keras** | KerasTuner (Bayesian, Hyperband) ou Optuna |
| **Transformers / LLM fine-tuning** | Unsloth + Optuna ou `transformers.Trainer.hyperparameter_search` |
<!-- #endregion -->

<!-- #region -->
## 7. Multi-objective
<!-- #endregion -->

<!-- #region -->
Optimiser plusieurs objectifs en même temps (perf + vitesse, perf + taille modèle, ...) — Optuna avec `NSGAIISampler` (algo génétique) renvoie un **front de Pareto** des solutions non-dominées.

```python
def multi_objective(trial):
    n_est = trial.suggest_int("n_estimators", 50, 500)
    depth = trial.suggest_int("max_depth", 3, 15)
    model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_tr, y_tr, cv=3).mean()
    n_params = n_est * (2**min(depth, 15))  # proxy taille
    return score, -n_params   # maximize score, maximize -size (= minimize size)


# study = optuna.create_study(directions=["maximize", "maximize"], sampler=optuna.samplers.NSGAIISampler())
# study.optimize(multi_objective, n_trials=50)
# pareto = study.best_trials
```
<!-- #endregion -->

<!-- #region -->
## 8. Pièges et bonnes pratiques
<!-- #endregion -->

<!-- #region -->
- ❌ Tuner sur le test set → biais d'optimisme. **Toujours** garder le test set vierge.
- ❌ Tuner sur 1 seul split → variance haute, choix instable. CV minimum 3-5 folds.
- ❌ Trop d'hyperparamètres → curse of dim. Commencer par les 3-5 plus importants.
- ❌ Espace trop large → gaspille du budget. Itérer : large random → zoomer.
- ❌ Oublier le **random_state** dans le sampler/study → résultats non reproductibles.
- ✅ Toujours **logger** chaque trial (Optuna le fait automatiquement dans son storage).
- ✅ **Early stopping** + Pruner = combo gagnant pour les modèles itératifs.
- ✅ Pour les LLMs / modèles très chers : **PBT** (Population Based Training) ou **ASHA** plutôt que TPE.
- ✅ Tracker l'exp avec **MLflow** ou **Optuna's MLflowCallback** pour visibilité long terme (voir `ML_MLFlow_Bench`).
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [Optuna — docs officielles](https://optuna.readthedocs.io/)
- [Hyperopt — docs](http://hyperopt.github.io/hyperopt/)
- [Ray Tune — docs](https://docs.ray.io/en/latest/tune/index.html)
- [Bergstra et al. (2011) — Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- [scikit-optimize](https://scikit-optimize.github.io/)
- Notebooks liés : `ML_Regression_Classification_CV_GridSearch`, `ML_Bagging_Boosting`, `ML_MLFlow_Bench`.
<!-- #endregion -->
