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
# Optimisation d'hyperparamètres
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Wiki** sur l'**optimisation d'hyperparamètres** (HPO) : le passage obligé pour transformer un modèle baseline en modèle compétitif.

Plan :

1. **Vocabulaire** — hyperparamètres vs paramètres, types de variables.
2. **Espaces et distributions** — uniform, log-uniform, catégoriel.
3. **Stratégies scikit-learn** — Grid / Random / Halving Search.
4. **Optimisation bayésienne** — modèle de substitution, fonction d'acquisition, TPE vs GP.
5. **Optuna** — prise en main, define-by-run, samplers, pruners.
6. **Multi-framework** — XGBoost, LightGBM, CatBoost (mêmes données).
7. **Visualisations** — historique, importance des hyperparamètres, slice.
8. **Multi-objectif** — front de Pareto (performance vs complexité).
9. **Pièges et bonnes pratiques**.

Toutes les démonstrations s'exécutent sur un **seul jeu de données** : **California Housing** (régression), sous-échantillonné pour la rapidité. Version Optuna ciblée : 4.x (2026).
<!-- #endregion -->

<!-- #region -->
## 1. Hyperparamètres vs paramètres
<!-- #endregion -->

<!-- #region -->
- **Paramètres** : appris par l'entraînement (coefficients d'une régression, poids d'un réseau, seuils de split d'un arbre).
- **Hyperparamètres** : fixés **avant** l'entraînement et qui pilotent l'apprentissage (`learning_rate`, `max_depth`, `n_estimators`, `dropout`, force de régularisation...).

L'**optimisation d'hyperparamètres** (HPO) cherche le tuple d'hyperparamètres qui maximise une métrique de **validation** (jamais le test set).

**Types de variables HPO** :

- **Continues** : `learning_rate` dans `[1e-4, 1e-1]` (souvent en log).
- **Entières** : `n_estimators` dans `[50, 500]`.
- **Catégorielles** : `kernel` dans `{linear, rbf, poly}`.
- **Conditionnelles** : `if kernel == 'rbf': gamma dans [...]`. Géré nativement par Optuna (define-by-run).
<!-- #endregion -->

<!-- #region -->
## 2. Espaces et distributions
<!-- #endregion -->

<!-- #region -->
Le choix de la **distribution** d'échantillonnage est aussi important que les bornes.

| Distribution | Quand l'utiliser |
|---|---|
| **Uniform** | Aucune préférence a priori sur la plage. |
| **Log-uniform** | Quand l'ordre de grandeur compte plus que la valeur (`learning_rate`, régularisation). |
| **Normal / TruncNormal** | Quand on connaît une zone probable autour d'une valeur. |
| **Categorical** | Choix discret sans ordre naturel. |
| **Int / IntLog** | Entiers, parfois en log (`n_estimators`, `num_leaves`). |

**Règle pratique** : un `learning_rate` se cherche en **log**. Chercher uniformément dans `[1e-4, 1e-1]` passerait 90 % du budget entre `1e-2` et `1e-1` ; le log distribue uniformément les **ordres de grandeur**.
<!-- #endregion -->

<!-- #region -->
## 3. Jeu de données commun : California Housing
<!-- #endregion -->

<!-- #region -->
On charge **California Housing** (régression : prédire le prix médian d'un quartier) et on en prend un sous-échantillon pour garder les recherches rapides. Le split sépare un **test set qui restera vierge** : il ne sert jamais à tuner, uniquement à estimer la généralisation finale.
<!-- #endregion -->

```python
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_california(n_sub: int = 3000, seed: int = 42):
    """Charge California Housing (régression) et renvoie un sous-échantillon splitté.

    Args:
        n_sub: taille du sous-échantillon (rapidité du tuning).
        seed: graine de reproductibilité.

    Returns:
        (X_tr, X_te, y_tr, y_te) en DataFrame/Series pandas.
    """
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X, y = X.iloc[:n_sub], y.iloc[:n_sub]
    return train_test_split(X, y, test_size=0.2, random_state=seed)


X_tr, X_te, y_tr, y_te = load_california()
print(f"X_tr = {X_tr.shape}   X_te = {X_te.shape}")
print(f"features = {list(X_tr.columns)}")
```

<!-- #region -->
Le premier appel à `fetch_california_housing` télécharge le jeu dans `~/scikit_learn_data/` (mis en cache ensuite). La métrique utilisée partout est le **R²** estimé en validation croisée.
<!-- #endregion -->

<!-- #region -->
## 4. Grid / Random / Halving Search (scikit-learn)
<!-- #endregion -->

<!-- #region -->
Trois stratégies natives de scikit-learn, par ordre de sophistication :

- **GridSearchCV** : teste **toutes** les combinaisons d'une grille. Exhaustif mais explose en dimension.
- **RandomizedSearchCV** : échantillonne `n_iter` combinaisons. Souvent plus efficace que la grille à budget égal (Bergstra & Bengio, 2012).
- **HalvingRandomSearchCV** : *successive halving*. Évalue beaucoup de candidats avec **peu de ressources** (peu de données / d'arbres), élimine les pires, et réalloue le budget aux survivants. Rapide, mais peut éliminer un bon candidat trop tôt si le schéma est agressif.

**Note technique** : l'estimateur est mis en `n_jobs=1` ; la parallélisation se fait au niveau de la recherche (`n_jobs=-1`). Imbriquer `n_jobs=-1` à deux niveaux provoque une **sur-souscription** des cœurs (ralentissements, voire blocages avec les GBM qui utilisent OpenMP).
<!-- #endregion -->

```python
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV

rf = RandomForestRegressor(random_state=42, n_jobs=1)

# 1. GridSearch : toutes les combinaisons de la grille.
grid = GridSearchCV(
    rf,
    param_grid={"n_estimators": [50, 100], "max_depth": [5, 10, None]},
    cv=3, scoring="r2", n_jobs=-1,
)
grid.fit(X_tr, y_tr)
print(f"Grid    R²={grid.best_score_:.4f}  {grid.best_params_}")

# 2. RandomSearch : échantillonnage de l'espace.
rand = RandomizedSearchCV(
    rf,
    param_distributions={"n_estimators": randint(50, 200), "max_depth": randint(3, 20)},
    n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42,
)
rand.fit(X_tr, y_tr)
print(f"Random  R²={rand.best_score_:.4f}  {rand.best_params_}")

# 3. HalvingRandomSearch : successive halving.
halving = HalvingRandomSearchCV(
    rf,
    param_distributions={"n_estimators": randint(50, 300), "max_depth": randint(3, 20)},
    n_candidates=20, factor=3, cv=3, scoring="r2", n_jobs=-1, random_state=42,
)
halving.fit(X_tr, y_tr)
print(f"Halving R²={halving.best_score_:.4f}  {halving.best_params_}")
```

<!-- #region -->
À budget comparable, Random tient tête à Grid pour bien moins de combinaisons. Halving est le plus rapide mais son score peut être en retrait : avec peu de données au premier palier, le classement précoce des candidats est bruité. C'est un compromis vitesse / optimalité.
<!-- #endregion -->

<!-- #region -->
## 5. Optimisation bayésienne
<!-- #endregion -->

<!-- #region -->
Grid et Random sont **sans mémoire** : chaque essai ignore les précédents. L'**optimisation bayésienne** exploite l'historique.

**Principe** : maintenir un **modèle de substitution** (surrogate) qui approxime la fonction objectif, puis choisir le prochain point en maximisant une **fonction d'acquisition** qui arbitre entre **exploration** (zones incertaines) et **exploitation** (zones prometteuses). Acquisitions classiques : *Expected Improvement* (EI), *Upper Confidence Bound* (UCB).

Deux familles de surrogate :

- **Gaussian Process** (`scikit-optimize`, `BoTorch`, `GPSampler` d'Optuna) : très efficace sur peu de dimensions continues, sensible au scaling, coûteux au-delà de quelques centaines d'essais.
- **TPE** (Tree-structured Parzen Estimator, Bergstra 2011) : modélise `p(x | y)` plutôt que `p(y | x)`. Passe bien à l'échelle, gère nativement les variables **conditionnelles** et **catégorielles**. C'est le sampler par défaut d'Optuna et d'Hyperopt.
<!-- #endregion -->

<!-- #region -->
## 6. Optuna : prise en main
<!-- #endregion -->

<!-- #region -->
**Optuna** est le standard de fait pour la HPO en 2026. Son API est **define-by-run** : on définit l'espace de recherche directement dans la fonction `objective`, en Python, ce qui rend les espaces conditionnels triviaux.

Concepts :

- **`Trial`** : une exécution de l'objective. Les `suggest_*` y tirent les hyperparamètres.
- **`Study`** : la campagne d'optimisation, qui mémorise tous les trials.
- **`optimize(objective, n_trials=...)`** : lance la recherche.

On passe la verbosité en `WARNING` pour ne pas noyer la sortie sous un log par trial.
<!-- #endregion -->

```python
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
```

<!-- #region -->
Premier contact avec un objectif jouet : trouver le `x` qui minimise `(x - 2)^2` (minimum évident en `x = 2`).
<!-- #endregion -->

```python
def objective_toy(trial: optuna.Trial) -> float:
    """Fonction jouet : on cherche x qui minimise (x - 2)^2."""
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study_toy = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study_toy.optimize(objective_toy, n_trials=30)
print(f"best x = {study_toy.best_params['x']:.4f}   best value = {study_toy.best_value:.5f}")
```

<!-- #region -->
Une étude expose tout son historique : meilleur jeu de paramètres, meilleure valeur, meilleur trial, nombre d'essais.
<!-- #endregion -->

```python
print("best_params :", study_toy.best_params)
print("best_value  :", study_toy.best_value)
print("best_trial  : #", study_toy.best_trial.number)
print("n_trials    :", len(study_toy.trials))
```

<!-- #region -->
## 7. Espace de recherche : suggest_* et conditionnels
<!-- #endregion -->

<!-- #region -->
Les trois familles de suggestions couvrent tous les types d'hyperparamètres :

- **`suggest_float(name, low, high, log=..., step=...)`** : continu (option `log` pour les échelles, `step` pour discrétiser).
- **`suggest_int(name, low, high, log=..., step=...)`** : entier.
- **`suggest_categorical(name, choices)`** : choix discret.

L'objectif ci-dessous (jouet, mais réellement exécuté) tire un hyperparamètre de chaque famille.
<!-- #endregion -->

```python
def objective_types(trial: optuna.Trial) -> float:
    """Démo des trois familles de suggestions (objectif jouet exécutable)."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)   # continu, log
    n_layers = trial.suggest_int("n_layers", 1, 4)                    # entier
    n_units = trial.suggest_int("n_units", 16, 128, step=16)          # entier discrétisé
    opt = trial.suggest_categorical("optimizer", ["sgd", "adam"])     # catégoriel
    penalty = 0.0 if opt == "adam" else 0.1
    return -(np.log10(lr) + 1.5) ** 2 - 0.01 * n_layers - 0.001 * n_units - penalty


study_types = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_types.optimize(objective_types, n_trials=15)
print("best params :", study_types.best_params)
```

<!-- #region -->
**Espace conditionnel** : le define-by-run permet de faire dépendre les hyperparamètres d'un choix antérieur. Ici Optuna sélectionne le **type de modèle** (`ridge` / `rf` / `gbr`) **et** ses hyperparamètres propres, dans une seule fonction objective.
<!-- #endregion -->

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def objective_conditional(trial: optuna.Trial) -> float:
    """Espace conditionnel : Optuna choisit le modèle ET ses hyperparamètres."""
    name = trial.suggest_categorical("model", ["ridge", "rf", "gbr"])
    if name == "ridge":
        model = Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True))
    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 50, 200),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            random_state=42, n_jobs=1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("gbr_n_estimators", 50, 200),
            learning_rate=trial.suggest_float("gbr_lr", 1e-2, 0.3, log=True),
            max_depth=trial.suggest_int("gbr_max_depth", 2, 5),
            random_state=42,
        )
    return cross_val_score(model, X_tr, y_tr, cv=3, scoring="r2", n_jobs=-1).mean()


study_cond = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_cond.optimize(objective_conditional, n_trials=12)
print(f"best R² = {study_cond.best_value:.4f}   {study_cond.best_params}")
```

<!-- #region -->
Optuna explore ainsi le **choix de modèle** comme un hyperparamètre à part entière : c'est de la sélection de modèle et de tuning en une seule étude.
<!-- #endregion -->

<!-- #region -->
## 8. Samplers (algorithmes d'échantillonnage)
<!-- #endregion -->

<!-- #region -->
Le **sampler** décide quel point essayer ensuite. Panorama 2026 :

- **TPESampler** — Tree-structured Parzen Estimator. Défaut, robuste, gère le conditionnel.
- **GPSampler** — Gaussian Process. Renforcé dans Optuna 4.x (contraintes + multi-objectif, échantillonnage accéléré). Excellent sur peu de dimensions continues.
- **CmaEsSampler** — CMA-ES, efficace sur les espaces continus de dimension moyenne.
- **NSGAIISampler / NSGAIIISampler** — algorithmes génétiques pour le **multi-objectif**.
- **GridSampler / BruteForceSampler** — recherche exhaustive pilotée par Optuna.
- **QMCSampler** — quasi-Monte-Carlo (meilleure couverture que l'aléatoire pur).
- **AutoSampler** — choisit automatiquement un sampler adapté au problème (nouveauté 4.x).

On instancie quelques samplers et on lit leur nom (sans lancer d'optimisation lourde).
<!-- #endregion -->

```python
for sampler in (
    optuna.samplers.TPESampler(seed=42),
    optuna.samplers.RandomSampler(seed=42),
    optuna.samplers.CmaEsSampler(seed=42),
):
    s = optuna.create_study(sampler=sampler)
    print(s.sampler.__class__.__name__)
print("défaut :", optuna.create_study().sampler.__class__.__name__)
```

<!-- #region -->
## 9. Pruners (arrêt précoce des trials)
<!-- #endregion -->

<!-- #region -->
Pour les modèles **itératifs** (boosting, réseaux), on peut reporter une métrique intermédiaire à chaque étape et laisser Optuna **abandonner** (prune) les essais qui démarrent mal. C'est l'analogue de l'early stopping, mais au niveau de la campagne d'optimisation.

Pruners disponibles : **MedianPruner** (sous la médiane des essais précédents au même palier), **SuccessiveHalvingPruner** et **HyperbandPruner** (plus agressifs, recommandés), **PercentilePruner**, **PatientPruner**, **ThresholdPruner**, **NopPruner**.

**Note** : les *callbacks* d'intégration (`LightGBMPruningCallback`, etc.) vivent désormais dans le paquet séparé `optuna-integration`. Pour rester autonome, on démontre le pruning avec une boucle manuelle `trial.report()` / `trial.should_prune()` sur un `SGDRegressor` entraîné par `partial_fit`. Le R² est mesuré sur un **split de validation interne au train** (le test set reste vierge).
<!-- #endregion -->

```python
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Split interne train/validation : le test set (X_te) reste vierge.
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=42)
scaler = StandardScaler().fit(X_fit)
Xfit_s, Xval_s = scaler.transform(X_fit), scaler.transform(X_val)


def objective_pruning(trial: optuna.Trial) -> float:
    """Modèle itératif : R² de validation reporté à chaque epoch -> élagage possible."""
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    eta0 = trial.suggest_float("eta0", 1e-4, 1e-1, log=True)
    model = SGDRegressor(alpha=alpha, eta0=eta0, learning_rate="invscaling", random_state=42)
    score = float("-inf")
    for epoch in range(30):
        model.partial_fit(Xfit_s, y_fit)
        score = r2_score(y_val, model.predict(Xval_s))
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return score


study_prune = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)
study_prune.optimize(objective_pruning, n_trials=20)
pruned = [t for t in study_prune.trials if t.state == optuna.trial.TrialState.PRUNED]
complete = [t for t in study_prune.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"complétés = {len(complete)}   élagués = {len(pruned)}   best R² = {study_prune.best_value:.4f}")
```

<!-- #region -->
Une part substantielle des essais est coupée avant la fin : autant de budget réaffecté aux trials prometteurs.
<!-- #endregion -->

<!-- #region -->
## 10. Tuning multi-framework sur California Housing
<!-- #endregion -->

<!-- #region -->
Même jeu de données, même protocole (R² en CV 3 folds), trois frameworks de gradient boosting (XGBoost, LightGBM, CatBoost). On factorise le lancement d'une étude dans une petite fonction réutilisable qui renvoie l'objet `Study` (réutilisé ensuite pour les visualisations).
<!-- #endregion -->

```python
from typing import Callable


def run_study(objective: Callable[[optuna.Trial], float], n_trials: int, name: str) -> optuna.Study:
    """Lance une étude TPE reproductible et renvoie l'objet Study.

    Args:
        objective: fonction objective Optuna à maximiser.
        n_trials: nombre d'essais.
        name: étiquette affichée dans le log.

    Returns:
        L'objet `optuna.Study` complet (best_value, best_params, historique des trials).
    """
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    print(f"[{name}] best R²={study.best_value:.4f}  best params={study.best_params}")
    return study
```

<!-- #region -->
### 10.1 XGBoost
<!-- #endregion -->

<!-- #region -->
Espace classique pour `XGBRegressor` : nombre d'arbres, `learning_rate` (log), profondeur, sous-échantillonnage des lignes et colonnes, régularisations L1/L2 (log). `tree_method="hist"` accélère l'entraînement.
<!-- #endregion -->

```python
from xgboost import XGBRegressor


def objective_xgb(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = XGBRegressor(tree_method="hist", random_state=42, n_jobs=1, **params)
    return cross_val_score(model, X_tr, y_tr, cv=3, scoring="r2", n_jobs=-1).mean()


study_xgb = run_study(objective_xgb, n_trials=20, name="xgboost")
r2_xgb = study_xgb.best_value
```

<!-- #region -->
### 10.2 LightGBM
<!-- #endregion -->

<!-- #region -->
Pour `LGBMRegressor`, le levier central est `num_leaves` (complexité de l'arbre) à coupler avec `learning_rate` ; `feature_fraction` / `bagging_fraction` apportent de la régularisation par échantillonnage. Un espace Optuna typique :

| Hyperparamètre | Suggestion Optuna |
|---|---|
| `n_estimators` | `suggest_int(100, 500)` |
| `learning_rate` | `suggest_float(1e-3, 0.3, log=True)` |
| `num_leaves` | `suggest_int(15, 255)` — **levier #1** |
| `feature_fraction` | `suggest_float(0.5, 1.0)` |
| `bagging_fraction` | `suggest_float(0.5, 1.0)` |
| `min_child_samples` | `suggest_int(5, 100)` |
| `reg_alpha` / `reg_lambda` | `suggest_float(1e-8, 10.0, log=True)` |

La fonction objective serait **strictement identique** à celle de XGBoost ci-dessus, en remplaçant le modèle par `LGBMRegressor(random_state=42, n_jobs=1, verbose=-1, **params)` puis `run_study(objective_lgbm, 20, "lightgbm")` — c'est tout l'intérêt du define-by-run : changer de framework ne change pas la mécanique d'optimisation.

> ⚠️ **Note d'environnement** : dans le runtime de ce notebook, la bibliothèque système **`libgomp` (OpenMP)** requise par LightGBM est absente (`OSError: libgomp.so.1` au `.fit()`). L'exemple exécuté de cette section se limite donc à **XGBoost** et **CatBoost** (dont l'OpenMP est embarqué dans le wheel). Sur une machine où LightGBM s'installe normalement, le bloc ci-dessus tourne tel quel.
<!-- #endregion -->

<!-- #region -->
### 10.3 CatBoost
<!-- #endregion -->

<!-- #region -->
`CatBoostRegressor` arrive souvent à de bons résultats avec peu de tuning. On garde `iterations` modeste pour la rapidité, `verbose=0` et `allow_writing_files=False` pour ne pas polluer le disque, `thread_count=1` pour éviter la sur-souscription avec la CV parallèle.
<!-- #endregion -->

```python
from catboost import CatBoostRegressor


def objective_cat(trial: optuna.Trial) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }
    model = CatBoostRegressor(
        random_state=42, verbose=0, allow_writing_files=False, thread_count=1, **params
    )
    return cross_val_score(model, X_tr, y_tr, cv=3, scoring="r2", n_jobs=-1).mean()


study_cat = run_study(objective_cat, n_trials=12, name="catboost")
r2_cat = study_cat.best_value
```

<!-- #region -->
Récapitulatif des frameworks exécutés sur le même jeu de données : les R² sont très proches, ce qui est attendu sur un tabulaire de cette taille (LightGBM donnerait un score du même ordre, cf. note d'environnement ci-dessus).
<!-- #endregion -->

```python
print("R² CV par framework :")
for name, val in [("XGBoost", r2_xgb), ("CatBoost", r2_cat)]:
    print(f"  {name:10s} {val:.4f}")
```

<!-- #region -->
## 11. Visualisations
<!-- #endregion -->

<!-- #region -->
L'**importance des hyperparamètres** (calculée par fANOVA) indique sur quels leviers s'est jouée la performance. Sur les GBM, `learning_rate` domine presque toujours.
<!-- #endregion -->

```python
importance = optuna.importance.get_param_importances(study_xgb)
print("Importance des hyperparamètres (XGBoost) :")
for k, v in importance.items():
    print(f"  {k:18s} {v:.3f}")
```

<!-- #region -->
Optuna fournit des graphiques prêts à l'emploi via `optuna.visualization.matplotlib` :

- **optimization_history** : la meilleure valeur au fil des trials.
- **param_importances** : l'importance fANOVA en barres.
- **slice** : la valeur objective en fonction de chaque hyperparamètre.
<!-- #endregion -->

```python
import optuna.visualization.matplotlib as ovm

ovm.plot_optimization_history(study_xgb)
plt.show()

ovm.plot_param_importances(study_xgb)
plt.show()

ovm.plot_slice(study_xgb)
plt.show()
```

<!-- #region -->
Le slice plot aide à resserrer l'espace : si une plage n'est jamais visitée par les bons trials, on peut la retirer et relancer une recherche plus ciblée.
<!-- #endregion -->

<!-- #region -->
## 12. Optimisation multi-objectif
<!-- #endregion -->

<!-- #region -->
On veut souvent optimiser **plusieurs objectifs antagonistes** : performance vs vitesse, performance vs taille du modèle. Il n'existe alors pas une solution unique mais un **front de Pareto** : l'ensemble des solutions qu'on ne peut pas améliorer sur un objectif sans dégrader l'autre.

Optuna gère ça avec `directions=[...]` et le **NSGAIISampler** (algorithme génétique). On maximise le R² et on minimise une **complexité** proxy (`n_estimators * max_depth`). `study.best_trials` renvoie le front.
<!-- #endregion -->

```python
def objective_multi(trial: optuna.Trial) -> tuple[float, float]:
    """Deux objectifs : maximiser le R² ET minimiser la complexité du modèle."""
    n_est = trial.suggest_int("n_estimators", 50, 400)
    depth = trial.suggest_int("max_depth", 2, 15)
    model = RandomForestRegressor(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=1)
    r2 = cross_val_score(model, X_tr, y_tr, cv=3, scoring="r2", n_jobs=-1).mean()
    complexity = n_est * depth  # proxy simple de la taille du modèle
    return r2, -complexity  # maximize r2, maximize -complexity (= minimiser la complexité)


study_mo = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler(seed=42),
)
study_mo.optimize(objective_multi, n_trials=25)
print(f"{len(study_mo.best_trials)} solutions sur le front de Pareto")
```

<!-- #region -->
On visualise tous les trials et on surligne le front de Pareto.
<!-- #endregion -->

```python
r2_all = [t.values[0] for t in study_mo.trials]
cplx_all = [-t.values[1] for t in study_mo.trials]
r2_par = [t.values[0] for t in study_mo.best_trials]
cplx_par = [-t.values[1] for t in study_mo.best_trials]

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(cplx_all, r2_all, c="lightgray", label="tous les trials")
ax.scatter(cplx_par, r2_par, c="crimson", label="front de Pareto")
ax.set_xlabel("Complexité (n_estimators × max_depth)")
ax.set_ylabel("R² (CV)")
ax.set_title("Front de Pareto : performance vs complexité")
ax.legend()
plt.show()
```

<!-- #region -->
Aucune solution du front n'en domine une autre : le choix final dépend de la contrainte métier (budget d'inférence, taille déployable).
<!-- #endregion -->

<!-- #region -->
## 13. Astuces HPO par framework (2026)
<!-- #endregion -->

<!-- #region -->
| Framework | Astuces HPO |
|---|---|
| **scikit-learn** | Pipeline + Optuna, ou `HalvingRandomSearchCV` natif pour un baseline rapide. |
| **XGBoost** | `tree_method="hist"`, early stopping sur un set de validation, `xgb.cv` natif possible. |
| **LightGBM** | Cible `num_leaves` + `learning_rate` ; intégration Optuna best-in-class (`optuna-integration`). |
| **CatBoost** | Defaults solides ; tuner surtout `depth`, `learning_rate`, `l2_leaf_reg`. |
| **PyTorch** | Optuna + pruning manuel, ou **Ray Tune** (ASHA) ; hooks PyTorch Lightning. |
| **Keras / TensorFlow** | **KerasTuner** (Bayesian, Hyperband) ou Optuna avec un callback de pruning. |
| **Transformers / LLM** | `transformers.Trainer.hyperparameter_search` (backend Optuna ou Ray) ; budgets très chers -> ASHA / PBT. |
<!-- #endregion -->

<!-- #region -->
## 14. Paysage des outils (2026)
<!-- #endregion -->

<!-- #region -->
- **Optuna** — single-node par défaut, parallélisable (storage partagé `postgresql://` / `redis://`), samplers et pruners variés. Référence pour le tabulaire et le ML classique.
- **Ray Tune** — orienté **distribué**, schedulers avancés (ASHA, PBT), remplaçant naturel d'Hyperopt à grande échelle.
- **scikit-learn Halving** — successive halving intégré, sans dépendance, idéal pour un premier balayage.
- **KerasTuner** — spécialisé Keras/TensorFlow (Hyperband natif).
- **Intégrations Optuna** — déplacées dans le paquet `optuna-integration` (callbacks LightGBM/XGBoost/Keras, `MLflowCallback`). Pour le suivi long terme des campagnes, coupler avec MLflow (voir `ML_MLFlow_Bench`).
<!-- #endregion -->

<!-- #region -->
## 15. Pièges et bonnes pratiques
<!-- #endregion -->

<!-- #region -->
- **Ne jamais tuner sur le test set.** Toujours garder un set d'évaluation vierge, sinon le score final est biaisé par optimisme.
- **Au moins 3 à 5 folds de CV.** Tuner sur un seul split donne une variance élevée et des choix instables.
- **Commencer par les 3 à 5 hyperparamètres les plus importants.** Trop de dimensions = malédiction de la dimensionnalité, budget gaspillé.
- **Itérer sur l'espace.** Large balayage aléatoire d'abord, puis resserrement (slice plot / importance) autour des zones prometteuses.
- **Fixer les graines** (`sampler`, `study`, modèles) pour des résultats reproductibles.
- **Logger chaque trial** : Optuna le fait automatiquement dans son storage ; persister l'étude permet de reprendre / analyser après coup.
- **Early stopping + pruner** : combo gagnant pour les modèles itératifs (GBM, réseaux).
- **Modèles très coûteux** (LLM, gros réseaux) : préférer **ASHA** ou **PBT** au TPE seul.
<!-- #endregion -->

<!-- #region -->
## 16. Sources
<!-- #endregion -->

<!-- #region -->
- [Optuna — documentation officielle](https://optuna.readthedocs.io/)
- [Ray Tune — documentation](https://docs.ray.io/en/latest/tune/index.html)
- [scikit-optimize](https://scikit-optimize.github.io/)
- [Bergstra et al. (2011) — Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- [Bergstra & Bengio (2012) — Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
- Notebooks liés : `ML_Regression_Classification_CV_GridSearch`, `ML_Bagging_Boosting`, `ML_MLFlow_Bench`.
<!-- #endregion -->
