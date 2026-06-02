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
# Bagging & Boosting — méthodes d'ensemble
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel + Cheat-sheet** sur les **méthodes d'ensemble** : du compromis biais-variance qui les justifie, jusqu'à la trinité moderne **XGBoost / LightGBM / CatBoost**.

**Fil rouge** : un modèle se trompe pour trois raisons — biais, variance, bruit irréductible. Le **bagging** attaque la variance, le **boosting** attaque le biais. On le montre empiriquement avant de dérouler chaque famille.

**Plan**

1. Décomposition biais-variance (maths).
2-3. Décomposition empirique sur un k-NN (processus génératif connu).
4-7. **Bagging** : bootstrap/OOB, Random Forest, Extra-Trees.
8-10. **Boosting** : AdaBoost, Gradient Boosting, HistGradientBoosting.
11. La **trinité 2026** : XGBoost / LightGBM / CatBoost (+ early stopping, importances).
12-13. **Combiner** : Voting, Stacking, benchmark récapitulatif.
14-16. **Classification** sur `breast_cancer`.
17-18. Conseils pratiques 2026 + sources.

**Datasets** : un générateur **synthétique** (pour la décomposition biais-variance, qui exige un processus génératif connu), **California Housing** (benchmark régression réel, mutualisé) et **breast_cancer** (classification).
<!-- #endregion -->

<!-- #region -->
La cellule suivante regroupe les imports transverses, la palette de couleurs, un helper `rmse`, et charge California Housing (réutilisé dans toute la partie régression).
<!-- #endregion -->

```python
import time

import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42

# Palette de couleurs cohérente
PRIMARY_1, MAUVAIS, MOYEN, ACCENT = "#00798c", "#d1495b", "#edae49", "#66a182"
ACCENT_DARK = "#2e4057"


def rmse(model, X: np.ndarray, y: np.ndarray) -> float:
    """Racine de l'erreur quadratique moyenne d'un modèle sur (X, y)."""
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


# Dataset régression réel mutualisé
cal = fetch_california_housing(as_frame=True)
X_cal, y_cal = cal.data, cal.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X_cal, y_cal, test_size=0.2, random_state=RANDOM_STATE
)

# Collecteur de scores pour le benchmark récapitulatif (§13)
reg_scores: dict[str, float] = {}
print(f"California Housing : train={X_tr.shape}, test={X_te.shape}")
```

<!-- #region -->
## 1. Décomposition biais-variance
<!-- #endregion -->

<!-- #region -->
Soit une relation vraie $Y = f^*(X) + \varepsilon$ avec $\mathbb{E}[\varepsilon]=0$ et $\text{Var}(\varepsilon)=\sigma^2$. Pour un estimateur $\hat f$ appris sur un échantillon aléatoire $D$, l'erreur quadratique moyenne en un point $x$ se décompose en **trois termes** :

$$
\mathbb{E}_D\!\left[(\hat f(x) - f^*(x))^2\right]
= \underbrace{\left(\mathbb{E}_D[\hat f(x)] - f^*(x)\right)^2}_{\text{Biais}^2}
+ \underbrace{\mathbb{E}_D\!\left[(\hat f(x) - \mathbb{E}_D[\hat f(x)])^2\right]}_{\text{Variance}}
+ \underbrace{\sigma^2}_{\text{Bruit}}
$$

- **Biais²** : écart entre la prédiction *moyenne* du modèle et la vérité. Élevé pour les modèles rigides (sous-ajustement).
- **Variance** : sensibilité du modèle au tirage de l'échantillon d'apprentissage. Élevée pour les modèles flexibles (sur-ajustement).
- **Bruit** $\sigma^2$ : limite incompressible, indépendante du modèle.

| Modèle | Biais | Variance | Risque |
|---|---|---|---|
| Très simple (sous-ajustement) | Élevé | Faible | Élevé |
| Très complexe (sur-ajustement) | Faible | Élevé | Élevé |
| Optimal | Équilibré | Équilibré | Minimum |

Tout l'art des ensembles est de **déplacer un seul terme** sans gonfler les autres : le bagging vise la variance, le boosting vise le biais.
<!-- #endregion -->

<!-- #region -->
## 2. Un processus génératif connu
<!-- #endregion -->

<!-- #region -->
Estimer empiriquement biais² et variance exige l'espérance $\mathbb{E}_D$ sur **plusieurs** jeux d'entraînement issus du *même* processus. Un dataset réel ne le permet pas (on n'a qu'un seul tirage) : on se donne donc un **générateur synthétique** dont on contrôle le bruit irréductible.

$$
y = \sin^2(x_1) + \cos^2(x_2) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\ \sigma^2=1/10)
$$

La fonction `bias_variance_decomposition` ré-entraîne un estimateur sur `n_train` jeux tirés du générateur et mesure ses prédictions sur un jeu de test fixe — d'où les estimations de biais² (variabilité de la moyenne vs vérité) et de variance (dispersion des prédictions).
<!-- #endregion -->

```python
def make_synthetic_regression(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Génère (X, y) depuis un processus génératif connu.

    y = sin²(x1) + cos²(x2) + ε,  ε ~ N(0, σ²=1/10),  x1, x2 ~ U(-1, 1).
    X est de forme (n_samples, 2).
    """
    rng = npr.default_rng(seed)
    x1 = rng.uniform(-1, 1, size=n_samples)
    x2 = rng.uniform(-1, 1, size=n_samples)
    noise = rng.normal(scale=np.sqrt(1 / 10), size=n_samples)
    y = np.sin(x1) ** 2 + np.cos(x2) ** 2 + noise
    return np.stack((x1, x2), axis=1), y


SIGMA2 = 1 / 10  # bruit irréductible du générateur


def bias_variance_decomposition(
    model_factory,
    *,
    param_values: np.ndarray,
    n_train: int,
    T_train: int,
    T_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estime biais² moyen et variance moyenne d'un estimateur paramétré.

    `model_factory(p)` renvoie un estimateur non entraîné pour le paramètre p
    (nb de voisins ou profondeur). Renvoie (bias2, var) alignés sur param_values.
    """
    X_test, y_test = make_synthetic_regression(T_test, seed=999)
    bias2 = np.zeros(len(param_values))
    var = np.zeros(len(param_values))
    for j, p in enumerate(param_values):
        preds = np.zeros((n_train, T_test))
        for i in range(n_train):
            X_train, y_train = make_synthetic_regression(T_train, seed=i)
            model = model_factory(int(p))
            model.fit(X_train, y_train)
            preds[i] = model.predict(X_test)
        mean_pred = preds.mean(axis=0)
        bias2[j] = np.mean((mean_pred - y_test) ** 2)
        var[j] = np.mean((preds - mean_pred) ** 2)
    return bias2, var
```

<!-- #region -->
## 3. Biais-variance empirique sur un k-NN
<!-- #endregion -->

<!-- #region -->
Le **k plus proches voisins** est idéal pour visualiser le compromis : `k=1` colle au bruit (variance maximale, biais minimal), un `k` grand lisse tout (biais qui monte, variance qui s'effondre). On trace biais², variance et leur somme (le risque) en fonction de `k`.
<!-- #endregion -->

```python
from sklearn.neighbors import KNeighborsRegressor

ks = np.arange(1, 11)
bias2_knn, var_knn = bias_variance_decomposition(
    lambda k: KNeighborsRegressor(n_neighbors=k),
    param_values=ks, n_train=20, T_train=200, T_test=50,
)
risk_knn = bias2_knn + var_knn + SIGMA2

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ks, bias2_knn, marker="o", color=PRIMARY_1, label="Biais² moyen")
ax.plot(ks, var_knn, marker="s", color=MAUVAIS, label="Variance moyenne")
ax.plot(ks, risk_knn, marker="^", color=ACCENT_DARK, label="Risque ≈ biais²+var+σ²")
ax.set_xlabel("Nombre de voisins k")
ax.set_ylabel("Erreur moyenne")
ax.set_title("Décomposition biais-variance — k-NN régression")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
print(f"k au risque minimal = {ks[np.argmin(risk_knn)]}")
```

<!-- #region -->
On lit directement le compromis : la **variance** chute quand `k` croît (on moyenne plus de voisins), tandis que le **biais²** remonte lentement (on lisse trop). Le risque est minimal à un `k` intermédiaire.
<!-- #endregion -->

<!-- #region -->
## 4. Bootstrap & Out-Of-Bag
<!-- #endregion -->

<!-- #region -->
Un **échantillon bootstrap** est un tirage **avec remise** de `n` éléments dans un dataset de taille `n`. La probabilité qu'une observation donnée ne soit *jamais* tirée est $(1 - 1/n)^n \to e^{-1} \approx 0{,}37$.

Conséquence : chaque bootstrap contient ~63 % des observations (le reste dupliqué), et ~37 % restent **hors-sac** (*Out-Of-Bag*, OOB). Ces OOB n'ont pas servi à entraîner l'arbre correspondant → ils fournissent une **estimation intégrée** de l'erreur de généralisation, sans set de validation séparé (`oob_score=True`).
<!-- #endregion -->

<!-- #region -->
## 5. Le bagging réduit la variance
<!-- #endregion -->

<!-- #region -->
**Bagging** (*Bootstrap Aggregating*, Breiman 1996) : entraîner `M` modèles sur `M` bootstraps différents, puis **moyenner** leurs prédictions. Si les modèles sont décorrélés, la variance de la moyenne est divisée par `M` (loi des grands nombres) tandis que le **biais reste inchangé**.

On le vérifie : on reprend la décomposition du k-NN, mais cette fois avec un `BaggingRegressor(KNeighborsRegressor)`, et on compare la variance à celle du k-NN seul.
<!-- #endregion -->

```python
from sklearn.ensemble import BaggingRegressor

bias2_bag, var_bag = bias_variance_decomposition(
    lambda k: BaggingRegressor(
        estimator=KNeighborsRegressor(n_neighbors=k),
        n_estimators=20, random_state=RANDOM_STATE,
    ),
    param_values=ks, n_train=20, T_train=200, T_test=50,
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ks, var_knn, marker="o", color=PRIMARY_1, label="Variance k-NN seul")
ax.plot(ks, var_bag, marker="s", color=ACCENT, label="Variance k-NN baggé (M=20)")
ax.set_xlabel("Nombre de voisins k")
ax.set_ylabel("Variance moyenne")
ax.set_title("Le bagging réduit la variance")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
print(f"Variance moyenne : k-NN seul={var_knn.mean():.4f} → baggé={var_bag.mean():.4f}")
```

<!-- #region -->
## 6. Arbres : seul vs baggé
<!-- #endregion -->

<!-- #region -->
Les **arbres de décision profonds** sont le candidat parfait au bagging : faible biais, **variance énorme**. On trace d'abord l'erreur train/test d'un arbre seul en fonction de la profondeur sur le jeu synthétique : l'écart train/test qui se creuse est la signature du surapprentissage (variance).
<!-- #endregion -->

```python
from sklearn.tree import DecisionTreeRegressor

X_syn_tr, y_syn_tr = make_synthetic_regression(600, seed=0)
X_syn_te, y_syn_te = make_synthetic_regression(200, seed=1)
depths = np.arange(1, 16)


def depth_curve(model_factory) -> tuple[np.ndarray, np.ndarray]:
    """Erreur MSE train/test vs profondeur sur le jeu synthétique."""
    train_err = np.zeros(len(depths))
    test_err = np.zeros(len(depths))
    for j, p in enumerate(depths):
        m = model_factory(int(p))
        m.fit(X_syn_tr, y_syn_tr)
        train_err[j] = mean_squared_error(y_syn_tr, m.predict(X_syn_tr))
        test_err[j] = mean_squared_error(y_syn_te, m.predict(X_syn_te))
    return train_err, test_err


tree_train, tree_test = depth_curve(
    lambda p: DecisionTreeRegressor(max_depth=p, random_state=RANDOM_STATE)
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, tree_train, marker="o", color=PRIMARY_1, label="Train")
ax.plot(depths, tree_test, marker="s", color=MAUVAIS, label="Test")
ax.set_xlabel("Profondeur max")
ax.set_ylabel("MSE")
ax.set_title("Arbre de régression seul — surapprentissage en profondeur")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

<!-- #region -->
Même expérience avec un **bagging d'arbres** (`M=20`) : l'erreur de test se stabilise et descend plus bas — la moyenne absorbe la variance des arbres individuels.
<!-- #endregion -->

```python
bag_train, bag_test = depth_curve(
    lambda p: BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=p, random_state=RANDOM_STATE),
        n_estimators=20, random_state=RANDOM_STATE,
    )
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, tree_test, marker="o", color=MAUVAIS, label="Arbre seul (test)")
ax.plot(depths, bag_test, marker="s", color=ACCENT, label="Bagging d'arbres (test)")
ax.set_xlabel("Profondeur max")
ax.set_ylabel("MSE test")
ax.set_title("Bagging : stabilise l'erreur de test en profondeur")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
print(f"Min MSE test : arbre={tree_test.min():.4f} | baggé={bag_test.min():.4f}")
```

<!-- #region -->
## 7. Random Forest & Extra-Trees
<!-- #endregion -->

<!-- #region -->
**Random Forest** (Breiman 2001) = bagging d'arbres **+** à chaque split, on ne considère qu'un **sous-ensemble aléatoire des features** ($\sqrt{d}$ en classification, $d/3$ en régression). Cette double randomisation **décorrèle** les arbres → la variance baisse plus efficacement que le bagging seul.

**Extra-Trees** (*Extremely Randomized Trees*) pousse plus loin : au lieu de chercher le meilleur seuil par feature, on en tire au hasard et on garde le meilleur. Plus rapide, souvent un poil meilleur sur données bruitées.

On passe maintenant au **dataset réel** (California Housing). Le `oob_score_` de la forêt est un R² estimé sur les observations hors-sac.

**Hyperparamètres-clés** : `n_estimators` (plus = mieux jusqu'à un plateau), `max_depth`/`min_samples_leaf` (complexité par arbre), `max_features` (taille du subset), `n_jobs=-1` (parallélisation).
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

rf = RandomForestRegressor(
    n_estimators=200, oob_score=True, random_state=RANDOM_STATE, n_jobs=-1,
).fit(X_tr, y_tr)
et = ExtraTreesRegressor(
    n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
).fit(X_tr, y_tr)

reg_scores["RandomForest"] = rmse(rf, X_te, y_te)
reg_scores["ExtraTrees"] = rmse(et, X_te, y_te)
print(f"Random Forest RMSE = {reg_scores['RandomForest']:.4f}  (OOB R² = {rf.oob_score_:.4f})")
print(f"Extra-Trees   RMSE = {reg_scores['ExtraTrees']:.4f}")
```

<!-- #region -->
## 8. AdaBoost
<!-- #endregion -->

<!-- #region -->
On change de stratégie : le **boosting** entraîne des modèles **faibles** (high bias) **séquentiellement**, chacun corrigeant les erreurs du précédent → il réduit le **biais**.

**AdaBoost** (*Adaptive Boosting*, Freund & Schapire 1995) :

1. Entraîner un classifieur faible sur les données pondérées.
2. Calculer son taux d'erreur pondéré $\varepsilon$.
3. Lui donner le poids $\alpha = \tfrac{1}{2}\ln\!\frac{1-\varepsilon}{\varepsilon}$ (plus il est bon, plus il pèse).
4. **Augmenter** le poids des exemples mal classés, baisser celui des bien classés.
5. Répéter. Prédiction finale = somme pondérée des $M$ modèles.

Limite : sensible aux **outliers**, qui finissent par monopoliser les poids — c'est pourquoi le gradient boosting l'a largement supplanté. (Ici en régression, AdaBoost est nettement plus faible que les autres — c'est attendu.)
<!-- #endregion -->

```python
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
    n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE,
).fit(X_tr, y_tr)
reg_scores["AdaBoost"] = rmse(ada, X_te, y_te)
print(f"AdaBoost RMSE = {reg_scores['AdaBoost']:.4f}")
```

<!-- #region -->
## 9. Gradient Boosting
<!-- #endregion -->

<!-- #region -->
**Gradient Boosting** (Friedman 1999) généralise AdaBoost : à chaque itération, le nouveau modèle est entraîné pour prédire le **gradient négatif de la loss** par rapport aux prédictions courantes (= les **résidus** pour la loss MSE). On construit une suite additive :

$$
F_{m+1}(x) = F_m(x) + \nu \cdot h_m(x), \qquad h_m \approx -\nabla_F\, L(y, F_m(x))
$$

- $\nu$ : **learning rate** (shrinkage), typiquement 0,01-0,1. Plus petit = plus précis mais plus d'arbres requis.
- $h_m$ : weak learner (arbre peu profond, `max_depth` 3-8).
- `subsample < 1` : *stochastic gradient boosting*, ré-introduit de l'aléa → réduit la variance.
<!-- #endregion -->

```python
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=RANDOM_STATE,
).fit(X_tr, y_tr)
reg_scores["GradientBoosting"] = rmse(gbm, X_te, y_te)
print(f"GradientBoosting RMSE = {reg_scores['GradientBoosting']:.4f}")
```

<!-- #region -->
Symétriquement à la section 5 (où le bagging écrasait la variance), on décompose biais-variance d'un GBM en fonction de la profondeur des arbres faibles : le **biais² diminue** à mesure que le boosting affine — c'est l'effet attendu du boosting.
<!-- #endregion -->

```python
gb_depths = np.arange(1, 9)
bias2_gb, var_gb = bias_variance_decomposition(
    lambda p: GradientBoostingRegressor(
        max_depth=p, n_estimators=50, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    param_values=gb_depths, n_train=10, T_train=200, T_test=50,
)
risk_gb = bias2_gb + var_gb + SIGMA2

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gb_depths, bias2_gb, marker="o", color=PRIMARY_1, label="Biais² moyen")
ax.plot(gb_depths, var_gb, marker="s", color=MAUVAIS, label="Variance moyenne")
ax.plot(gb_depths, risk_gb, marker="^", color=ACCENT_DARK, label="Risque")
ax.set_xlabel("Profondeur des arbres faibles")
ax.set_ylabel("Erreur moyenne")
ax.set_title("Boosting : décomposition biais-variance")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
print(f"Biais² GBM : {bias2_gb[0]:.3f} → {bias2_gb[-1]:.3f}")
```

<!-- #region -->
## 10. HistGradientBoosting
<!-- #endregion -->

<!-- #region -->
`GradientBoostingRegressor` est pédagogique mais lent. `HistGradientBoostingRegressor` est l'implémentation **moderne** de sklearn : elle **binne** les features en histogrammes (comme LightGBM), supporte nativement les **catégorielles** et les **NaN**, et passe à l'échelle sur de gros volumes. C'est le pont naturel vers les bibliothèques dédiées.
<!-- #endregion -->

```python
from sklearn.ensemble import HistGradientBoostingRegressor

t0 = time.perf_counter()
hgb = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE,
).fit(X_tr, y_tr)
reg_scores["HistGradientBoosting"] = rmse(hgb, X_te, y_te)
print(f"HistGradientBoosting RMSE = {reg_scores['HistGradientBoosting']:.4f} "
      f"(entraîné en {time.perf_counter() - t0:.2f}s)")
```

<!-- #region -->
## 11. La trinité 2026 — XGBoost / LightGBM / CatBoost
<!-- #endregion -->

<!-- #region -->
Trois implémentations dominent le tabulaire. Un benchmark 2025 sur 111 datasets confirme qu'elles **égalent ou dépassent le deep learning** sur données tabulaires.

| Aspect | **XGBoost** | **LightGBM** | **CatBoost** |
|---|---|---|---|
| Auteur (année) | DMLC (2014) | Microsoft (2016) | Yandex (2017) |
| Croissance des arbres | level-wise | **leaf-wise** (profond, rapide) | symétrique (oblivious) |
| Vitesse training | bonne | **très rapide** | bonne |
| Catégorielles natives | non (encodage requis) | partiel | **oui** (ordered target encoding) |
| GPU | oui | oui | oui |
| Régularisation | L1 + L2 | L1 + L2 | L2 + ordered boosting |
| Résistance à l'overfit | bonne | sensible (à régulariser) | **très bonne** |
| Quand l'utiliser | baseline robuste, large communauté | gros volumes numériques, itération rapide | beaucoup de catégorielles |

**Pratique 2026** : démarrer par **LightGBM** (rapide, pair bien avec Optuna). Beaucoup de catégorielles à forte cardinalité → **CatBoost**. Baseline polyvalente avec forte régularisation → **XGBoost**.
<!-- #endregion -->

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

xgb_model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
).fit(X_tr, y_tr)
reg_scores["XGBoost"] = rmse(xgb_model, X_te, y_te)

lgb_model = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
).fit(X_tr, y_tr)
reg_scores["LightGBM"] = rmse(lgb_model, X_te, y_te)

cb_model = CatBoostRegressor(
    iterations=300, learning_rate=0.05, depth=6,
    random_state=RANDOM_STATE, verbose=False,
).fit(X_tr, y_tr)
reg_scores["CatBoost"] = rmse(cb_model, X_te, y_te)

print(f"XGBoost  RMSE = {reg_scores['XGBoost']:.4f}")
print(f"LightGBM RMSE = {reg_scores['LightGBM']:.4f}")
print(f"CatBoost RMSE = {reg_scores['CatBoost']:.4f}")
```

<!-- #region -->
### 11.1 Early stopping
<!-- #endregion -->

<!-- #region -->
Toutes supportent l'**early stopping** : on fixe `n_estimators` très grand et on arrête dès que la métrique sur un set de validation stagne pendant `early_stopping_rounds` itérations. C'est la façon propre de calibrer le nombre d'arbres sans grid-search.
<!-- #endregion -->

```python
X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE
)

lgb_es = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.05, num_leaves=63,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)
lgb_es.fit(
    X_tr2, y_tr2, eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

xgb_es = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.05, max_depth=6, tree_method="hist",
    early_stopping_rounds=50, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)
xgb_es.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)

print(f"LightGBM : arrêt à {lgb_es.best_iteration_} arbres (sur 2000)")
print(f"XGBoost  : arrêt à {xgb_es.best_iteration} arbres (sur 2000)")
```

<!-- #region -->
### 11.2 Importance des features
<!-- #endregion -->

<!-- #region -->
Les modèles d'arbres exposent une importance par feature (gain ou nombre de splits). Utile pour un premier tri, mais pour une attribution fiable (effets, interactions, signe), préférer **SHAP** — voir `ML_Explication_Feature_Importance_Selection`.
<!-- #endregion -->

```python
imp = (
    pd.DataFrame({
        "feature": X_cal.columns,
        "xgb": xgb_model.feature_importances_,
        "lgb": lgb_model.feature_importances_ / lgb_model.feature_importances_.sum(),
    })
    .sort_values("xgb", ascending=False)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(imp))
ax.barh(y_pos - 0.2, imp["xgb"], height=0.4, color=PRIMARY_1, label="XGBoost")
ax.barh(y_pos + 0.2, imp["lgb"], height=0.4, color=ACCENT, label="LightGBM")
ax.set_yticks(y_pos)
ax.set_yticklabels(imp["feature"])
ax.invert_yaxis()
ax.set_xlabel("Importance (normalisée)")
ax.set_title("Importance des features — XGBoost vs LightGBM")
ax.legend()
plt.show()
print(imp.to_string(index=False))
```

<!-- #region -->
## 12. Voting & Stacking
<!-- #endregion -->

<!-- #region -->
Au-delà d'une seule famille, on peut **combiner des modèles diversifiés** :

- **Voting** (`VotingRegressor`) : moyenne brute des prédictions. Simple, robuste.
- **Stacking** (`StackingRegressor`) : un **méta-modèle** (souvent linéaire, ici `RidgeCV`) apprend *comment* combiner les modèles de base, à partir de prédictions **out-of-fold** (CV interne) pour éviter le leak.

Le stacking gagne typiquement +0,5 à +2 % vs le meilleur modèle individuel, **à condition** que les modèles de base soient diversifiés (familles différentes). Coût : on multiplie le temps d'entraînement.
<!-- #endregion -->

```python
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

base_estimators = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    ("lgb", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ("hgb", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          random_state=RANDOM_STATE)),
]

voting = VotingRegressor(estimators=base_estimators, n_jobs=-1).fit(X_tr, y_tr)
reg_scores["Voting"] = rmse(voting, X_te, y_te)

stacking = StackingRegressor(
    estimators=base_estimators, final_estimator=RidgeCV(), cv=5, n_jobs=-1,
).fit(X_tr, y_tr)
reg_scores["Stacking"] = rmse(stacking, X_te, y_te)

print(f"Voting   RMSE = {reg_scores['Voting']:.4f}")
print(f"Stacking RMSE = {reg_scores['Stacking']:.4f}")
```

<!-- #region -->
## 13. Benchmark récapitulatif
<!-- #endregion -->

<!-- #region -->
Tous les modèles de régression entraînés sur California Housing, classés par RMSE croissante (plus bas = meilleur). Le meilleur est mis en évidence.
<!-- #endregion -->

```python
ranked = sorted(reg_scores.items(), key=lambda kv: kv[1])
names = [n for n, _ in ranked]
vals = [v for _, v in ranked]
colors = [ACCENT if i == 0 else PRIMARY_1 for i in range(len(names))]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, vals, color=colors)
ax.set_ylabel("RMSE test (California Housing)")
ax.set_title("Benchmark des méthodes d'ensemble — régression")
ax.set_ylim(min(vals) * 0.95, max(vals) * 1.02)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.show()
print(f"Meilleur modèle : {names[0]} (RMSE = {vals[0]:.4f})")
```

<!-- #region -->
## 14. Classification — breast_cancer
<!-- #endregion -->

<!-- #region -->
On bascule sur un problème de **classification binaire** : le dataset `breast_cancer` (569 échantillons, 30 features numériques, 2 classes — *malignant* / *benign*). Réel, sans valeurs manquantes ni catégorielles → on reste concentré sur les ensembles. Split stratifié pour préserver les proportions de classes.
<!-- #endregion -->

```python
bc = load_breast_cancer(as_frame=True)
Xc, yc = bc.data, bc.target  # 0 = malignant, 1 = benign
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc
)
print(f"train={Xc_tr.shape}, répartition classes = "
      f"{dict(zip(bc.target_names, np.bincount(yc)))}")
```

<!-- #region -->
## 15. Bagging en classification
<!-- #endregion -->

<!-- #region -->
Le bagging fonctionne avec n'importe quel estimateur de base, pas seulement les arbres. Ici un **SVC** baggé : chaque SVC voit un bootstrap différent, et on vote à la majorité.
<!-- #endregion -->

```python
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

bag_clf = BaggingClassifier(
    estimator=SVC(), n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1,
).fit(Xc_tr, yc_tr)
print(f"Bagging(SVC) accuracy = {bag_clf.score(Xc_te, yc_te):.4f}\n")
print(classification_report(yc_te, bag_clf.predict(Xc_te), target_names=bc.target_names))
```

<!-- #region -->
Effet du **nombre d'estimateurs** sur l'erreur (0-1 loss) en train et test : l'erreur de test se stabilise rapidement — au-delà d'une dizaine d'estimateurs, le gain est marginal.
<!-- #endregion -->

```python
from sklearn.metrics import zero_one_loss

n_max = 30
err_test = np.zeros(n_max)
err_train = np.zeros(n_max)
for n in range(1, n_max + 1):
    m = BaggingClassifier(
        estimator=SVC(), n_estimators=n, random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(Xc_tr, yc_tr)
    err_test[n - 1] = zero_one_loss(yc_te, m.predict(Xc_te))
    err_train[n - 1] = zero_one_loss(yc_tr, m.predict(Xc_tr))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.arange(1, n_max + 1), err_train, marker="o", color=PRIMARY_1, label="Train error")
ax.plot(np.arange(1, n_max + 1), err_test, marker="s", color=MAUVAIS, label="Test error")
ax.set_xlabel("Nombre d'estimateurs")
ax.set_ylabel("0-1 loss")
ax.set_title("Bagging(SVC) — erreur vs nombre d'estimateurs")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

<!-- #region -->
## 16. Comparaison des ensembles en classification
<!-- #endregion -->

<!-- #region -->
On entraîne les principales familles (AdaBoost, Gradient Boosting, Random Forest, Extra-Trees, LightGBM) sur breast_cancer et on les compare sur l'accuracy et le F1-score du set de test. Sur ce dataset propre et discriminant, tous atteignent des scores élevés et proches — le choix se fait alors sur la vitesse et la robustesse au tuning.
<!-- #endregion -->

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, ExtraTreesClassifier,
)

clf_models = {
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    ),
}

rows = []
for name, model in clf_models.items():
    model.fit(Xc_tr, yc_tr)
    pred = model.predict(Xc_te)
    rows.append({"modèle": name,
                 "accuracy": accuracy_score(yc_te, pred),
                 "f1": f1_score(yc_te, pred)})
clf_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
print(clf_df.to_string(index=False))
```

<!-- #region -->
## 17. Conseils pratiques 2026
<!-- #endregion -->

<!-- #region -->
- **Benchmarker LightGBM en premier** sur tabulaire : souvent dans le top 3 sans tuning.
- **Optuna + early stopping** = la combo de tuning en 2026 (voir `ML_Optimisation_de_Modèles`).
- **CatBoost** dès qu'il y a beaucoup de catégorielles à forte cardinalité (target encoding ordonné contrôlé).
- En cas d'overfit LightGBM sur données bruitées : monter `min_child_samples` à 50-100.
- **GPU** rentable à partir de ~1M observations × 100 features ; en deçà, CPU + `n_jobs=-1` suffit.
- **Random Forest** reste un excellent baseline robuste et quasi sans tuning.
- **Stacking** : le dernier 1-2 % de perf, si le budget de calcul le permet.
- **Interprétabilité** : SHAP (voir `ML_Explication_Feature_Importance_Selection`).
<!-- #endregion -->

<!-- #region -->
## 18. Sources
<!-- #endregion -->

<!-- #region -->
- [XGBoost — documentation](https://xgboost.readthedocs.io/)
- [LightGBM — documentation](https://lightgbm.readthedocs.io/)
- [CatBoost — documentation](https://catboost.ai/docs/)
- Breiman, L. (2001), *Random Forests*, Machine Learning 45(1).
- Friedman, J. (1999/2001), *Greedy Function Approximation: A Gradient Boosting Machine*.
- [XGBoost vs LightGBM vs CatBoost (2026)](https://pythondatabench.com/article/gradient-boosting-python-xgboost-lightgbm-catboost-2026)
- [GradientBoosting vs AdaBoost vs XGBoost vs CatBoost vs LightGBM — 2026](https://thelinuxcode.com/gradientboosting-vs-adaboost-vs-xgboost-vs-catboost-vs-lightgbm-what-really-changes-and-what-i-pick-in-2026/)
- Notebooks liés : `ML_Regression_Classification_CV_GridSearch`, `ML_Optimisation_de_Modèles`, `ML_Explication_Feature_Importance_Selection`.
<!-- #endregion -->
