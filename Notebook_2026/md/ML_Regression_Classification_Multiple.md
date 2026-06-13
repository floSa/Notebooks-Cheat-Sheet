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
# 🎯 Régression & Classification à sorties multiples
<!-- #endregion -->

<!-- #region -->
La plupart des modèles supervisés prédisent **une seule** sortie. En pratique on doit souvent en
prédire **plusieurs en même temps** : trois grandeurs physiques corrélées, l'appartenance simultanée
à plusieurs étiquettes, une cible binaire **et** une cible multiclasse à partir des mêmes features.

Ce notebook est un **tutoriel + wiki** dédié à ces problèmes « multi-sorties ». Il couvre :

1. La **taxonomie** des problèmes à sorties multiples et les **3 stratégies** scikit-learn (natif,
   `MultiOutput*`, `*Chain`).
2. La **régression multi-cibles** : benchmark de modèles, comparaison wrapper vs chaîne, réseaux de
   neurones à sorties multiples.
3. La **classification multi-sorties / multi-labels** : benchmark, chaînes de classifieurs et leur
   ensemble, réseaux **multi-têtes** avec matrices de confusion et courbes ROC.

Les données sont **synthétiques** (`make_regression` multi-cibles, `make_multilabel_classification`) :
c'est le seul moyen d'illustrer proprement plusieurs sorties corrélées — un dataset mono-cible comme
California Housing ne s'y prête pas. Tout est reproductible (`random_state=42`).
<!-- #endregion -->

<!-- #region -->
## 0. Imports & configuration
<!-- #endregion -->

<!-- #region -->
Imports communs, graine aléatoire et palette de couleurs partagée par toutes les figures.
<!-- #endregion -->

```python
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")

# Palette commune aux graphiques
PALETTE = ["#00798c", "#d1495b", "#edae49", "#66a182", "#2e4057", "#9d83b8"]
```

<!-- #region -->
## 1. Taxonomie des problèmes à sorties multiples
<!-- #endregion -->

<!-- #region -->
Il faut distinguer cinq cas, selon la **forme de la cible** `y` :

| Problème | Forme de `y` | Exemple |
|---|---|---|
| **Régression** simple | `(n,)` continue | prix d'une maison |
| **Classification binaire / multiclasse** | `(n,)` entière, $K$ classes | chiffre 0–9 |
| **Régression multi-sorties** (*multi-output regression*) | `(n, t)` continue | 3 grandeurs physiques |
| **Multi-label** (*multilabel*) | `(n, L)` binaire | tags d'un article (plusieurs vrais à la fois) |
| **Multiclasse multi-sorties** (*multiclass-multioutput*) | `(n, t)` entière | prédire **simultanément** une cible binaire et une cible à 4 classes |

⚠️ Ne pas confondre **multiclasse** (1 cible, $K>2$ classes, **exclusives**) et **multilabel**
($L$ étiquettes binaires **non exclusives**).

**Trois stratégies** dans scikit-learn pour gérer plusieurs sorties :

1. **Estimateur nativement multi-sorties** — `LinearRegression`, `Ridge`, `KNeighbors*`, arbres et
   forêts, `MLP*` acceptent directement un `y` de forme `(n, t)`. Une seule structure de modèle.
2. **`MultiOutputRegressor` / `MultiOutputClassifier`** — entraînent **un modèle indépendant par
   cible**. Permet d'utiliser un estimateur mono-sortie (SVR, CatBoost…). Ignore les corrélations
   entre cibles.
3. **`RegressorChain` / `ClassifierChain`** — entraînent une **chaîne** : la prédiction de la cible
   $k$ reçoit en feature supplémentaire les prédictions des cibles précédentes,
   $$\hat{y}_k = f_k\big(x,\ \hat{y}_1, \dots, \hat{y}_{k-1}\big).$$
   La chaîne **capture les corrélations** entre cibles, au prix d'une sensibilité à l'ordre.
<!-- #endregion -->

<!-- #region -->
# Partie I — Régression multi-cibles
<!-- #endregion -->

<!-- #region -->
## 2. Construction du jeu de données
<!-- #endregion -->

<!-- #region -->
Le cas le plus simple : `make_regression` sait générer plusieurs cibles d'un coup avec `n_targets`.
Ici 3 cibles, combinaisons linéaires bruitées des features.
<!-- #endregion -->

```python
from sklearn.datasets import make_regression

X_demo, Y_demo = make_regression(
    n_samples=4000, n_features=10, n_targets=3, noise=1.0, random_state=RANDOM_STATE
)
cols = [f"Feature_{i}" for i in range(1, 11)] + [f"Target_{i}" for i in range(1, 4)]
df_demo = pd.DataFrame(np.concatenate([X_demo, Y_demo], axis=1), columns=cols)
df_demo.head(2)
```

<!-- #region -->
`make_regression` ne produit que des relations **linéaires**. Pour un benchmark plus parlant, on
fabrique un jeu où les 3 cibles mêlent une relation **linéaire**, une relation **trigonométrique** et
une relation **quadratique** — de quoi départager modèles linéaires et non linéaires. Ce jeu devient
notre dataset de travail.
<!-- #endregion -->

```python
def make_multitarget_regression(
    n_samples: int = 2500, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Génère un jeu à 3 cibles mêlant relations linéaire, trigonométrique et quadratique.

    Returns:
        (df, feature_cols, target_cols)
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, 10)) * 10
    Y = np.column_stack([
        X[:, 0] + 2 * X[:, 1] - X[:, 2] + rng.standard_normal(n_samples),            # linéaire
        np.sin(X[:, 2]) + np.cos(X[:, 3]) + rng.standard_normal(n_samples),          # non linéaire
        2 * X[:, 2] ** 2 - 1 / (X[:, 0] + 0.1) + rng.standard_normal(n_samples),     # quadratique
    ])
    feat = [f"Feature_{i}" for i in range(X.shape[1])]
    targ = [f"Target_{j}" for j in range(Y.shape[1])]
    df = pd.DataFrame(np.column_stack([X, Y]), columns=feat + targ)
    return df, feat, targ


df_reg, FEAT_REG, TARG_REG = make_multitarget_regression()
df_reg.head(2)
```

<!-- #region -->
## 3. Modèles nativement multi-sorties ?
<!-- #endregion -->

<!-- #region -->
**Nativement** multi-sorties (un `y` de forme `(n, 3)` passe directement) : `LinearRegression`,
`Ridge`, `Lasso`, `ElasticNet`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForest`,
`ExtraTrees`, `MLPRegressor`.

**Mono-sortie uniquement** (il faut envelopper) : `SVR`, `GradientBoostingRegressor`, XGBoost,
CatBoost. On les enveloppe dans `MultiOutputRegressor`, qui entraîne **un modèle par cible**.

La démo ci-dessous le montre : `LinearRegression` accepte directement les 3 cibles, tandis que
`SVR` doit être enveloppé (on en met aussi un `StandardScaler` devant — un noyau RBF sur des
features non standardisées est lent et de mauvaise qualité).
<!-- #endregion -->

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_reg = df_reg[FEAT_REG]
Y_reg = df_reg[TARG_REG]
Xr_tr, Xr_te, Yr_tr, Yr_te = train_test_split(X_reg, Y_reg, test_size=0.2, random_state=RANDOM_STATE)

# Natif : LinearRegression accepte directement un Y de forme (n, 3)
native = LinearRegression().fit(Xr_tr, Yr_tr)
print("LinearRegression natif      → predict shape :", native.predict(Xr_te).shape)

# Non natif : SVR ne gère qu'une cible → on l'enveloppe (un modèle par cible).
wrapped = MultiOutputRegressor(make_pipeline(StandardScaler(), SVR())).fit(Xr_tr, Yr_tr)
print("MultiOutputRegressor(SVR)   → predict shape :", wrapped.predict(Xr_te).shape)
print("Sous-modèles entraînés      :", len(wrapped.estimators_))
```

<!-- #region -->
## 4. Benchmark de régression multi-sorties
<!-- #endregion -->

<!-- #region -->
On compare une dizaine de modèles sur les 3 cibles. La métrique est le **R²** moyenné sur les cibles
(`r2_score` applique par défaut `multioutput="uniform_average"`, c.-à-d. la moyenne arithmétique des
R² par cible ; l'alternative `variance_weighted` pondère par la variance de chaque cible).

> **Note sur le gradient boosting** : l'original utilisait XGBoost **et** LightGBM. Dans cet
> environnement WSL, LightGBM ne se charge pas (`libgomp` absent) et XGBoost provoque un *segfault*
> OpenMP. On garde donc **CatBoost** comme représentant du gradient boosting ; XGBoost/LightGBM
> seraient des choix équivalents sur une machine où leur OpenMP fonctionne.
<!-- #endregion -->

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


def bench_regression(models: dict, X_tr, Y_tr, X_te, Y_te) -> pd.DataFrame:
    """Entraîne chaque modèle et renvoie le R² (moyenne sur les cibles) trié décroissant."""
    rows = []
    for name, model in models.items():
        model.fit(X_tr, Y_tr)
        rows.append({"model": name, "r2": r2_score(Y_te, model.predict(X_te))})
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


reg_models = {
    "Elastic Net": ElasticNet(),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "MLP": make_pipeline(StandardScaler(), MLPRegressor(max_iter=300, random_state=RANDOM_STATE)),
    "SVR": MultiOutputRegressor(make_pipeline(StandardScaler(), SVR())),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "CatBoost": MultiOutputRegressor(CatBoostRegressor(iterations=200, verbose=0, random_state=RANDOM_STATE)),
}
reg_scores = bench_regression(reg_models, Xr_tr, Yr_tr, Xr_te, Yr_te)
print(reg_scores.to_string(index=False))
```

<!-- #region -->
Un R² global masque les disparités entre cibles. On visualise le R² **par cible** du meilleur modèle :
la cible non linéaire (trigonométrique) est généralement la plus difficile.
<!-- #endregion -->

```python
best_name = reg_scores.iloc[0]["model"]
best_model = reg_models[best_name]
Yr_pred = best_model.predict(Xr_te)
r2_per_target = [r2_score(Yr_te.iloc[:, k], Yr_pred[:, k]) for k in range(len(TARG_REG))]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(TARG_REG, r2_per_target, color=PALETTE[: len(TARG_REG)])
ax.set_ylabel("R²")
ax.set_ylim(0, 1)
ax.set_title(f"R² par cible — {best_name}")
for i, v in enumerate(r2_per_target):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
```

<!-- #region -->
## 5. MultiOutputRegressor vs RegressorChain
<!-- #endregion -->

<!-- #region -->
`MultiOutputRegressor` traite chaque cible **indépendamment**. `RegressorChain` les enchaîne : la
cible $k$ reçoit en entrée les prédictions des cibles $1\dots k-1$. Quand les cibles sont corrélées,
la chaîne peut **récupérer un peu de R²**. Comparaison sur une base `GradientBoostingRegressor`
(mono-sortie, donc à envelopper dans les deux cas).
<!-- #endregion -->

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import RegressorChain


def gbr() -> GradientBoostingRegressor:
    """Base mono-sortie partagée par les deux stratégies."""
    return GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)


wrapper = MultiOutputRegressor(gbr()).fit(Xr_tr, Yr_tr)
chain = RegressorChain(gbr(), order=[0, 1, 2], random_state=RANDOM_STATE).fit(Xr_tr, Yr_tr)

print(f"MultiOutputRegressor  R² = {r2_score(Yr_te, wrapper.predict(Xr_te)):.4f}")
print(f"RegressorChain        R² = {r2_score(Yr_te, chain.predict(Xr_te)):.4f}")
```

<!-- #region -->
# Partie II — Classification multi-sorties / multi-labels
<!-- #endregion -->

<!-- #region -->
## 6. Construction du jeu de données
<!-- #endregion -->

<!-- #region -->
`make_multilabel_classification` génère 3 étiquettes binaires. Pour obtenir un problème
**multiclasse multi-sorties**, on **fusionne** deux de ces étiquettes en une seule cible multiclasse
(les $2 \times 2 = 4$ combinaisons), via `pd.factorize`. On garde donc :

- `Target_0` : cible **binaire** (2 classes),
- `Target_1` : cible **multiclasse** (4 classes).

> `pd.factorize` sur le tuple des deux labels, `df.groupby([...]).ngroup()` ou un mapping
> dictionnaire explicite produisent **exactement le même** ré-encodage — on retient la plus
> idiomatique.
<!-- #endregion -->

```python
from sklearn.datasets import make_multilabel_classification


def make_multitarget_classification(
    n_samples: int = 2500, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Construit 2 cibles : Target_0 binaire + Target_1 multiclasse (fusion de 2 labels → 4 combos)."""
    X, Y = make_multilabel_classification(
        n_samples=n_samples, n_features=10, n_classes=3, n_labels=2, random_state=seed
    )
    feat = [f"Feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(np.column_stack([X, Y]), columns=feat + ["Target_0", "Target_1", "Target_2"])
    # Fusion de Target_1 × Target_2 en une cible multiclasse unique :
    df["Target_1"], _ = pd.factorize(df[["Target_1", "Target_2"]].apply(tuple, axis=1), sort=True)
    # alternative strictement équivalente : df.groupby(["Target_1", "Target_2"]).ngroup()
    df = df.drop(columns=["Target_2"])
    return df, feat, ["Target_0", "Target_1"]


df_clf, FEAT_CLF, TARG_CLF = make_multitarget_classification()
df_clf.head(2)
```

<!-- #region -->
La cible multiclasse fusionnée est **déséquilibrée** (certaines combinaisons sont rares). On le
vérifie avant de modéliser — c'est ce qui justifiera plus loin les métriques par classe (F1 macro)
plutôt que la seule accuracy.
<!-- #endregion -->

```python
print("Target_0 (binaire) :")
print(df_clf["Target_0"].value_counts().sort_index().to_string())
print("\nTarget_1 (multiclasse) :")
print(df_clf["Target_1"].value_counts().sort_index().to_string())
```

<!-- #region -->
## 7. Benchmark de classification multi-sorties
<!-- #endregion -->

<!-- #region -->
`MultiOutputClassifier` entraîne un classifieur par cible et prédit les deux d'un coup. On reporte
l'**accuracy par label**. (`SVC` est laissé sans `probability=True` : inutile pour l'accuracy et
coûteux ; un `StandardScaler` précède les modèles à noyau / distance.)
<!-- #endregion -->

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

Xc = df_clf[FEAT_CLF]
Yc = df_clf[TARG_CLF]
Xc_tr, Xc_te, Yc_tr, Yc_te = train_test_split(Xc, Yc, test_size=0.2, random_state=RANDOM_STATE)


def bench_classification(models: dict, X_tr, Y_tr, X_te, Y_te, targets: list[str]) -> pd.DataFrame:
    """Entraîne chaque modèle multi-sorties et renvoie l'accuracy par label."""
    rows = []
    for name, model in models.items():
        model.fit(X_tr, Y_tr)
        pred = model.predict(X_te)
        row = {"model": name}
        for k, t in enumerate(targets):
            row[f"acc_{t}"] = accuracy_score(Y_te.iloc[:, k], pred[:, k])
        rows.append(row)
    return pd.DataFrame(rows)


clf_models = {
    "Logistic Regression": MultiOutputClassifier(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))),
    "KNN": MultiOutputClassifier(make_pipeline(StandardScaler(), KNeighborsClassifier())),
    "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)),
    "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
    "MLP": MultiOutputClassifier(make_pipeline(StandardScaler(), MLPClassifier(max_iter=300, random_state=RANDOM_STATE))),
    "SVM": MultiOutputClassifier(make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE))),
}
clf_scores = bench_classification(clf_models, Xc_tr, Yc_tr, Xc_te, Yc_te, TARG_CLF)
print(clf_scores.to_string(index=False))
```

<!-- #region -->
Comme la cible multiclasse est déséquilibrée, on regarde le **F1 macro par label** (moyenne non
pondérée des F1 par classe — chaque classe compte autant) pour le meilleur modèle.
<!-- #endregion -->

```python
clf_scores["acc_mean"] = clf_scores[[f"acc_{t}" for t in TARG_CLF]].mean(axis=1)
best_clf_name = clf_scores.sort_values("acc_mean", ascending=False).iloc[0]["model"]
best_clf = clf_models[best_clf_name]
clf_pred = best_clf.predict(Xc_te)
f1_per_label = [f1_score(Yc_te.iloc[:, k], clf_pred[:, k], average="macro") for k in range(len(TARG_CLF))]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(TARG_CLF, f1_per_label, color=PALETTE[: len(TARG_CLF)])
ax.set_ylabel("F1 (macro)")
ax.set_ylim(0, 1)
ax.set_title(f"F1 macro par label — {best_clf_name}")
for i, v in enumerate(f1_per_label):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()
```

<!-- #region -->
## 8. ClassifierChain & ensemble de chaînes
<!-- #endregion -->

<!-- #region -->
Sur un **vrai multilabel** (les 3 étiquettes binaires, sans fusion), `ClassifierChain` modélise les
dépendances entre labels : la probabilité du label $k$ est conditionnée aux labels précédents,
$$P(y_k \mid x,\ y_1, \dots, y_{k-1}).$$

La chaîne dépend de l'**ordre** des labels. La pratique recommandée (scikit-learn) est un **ensemble
de chaînes** entraînées sur des ordres aléatoires, dont on **moyenne les probabilités** — plus
robuste qu'une chaîne unique. On mesure le **Jaccard score** (moyenne par échantillon) :
$$J(y, \hat{y}) = \frac{|y \cap \hat{y}|}{|y \cup \hat{y}|}.$$
<!-- #endregion -->

```python
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_score

# Vrai multilabel : 3 labels binaires simultanés (sans fusion)
Xml, Yml = make_multilabel_classification(
    n_samples=2500, n_features=10, n_classes=3, n_labels=2, random_state=RANDOM_STATE
)
Xml_tr, Xml_te, Yml_tr, Yml_te = train_test_split(Xml, Yml, test_size=0.2, random_state=RANDOM_STATE)

single_chain = ClassifierChain(
    LogisticRegression(max_iter=1000), order="random", random_state=RANDOM_STATE
).fit(Xml_tr, Yml_tr)

chains = [
    ClassifierChain(LogisticRegression(max_iter=1000), order="random", random_state=i).fit(Xml_tr, Yml_tr)
    for i in range(5)
]
proba_ensemble = np.mean([c.predict_proba(Xml_te) for c in chains], axis=0)
pred_ensemble = (proba_ensemble >= 0.5).astype(int)

print(f"Chaîne unique   — Jaccard (samples) = {jaccard_score(Yml_te, single_chain.predict(Xml_te), average='samples'):.4f}")
print(f"Ensemble de 5   — Jaccard (samples) = {jaccard_score(Yml_te, pred_ensemble, average='samples'):.4f}")
```

<!-- #region -->
# Partie III — Réseaux de neurones multi-sorties
<!-- #endregion -->

<!-- #region -->
Un réseau de neurones gère naturellement plusieurs sorties : il suffit d'une **couche de sortie à
plusieurs unités** (régression), ou de **plusieurs têtes** branchées sur un tronc commun
(classification de plusieurs cibles). Le tronc partagé apprend une représentation commune, chaque
tête se spécialise.

> ⚠️ **Contrainte d'environnement.** Dans ce kernel WSL, TensorFlow et PyTorch ne **cohabitent pas**
> dans le même process : une fois TensorFlow importé, tout calcul PyTorch déclenche un *segfault*
> (conflit de runtimes OpenMP/MKL). On exécute donc **tout PyTorch d'abord**, puis on importe
> TensorFlow. Hors de ce piège, l'ordre n'aurait pas d'importance.

On prépare d'abord les éléments communs : cibles de régression **normalisées** (un réseau converge
mal sur des cibles d'amplitudes très différentes), et le nombre de classes de chaque cible de
classification.
<!-- #endregion -->

```python
y_scaler = StandardScaler().fit(Yr_tr)
classes_1 = np.sort(df_clf["Target_1"].unique())
n_class_0 = int(df_clf["Target_0"].nunique())
n_class_1 = int(len(classes_1))
print(f"Target_0 : {n_class_0} classes · Target_1 : {n_class_1} classes")
```

<!-- #region -->
### PyTorch — régression multi-sorties
<!-- #endregion -->

<!-- #region -->
Un MLP dont la **dernière couche a 3 unités** (une par cible). On entraîne sur les cibles
normalisées et on évalue le R² après avoir ré-inversé la normalisation.
<!-- #endregion -->

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(RANDOM_STATE)
Xtr_t = torch.tensor(Xr_tr.values, dtype=torch.float32)
Xte_t = torch.tensor(Xr_te.values, dtype=torch.float32)
Ytr_t = torch.tensor(y_scaler.transform(Yr_tr), dtype=torch.float32)
reg_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=32, shuffle=True)


class RegNet(nn.Module):
    """Perceptron multi-couches à sorties multiples (régression)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


reg_net = RegNet(Xtr_t.shape[1], len(TARG_REG))
opt = torch.optim.Adam(reg_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

reg_losses: list[float] = []
for epoch in range(15):
    reg_net.train()
    running = 0.0
    for xb, yb in reg_loader:
        opt.zero_grad()
        loss = loss_fn(reg_net(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * len(xb)
    reg_losses.append(running / len(Xtr_t))

reg_net.eval()
with torch.no_grad():
    pred_s = reg_net(Xte_t).numpy()
pred = y_scaler.inverse_transform(pred_s)
print("R² (PyTorch, 3 cibles) :", round(r2_score(Yr_te, pred), 3))
```

<!-- #region -->
La **courbe d'apprentissage** (MSE par epoch sur les cibles normalisées) confirme la convergence.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(reg_losses) + 1), reg_losses, color=PALETTE[0], marker="o", ms=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (cibles normalisées)")
ax.set_title("Courbe d'apprentissage — PyTorch régression multi-sorties")
plt.tight_layout()
plt.show()
```

<!-- #region -->
### PyTorch — réseau multi-têtes (classification)
<!-- #endregion -->

<!-- #region -->
Un **tronc partagé** alimente **deux têtes** : une pour `Target_0` (2 classes), une pour `Target_1`
(4 classes). La perte totale est la **somme** des deux entropies croisées. (On corrige ici l'original
qui traitait la tête multiclasse en `sigmoid`/`BCE` — incorrect pour des classes exclusives — au
profit d'un `CrossEntropyLoss` sur chaque tête.)
<!-- #endregion -->

```python
torch.manual_seed(RANDOM_STATE)
Xc_tr_t = torch.tensor(Xc_tr.values, dtype=torch.float32)
Xc_te_t = torch.tensor(Xc_te.values, dtype=torch.float32)
y0_tr_t = torch.tensor(Yc_tr.values[:, 0], dtype=torch.long)
y1_tr_t = torch.tensor(Yc_tr.values[:, 1], dtype=torch.long)


class DualHead(nn.Module):
    """Tronc partagé + deux têtes de classification (logits)."""

    def __init__(self, n_in: int, hidden: int, n0: int, n1: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
        self.head0 = nn.Linear(hidden, n0)
        self.head1 = nn.Linear(hidden, n1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.head0(h), self.head1(h)


clf_net = DualHead(Xc_tr_t.shape[1], 64, n_class_0, n_class_1)
opt = torch.optim.Adam(clf_net.parameters())
ce = nn.CrossEntropyLoss()
clf_loader = DataLoader(TensorDataset(Xc_tr_t, y0_tr_t, y1_tr_t), batch_size=32, shuffle=True)

for epoch in range(15):
    clf_net.train()
    for xb, yb0, yb1 in clf_loader:
        opt.zero_grad()
        o0, o1 = clf_net(xb)
        loss = ce(o0, yb0) + ce(o1, yb1)
        loss.backward()
        opt.step()

clf_net.eval()
with torch.no_grad():
    logit0, logit1 = clf_net(Xc_te_t)
    pt_proba0 = torch.softmax(logit0, dim=1).numpy()
    pt_proba1 = torch.softmax(logit1, dim=1).numpy()
print("Sorties PyTorch (probas) :", pt_proba0.shape, pt_proba1.shape)
```

<!-- #region -->
**Matrices de confusion** normalisées par ligne, une par tête (le pourcentage sur la diagonale = le
rappel de chaque classe).
<!-- #endregion -->

```python
from sklearn.metrics import confusion_matrix

pt_pred = [pt_proba0, pt_proba1]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for lab, ax in enumerate(axes):
    true = Yc_te.values[:, lab]
    pred = np.argmax(pt_pred[lab], axis=1)
    cm = confusion_matrix(true, pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(true, pred)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
        xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]),
    )
    ax.set_title(f"{TARG_CLF[lab]} — Accuracy {acc:.3f}")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
plt.tight_layout()
plt.show()
```

<!-- #region -->
**Courbes ROC** par label : directe pour `Target_0` (binaire), et en **micro-moyenne one-vs-rest**
pour `Target_1` (multiclasse), avec l'AUC macro.
<!-- #endregion -->

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

fig, ax = plt.subplots(figsize=(6, 5))
fpr0, tpr0, _ = roc_curve(Yc_te.values[:, 0], pt_proba0[:, 1])
ax.plot(fpr0, tpr0, color=PALETTE[0], lw=2, label=f"Target_0 (AUC={auc(fpr0, tpr0):.2f})")
y1_bin = label_binarize(Yc_te.values[:, 1], classes=classes_1)
macro_auc = roc_auc_score(Yc_te.values[:, 1].astype(int), pt_proba1, multi_class="ovr", average="macro")
fpr1, tpr1, _ = roc_curve(y1_bin.ravel(), pt_proba1.ravel())
ax.plot(fpr1, tpr1, color=PALETTE[1], lw=2, label=f"Target_1 micro-OvR (AUC macro={macro_auc:.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax.set_xlabel("Taux de faux positifs")
ax.set_ylabel("Taux de vrais positifs")
ax.set_title("Courbes ROC par label — PyTorch")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

<!-- #region -->
### TensorFlow / Keras — régression multi-sorties
<!-- #endregion -->

<!-- #region -->
Même réseau côté Keras : un `Sequential` dont la dernière `Dense` a 3 unités. **TensorFlow n'est
importé qu'ici**, après tout le code PyTorch (cf. la note de la Partie III).
<!-- #endregion -->

```python
import tensorflow as tf

Yr_tr_s = y_scaler.transform(Yr_tr)
tf.random.set_seed(RANDOM_STATE)
tf_reg = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xr_tr.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(TARG_REG)),
])
tf_reg.compile(optimizer="adam", loss="mse")
tf_reg.fit(Xr_tr, Yr_tr_s, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

pred = y_scaler.inverse_transform(tf_reg.predict(Xr_te, verbose=0))
print("R² (TensorFlow, 3 cibles) :", round(r2_score(Yr_te, pred), 3))
```

<!-- #region -->
### TensorFlow / Keras — réseau multi-têtes (classification)
<!-- #endregion -->

<!-- #region -->
Avec l'**API fonctionnelle** Keras : un tronc `Dense(64)` partagé, puis **deux sorties** `softmax`
nommées (une par cible), chacune avec sa `categorical_crossentropy`. Les cibles sont encodées en
one-hot.
<!-- #endregion -->

```python
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

enc0 = OneHotEncoder(sparse_output=False).fit(Yc.values[:, [0]])
enc1 = OneHotEncoder(sparse_output=False).fit(Yc.values[:, [1]])
y0_tr = enc0.transform(Yc_tr.values[:, [0]])
y1_tr = enc1.transform(Yc_tr.values[:, [1]])

tf.random.set_seed(RANDOM_STATE)
inp = Input(shape=(Xc_tr.shape[1],))
hidden = Dense(64, activation="relu")(inp)
out0 = Dense(n_class_0, activation="softmax", name="Target_0")(hidden)   # tête binaire
out1 = Dense(n_class_1, activation="softmax", name="Target_1")(hidden)   # tête multiclasse (4 combos)
tf_clf = keras.Model(inputs=inp, outputs=[out0, out1])
tf_clf.compile(
    optimizer="adam",
    loss={"Target_0": "categorical_crossentropy", "Target_1": "categorical_crossentropy"},
    metrics={"Target_0": "accuracy", "Target_1": "accuracy"},
)
tf_clf.fit(Xc_tr, {"Target_0": y0_tr, "Target_1": y1_tr}, epochs=10, batch_size=32, verbose=0)

tf_pred = tf_clf.predict(Xc_te, verbose=0)
print("Sorties TensorFlow (probas) :", tf_pred[0].shape, tf_pred[1].shape)
```

<!-- #region -->
Les mêmes diagnostics que pour PyTorch — **matrices de confusion** par tête.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for lab, ax in enumerate(axes):
    true = Yc_te.values[:, lab]
    pred = np.argmax(tf_pred[lab], axis=1)
    cm = confusion_matrix(true, pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(true, pred)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
        xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]),
    )
    ax.set_title(f"{TARG_CLF[lab]} — Accuracy {acc:.3f}")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
plt.tight_layout()
plt.show()
```

<!-- #region -->
… et les **courbes ROC** par label.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(6, 5))
fpr0, tpr0, _ = roc_curve(Yc_te.values[:, 0], tf_pred[0][:, 1])
ax.plot(fpr0, tpr0, color=PALETTE[0], lw=2, label=f"Target_0 (AUC={auc(fpr0, tpr0):.2f})")
y1_bin = label_binarize(Yc_te.values[:, 1], classes=classes_1)
macro_auc = roc_auc_score(Yc_te.values[:, 1].astype(int), tf_pred[1], multi_class="ovr", average="macro")
fpr1, tpr1, _ = roc_curve(y1_bin.ravel(), tf_pred[1].ravel())
ax.plot(fpr1, tpr1, color=PALETTE[1], lw=2, label=f"Target_1 micro-OvR (AUC macro={macro_auc:.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax.set_xlabel("Taux de faux positifs")
ax.set_ylabel("Taux de vrais positifs")
ax.set_title("Courbes ROC par label — TensorFlow")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

<!-- #region -->
## 9. Récapitulatif — quelle stratégie choisir ?
<!-- #endregion -->

<!-- #region -->
| Stratégie | Capte les corrélations entre cibles ? | Coût | Quand l'utiliser |
|---|---|---|---|
| **Estimateur natif** (`LinearRegression`, RF, MLP…) | partiellement (structure partagée) | 1 modèle | défaut : simple et rapide quand l'algo le supporte |
| **`MultiOutput*`** | ❌ (cibles indépendantes) | $t$ modèles | pour utiliser un algo mono-sortie (SVR, CatBoost) sur plusieurs cibles |
| **`*Chain`** | ✅ | $t$ modèles (séquentiels) | cibles corrélées ; **ensemble de chaînes** pour lisser l'ordre |
| **Réseau multi-têtes** | ✅ (tronc partagé) | 1 réseau | beaucoup de cibles, features riches, apprentissage de représentation commun |

Règles rapides :

- **Régression de cibles indépendantes** → estimateur natif (Random Forest, CatBoost via wrapper).
- **Cibles corrélées** → `RegressorChain` / ensemble de `ClassifierChain`, ou réseau multi-têtes.
- **Multilabel vrai** (étiquettes non exclusives) → `MultiOutputClassifier` ou `ClassifierChain` +
  réglage du seuil par label.
- **Plusieurs cibles de natures différentes** (binaire + multiclasse + régression) → **réseau
  multi-têtes**, une tête et une perte par cible.
<!-- #endregion -->

<!-- #region -->
## 10. Sources
<!-- #endregion -->

<!-- #region -->
- [scikit-learn — Multiclass and multioutput algorithms (User Guide)](https://scikit-learn.org/stable/modules/multiclass.html)
- [scikit-learn — `RegressorChain` / `ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)
- [scikit-learn — exemple « Classifier Chain » (ensemble de chaînes)](https://scikit-learn.org/stable/auto_examples/multioutput/plot_classifier_chain_yeast.html)
- [CatBoost — multi-régression (`MultiRMSE`)](https://catboost.ai/docs/)
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (métriques, biais/variance).
- Notebooks liés : `ML_Regression_Classification_CV_GridSearch` (CV/GridSearch), `ML_Bagging_Boosting`
  (ensembles), `DL_PyTorch` / `DL_TensorFlow` (réseaux en détail).
<!-- #endregion -->
