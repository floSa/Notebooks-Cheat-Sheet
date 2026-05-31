---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python (notebooks-refonte)
    language: python
    name: notebooks-refonte
---

<!-- #region -->
# 🔭 Analyse multivariée — méthodes factorielles, tests & réduction de dimension
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel + Cheat-sheet** sur l'**analyse multivariée** : comment
résumer, cartographier et tester un jeu de données décrit par plusieurs variables.

**Trois blocs** :

1. **Tests statistiques multivariés** — régression (linéaire/logistique), ANOVA, MANOVA.
2. **Analyse factorielle** — PCA, CA, MCA, MFA, FAMD, GPA : réduire l'information à
   quelques axes (« facteurs ») et lire la proximité entre individus / modalités.
3. **Réduction de dimension non-linéaire** — Isomap, LLE, MDS, t-SNE, UMAP.

**Choix des libs (2026)** :

- **`prince` 0.19** est le backbone unique des méthodes factorielles (API scikit-learn,
  couvre PCA/CA/MCA/MFA/FAMD/GPA). ⚠️ Son API a été **entièrement refondue en 2022** —
  tous les vieux tutos (`plot_row_coordinates`, `mapping`, `explained_inertia_`) sont obsolètes.
- **`fanalysis`** (utilisé dans la 1ʳᵉ version de ce notebook) est **abandonné** et
  n'installe plus proprement → ses graphes utiles sont **réimplémentés** en helpers matplotlib.
- **`scikit-learn`** pour la PCA « pipeline ML » et tout le `manifold` ; **`umap-learn`** ajouté.
- **`statsmodels`** pour les tests (p-values, traces de Wilks/Pillai).

**Datasets** (tous programmatiques ou inline — rien à télécharger) : `iris`, `tips`,
`titanic` (catégoriel, pour la MCA), + 2 tables inline (hair×eye, dégustation de vins).

**Renvois** : visualisation EDA → `EDA_Visualisation_Introduction.ipynb` ;
preprocessing (encodage, scalers) → `Structures_Preprocessing.ipynb`.
<!-- #endregion -->

<!-- #region -->
## 1. Setup, imports et charte graphique
<!-- #endregion -->

<!-- #region -->
On centralise les imports en tête et on affiche les versions pour la reproductibilité.
<!-- #endregion -->

```python
import warnings
warnings.filterwarnings("ignore")

from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import prince
import statsmodels.api as sm
import umap
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA

print(f"pandas      : {pd.__version__}")
print(f"scikit-learn: {version('scikit-learn')}")
print(f"statsmodels : {version('statsmodels')}")
print(f"prince      : {version('prince')}")
print(f"umap-learn  : {version('umap-learn')}")
```

<!-- #region -->
**Charte graphique unique** (`CHART`) : 8 couleurs nommées sémantiquement, appliquées
partout via `apply_chart_style()`. Bonne pratique de reporting — voir
`EDA_Visualisation_Introduction.ipynb` pour la justification détaillée.
<!-- #endregion -->

```python
CHART: dict[str, str] = {
    "primary_1":   "#00798c",  # Teal     — info / catégorie principale
    "mauvais":     "#d1495b",  # Crimson  — bad / nul / critique
    "moyen":       "#edae49",  # Saffron  — moyen / warning
    "accent":      "#66a182",  # Sage     — accent / bon / highlight
    "accent_dark": "#2e4057",  # Navy     — texte fort, valeur max highlight
    "lavender":    "#9d83b8",  # Violet                — secondaire 1
    "dusty_rose":  "#b8848e",  # Rose terne            — secondaire 2
    "beige":       "#c9b78b",  # Beige                 — neutre / background
}
PALETTE: list[str] = list(CHART.values())


def apply_chart_style() -> None:
    """Applique la charte graphique au runtime matplotlib + seaborn (idempotent)."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": CHART["accent_dark"],
        "axes.labelcolor": CHART["accent_dark"],
        "axes.titleweight": "bold",
        "grid.alpha": 0.3,
        "font.size": 11,
    })


apply_chart_style()
```

<!-- #region -->
Le dataset fil rouge est **iris** (150 fleurs × 4 mesures numériques + l'espèce).
On le charge une fois et on isole la partie numérique.
<!-- #endregion -->

```python
iris = sns.load_dataset("iris")
iris_num = iris.drop(columns=["species"])
iris_species = iris["species"]
print("iris:", iris.shape, "| espèces:", list(iris_species.unique()))
iris.head()
```

<!-- #region -->
## 2. Tests statistiques multivariés
<!-- #endregion -->

<!-- #region -->
Avant de *réduire*, on *teste* les relations entre variables. Deux familles :

- **Régression** : expliquer/prédire une variable cible à partir des autres.
- **ANOVA / MANOVA** : tester si les moyennes d'une (ANOVA) ou plusieurs (MANOVA)
  variables numériques diffèrent selon les modalités d'un facteur catégoriel.
<!-- #endregion -->

<!-- #region -->
### 2.1 Régression linéaire
<!-- #endregion -->

<!-- #region -->
On prédit le pourboire (`tip`) du dataset `tips` à partir des autres variables
numériques. On utilise **deux backends complémentaires** :

- **scikit-learn** — orienté *prédiction* (coefficients, R², RMSE).
- **statsmodels** — orienté *inférence* (p-values, intervalles de confiance, $R^2$ ajusté).

$$\hat{y} = \beta_0 + \sum_{j} \beta_j x_j \qquad
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
<!-- #endregion -->

```python
tips = sns.load_dataset("tips")
tips_num = tips.select_dtypes(include="number")
features = tips_num.columns.drop("tip")
X_lin = tips_num[features]
y_lin = tips_num["tip"]
print("X", X_lin.shape, "| y", y_lin.shape)
X_lin.head()
```

<!-- #region -->
On encapsule l'ajustement sklearn dans une fonction typée réutilisable.
<!-- #endregion -->

```python
def fit_linear_sklearn(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Ajuste une régression linéaire OLS (scikit-learn) et affiche R²/RMSE."""
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    rmse = float(np.sqrt(((y - model.predict(X)) ** 2).mean()))
    print(f"intercept = {model.intercept_:.3f}")
    print(f"coefs     = {dict(zip(X.columns, np.round(model.coef_, 3)))}")
    print(f"R²        = {r2:.3f} | RMSE = {rmse:.3f}")
    return model


reg = fit_linear_sklearn(X_lin, y_lin)
```

<!-- #region -->
Visualisation : nuage de points sur la variable la plus explicative + droite de
régression (les autres variables fixées à leur moyenne).
<!-- #endregion -->

```python
x0 = X_lin.iloc[:, 0]
fig, ax = plt.subplots()
ax.scatter(x0, y_lin, color=CHART["primary_1"], alpha=0.5, label="observations")
xseq = np.linspace(x0.min(), x0.max(), 100)
others_mean = (reg.coef_[1:] * X_lin.iloc[:, 1:].mean().to_numpy()).sum()
ax.plot(xseq, reg.intercept_ + others_mean + reg.coef_[0] * xseq,
        color=CHART["mauvais"], lw=2.5, label="droite de régression")
ax.set(xlabel=X_lin.columns[0], ylabel="tip", title="Régression linéaire (sklearn)")
ax.legend()
plt.show()
```

<!-- #region -->
La version **statsmodels** ajoute la constante explicitement et donne une table
d'inférence complète (`summary()`) : p-values, IC à 95 %, tests de significativité.
<!-- #endregion -->

```python
X_cst = sm.add_constant(X_lin, prepend=False)
ols_res = sm.OLS(y_lin, X_cst).fit()
print("R²_adj =", round(ols_res.rsquared_adj, 3))
print(ols_res.summary())
```

<!-- #region -->
### 2.2 Régression logistique
<!-- #endregion -->

<!-- #region -->
Cible **catégorielle** (le sexe du payeur). On modélise la probabilité via la
fonction logistique. `penalty=None` désactive la régularisation (l'ancien
`penalty='none'` — une chaîne — est supprimé des versions récentes de scikit-learn).

$$P(y=1 \mid x) = \sigma(\beta_0 + \beta^\top x), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$
<!-- #endregion -->

```python
X_log = tips[["total_bill", "tip", "size"]]
encoder = LabelEncoder()
y_log = encoder.fit_transform(tips["sex"])
print("classes:", list(encoder.classes_))

logit = LogisticRegression(penalty=None, solver="newton-cg", max_iter=200).fit(X_log, y_log)
coef_table = pd.DataFrame(
    np.concatenate([logit.intercept_.reshape(-1, 1), logit.coef_], axis=1),
    index=["coef"],
    columns=["constante"] + list(X_log.columns),
).T
coef_table.round(4)
```

<!-- #region -->
La version statsmodels (`Logit`) fournit les p-values et le pseudo-$R^2$ de McFadden.
<!-- #endregion -->

```python
logit_sm = sm.Logit(y_log, sm.add_constant(X_log.to_numpy())).fit(disp=0)
print("pseudo-R² (McFadden) =", round(logit_sm.prsquared, 3))
print(logit_sm.summary())
```

<!-- #region -->
### 2.3 ANOVA — une variable numérique vs un facteur
<!-- #endregion -->

<!-- #region -->
L'**ANOVA** décompose la variance totale d'une variable numérique en variance
**inter-groupes** (expliquée par le facteur) et **intra-groupes** (résiduelle). Le
test $F$ confronte les deux ; $H_0$ = « toutes les moyennes de groupe sont égales ».

$$F = \frac{\text{SCE}_{\text{inter}} / (k-1)}{\text{SCE}_{\text{intra}} / (n-k)}$$

Le boxplot ci-dessous montre déjà la séparation des espèces sur `sepal_length`.
<!-- #endregion -->

```python
fig, ax = plt.subplots()
sns.boxplot(x="species", y="sepal_length", data=iris,
            hue="species", palette=PALETTE[:3], legend=False, ax=ax)
ax.set_title("Distribution de sepal_length par espèce")
plt.show()
```

<!-- #region -->
On ajuste un modèle linéaire `sepal_length ~ species` puis on lit la table ANOVA
(type II). La p-value minuscule confirme que l'espèce explique la longueur du sépale.
<!-- #endregion -->

```python
lm = ols("sepal_length ~ species", data=iris).fit()
aov = sm.stats.anova_lm(lm, typ=2)
print(aov)
```

<!-- #region -->
### 2.4 MANOVA — plusieurs variables numériques vs un facteur
<!-- #endregion -->

<!-- #region -->
La **MANOVA** généralise l'ANOVA à **plusieurs variables dépendantes simultanément** :
elle teste l'égalité des *vecteurs* de moyennes entre groupes. Les statistiques
usuelles (lambda de **Wilks**, trace de **Pillai**) reposent sur les valeurs propres
de $\mathbf{H}\mathbf{E}^{-1}$ (matrices de covariance inter / intra-groupes).
<!-- #endregion -->

```python
maov = MANOVA.from_formula(
    "sepal_length + sepal_width + petal_length + petal_width ~ species",
    data=iris,
)
mv = maov.mv_test()
wilks = mv.results["species"]["stat"].loc["Wilks' lambda"].astype(float)
print("Wilks' lambda (species):", wilks.round(5).to_dict())
```

<!-- #region -->
## 3. Analyse factorielle — réduire et cartographier
<!-- #endregion -->

<!-- #region -->
L'analyse factorielle cherche **peu d'axes** (« facteurs ») qui résument l'essentiel
de l'information. Le choix de la méthode dépend du **type de données**. L'arbre de
décision ci-dessous (à garder sous le coude) résume la logique :
<!-- #endregion -->

<!-- #region -->
![Arbre de décision — analyse multivariée](images/arbre_decision_multivarie.png)
<!-- #endregion -->

<!-- #region -->
**Lecture de l'arbre** :

- **Données catégorielles ?**
  - **Oui + numériques aussi** → **FAMD** (données mixtes).
  - **Oui, que du catégoriel** → **MCA** (>2 variables) ou **CA** (table de contingence 2 variables).
  - **Non (que du numérique)** :
    - **groupes de colonnes** → **MFA**.
    - **analyse de formes** → **GPA**.
    - sinon → **PCA**.
<!-- #endregion -->

<!-- #region -->
### 3.0 Rappels mathématiques
<!-- #endregion -->

<!-- #region -->
Toutes ces méthodes partagent la même mécanique : on construit une matrice
(individus × variables ou table de contingence), on la **standardise**, puis on
en extrait les **valeurs/vecteurs propres** (≈ une SVD). Vocabulaire commun :

- **Inertie** = variance totale du nuage de points. Chaque axe en capte une part.
- **Valeur propre** $\lambda_k$ = inertie portée par l'axe $k$ ;
  $\%\text{var}_k = \lambda_k / \sum_j \lambda_j$.
- **Coordonnées factorielles** = projections des individus / modalités sur les axes.
- **Contribution** d'un point à un axe = sa part dans l'inertie de l'axe (qui *fait* l'axe).
- **cos²** = qualité de représentation d'un point sur un axe (proche de 1 = bien projeté).

| Méthode | Type de données | Question répondue |
|---|---|---|
| **PCA** | numérique | axes de variance maximale |
| **CA** | table de contingence (2 var. cat.) | associations lignes ↔ colonnes |
| **MCA** | ≥ 2 variables catégorielles | proximité des modalités |
| **FAMD** | mixte (num + cat) | PCA ⊕ MCA |
| **MFA** | numériques en **groupes** | équilibre entre groupes de variables |
| **GPA** | configurations de **formes** | alignement (rotation/échelle/translation) |
<!-- #endregion -->

<!-- #region -->
### 3.1 Helpers de visualisation factorielle
<!-- #endregion -->

<!-- #region -->
Quatre helpers matplotlib typés (ils **remplacent les graphes de `fanalysis`**) :
`scree_plot` (éboulis), `plot_factor_map` (carte des individus), `correlation_circle`
(cercle des corrélations) et `plot_contributions` (barres de contribution). Réutilisables
tels quels dans un autre projet.
<!-- #endregion -->

```python
def scree_plot(pov: np.ndarray, title: str = "Éboulis des valeurs propres",
               ax: plt.Axes | None = None) -> plt.Axes:
    """Diagramme d'éboulis : % de variance/inertie expliquée par axe + cumul."""
    if ax is None:
        _, ax = plt.subplots()
    n = len(pov)
    axes_lbl = [f"Axe {i + 1}" for i in range(n)]
    ax.bar(axes_lbl, pov, color=CHART["primary_1"], label="% par axe")
    ax2 = ax.twinx()
    ax2.plot(axes_lbl, np.cumsum(pov), color=CHART["mauvais"], marker="o", lw=2)
    ax2.set_ylim(0, 105)
    ax.set(ylabel="% variance", title=title)
    ax2.set_ylabel("% cumulé")
    return ax


def plot_factor_map(coords: pd.DataFrame, labels: pd.Series | None = None,
                    x: int = 0, y: int = 1, title: str = "Carte factorielle",
                    ax: plt.Axes | None = None) -> plt.Axes:
    """Projette des individus sur 2 axes factoriels, colorés par un label optionnel."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(coords.iloc[:, x], coords.iloc[:, y],
                   color=CHART["primary_1"], s=40, alpha=0.7)
    else:
        for i, lvl in enumerate(pd.unique(labels)):
            mask = np.asarray(labels) == lvl
            ax.scatter(coords.iloc[:, x][mask], coords.iloc[:, y][mask],
                       color=PALETTE[i % len(PALETTE)], s=40, alpha=0.7, label=str(lvl))
        ax.legend(title=getattr(labels, "name", None))
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set(xlabel=f"Axe {x + 1}", ylabel=f"Axe {y + 1}", title=title)
    return ax


def correlation_circle(corr: pd.DataFrame, x: int = 0, y: int = 1,
                       ax: plt.Axes | None = None) -> plt.Axes:
    """Cercle des corrélations : flèches variables → axes (PCA/FAMD)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=CHART["accent_dark"], lw=1))
    for var in corr.index:
        cx, cy = corr.iloc[:, x][var], corr.iloc[:, y][var]
        ax.arrow(0, 0, cx, cy, head_width=0.03, color=CHART["primary_1"])
        ax.text(cx * 1.1, cy * 1.1, var, ha="center", color=CHART["accent_dark"])
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xlabel=f"Axe {x + 1}",
           ylabel=f"Axe {y + 1}", title="Cercle des corrélations")
    ax.set_aspect("equal")
    return ax


def plot_contributions(contrib: pd.DataFrame, axis: int = 0, top: int = 10,
                       ax: plt.Axes | None = None) -> plt.Axes:
    """Barres horizontales des contributions des variables à un axe."""
    if ax is None:
        _, ax = plt.subplots()
    s = contrib.iloc[:, axis].sort_values(ascending=True).tail(top)
    ax.barh(s.index.astype(str), s.to_numpy(), color=CHART["primary_1"])
    ax.set(xlabel="contribution", title=f"Contributions — Axe {axis + 1}")
    return ax
```

<!-- #region -->
### 3.2 ACP / PCA — variables numériques
<!-- #endregion -->

<!-- #region -->
La **PCA** projette des variables numériques sur des axes orthogonaux de variance
décroissante. On **centre-réduit** d'abord (sinon les variables à grande échelle
dominent). Deux vues : scikit-learn (pipeline ML) puis prince (API factorielle).
<!-- #endregion -->

<!-- #region -->
#### 3.2.1 Avec scikit-learn
<!-- #endregion -->

<!-- #region -->
Deux fonctions typées : `pca_sklearn` (standardisation + ajustement) et
`variance_table` (tableau valeur propre / % variance / % cumulé).
<!-- #endregion -->

```python
def pca_sklearn(df: pd.DataFrame, n_components: int) -> tuple[SkPCA, np.ndarray, list[str]]:
    """Standardise les variables numériques et applique une PCA scikit-learn.

    Retourne (modèle ajusté, données standardisées, noms des variables).
    """
    num = df.select_dtypes(include="number")
    cols = list(num.columns)
    X = StandardScaler().fit_transform(num.to_numpy())
    model = SkPCA(n_components=n_components).fit(X)
    return model, X, cols


def variance_table(model: SkPCA) -> pd.DataFrame:
    """Tableau valeur propre / % variance / % cumulé par dimension."""
    n = len(model.explained_variance_ratio_)
    return pd.DataFrame({
        "Dim": [f"Dim {i + 1}" for i in range(n)],
        "Val propre": np.round(model.explained_variance_, 3),
        "% var": np.round(model.explained_variance_ratio_ * 100, 2),
        "% cumulé": np.round(np.cumsum(model.explained_variance_ratio_ * 100), 2),
    })


sk_pca, X_std, var_names = pca_sklearn(iris, n_components=3)
variance_table(sk_pca)
```

<!-- #region -->
Projection des 150 fleurs sur les 2 premiers axes, colorées par espèce : `setosa`
se détache nettement (axe 1), `versicolor`/`virginica` se chevauchent partiellement.
<!-- #endregion -->

```python
scores = pd.DataFrame(sk_pca.transform(X_std),
                      columns=[f"Dim {i + 1}" for i in range(sk_pca.n_components_)])
fig, ax = plt.subplots(figsize=(7, 6))
plot_factor_map(scores, iris_species, title="PCA iris (sklearn) — individus", ax=ax)
plt.show()
```

<!-- #region -->
#### 3.2.2 Avec prince (API 2026)
<!-- #endregion -->

<!-- #region -->
`prince.PCA` suit l'API scikit-learn (`.fit`) et expose directement
`eigenvalues_summary`, `row_coordinates`, `column_correlations` (corrélations
variables↔axes, pour le cercle) et `column_contributions_`.
<!-- #endregion -->

```python
prince_pca = prince.PCA(n_components=3, random_state=42).fit(iris_num)
prince_pca.eigenvalues_summary
```

<!-- #region -->
Le **cercle des corrélations** lit l'influence de chaque variable : `petal_length`
et `petal_width` portent l'axe 1, `sepal_width` l'axe 2.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(6, 6))
correlation_circle(prince_pca.column_correlations, ax=ax)
plt.show()
```

<!-- #region -->
Éboulis (à gauche) et contributions des variables à l'axe 1 (à droite).
<!-- #endregion -->

```python
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
scree_plot(prince_pca.percentage_of_variance_, ax=a1)
plot_contributions(prince_pca.column_contributions_, axis=0, ax=a2)
plt.show()
```

<!-- #region -->
### 3.3 AFC / CA — table de contingence (2 variables catégorielles)
<!-- #endregion -->

<!-- #region -->
L'**analyse des correspondances** décrit l'association entre les lignes et les
colonnes d'une **table de contingence**. Elle décompose l'inertie du $\chi^2$ et
place lignes et colonnes sur un même plan : deux modalités proches sont associées.
Exemple classique : couleur des **yeux** × couleur des **cheveux**.
<!-- #endregion -->

```python
hair_eye = pd.DataFrame(
    data=[[326, 38, 241, 110, 3],
          [688, 116, 584, 188, 4],
          [343, 84, 909, 412, 26],
          [98, 48, 403, 681, 85]],
    columns=["Fair", "Red", "Medium", "Dark", "Black"],
    index=["Blue", "Light", "Medium", "Dark"],
)
hair_eye.columns.name = "Cheveux"
hair_eye.index.name = "Yeux"
hair_eye
```

<!-- #region -->
`prince.CA` donne l'inertie par axe et les coordonnées des lignes (`row_coordinates`)
et colonnes (`column_coordinates`).
<!-- #endregion -->

```python
ca = prince.CA(n_components=2, random_state=42).fit(hair_eye)
ca.eigenvalues_summary
```

<!-- #region -->
Carte simultanée : yeux (teal) et cheveux (rouge). On lit le gradient clair→foncé
le long de l'axe 1 (yeux clairs/cheveux blonds à gauche, foncés à droite).
<!-- #endregion -->

```python
row_c = ca.row_coordinates(hair_eye)
col_c = ca.column_coordinates(hair_eye)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(row_c.iloc[:, 0], row_c.iloc[:, 1], color=CHART["primary_1"], s=60)
for lbl, (px, py) in zip(row_c.index, row_c.to_numpy()):
    ax.text(px, py, str(lbl), color=CHART["primary_1"], weight="bold")
ax.scatter(col_c.iloc[:, 0], col_c.iloc[:, 1], color=CHART["mauvais"], s=60, marker="^")
for lbl, (px, py) in zip(col_c.index, col_c.to_numpy()):
    ax.text(px, py, str(lbl), color=CHART["mauvais"], weight="bold")
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set(title="CA — yeux (teal) × cheveux (rouge)", xlabel="Axe 1", ylabel="Axe 2")
plt.show()
```

<!-- #region -->
### 3.4 ACM / MCA — plusieurs variables catégorielles
<!-- #endregion -->

<!-- #region -->
La **MCA** étend la CA à **plus de deux variables catégorielles** : c'est une approche
« datamining » qui cartographie les **modalités** pour repérer celles qui vont ensemble.
La 1ʳᵉ version de ce notebook lisait un fichier depuis Google Drive (non reproductible) ;
on le remplace par un sous-ensemble **catégoriel du Titanic** (chargé programmatiquement).
<!-- #endregion -->

```python
titanic = sns.load_dataset("titanic")
cat_cols = ["sex", "class", "embark_town", "who", "alive"]
titanic_cat = titanic[cat_cols].dropna().astype(str)
print("Titanic catégoriel:", titanic_cat.shape)
titanic_cat.head()
```

<!-- #region -->
`prince.MCA` encode automatiquement les variables en one-hot puis applique l'analyse.
<!-- #endregion -->

```python
mca = prince.MCA(n_components=2, random_state=42).fit(titanic_cat)
mca.eigenvalues_summary
```

<!-- #region -->
Carte des **modalités** : on retrouve l'opposition survie (`alive=yes` / `no`) corrélée
au sexe et à la classe — la signature bien connue du naufrage.
<!-- #endregion -->

```python
mca_cols = mca.column_coordinates(titanic_cat)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mca_cols.iloc[:, 0], mca_cols.iloc[:, 1], color=CHART["mauvais"], s=50)
for lbl, (px, py) in zip(mca_cols.index, mca_cols.to_numpy()):
    ax.text(px, py, str(lbl), fontsize=8, color=CHART["accent_dark"])
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set(title="MCA Titanic — carte des modalités", xlabel="Axe 1", ylabel="Axe 2")
plt.show()
```

<!-- #region -->
### 3.5 AFM / MFA — groupes de variables
<!-- #endregion -->

<!-- #region -->
L'**analyse factorielle multiple** traite des variables organisées en **groupes**
(ici : 3 experts notent 6 vins sur plusieurs descripteurs). Chaque groupe est
normalisé par sa 1ʳᵉ valeur propre pour qu'**aucun groupe ne domine**, puis on fait
une PCA globale. On peut lire la contribution de chaque groupe aux axes.
<!-- #endregion -->

```python
wine = pd.DataFrame(
    data=[[1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
          [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
          [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
          [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
          [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
          [3, 4, 4, 3, 5, 4, 5, 1, 7, 5]],
    columns=["E1 fruity", "E1 woody", "E1 coffee",
             "E2 red fruit", "E2 roasted", "E2 vanillin", "E2 woody",
             "E3 fruity", "E3 butter", "E3 woody"],
    index=[f"Wine {i + 1}" for i in range(6)],
)
groups = {
    f"Expert {no + 1}": [c for c in wine.columns if c.startswith(f"E{no + 1}")]
    for no in range(3)
}
groups
```

<!-- #region -->
Les groupes sont passés à **`.fit(X, groups=...)`** (et non plus au constructeur,
comme dans l'ancienne API). `group_contributions_` donne le poids de chaque expert.
<!-- #endregion -->

```python
mfa = prince.MFA(n_components=2, random_state=42).fit(wine, groups=groups)
print(mfa.eigenvalues_summary)
mfa.group_contributions_.round(3)
```

<!-- #region -->
Carte des individus (vins) sur les 2 premiers axes globaux.
<!-- #endregion -->

```python
mfa_rows = mfa.row_coordinates(wine)
fig, ax = plt.subplots(figsize=(7, 6))
plot_factor_map(mfa_rows, pd.Series(wine.index, name="Vin"),
                title="MFA — individus (vins)", ax=ax)
plt.show()
```

<!-- #region -->
### 3.6 AFDM / FAMD — données mixtes (numérique + catégoriel)
<!-- #endregion -->

<!-- #region -->
La **FAMD** gère un tableau qui mélange variables **numériques et catégorielles** :
mathématiquement, **FAMD = PCA (sur le numérique) ⊕ MCA (sur le catégoriel)**, avec
une pondération qui met les deux types sur un pied d'égalité.

⚠️ Piège pratique : `prince.FAMD` ne reconnaît comme numériques que les colonnes de
dtype **`float`** — on caste donc explicitement les colonnes numériques.
<!-- #endregion -->

```python
wine_mixed = pd.DataFrame(
    data=[["A", "A", "A", 2, 5, 7, 6, 3, 6, 7],
          ["A", "A", "A", 4, 4, 4, 2, 4, 4, 3],
          ["B", "A", "B", 5, 2, 1, 1, 7, 1, 1],
          ["B", "A", "B", 7, 2, 1, 2, 2, 2, 2],
          ["B", "B", "B", 3, 5, 6, 5, 2, 6, 6],
          ["B", "B", "A", 3, 5, 4, 5, 1, 7, 5]],
    columns=["c1", "c2", "c3", "n1", "n2", "n3", "n4", "n5", "n6", "n7"],
    index=[f"Wine {i + 1}" for i in range(6)],
)
num_cols = ["n1", "n2", "n3", "n4", "n5", "n6", "n7"]
wine_mixed[num_cols] = wine_mixed[num_cols].astype(float)
oak = pd.Series([1, 2, 2, 2, 1, 1], index=wine_mixed.index, name="Oak type")
wine_mixed
```

<!-- #region -->
Ajustement et inertie expliquée. La carte des individus colorée par `Oak type`
montre une séparation nette des deux types de chêne sur l'axe 1.
<!-- #endregion -->

```python
famd = prince.FAMD(n_components=2, random_state=42).fit(wine_mixed)
print(famd.eigenvalues_summary)
famd_rows = famd.row_coordinates(wine_mixed)
fig, ax = plt.subplots(figsize=(7, 6))
plot_factor_map(famd_rows, oak.astype(str), title="FAMD — individus (Oak type)", ax=ax)
plt.show()
```

<!-- #region -->
### 3.7 GPA — analyse procustéenne généralisée (formes)
<!-- #endregion -->

<!-- #region -->
La **GPA** aligne plusieurs **configurations de points** (formes) en supprimant les
différences de **translation, rotation et échelle**, pour ne garder que les
différences de *forme* réelles. Usages : morphométrie, comparaison de capteurs,
consensus de notations spatiales.

La 1ʳᵉ version utilisait une implémentation maison incorrecte (rotation 1-D bricolée
qui mutait le DataFrame en place) ; on la remplace par **`prince.GPA`**. L'entrée est
un tableau 3D `(n_formes, n_points, n_dimensions)`.
<!-- #endregion -->

```python
rng = np.random.RandomState(0)
base_shape = np.array([[0, 0], [1, 0], [1.2, 1], [0.4, 1.4], [-0.3, 0.8]], dtype=float)


def _rot(theta: float) -> np.ndarray:
    """Matrice de rotation 2D d'angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# 3 copies de la même forme : translatées, tournées, redimensionnées + bruit.
shapes = np.stack([
    base_shape @ _rot(0.0).T + np.array([2.0, 2.0]) + rng.normal(0, 0.03, base_shape.shape),
    base_shape * 1.4 @ _rot(0.7).T + np.array([-1.0, 1.0]) + rng.normal(0, 0.03, base_shape.shape),
    base_shape * 0.7 @ _rot(-0.5).T + np.array([0.5, -1.5]) + rng.normal(0, 0.03, base_shape.shape),
])
print("formes (n_formes, n_points, n_dims):", shapes.shape)
```

<!-- #region -->
`fit_transform` renvoie les formes **alignées** : avant, 3 pentagones éparpillés ;
après, ils se superposent presque parfaitement (seul le bruit subsiste).
<!-- #endregion -->

```python
gpa = prince.GPA()
aligned = np.asarray(gpa.fit_transform(shapes))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for i in range(shapes.shape[0]):
    a1.plot(*np.vstack([shapes[i], shapes[i][0]]).T, marker="o",
            color=PALETTE[i], label=f"forme {i + 1}")
    a2.plot(*np.vstack([aligned[i], aligned[i][0]]).T, marker="o",
            color=PALETTE[i], label=f"forme {i + 1}")
a1.set_title("Avant alignement")
a2.set_title("Après GPA")
for a in (a1, a2):
    a.legend()
    a.set_aspect("equal")
plt.show()
```

<!-- #region -->
## 4. Réduction de dimension (manifold learning)
<!-- #endregion -->

<!-- #region -->
Au-delà de la PCA (**linéaire**), les méthodes de **manifold learning** capturent des
structures **non-linéaires**. On les compare toutes sur iris via un helper unifié.
<!-- #endregion -->

```python
X_iris = iris_num.to_numpy()


def reduce(method: str, X: np.ndarray) -> np.ndarray:
    """Réduit X à 2D selon la méthode demandée (sklearn manifold + umap)."""
    reducers = {
        "PCA": SkPCA(n_components=2),
        "Isomap": Isomap(n_neighbors=10, n_components=2),
        "LLE": LocallyLinearEmbedding(n_neighbors=10, n_components=2),
        "MDS": MDS(n_components=2, normalized_stress="auto", random_state=42),
        "t-SNE": TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42),
        "UMAP": umap.UMAP(n_components=2, random_state=42),
    }
    return reducers[method].fit_transform(X)
```

<!-- #region -->
**Intuition de chaque méthode** :

- **PCA** — projection **linéaire** de variance maximale (rapide, interprétable).
- **Isomap** — préserve les **distances géodésiques** le long de la variété.
- **LLE** — reconstruit chaque point depuis ses **voisins** (relations locales).
- **MDS** — préserve au mieux les **distances deux-à-deux** (minimise le *stress*).
- **t-SNE** — préserve les **voisinages** locaux (divergence KL) ; superbe pour visualiser des clusters.
- **UMAP** — graphe flou de voisinage ; **rapide**, préserve mieux la structure globale que t-SNE. Standard 2026.
<!-- #endregion -->

```python
methods = ["PCA", "Isomap", "LLE", "MDS", "t-SNE", "UMAP"]
embeddings = {m: reduce(m, X_iris) for m in methods}
{m: emb.shape for m, emb in embeddings.items()}
```

<!-- #region -->
Grille comparative : `setosa` (teal) toujours isolée ; t-SNE et UMAP séparent
nettement les trois espèces, là où les méthodes linéaires laissent
`versicolor`/`virginica` se chevaucher.
<!-- #endregion -->

```python
species_codes = pd.Categorical(iris_species).codes
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, m in zip(axes.ravel(), methods):
    emb = embeddings[m]
    for i, lvl in enumerate(iris_species.unique()):
        mask = species_codes == i
        ax.scatter(emb[mask, 0], emb[mask, 1], color=PALETTE[i], s=20, alpha=0.7, label=lvl)
    ax.set_title(m)
    ax.set_xticks([])
    ax.set_yticks([])
axes.ravel()[0].legend(title="espèce", fontsize=8)
fig.suptitle("Réduction de dimension d'iris — 6 méthodes", weight="bold")
plt.show()
```

<!-- #region -->
## 5. Récapitulatif — quelle méthode choisir
<!-- #endregion -->

<!-- #region -->
| Situation | Méthode | Section |
|---|---|---|
| Expliquer/prédire une variable numérique | Régression linéaire | 2.1 |
| Expliquer/prédire une variable catégorielle | Régression logistique | 2.2 |
| Comparer des moyennes (1 var.) entre groupes | ANOVA | 2.3 |
| Comparer des moyennes (≥2 var.) entre groupes | MANOVA | 2.4 |
| Résumer des variables **numériques** | **PCA** | 3.2 |
| Associer 2 variables **catégorielles** (contingence) | **CA** | 3.3 |
| Cartographier **≥2 variables catégorielles** | **MCA** | 3.4 |
| Variables numériques en **groupes** | **MFA** | 3.5 |
| Données **mixtes** (num + cat) | **FAMD** | 3.6 |
| Aligner des **formes** | **GPA** | 3.7 |
| Visualiser des clusters **non-linéaires** | t-SNE / UMAP | 4 |

Garder l'**arbre de décision** de la section 3 comme aide-mémoire principal.
<!-- #endregion -->
