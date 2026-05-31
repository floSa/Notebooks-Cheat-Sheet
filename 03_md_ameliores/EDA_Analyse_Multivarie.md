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
# Analyse multivariée — méthodes factorielles, tests & réduction de dimension
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
  couvre PCA/CA/MCA/MFA/FAMD/GPA). Son API a été **entièrement refondue en 2022** —
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
import pacmap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import FactorAnalysis, KernelPCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding
from sklearn.metrics import adjusted_rand_score
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
**Quand & pourquoi.** On veut **expliquer ou prédire une variable numérique
continue** à partir d'autres variables. Ici : prédire le pourboire (`tip`) à partir
de la facture et de la taille de la table. Sert aussi à **quantifier l'effet** de
chaque variable, toutes choses égales par ailleurs.

**Données.** Cible **numérique continue** ; prédicteurs numériques (ou catégoriels
encodés). On travaille sur les colonnes numériques de `tips`.

**Principe & maths.** On cherche les coefficients $\beta$ qui **minimisent la somme
des carrés des résidus** (moindres carrés ordinaires, OLS) :

$$\hat{y} = \beta_0 + \sum_j \beta_j x_j, \qquad
\hat{\beta} = \arg\min_\beta \sum_i (y_i - \hat{y}_i)^2 = (X^\top X)^{-1} X^\top y$$

Le coefficient $\beta_j$ se lit : *« +1 unité de $x_j$ ⟹ $+\beta_j$ unités de $y$,
les autres variables fixées »*. La qualité globale se mesure par le
$R^2 = 1 - \frac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y)^2}$ (part de variance expliquée).

**Deux backends complémentaires** : **scikit-learn** (prédiction : coefs, R², RMSE)
et **statsmodels** (inférence : p-values, IC à 95 %, $R^2$ ajusté).
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
**Comment lire (règles générales).**

- **$R^2$** (part de variance expliquée, entre 0 et 1) :
  - **proche de 1** → le modèle capture presque toute la variabilité de $y$ ;
  - **0,3 – 0,7** → effet réel mais **partiel** (d'autres facteurs jouent) ;
  - **proche de 0** → prédicteurs non pertinents, relation **non linéaire**, ou variables manquantes ;
  - **négatif (en validation)** → le modèle fait **pire que la moyenne** = sur-ajustement.
- **Coefficient $\beta_j$** : le **signe** donne le sens de l'effet (+ augmente $y$, − le diminue),
  la **valeur** l'ampleur (en unités de $y$ par unité de $x_j$, autres variables fixées).
  Ne comparer des $\beta_j$ entre eux que si les $x$ sont à la **même échelle** (sinon standardiser).
- **p-value du coefficient** : **< 0,05** → effet **significatif** ; **≥ 0,05** → effet **non
  démontré** (ce n'est *pas* « effet nul », juste pas de preuve).
- **RMSE** : erreur typique de prédiction dans l'unité de $y$ ; à juger **relativement** à
  l'écart-type de $y$ (RMSE ≪ écart-type = bon).

**Sur cet exemple** : $R^2 \approx 0{,}47$ (effet partiel), `total_bill` significatif
(p ≈ 0 ; +1 \$ de facture → +0,09 \$ de pourboire), RMSE ≈ 1 \$.

**Hypothèses OLS** (sinon l'inférence est biaisée) : relation **linéaire**, résidus
**indépendants**, **homoscédastiques** et **normaux** — inspecter via *residual plot* + *QQ-plot*.

**À retenir** : la régression linéaire **quantifie** un effet et le **teste** ; lire le triplet
**(R², signe+ampleur du coef, p-value)** ensemble, jamais isolément.
<!-- #endregion -->

<!-- #region -->
### 2.2 Régression logistique
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** On veut **prédire/expliquer une variable catégorielle**
(binaire ici : le sexe du payeur). Au lieu d'une valeur continue, on modélise la
**probabilité** d'appartenir à une classe.

**Données.** Cible **catégorielle** (binaire ou multiclasse) ; prédicteurs numériques
ou encodés. Ici `total_bill, tip, size` → `sex`.

**Principe & maths.** On passe la combinaison linéaire dans la **sigmoïde** pour la
ramener dans $[0,1]$ :

$$P(y=1 \mid x) = \sigma(\beta_0 + \beta^\top x), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Les coefficients s'interprètent en **log-odds** ; leur exponentielle donne l'**odds-ratio**
$e^{\beta_j}$ : facteur multiplicatif sur la **cote** $\frac{P(1)}{P(0)}$ quand $x_j$
augmente de 1. $e^{\beta_j} > 1$ ⟹ augmente la probabilité, $< 1$ ⟹ la diminue.

`penalty=None` désactive la régularisation (l'ancien `penalty='none'` — une chaîne —
est supprimé des versions récentes de scikit-learn).
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
Les **odds-ratios** ($e^{\beta}$) rendent les coefficients lisibles : un OR de 1,04 sur
`total_bill` signifie *« +1 \$ de facture multiplie par 1,04 la cote d'être un homme »*.
<!-- #endregion -->

```python
odds = pd.DataFrame({
    "coef (log-odds)": logit_sm.params,
    "odds-ratio": np.exp(logit_sm.params),
    "p-value": logit_sm.pvalues,
}, index=["constante"] + list(X_log.columns))
odds.round(3)
```

<!-- #region -->
**Comment lire (règles générales).**

- **Odds-ratio $e^{\beta_j}$** (effet multiplicatif sur la cote $P(1)/P(0)$ quand $x_j$ +1) :
  - **> 1** → $x_j$ **augmente** la probabilité de la classe 1 (ex. OR = 1,5 → +50 % de cote) ;
  - **< 1** → $x_j$ la **diminue** (ex. OR = 0,5 → cote divisée par 2) ;
  - **≈ 1** → **pas d'effet**. Plus l'OR s'éloigne de 1, plus l'effet est fort.
- **p-value du coefficient** : **< 0,05** → effet significatif ; **≥ 0,05** → non démontré
  (un OR « parlant » mais non significatif ne prouve rien).
- **Pseudo-$R^2$ de McFadden** : **> 0,2–0,4** → bon ajustement ; **0,1–0,2** → modéré ;
  **< 0,1** → faible pouvoir explicatif.
- **Matrice de confusion / AUC** (à ajouter en pratique) : pour juger la **qualité de
  classification**, pas seulement l'ajustement.

**Sur cet exemple** : tous les OR ≈ 1, p-values > 0,05, pseudo-$R^2 \approx 0{,}02$ →
facture/pourboire/taille **ne prédisent pas** le sexe. C'est le **bon** enseignement : ne pas
sur-interpréter un modèle non significatif.

**Pièges** : suppose une **séparation linéaire en log-odds** ; sensible à la
**multicolinéarité** (ici `tip`/`total_bill` corrélés) et au **déséquilibre des classes**.

**À retenir** : lire les coefficients en **odds-ratios**, et **toujours** croiser avec la
**significativité** avant de conclure à un effet.
<!-- #endregion -->

<!-- #region -->
### 2.3 ANOVA — une variable numérique vs un facteur
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Comparer la **moyenne d'une variable numérique** entre **plusieurs
groupes** définis par une variable catégorielle. Question : *« la longueur du sépale
diffère-t-elle selon l'espèce ? »* C'est le test de référence pour « 1 numérique × 1 facteur ».

**Données.** Une variable **numérique** (cible) + un **facteur catégoriel** (≥ 2 modalités).

**Principe & maths.** On **décompose la variance totale** en variance **inter-groupes**
(expliquée par le facteur) et **intra-groupes** (résiduelle). Le test $F$ confronte les deux ;
$H_0$ = « toutes les moyennes de groupe sont égales » :

$$F = \frac{\text{SCE}_{\text{inter}} / (k-1)}{\text{SCE}_{\text{intra}} / (n-k)}, \qquad
\eta^2 = \frac{\text{SCE}_{\text{inter}}}{\text{SCE}_{\text{totale}}} \;(\text{taille d'effet})$$

Le boxplot ci-dessous **visualise** déjà la séparation : si les boîtes ne se chevauchent
pas, l'ANOVA sera très significative.
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
**Comment lire (règles générales).**

- **p-value (colonne `PR(>F)`)** : **< 0,05** → on **rejette $H_0$** : au moins une moyenne
  de groupe diffère ; **≥ 0,05** → pas de différence démontrée.
- **Statistique $F$** : plus elle est **grande**, plus la variance **inter-groupes** domine
  la variance **intra** (groupes bien séparés). Un $F$ proche de 1 → groupes indistincts.
- **Taille d'effet $\eta^2 = \frac{\text{SCE}_{\text{inter}}}{\text{SCE}_{\text{totale}}}$**
  (conventions de Cohen) : **≈ 0,01** petit · **≈ 0,06** moyen · **≥ 0,14** grand effet.
  C'est elle qui dit l'**ampleur** ; avec un gros échantillon une p-value peut être minuscule
  pour un effet trivial.
- **Significatif globalement ?** → enchaîner avec un **post-hoc** (Tukey) pour savoir
  *quelles paires* de groupes diffèrent.

**Sur cet exemple** : $F \approx 119$, p $\approx 10^{-31}$, $\eta^2 \approx 0{,}62$ → l'espèce
explique 62 % de la variance de `sepal_length` : effet **massif**.

**Hypothèses** : résidus **normaux** par groupe (Shapiro), **homoscédasticité** (Levene),
**indépendance**. Homoscédasticité violée → **ANOVA de Welch**.

**À retenir** : toujours accompagner la **p-value** (« y a-t-il un effet ? ») d'une **taille
d'effet** (« quelle ampleur ? »).
<!-- #endregion -->

<!-- #region -->
### 2.4 MANOVA — plusieurs variables numériques vs un facteur
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Comme l'ANOVA, mais avec **plusieurs variables numériques
dépendantes en même temps**. Question : *« les espèces diffèrent-elles sur l'ensemble
{longueur/largeur sépale + pétale} pris conjointement ? »* Utile quand les variables sont
**corrélées** (les tester séparément ignorerait leurs liens et gonflerait le risque d'erreur).

**Données.** **Plusieurs** variables **numériques** (cibles) + un **facteur catégoriel**.

**Principe & maths.** On teste l'égalité des **vecteurs de moyennes** entre groupes. On
forme les matrices de covariance **inter-groupes** $\mathbf{H}$ et **intra** $\mathbf{E}$,
et on résume $\mathbf{H}\mathbf{E}^{-1}$ par ses valeurs propres. Les statistiques usuelles :

- **Lambda de Wilks** $\Lambda = \prod \frac{1}{1+\lambda_i} \in [0,1]$ — **proche de 0 = forte séparation**.
- **Trace de Pillai** — la plus **robuste** aux écarts d'hypothèses.

On lit la **p-value** associée (via une approximation en $F$).
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
**Comment lire (règles générales).**

- **Lambda de Wilks $\Lambda \in [0,1]$** :
  - **proche de 0** → les groupes ont des **vecteurs de moyennes très différents** (forte séparation) ;
  - **proche de 1** → moyennes multivariées **quasi identiques** (pas de différence).
- **p-value** (via l'approximation en $F$) : **< 0,05** → différence multivariée significative.
- **Plusieurs statistiques** dans la sortie (Wilks, **Pillai**, Hotelling-Lawley, Roy) :
  elles concordent en général ; en cas de doute (hypothèses limites), se fier à **Pillai**,
  la plus **robuste**. Roy est la plus puissante mais la moins robuste.
- **Si significatif** → revenir aux **ANOVA univariées** (par variable) pour localiser
  *quelles* variables portent la différence.

**Sur cet exemple** : $\Lambda \approx 0{,}023$ (≈ 0), $F \approx 199$, p $\approx 0$ → les
espèces ont des profils multivariés radicalement différents — ce qui rend iris si séparable en PCA (§3.2).

**Hypothèses** : **normalité multivariée** et **égalité des matrices de covariance** (Box's M).

**À retenir** : la MANOVA teste les variables **conjointement** ; un $\Lambda$ proche de 0 =
forte séparation dans l'espace multivarié.
<!-- #endregion -->

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
**L'idée commune.** Un tableau de données est un **nuage de points** dans un espace à
$p$ dimensions (une par variable). Ce nuage a une **forme** (sa dispersion = son
**inertie**). L'analyse factorielle cherche les **directions selon lesquelles le nuage
s'étire le plus** : on les ordonne, on n'en garde que quelques-unes, et on **projette**
dessus pour voir l'essentiel en 2D.

**La mécanique (commune à toutes les méthodes).** On construit une matrice (individus ×
variables, ou table de contingence), on la **standardise/pondère** selon la méthode, puis
on en extrait les **valeurs et vecteurs propres** — concrètement une **SVD** $X = U\Sigma V^\top$ :

- les **vecteurs propres** ($V$) donnent les **axes factoriels** (directions d'étirement) ;
- les **valeurs propres** $\lambda_k = \sigma_k^2$ donnent l'**inertie portée par chaque axe**.

**Vocabulaire et formules à connaître** (pour un individu $i$, un axe $k$) :

- **Inertie** = variance totale du nuage = $\sum_k \lambda_k$. Chaque axe en capte $\lambda_k$.
- **% de variance** de l'axe $k$ : $\dfrac{\lambda_k}{\sum_j \lambda_j}$ — combien l'axe « résume ».
- **Coordonnée factorielle** $F_{ik}$ = projection de l'individu sur l'axe (sa position sur la carte).
- **Contribution** $\text{ctr}_{ik} = \dfrac{m_i\,F_{ik}^2}{\lambda_k}$ : part de l'individu dans
  l'inertie de l'axe — **qui *construit* l'axe** (à lire pour nommer l'axe).
- **cos²** $\cos^2_{ik} = \dfrac{F_{ik}^2}{\sum_j F_{ij}^2}$ : **qualité de représentation**
  (proche de 1 = le point est fidèlement projeté sur cet axe ; proche de 0 = méfiance).

> **Contribution vs cos²** : la **contribution** dit *qui fait l'axe* (pour l'interpréter) ;
> le **cos²** dit *si un point est bien représenté* (pour savoir si on peut lui faire confiance).

**Choisir la méthode selon les données** (voir l'arbre ci-dessus) :

| Méthode | Type de données | Question répondue | Inertie totale |
|---|---|---|---|
| **PCA** | numériques | axes de variance maximale | $p$ (vars standardisées) |
| **FA** | numériques | facteurs **latents** + bruit | — (modèle proba) |
| **CA** | table de contingence (2 var. cat.) | associations lignes ↔ colonnes | $\phi^2 = \chi^2/n$ |
| **MCA** | ≥ 2 variables catégorielles | proximité des modalités | $\frac{J}{Q}-1$ (gonflée) |
| **FAMD** | mixte (num + cat) | PCA ⊕ MCA | mixte |
| **MFA** | numériques en **groupes** | équilibre entre groupes | groupes pondérés |
| **GPA** | configurations de **formes** | alignement (rotation/échelle/translation) | distance de Procruste |
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
**Quand & pourquoi.** Vous avez beaucoup de variables numériques **corrélées** et vous
voulez (a) **résumer** l'information en 2-3 axes, (b) **visualiser** la structure des
individus, (c) **débruiter** avant un autre modèle. C'est la méthode de référence du
tableau « tout numérique ».

**Données.** Uniquement des variables **numériques** (continues). Pré-requis :
**centrer-réduire** (sinon une variable en kilomètres écrase une variable en grammes).

**Principe & maths.** On cherche les axes orthogonaux qui **maximisent la variance projetée**.
Le 1ᵉʳ axe est la direction de variance maximale ; le 2ᵉ, orthogonal, capte le maximum du
reste, etc. Algébriquement : **vecteurs propres de la matrice de corrélation** (données
standardisées), valeurs propres = variance portée.

$$\text{Axe}_1 = \arg\max_{\|w\|=1}\ \mathrm{Var}(Xw)$$

**Comment lire.** (1) Le **% de variance** par axe dit combien on a « compressé » ;
(2) le **cercle des corrélations** dit quelles variables font chaque axe ;
(3) la **carte des individus** montre les groupes ;
(4) **contributions** (qui fait l'axe) et **cos²** (qualité) affinent l'interprétation.
On présente deux vues : **scikit-learn** (pipeline ML) puis **prince** (API factorielle complète).
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
**Comment lire (règles générales).**

- **% de variance de l'axe 1** :
  - **élevé (> 70 %)** → une **structure dominante** : un seul facteur explique presque tout ;
  - **réparti** (axes 1 et 2 proches) → information **multidimensionnelle**, il faut plus d'axes ;
  - **% cumulé du plan 1-2 ≥ 80 %** → le plan suffit ; **< 50 %** → la carte 2D est trompeuse,
    examiner les axes 3+.
- **Cercle des corrélations** : une variable dont la **flèche est longue et proche du cercle**
  est bien représentée ; deux flèches **dans le même sens** = variables corrélées ; à **90°** =
  indépendantes ; **opposées** = anticorrélées. Une flèche alignée sur un axe **définit** cet axe.
- **Contributions** : la (les) variable(s) à **forte contribution** *construisent* l'axe → on
  s'en sert pour **nommer** l'axe.

**Sur cet exemple** : axe 1 = 73 %, axe 2 = 23 % (96 % cumulé → plan suffisant) ;
`petal_length`/`petal_width` font l'axe 1 (→ « taille des pétales »), `sepal_width` l'axe 2.

**Pièges.** PCA **linéaire** : rate les structures courbes (→ Kernel PCA §3.8, manifold §4).
Sensible aux **outliers** et à l'**absence de standardisation**.

**À retenir.** Lire dans l'ordre : **% variance** → **contributions (nommer les axes)** →
**cercle** → seulement ensuite la carte des individus.
<!-- #endregion -->

<!-- #region -->
#### 3.2.3 Combien d'axes retenir ?
<!-- #endregion -->

<!-- #region -->
Trois critères usuels pour choisir le nombre de composantes :

- **Kaiser** : garder les axes de **valeur propre > 1** (sur données standardisées ;
  un axe doit porter plus qu'une variable d'origine). Souvent **trop sévère**.
- **Coude (scree)** : couper là où l'éboulis « décroche » (cf. graphe ci-dessus).
- **Seuil de variance cumulée** : garder assez d'axes pour atteindre p. ex. **80 %**.
<!-- #endregion -->

```python
eigvals = prince_pca.eigenvalues_
cumpov = np.cumsum(prince_pca.percentage_of_variance_)
n_kaiser = int((np.asarray(eigvals) > 1).sum())
n_seuil80 = int(np.argmax(cumpov >= 80) + 1)
print(f"Kaiser (valeur propre > 1) : {n_kaiser} axe(s)")
print(f"Seuil 80% variance cumulée : {n_seuil80} axe(s)")
print(f"Variance cumulée : {np.round(cumpov, 1)}")
```

<!-- #region -->
**Comment lire (règles générales).**

- **Si les critères concordent** (Kaiser = coude = seuil) → choix **évident**, prendre ce nombre.
- **Si Kaiser < seuil 80 %** (cas fréquent) → Kaiser est **conservateur** ; privilégier le
  **coude** et le besoin métier (lisibilité d'une carte 2D, débruitage…).
- **Si le coude est net** (une marche d'escalier dans l'éboulis) → couper **juste après** ;
  **s'il n'y a pas de coude** (décroissance régulière) → la variance est diffuse, se rabattre
  sur le seuil cumulé.
- Garder **≥ 2 axes** si l'on veut une **carte** ; pour un simple débruitage avant un modèle,
  garder ce qui atteint le seuil cumulé voulu.

**Sur cet exemple** : Kaiser = **1** ($\lambda_1=2{,}9$ seul > 1), seuil 80 % = **2** ; on retient
**2 axes** (plan lisible, 96 % capté).
<!-- #endregion -->

<!-- #region -->
#### 3.2.4 cos² — qualité de représentation
<!-- #endregion -->

<!-- #region -->
Le **cos²** d'un individu sur un axe mesure **à quel point il est bien projeté** :
proche de 1, le point est fidèlement représenté ; proche de 0, il « vit » sur d'autres
axes et son interprétation sur ce plan est trompeuse. On colore le nuage par la qualité
sur le plan 1-2 (somme des cos² des deux axes).
<!-- #endregion -->

```python
pc_coords = prince_pca.row_coordinates(iris_num)
cos2 = prince_pca.row_cosine_similarities(iris_num)
quality = cos2[0] + cos2[1]  # qualité sur le plan factoriel 1-2
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(pc_coords[0], pc_coords[1], c=quality, cmap="viridis", s=40)
fig.colorbar(sc, ax=ax, label="cos² (qualité plan 1-2)")
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set(title="PCA iris — individus colorés par qualité (cos²)", xlabel="Axe 1", ylabel="Axe 2")
plt.show()
```

<!-- #region -->
#### 3.2.5 Biplot
<!-- #endregion -->

<!-- #region -->
Le **biplot** superpose sur un même plan les **individus** (points) et les **variables**
(flèches = loadings). On lit d'un coup d'œil quelles variables « tirent » quels individus :
ici `petal_length`/`petal_width` pointent vers `virginica`, `sepal_width` vers le haut.
<!-- #endregion -->

```python
def biplot(scores: pd.DataFrame, loadings: np.ndarray, var_names: list[str],
           labels: pd.Series | None = None, ax: plt.Axes | None = None) -> plt.Axes:
    """Biplot PCA : individus (points) + variables (flèches) sur le même plan."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    plot_factor_map(scores, labels, title="Biplot PCA", ax=ax)
    scale = np.abs(scores.iloc[:, :2].to_numpy()).max() / np.abs(loadings).max()
    for i, name in enumerate(var_names):
        ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
                 color=CHART["accent_dark"], head_width=0.08, alpha=0.8)
        ax.text(loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1,
                name, color=CHART["accent_dark"], weight="bold")
    return ax


loadings = sk_pca.components_[:2].T  # (n_vars, 2)
fig, ax = plt.subplots(figsize=(7, 7))
biplot(scores, loadings, var_names, iris_species, ax=ax)
plt.show()
```

<!-- #region -->
#### 3.2.6 Variables et individus supplémentaires
<!-- #endregion -->

<!-- #region -->
Un atout des méthodes factorielles : projeter des **individus supplémentaires** (ou des
variables) **a posteriori**, sans qu'ils n'influencent la construction des axes. Utile
pour positionner de nouvelles observations dans un repère figé. Ici on ajuste la PCA sur
135 fleurs et on **projette** les 15 restantes via `row_coordinates`.
<!-- #endregion -->

```python
rng_sup = np.random.RandomState(1)
sup_idx = rng_sup.choice(iris_num.index, size=15, replace=False)
train_idx = iris_num.index.difference(sup_idx)
pca_train = prince.PCA(n_components=2, random_state=42).fit(iris_num.loc[train_idx])
coords_train = pca_train.row_coordinates(iris_num.loc[train_idx])
coords_sup = pca_train.row_coordinates(iris_num.loc[sup_idx])  # projection sans réajuster
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(coords_train[0], coords_train[1], color=CHART["beige"], s=30, label="actifs")
ax.scatter(coords_sup[0], coords_sup[1], color=CHART["mauvais"], s=70,
           marker="*", label="supplémentaires")
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set(title="PCA — individus supplémentaires (projetés a posteriori)",
       xlabel="Axe 1", ylabel="Axe 2")
ax.legend()
plt.show()
```

<!-- #region -->
#### 3.2.7 Clustering sur les composantes
<!-- #endregion -->

<!-- #region -->
Workflow classique (esprit **HCPC** de FactoMineR) : **réduire** d'abord avec une PCA,
**puis regrouper** sur les premières composantes (débruitées). On compare le KMeans aux
espèces réelles via l'**ARI** (Adjusted Rand Index, 1 = identique). Pour le clustering
approfondi (silhouette, dendrogrammes, DBSCAN…), voir le notebook dédié.
<!-- #endregion -->

```python
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(pc_coords[[0, 1]])
ari = adjusted_rand_score(iris_species, km.labels_)
print(f"KMeans(3) sur axes 1-2 — ARI vs espèces réelles : {ari:.3f}")
fig, ax = plt.subplots(figsize=(7, 6))
plot_factor_map(pc_coords, pd.Series(km.labels_, name="cluster"),
                title=f"KMeans sur composantes PCA (ARI={ari:.2f})", ax=ax)
plt.show()
```

<!-- #region -->
**Comment lire (règles générales).**

- **ARI (Adjusted Rand Index, accord avec une vérité terrain)** :
  - **≈ 1** → clusters quasi identiques aux classes réelles ;
  - **0,5 – 0,8** → correspondance **partielle** (groupes en partie retrouvés) ;
  - **≈ 0** → accord **aléatoire** (le clustering ne capte pas les classes) ;
  - **< 0** → pire que le hasard.
- **Sans vérité terrain** (cas réel), juger la qualité par la **silhouette** ou l'inertie
  intra (méthode du coude), pas par l'ARI.
- **Choix de $k$** : autant de clusters que de structure visible sur la carte ; valider avec
  silhouette / gap statistic.

**Sur cet exemple** : ARI ≈ 0,62 → `setosa` (isolée) parfaitement retrouvée, `versicolor`/
`virginica` (chevauchantes) en partie confondues. On aurait quand même découvert 3 groupes plausibles.

**À retenir.** Réduire **avant** de clusteriser débruite et accélère ; l'ARI ne sert que
**quand on connaît** déjà les vraies classes.
<!-- #endregion -->

<!-- #region -->
#### 3.2.8 PCA vs Factor Analysis (FA)
<!-- #endregion -->

<!-- #region -->
On confond souvent PCA et **analyse factorielle (FA)**, mais ce sont deux modèles
distincts :

- **PCA** — purement géométrique : décompose la **variance totale** en axes orthogonaux.
  Pas de modèle probabiliste.
- **FA** — modèle **à variables latentes** : suppose que les variables observées sont
  générées par quelques **facteurs cachés** + un **bruit spécifique** à chaque variable.
  $x = \mathbf{L}f + \varepsilon$. Idéale quand on cherche à *interpréter* des facteurs
  sous-jacents (psychométrie, questionnaires). La rotation **varimax** rend les loadings
  plus lisibles (chaque variable « charge » surtout un facteur).
<!-- #endregion -->

```python
X_std_fa = StandardScaler().fit_transform(iris_num.to_numpy())
fa = FactorAnalysis(n_components=2, rotation="varimax", random_state=42).fit(X_std_fa)
fa_loadings = pd.DataFrame(fa.components_.T, index=var_names, columns=["Facteur 1", "Facteur 2"])
print(fa_loadings.round(3))
fa_scores = fa.transform(X_std_fa)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
for i, lvl in enumerate(iris_species.unique()):
    m = pd.Categorical(iris_species).codes == i
    a1.scatter(scores.iloc[:, 0][m], scores.iloc[:, 1][m], color=PALETTE[i], s=25, label=lvl)
    a2.scatter(fa_scores[m, 0], fa_scores[m, 1], color=PALETTE[i], s=25, label=lvl)
a1.set_title("PCA (variance)")
a2.set_title("Factor Analysis (facteurs latents + bruit)")
a1.legend(title="espèce")
plt.show()
```

<!-- #region -->
**Comment lire (règles générales).** On lit la **matrice des loadings** (variables × facteurs) :

- un **loading proche de ±1** → la variable **définit** ce facteur (signe = sens) ;
- un **loading proche de 0** → la variable n'a rien à voir avec ce facteur ;
- on **nomme** chaque facteur par les variables qui le chargent fortement ;
- après **varimax**, l'idéal est que **chaque variable charge surtout UN facteur**
  (structure « simple », facile à interpréter). Si une variable charge fort **deux** facteurs,
  elle est **ambiguë**.

**Sur cet exemple** : Facteur 1 chargé par `sepal_length` (0,99), `petal_length` (0,91),
`petal_width` (0,86) → **« taille générale »** ; Facteur 2 par `sepal_width` (−0,67) → **forme
du sépale**. Projections PCA et FA proches ici (données corrélées), mais la FA est plus explicite.

**Quand préférer FA à la PCA.** Si l'on **postule des causes latentes** (traits, satisfaction)
et qu'on veut les **nommer** ; PCA pour la **compression** et la visualisation pures.
<!-- #endregion -->

<!-- #region -->
### 3.3 AFC / CA — table de contingence (2 variables catégorielles)
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Vous avez **deux variables catégorielles croisées** dans une
**table de contingence** (effectifs) et vous voulez voir **quelles modalités s'attirent**
(ex. quel type de client achète quel type de produit). C'est la PCA des tableaux croisés.

**Données.** Une **table de contingence** (comptages ≥ 0), ici couleur des **yeux** ×
couleur des **cheveux**.

**Principe & maths.** On mesure l'écart à l'**indépendance** via le **$\chi^2$**, et on
décompose l'**inertie** $\phi^2 = \chi^2/n$ en axes. On compare des **profils** (lignes
divisées par leur total) avec la **distance du $\chi^2$**. Lignes et colonnes se placent
sur le **même plan**.

**Comment lire.** Une modalité **loin de l'origine** = profil atypique ; une ligne et une
colonne **proches et dans la même direction** = **sur-représentées ensemble** (association
positive) ; **opposées** = elles s'excluent.
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
**Comment lire (règles générales).**

- **% d'inertie de l'axe 1** : **élevé** → association **simple et forte** (un seul gradient
  structure le tableau) ; **réparti sur plusieurs axes** → associations **multiples/complexes**.
- **Position d'une modalité** : **loin de l'origine** → profil **atypique** (se démarque) ;
  **proche du centre** → profil **moyen**, peu informatif.
- **Proximité ligne ↔ colonne** : une ligne et une colonne **proches et dans la même direction
  depuis l'origine** → **sur-représentées ensemble** (association positive) ; **opposées** →
  elles **s'évitent** ; **à 90°** → pas de lien particulier.

**Sur cet exemple** : axe 1 = 87 % (structure simple) ; gradient clair → foncé, `Fair`+yeux
bleus à une extrémité, `Dark`/`Black`+yeux foncés à l'autre : forte association de teinte.

**Pièges.** La CA décrit une **association**, pas une **causalité** ; uniquement des
**comptages** (≥ 0). **À retenir.** Lire le % d'inertie (simple vs complexe) puis les
**proximités ligne↔colonne**.
<!-- #endregion -->

<!-- #region -->
### 3.4 ACM / MCA — plusieurs variables catégorielles
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Vous avez un questionnaire / des données **100 % qualitatives**
(sondage, profils clients, dossiers codés) et vous voulez repérer les **profils-types** :
quelles modalités vont ensemble, lesquelles s'opposent. C'est l'équivalent catégoriel de la PCA.

**Données.** **≥ 3 variables catégorielles** (ici `sex, class, embark_town, who, alive` du
Titanic). Des variables numériques devraient être **discrétisées** au préalable. *(La 1ʳᵉ
version lisait un fichier Google Drive non reproductible — remplacé par le Titanic chargé
programmatiquement.)*

**Principe & maths.** On code chaque modalité en indicatrice 0/1 (matrice **disjonctive
complète**, ou matrice de **Burt** = tous les croisements deux-à-deux), puis on applique une
CA dessus. L'inertie totale vaut $\frac{J}{Q}-1$ ($J$ modalités, $Q$ variables) — donc
**mécaniquement gonflée**, d'où des % de variance par axe faibles (ne pas s'en alarmer).

**Comment lire.** Une modalité **proche du centre** = fréquente / peu discriminante ;
**loin du centre** = rare ou structurante. Deux modalités **proches** co-occurrent ;
**opposées par rapport à l'origine** = exclusives.
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
**Lecture des valeurs propres (règle générale).** En MCA les **% bruts sont toujours faibles**
(inertie gonflée par le codage disjonctif) : un axe 1 à 20-35 % n'est **pas** un échec, alors
qu'en PCA ce serait médiocre. **Ne jamais comparer** ces % à ceux d'une PCA ; pour des
pourcentages comparables, appliquer la **correction de Benzécri/Greenacre**. On se concentre
donc sur les **oppositions** de modalités, pas sur les % bruts.
<!-- #endregion -->

<!-- #region -->
Carte des **modalités** : chaque point est une modalité ; on cherche les regroupements et
les oppositions le long des axes.
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
**Comment lire la carte des modalités (règles générales).**

- **Modalités proches** (même zone) → elles **co-occurrent** chez les mêmes individus (profil-type).
- **Modalités opposées** par rapport à l'origine → elles **s'excluent**.
- **Modalité loin du centre** → **rare ou très structurante** (à confronter à son effectif) ;
  **proche du centre** → fréquente / **peu discriminante**.
- **Nommer chaque axe** par les modalités extrêmes qui le portent (comme pour les variables en PCA).

**Sur cet exemple** : l'axe 1 oppose `alive=yes`/`female`/`First`/`woman` à
`alive=no`/`male`/`Third` → la signature du « women and children first » ; le port
d'embarquement, central, est peu discriminant.

**Pièges.** Une modalité **rare** attire l'œil mais repose sur peu d'individus → **vérifier
les effectifs**. **À retenir.** MCA = PCA des catégories ; lire **oppositions** et
**co-occurrences**, pas les distances absolues ni les % bruts.
<!-- #endregion -->

<!-- #region -->
### 3.5 AFM / MFA — groupes de variables
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Vos variables sont **structurées en groupes** mesurant le même objet
sous plusieurs angles : capteurs par station, blocs d'un questionnaire, **plusieurs experts**
notant les mêmes produits. On veut une vue **globale** où **aucun groupe n'écrase** les autres.

**Données.** Plusieurs **groupes** de variables (ici numériques : 3 experts × descripteurs
sensoriels, 6 vins). MFA gère aussi des groupes catégoriels.

**Principe & maths.** Chaque groupe est d'abord analysé par une PCA et **divisé par sa 1ʳᵉ
valeur propre** $\lambda_1^{(g)}$ : cette normalisation met tous les groupes sur la **même
échelle d'inertie**. On fait ensuite une PCA pondérée sur l'ensemble. On peut lire la
**contribution de chaque groupe** à chaque axe et les **coordonnées partielles** (la vision
d'un groupe pour un individu).

**Comment lire.** `group_contributions_` : un groupe à forte contribution **oriente** l'axe.
Coordonnées partielles proches entre groupes = les experts sont **d'accord** sur cet individu.
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
**Comment lire (règles générales).**

- **% d'inertie de l'axe 1** : **élevé** → les groupes **s'accordent** sur une structure
  commune dominante ; **faible/réparti** → les groupes voient des choses **différentes**.
- **`group_contributions_` (un groupe × un axe)** :
  - contributions **équilibrées** sur un axe → tous les groupes **concourent** à cet axe
    (consensus, la normalisation MFA a empêché qu'un bloc domine) ;
  - un groupe **nettement plus contributif** sur un axe → cet axe traduit **sa spécificité**
    (là où ce groupe se démarque).
- **Coordonnées partielles** d'un individu : si les points des différents groupes sont
  **proches**, les groupes sont **d'accord** sur cet individu ; **dispersés** → désaccord.

**Sur cet exemple** : axe 1 = 85 % avec contributions ~0,32–0,34 (consensus des 3 experts) ;
axe 2 porté par l'**Expert 3** (≈ 0,77) = là où il se démarque.

**Pièges.** MFA n'a de sens que si les **groupes sont pertinents** ; ici 6 vins → illustratif.
**À retenir.** Lire d'abord les **contributions de groupes** pour savoir quel bloc fait quel axe.
<!-- #endregion -->

<!-- #region -->
### 3.6 AFDM / FAMD — données mixtes (numérique + catégoriel)
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Cas le plus fréquent en vrai : un tableau qui **mélange numérique et
catégoriel** (âge + sexe + revenu + catégorie socio…). On veut une PCA qui traite les deux
types **équitablement**, sans avoir à tout binariser ou tout discrétiser.

**Données.** Tableau **mixte** : au moins une colonne numérique **et** une catégorielle.

**Principe & maths.** **FAMD = PCA $\oplus$ MCA** : les variables numériques sont
centrées-réduites (comme en PCA), les catégorielles codées en indicatrices et pondérées
(comme en MCA), de sorte que **chaque variable apporte la même inertie**. On obtient des
axes communs aux deux types.

**Comment lire.** Comme une PCA pour les individus ; les variables numériques se lisent via
un cercle de corrélations, les modalités catégorielles comme des points (barycentres des
individus qui les portent).

Piège pratique : `prince.FAMD` ne reconnaît comme numériques que les colonnes de dtype
**`float`** — on caste donc explicitement les colonnes numériques.
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
**Comment lire (règles générales).** S'interprète **comme une PCA** :

- **% d'inertie** du plan → a-t-on capté l'essentiel en 2D ?
- **carte des individus** → groupes et gradients ; si on dispose d'une variable de coloration
  (illustrative), **une séparation nette** indique que les axes la capturent.
- **rôle des variables** : les **numériques** via leurs corrélations aux axes (comme un cercle),
  les **modalités catégorielles** comme des points (barycentre des individus qui les portent) ;
  une modalité **excentrée** caractérise un côté de l'axe.

**Sur cet exemple** : plan 1-2 = 87 % (66 % + 20 %) ; coloriés par `Oak type`, les vins se
séparent nettement sur l'axe 1 → FAMD a combiné numériques **et** catégorielles `c1/c2/c3`.

**Pièges.** Sensible au **déséquilibre** num/cat (beaucoup de modalités rares gonflent la part
catégorielle) ; dtypes `float` obligatoires pour le numérique. **À retenir.** FAMD = PCA des
**données mixtes**, la pondération interne fait l'équilibre.
<!-- #endregion -->

<!-- #region -->
### 3.7 GPA — analyse procustéenne généralisée (formes)
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Cas particulier : vos individus sont des **formes** (jeux de points
correspondants) et vous voulez les **comparer indépendamment de leur position, orientation
et taille**. Usages : morphométrie (crânes, visages), alignement de capteurs, consensus de
plusieurs annotateurs plaçant des points sur une même image.

**Données.** Un ensemble de **configurations de points appariés** : tableau 3D
`(n_formes, n_points, n_dimensions)`. Les points doivent se **correspondre** d'une forme à l'autre.

**Principe & maths.** On superpose les formes en minimisant la **distance de Procruste** (somme
des carrés des écarts entre points homologues), par **translation + rotation + mise à l'échelle**
optimales, itérativement autour d'une **forme moyenne (consensus)**.

$$\min_{\,s,\,R,\,t}\ \sum_k \lVert s\,R\,x_k + t - \bar{x}_k \rVert^2$$

**Comment lire.** Après alignement, ce qu'il **reste** comme écart entre formes = la
**variabilité de forme réelle** (le signal d'intérêt). *(La 1ʳᵉ version utilisait une
implémentation maison fausse qui mutait le DataFrame en place ; remplacée par `prince.GPA`.)*
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
**Comment lire (règles générales).** On compare les formes **avant** et **après** alignement :

- **si les formes se superposent presque parfaitement** après GPA → elles ont la **même forme**,
  elles ne différaient que par position/orientation/taille (cas de notre exemple : il ne reste
  que le bruit injecté) ;
- **s'il subsiste des écarts structurés** (un sommet systématiquement décalé) → c'est une
  **vraie différence de forme** = le signal d'intérêt ;
- l'**écart résiduel moyen** (distance de Procruste) **quantifie** la variabilité de forme ;
  la **forme moyenne** (consensus) sert de référence.

**Pièges.** Exige une **correspondance point-à-point** (le point $k$ = même repère partout) ;
sensible aux **points aberrants**. **À retenir.** GPA = aligner pour **isoler la forme** ; ce
qui **reste** après superposition est le seul signal interprétable.
<!-- #endregion -->

<!-- #region -->
### 3.8 Kernel PCA — PCA non-linéaire
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Quand vos données ont une structure **non-linéaire** (spirales,
cercles concentriques) que la PCA linéaire « aplatit ». La Kernel PCA garde l'esprit PCA
(axes, projection) mais dans un espace transformé.

**Données.** Numériques, standardisées. Choix d'un **noyau** (RBF, polynomial…) et de ses
hyperparamètres.

**Principe & maths.** *Kernel trick* : au lieu de calculer les produits scalaires dans
l'espace d'origine, on les remplace par un **noyau** $k(x,x') = \langle \phi(x),\phi(x')\rangle$
(ex. RBF $k(x,x')=e^{-\gamma\lVert x-x'\rVert^2}$). On fait alors la PCA dans l'espace
$\phi$ **sans jamais le calculer explicitement**, via la **matrice de Gram** $K$.

**Comment lire.** Comme une PCA, mais les axes ne sont plus interprétables en termes de
variables d'origine (l'espace est implicite). `gamma` règle la « courbure » : trop grand →
sur-ajustement, trop petit → redevient linéaire.
<!-- #endregion -->

```python
X_iris_std = StandardScaler().fit_transform(iris_num.to_numpy())
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5)
kpca_emb = kpca.fit_transform(X_iris_std)
lin_emb = SkPCA(n_components=2).fit_transform(X_iris_std)
codes = pd.Categorical(iris_species).codes
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
for i, lvl in enumerate(iris_species.unique()):
    m = codes == i
    a1.scatter(lin_emb[m, 0], lin_emb[m, 1], color=PALETTE[i], s=25, label=lvl)
    a2.scatter(kpca_emb[m, 0], kpca_emb[m, 1], color=PALETTE[i], s=25, label=lvl)
a1.set_title("PCA linéaire")
a2.set_title("Kernel PCA (RBF)")
a1.legend(title="espèce")
plt.show()
```

<!-- #region -->
**Comment lire (règles générales).** On compare la projection au **PCA linéaire** :

- **si Kernel PCA sépare nettement mieux** les groupes → il y avait une **structure
  non-linéaire** que la PCA ratait (gros gain) ;
- **si les deux se ressemblent** → les données sont **déjà ~linéaires**, la Kernel PCA
  n'apporte rien (cas d'iris : gain modeste) ;
- **effet de `gamma`** : trop **grand** → chaque point s'isole (sur-ajustement, amas éclaté) ;
  trop **petit** → on **retombe sur la PCA linéaire**. Le régler par essais.

Attention : les **axes ne sont pas interprétables** en variables d'origine (espace implicite).

**À retenir.** Kernel PCA = PCA dans un espace transformé ; à comparer **systématiquement** au
PCA linéaire pour juger si le non-linéaire apporte quelque chose.
<!-- #endregion -->

<!-- #region -->
## 4. Réduction de dimension (manifold learning)
<!-- #endregion -->

<!-- #region -->
**Quand & pourquoi.** Quand le but est de **visualiser** en 2D un nuage à structure
**non-linéaire** (la PCA aplatirait une spirale). Surtout pour **explorer** et **repérer des
clusters**, pas pour produire des features interprétables.

**Données.** Numériques (ou une matrice de distances pour MDS/Isomap). Standardiser au préalable.

**Le principe commun.** Toutes ces méthodes préservent une **notion de proximité** : globale
(distances — PCA, MDS, Isomap) ou locale (voisinages — LLE, t-SNE, UMAP, PaCMAP). Elles
diffèrent par *quelle* proximité elles cherchent à conserver. On les compare sur iris via un
helper unifié.
<!-- #endregion -->

```python
X_iris = iris_num.to_numpy()


def reduce(method: str, X: np.ndarray) -> np.ndarray:
    """Réduit X à 2D selon la méthode demandée (sklearn manifold + umap + pacmap)."""
    if method == "PaCMAP":
        return np.asarray(pacmap.PaCMAP(n_components=2, random_state=42).fit_transform(X, init="pca"))
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
- **PaCMAP** — méthode 2026 qui **équilibre structures locale et globale** via 3 types de paires (voisins / mi-distance / lointains) ; souvent plus fidèle que t-SNE/UMAP et peu sensible à l'init.
<!-- #endregion -->

```python
methods = ["PCA", "Isomap", "LLE", "MDS", "t-SNE", "UMAP", "PaCMAP"]
embeddings = {m: reduce(m, X_iris) for m in methods}
{m: emb.shape for m, emb in embeddings.items()}
```

<!-- #region -->
Grille comparative : `setosa` (teal) toujours isolée ; t-SNE, UMAP et PaCMAP séparent
nettement les trois espèces, là où les méthodes linéaires laissent
`versicolor`/`virginica` se chevaucher.
<!-- #endregion -->

```python
species_codes = pd.Categorical(iris_species).codes
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, m in zip(axes.ravel(), methods):
    emb = embeddings[m]
    for i, lvl in enumerate(iris_species.unique()):
        mask = species_codes == i
        ax.scatter(emb[mask, 0], emb[mask, 1], color=PALETTE[i], s=20, alpha=0.7, label=lvl)
    ax.set_title(m)
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes.ravel()[len(methods):]:
    ax.set_visible(False)
axes.ravel()[0].legend(title="espèce", fontsize=8)
fig.suptitle("Réduction de dimension d'iris — 7 méthodes", weight="bold")
plt.show()
```

<!-- #region -->
**Comment lire (règles générales).**

- **Un groupe séparé sur (presque) toutes les méthodes** → séparation **robuste et réelle**
  (ex. `setosa` ici). **Un groupe séparé par une seule méthode** → possible **artefact**, à
  confirmer.
- **Méthodes linéaires** (PCA, MDS) **et** non-linéaires donnent la **même image** → la
  structure est essentiellement **linéaire**. Elles **divergent** → il y a du **non-linéaire**
  que seules t-SNE/UMAP/PaCMAP capturent.
- **Une méthode dégénère** (points alignés/écrasés, ex. **LLE** ici) → instabilité de **cette**
  méthode sur **ces** données, pas une propriété des données : ne pas en tirer de conclusion.
- **Choix selon le but** : **PCA** si l'on veut interpréter/quantifier ; **UMAP/PaCMAP** si l'on
  veut *visualiser* des clusters.

**Sur cet exemple** : `setosa` isolée partout (robuste) ; linéaires laissent
`versicolor`/`virginica` se chevaucher, t-SNE/UMAP/PaCMAP les séparent ; LLE dégénère.

**À retenir.** Aucune méthode n'est « la meilleure » ; **comparer plusieurs projections** évite
de conclure sur un **artefact** d'une seule.
<!-- #endregion -->

<!-- #region -->
### 4.1 Pièges de t-SNE / UMAP
<!-- #endregion -->

<!-- #region -->
Ces méthodes sont superbes pour **visualiser** des clusters, mais traîtres si on
sur-interprète :

- **Les distances n'ont pas de sens global** : deux clusters éloignés sur le plot ne
  sont pas forcément « plus différents » que deux proches.
- **La taille et la densité des clusters sont arbitraires** (t-SNE les égalise).
- **Forte sensibilité aux hyperparamètres** : `perplexity` (t-SNE) / `n_neighbors` (UMAP)
  changent radicalement la figure — voir ci-dessous.
- **Stochastiques** : fixer `random_state` pour la reproductibilité.

Règle : utiliser t-SNE/UMAP pour *explorer*, jamais comme seule preuve d'une structure.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, perp in zip(axes, [5, 30, 50]):
    emb = TSNE(n_components=2, perplexity=perp, max_iter=1000,
               random_state=42).fit_transform(X_iris)
    for i, lvl in enumerate(iris_species.unique()):
        m = codes == i
        ax.scatter(emb[m, 0], emb[m, 1], color=PALETTE[i], s=20, label=lvl)
    ax.set_title(f"perplexity = {perp}")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].legend(title="espèce", fontsize=8)
fig.suptitle("t-SNE — la forme dépend fortement de perplexity", weight="bold")
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
| Interpréter des **facteurs latents** (+ bruit) | **Factor Analysis** | 3.2.8 |
| Associer 2 variables **catégorielles** (contingence) | **CA** | 3.3 |
| Cartographier **≥2 variables catégorielles** | **MCA** | 3.4 |
| Variables numériques en **groupes** | **MFA** | 3.5 |
| Données **mixtes** (num + cat) | **FAMD** | 3.6 |
| Aligner des **formes** | **GPA** | 3.7 |
| Capturer du **non-linéaire** en gardant l'esprit PCA | **Kernel PCA** | 3.8 |
| Visualiser des clusters **non-linéaires** | t-SNE / UMAP / PaCMAP | 4 |
| **Regrouper** après réduction | PCA + KMeans (HCPC) | 3.2.7 |

**Bonnes pratiques transverses** : choisir le nombre d'axes (Kaiser / coude / 80 %, §3.2.3),
vérifier la qualité de projection (**cos²**, §3.2.4), projeter des **individus
supplémentaires** sans réajuster (§3.2.6), et ne jamais sur-interpréter t-SNE/UMAP (§4.1).
Garder l'**arbre de décision** de la section 3 comme aide-mémoire principal.
<!-- #endregion -->

<!-- #region -->
### 5.1 Une fois la méthode choisie — le réflexe de lecture
<!-- #endregion -->

<!-- #region -->
Pour **toutes** les méthodes factorielles, lire **dans cet ordre** :

1. **% de variance / inertie par axe** → a-t-on le droit de se limiter à 2 axes ?
2. **Contributions** → *qui construit chaque axe* → on **nomme** l'axe.
3. **cos²** → les points/variables sont-ils **bien représentés** sur ce plan ?
4. **Carte** → seulement maintenant, lire les **proximités et oppositions**.
<!-- #endregion -->

<!-- #region -->
### 5.2 Erreurs courantes à éviter
<!-- #endregion -->

<!-- #region -->
| Erreur | Pourquoi c'est faux | Le bon réflexe |
|---|---|---|
| Faire une PCA **sans standardiser** | une variable à grande échelle capte toute la variance | centrer-réduire (ou `prince`, qui le fait) |
| Comparer les **% d'inertie** d'une MCA à ceux d'une PCA | l'inertie MCA est mécaniquement gonflée | correction de Benzécri, ou lire les oppositions |
| Interpréter un point **mal représenté** (cos² faible) | sa position sur le plan est trompeuse | vérifier le cos² avant de commenter |
| Lire les **distances** sur un plot t-SNE/UMAP | seules les **voisinages locaux** sont fiables | comparer plusieurs hyperparamètres / méthodes |
| Conclure d'un modèle **non significatif** (régression, ANOVA) | p-value > 0,05 = pas d'effet démontré | rapporter l'absence d'effet honnêtement |
| Mettre des **comptages négatifs / des moyennes** dans une CA | la CA exige une table de contingence (effectifs ≥ 0) | utiliser la bonne méthode selon le type |
| Confondre **PCA** et **Factor Analysis** | la FA modélise des facteurs latents + bruit | choisir selon l'objectif (compresser vs expliquer) |
<!-- #endregion -->
