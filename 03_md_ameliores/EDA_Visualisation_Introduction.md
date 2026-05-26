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
# 📊 Visualisation pour l'EDA — Cheat-sheet & Tutoriel
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Cheat-sheet** : référence visuelle complète pour préparer un rapport / explorer un dataset.

**Objectif** : 45 patterns de visualisation couvrant univariée → bivariée → multivariée → patterns avancés (bar, line) → visualisations expertes multi-catégorielles.

**Particularité de ce notebook** : une **charte graphique unique** (8 couleurs nommées + helpers) appliquée **partout** — démonstration d'une bonne pratique cruciale pour des rapports professionnels.

**Datasets utilisés** :

- `sns.load_dataset('titanic')` (mutualisé avec les notebooks NLP/ML).
- `sns.load_dataset('diamonds')` pour les patterns avancés bar/line.
- Datasets synthétiques (capteurs temporels, ventes multi-catégorielles).

**Renvois** :

- Preprocessing complet (1-hot, scalers, transformations) → `Structures_Preprocessing.ipynb`.
- Auto-corrélation, lag plots, séries temporelles → `TS_Time_Series_Intro.ipynb`.
- PCA / Analyse Multivariée (CA / MCA / FAMD) → `EDA_Analyse_Multivarie.ipynb`.
<!-- #endregion -->

<!-- #region -->
## 1. Setup et imports
<!-- #endregion -->

<!-- #region -->
On centralise tous les imports en haut du notebook. Convention : `pd`, `np`, `plt`, `sns`, `px` (plotly express). Les versions sont affichées pour la reproductibilité.
<!-- #endregion -->

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.graphics.mosaicplot import mosaic

print(f"pandas     : {pd.__version__}")
print(f"numpy      : {np.__version__}")
print(f"matplotlib : {plt.matplotlib.__version__}")
print(f"seaborn    : {sns.__version__}")
```

<!-- #region -->
Tu dois voir des versions récentes (`pandas >= 2.2`, `seaborn >= 0.13`). `plotly` est utilisé pour les visualisations interactives en section 13.
<!-- #endregion -->

<!-- #region -->
## 2. Charte graphique unique
<!-- #endregion -->

<!-- #region -->
**C'est LA bonne pratique fondamentale** de ce notebook. Pourquoi définir et respecter une charte couleurs ?

- **Lisibilité cross-figures** : un lecteur retrouve instantanément "la classe 1 est toujours teal", sans relire chaque légende.
- **Identité visuelle du rapport** : 4 à 8 couleurs maximum, cohérent avec un branding éventuel.
- **Accessibilité daltonien (~8% hommes)** : éviter les couples rouge/vert seuls. Privilégier les couples teal/orange, bleu/orange, bleu foncé/orange (sûrs pour la grande majorité).
- **Hiérarchie visuelle** : les couleurs d'**accent** (rose vif, navy) sont réservées aux **highlights** (valeur max, outlier, classe critique). Les couleurs **neutres** (gris) habillent les séries secondaires sans attirer l'œil.
- **Anti-pattern combattu** : 30 couleurs aléatoires dans un rapport = lecteur saturé en 3 figures, perd toute confiance dans la donnée.

Référence essentielle : *The Visual Display of Quantitative Information* (Tufte, 1983) — concept de **data-ink ratio**, chaque trait d'encre doit servir la donnée.
<!-- #endregion -->

<!-- #region -->
### 2.1 Définition de la palette
<!-- #endregion -->

<!-- #region -->
Palette de **8 couleurs** : 4 primaires (teal, crimson, saffron, sage — les 4 originales de l'utilisateur) + 4 d'extension (navy pour accent foncé, gris neutre, rose vif pour warning, mint pour positif).

| Nom | Hex | Rôle |
|---|---|---|
| `primary_1` | `#00798c` | Teal — catégorie principale |
| `primary_2` | `#d1495b` | Crimson — catégorie secondaire / contraste |
| `primary_3` | `#edae49` | Saffron — catégorie tertiaire |
| `primary_4` | `#66a182` | Sage — catégorie quaternaire |
| `accent_dark` | `#2e4057` | Navy — axes forts, valeur max highlight |
| `neutral` | `#8d99ae` | Cool Gray — background, séries secondaires |
| `accent_warn` | `#ef476f` | Vivid Pink — warning, outlier highlight |
| `accent_ok` | `#06d6a0` | Vivid Mint — positif, "succès" |
<!-- #endregion -->

```python
CHART: dict[str, str] = {
    "primary_1": "#00798c",   # Teal      — catégorie principale
    "primary_2": "#d1495b",   # Crimson   — catégorie secondaire
    "primary_3": "#edae49",   # Saffron   — catégorie tertiaire
    "primary_4": "#66a182",   # Sage      — catégorie quaternaire
    "accent_dark": "#2e4057", # Navy      — accent foncé, texte fort, highlight
    "neutral": "#8d99ae",     # Cool Gray — neutre, background, séries secondaires
    "accent_warn": "#ef476f", # Vivid Pink — warning, outlier highlight
    "accent_ok": "#06d6a0",   # Vivid Mint — positif, success
}
PALETTE: list[str] = list(CHART.values())

# Swatch visuel
fig, ax = plt.subplots(figsize=(11, 1.8))
for i, (name, hex_) in enumerate(CHART.items()):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=hex_))
    ax.text(i + 0.5, -0.35, name, ha="center", va="top", fontsize=8)
    ax.text(i + 0.5, 0.5, hex_, ha="center", va="center", fontsize=9,
            color="white" if i not in (2,) else "black", weight="bold")
ax.set_xlim(0, len(CHART)); ax.set_ylim(-0.6, 1)
ax.set_xticks([]); ax.set_yticks([])
for s in ["top", "right", "bottom", "left"]:
    ax.spines[s].set_visible(False)
ax.set_title("Charte graphique — 8 couleurs", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()
```

<!-- #region -->
Le swatch ci-dessus est la **référence visuelle** de toutes les figures du notebook. Quand tu vois la couleur `#00798c` (teal), c'est toujours la catégorie principale.
<!-- #endregion -->

<!-- #region -->
### 2.2 Helpers — `apply_chart_style()` et `scatter_by_category()`
<!-- #endregion -->

<!-- #region -->
Deux helpers pour ne **jamais** répéter le code de styling :

- `apply_chart_style()` — appliqué une seule fois en début de notebook, configure `seaborn` + `matplotlib` pour utiliser la charte par défaut partout.
- `scatter_by_category(df, x, y, hue, style=None)` — scatter encodant 2 dimensions catégorielles via couleur **et** marker. Pattern utile en monitoring industriel (capteur × état machine).
<!-- #endregion -->

```python
def apply_chart_style() -> None:
    """Applique la charte graphique au runtime matplotlib + seaborn.

    Idempotente. À appeler une fois en début de notebook.
    """
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


def scatter_by_category(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    style: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Scatter encodant 2 dimensions catégorielles via couleur ET marker.

    Args:
        df: DataFrame source.
        x, y: colonnes numériques pour les axes.
        hue: colonne catégorielle encodée par couleur (mapping sur PALETTE).
        style: colonne catégorielle encodée par marker. Optionnel.
        ax: Axes matplotlib existant. Si None, en crée un.
        title: titre.

    Returns:
        l'axe matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    hue_levels = df[hue].unique()
    color_map = {lvl: PALETTE[i % len(PALETTE)] for i, lvl in enumerate(hue_levels)}
    markers = ["o", "s", "^", "D", "v", "*", "P", "X"]
    if style is None:
        for lvl in hue_levels:
            d = df[df[hue] == lvl]
            ax.scatter(d[x], d[y], color=color_map[lvl], label=str(lvl),
                       alpha=0.7, edgecolor="white", s=70)
    else:
        style_levels = df[style].unique()
        marker_map = {lvl: markers[i % len(markers)] for i, lvl in enumerate(style_levels)}
        for h in hue_levels:
            for s in style_levels:
                d = df[(df[hue] == h) & (df[style] == s)]
                if len(d) == 0:
                    continue
                ax.scatter(d[x], d[y], color=color_map[h], marker=marker_map[s],
                           label=f"{h} — {s}", alpha=0.7, edgecolor="white", s=70)
    ax.set_xlabel(x); ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    return ax
```

<!-- #region -->
On applique la charte tout de suite. À partir de maintenant, **toutes** les figures héritent du style.
<!-- #endregion -->

```python
apply_chart_style()
print("Charte graphique appliquée.")
```

<!-- #region -->
## 3. Chargement Titanic et premier regard
<!-- #endregion -->

<!-- #region -->
**Titanic** est le dataset mutualisé du projet (voir `00_datasets.md`). Disponible directement via `seaborn`.

Colonnes principales :

- `survived` (0/1) — la target.
- `pclass` (1/2/3) — classe billet.
- `sex` (male/female), `age`, `sibsp` (frères/sœurs/conjoints), `parch` (parents/enfants), `fare`.
- `embarked` (C/Q/S) — port d'embarquement.
- `class`, `deck`, `embark_town`, `alive`, `alone` — redondants ou texte (à ignorer le plus souvent).
<!-- #endregion -->

```python
df = sns.load_dataset("titanic")
df.sample(2)
```

<!-- #region -->
Premier diagnostic systématique : `shape`, `missing values`, `describe`.
<!-- #endregion -->

```python
print(f"Shape : {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum().sort_values(ascending=False).head(8)}")
print(f"\nDescribe numérique:\n{df.describe()}")
```

<!-- #region -->
**Lecture** : `age` 20% manquant (à imputer), `deck` 77% manquant (à drop ou ré-encoder en "deck connu / inconnu"). Pattern classique d'EDA.
<!-- #endregion -->

<!-- #region -->
## 4. Univariée — variables quantitatives
<!-- #endregion -->

<!-- #region -->
Pour **1 variable numérique**, on cherche : forme de la **distribution**, **tendance centrale** (médiane), **dispersion** (IQR), **queues** (outliers).
<!-- #endregion -->

<!-- #region -->
### 4.1 Histogramme
<!-- #endregion -->

<!-- #region -->
Le grand classique. Variante `seaborn.histplot` avec KDE superposée. Choix du nombre de bins critique : trop peu lisse l'info, trop introduit du bruit (règle de Freedman-Diaconis ou Sturges en pratique).
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["age"].dropna(), bins=20, color=CHART["primary_1"], edgecolor="white")
axes[0].set_title("plt.hist — bins=20"); axes[0].set_xlabel("age"); axes[0].set_ylabel("count")
sns.histplot(df["age"].dropna(), kde=True, color=CHART["primary_1"], ax=axes[1])
axes[1].set_title("sns.histplot — avec KDE")
plt.tight_layout(); plt.show()
```

<!-- #region -->
La couleur primaire 1 (teal) est utilisée pour la distribution principale — convention de ce notebook.
<!-- #endregion -->

<!-- #region -->
### 4.2 Boxplot et boxenplot
<!-- #endregion -->

<!-- #region -->
Le **boxplot** affiche médiane, quartiles (Q1, Q3), moustaches (1.5×IQR), et points outliers. Le **boxenplot** ajoute des quantiles supplémentaires — utile pour distributions à queues lourdes (ex : `fare` du Titanic).
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["fare"], color=CHART["primary_2"], ax=axes[0])
axes[0].set_title("sns.boxplot — fare")
sns.boxenplot(y=df["fare"], color=CHART["primary_2"], ax=axes[1])
axes[1].set_title("sns.boxenplot — fare (queues lourdes)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
`fare` a une queue droite massive (riches sur le Titanic) → le **boxenplot révèle la structure** des outliers de manière plus informative qu'un boxplot classique.
<!-- #endregion -->

<!-- #region -->
### 4.3 KDE et rugplot
<!-- #endregion -->

<!-- #region -->
La **KDE** (kernel density estimate) est l'alternative continue à l'histogramme : moins biaisée par le choix des bins, mais introduit un paramètre de bande passante. Le **rugplot** ajoute un tick par observation — utile pour voir la densité brute.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(df["fare"].dropna(), fill=True, color=CHART["primary_3"], ax=ax, alpha=0.5)
sns.rugplot(df["fare"].dropna(), color=CHART["accent_dark"], ax=ax, height=0.05)
ax.set_title("KDE + rugplot — fare")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Pour petits échantillons (n < 50), préférer hist+rug ; pour grands, KDE seul suffit.
<!-- #endregion -->

<!-- #region -->
## 5. Univariée — variables qualitatives
<!-- #endregion -->

<!-- #region -->
Pour 1 catégorielle : **comptage par modalité** (countplot/barplot via `value_counts`). Toujours **trier par fréquence** sauf si un ordre métier existe (S < M < L, classes ordinales). Highlight optionnel de la modalité modale en couleur d'accent.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order_emb = df["embarked"].value_counts().index.tolist()
sns.countplot(x="embarked", data=df, order=order_emb, ax=axes[0],
              palette=[CHART["primary_1"], CHART["neutral"], CHART["neutral"]])
axes[0].set_title("countplot — embarked (modale en couleur primaire)")
order_pcl = df["pclass"].value_counts().index.tolist()
sns.countplot(x="pclass", data=df, order=order_pcl, ax=axes[1],
              palette=[CHART["primary_1"], CHART["neutral"], CHART["neutral"]])
axes[1].set_title("countplot — pclass")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Pattern **highlight modale** : on met la barre la plus grande en couleur primaire, les autres en gris neutre. L'œil va directement à l'info principale.
<!-- #endregion -->

<!-- #region -->
## 6. Bivariée — quanti × quanti
<!-- #endregion -->

<!-- #region -->
### 6.1 Scatter et regplot
<!-- #endregion -->

<!-- #region -->
`scatter` montre les paires de points. `regplot` ajoute une **régression linéaire** (et son intervalle de confiance) pour quantifier la tendance. Pour des non-linéarités : `regplot(lowess=True)`.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=df, x="age", y="fare", color=CHART["primary_1"], alpha=0.5, ax=axes[0])
axes[0].set_title("scatter — age vs fare")
sns.regplot(data=df, x="age", y="fare",
            scatter_kws={"color": CHART["primary_1"], "alpha": 0.3},
            line_kws={"color": CHART["accent_warn"]}, ax=axes[1])
axes[1].set_title("regplot — fit linéaire")
plt.tight_layout(); plt.show()
```

<!-- #region -->
La droite est en couleur d'accent (`accent_warn` = rose vif) pour ressortir nettement sur les points. Relation âge↔fare faible — comme attendu.
<!-- #endregion -->

<!-- #region -->
### 6.2 Jointplot — scatter + marginales
<!-- #endregion -->

<!-- #region -->
Le `jointplot` combine scatter (ou hex / kde) avec les **distributions marginales** de chaque axe. Idéal pour voir simultanément la jointe et les marginales.
<!-- #endregion -->

```python
g = sns.jointplot(data=df, x="age", y="fare", kind="hex", color=CHART["primary_1"], height=5)
g.fig.suptitle("jointplot kind='hex'", y=1.02)
plt.show()
```

<!-- #region -->
- `kind='hex'` : pour gros volumes (densité en hexagones).
- `kind='kde'` : pour estimer la densité 2D continue.
- `kind='scatter'` : pour petits volumes (par défaut).
<!-- #endregion -->

<!-- #region -->
### 6.3 Heatmap des corrélations
<!-- #endregion -->

<!-- #region -->
La matrice des corrélations Pearson visualisée. **Cmap divergente** (`RdBu_r`) centrée sur 0 : rouge = corrélation positive, bleu = négative. Toujours annoter les valeurs.
<!-- #endregion -->

```python
corr = df.select_dtypes(include="number").corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("Corrélations Pearson — Titanic numérique")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Cellules à investiguer : |corr| > 0.5. Ici on voit corr(pclass, fare) ≈ -0.55 (1ère classe = fare élevé, logique). Cmap divergente centrée 0 permet de **lire le signe instantanément**.
<!-- #endregion -->

<!-- #region -->
## 7. Bivariée — quanti × quali
<!-- #endregion -->

<!-- #region -->
Quand une catégorie agit sur une numérique, **6 alternatives principales** à comparer (boxplot, violinplot, swarmplot, boxenplot, pointplot, barplot). Une figure 2×3 met les 6 côte à côte pour le même couple `sex × age`.
<!-- #endregion -->

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sns.boxplot(data=df, x="sex", y="age", palette=[CHART["primary_1"], CHART["primary_2"]], ax=axes[0, 0])
axes[0, 0].set_title("boxplot")
sns.violinplot(data=df, x="sex", y="age", palette=[CHART["primary_1"], CHART["primary_2"]], ax=axes[0, 1])
axes[0, 1].set_title("violinplot")
sns.swarmplot(data=df.sample(min(300, len(df)), random_state=0), x="sex", y="age",
              palette=[CHART["primary_1"], CHART["primary_2"]], size=3, ax=axes[0, 2])
axes[0, 2].set_title("swarmplot (n=300)")
sns.boxenplot(data=df, x="sex", y="age", palette=[CHART["primary_1"], CHART["primary_2"]], ax=axes[1, 0])
axes[1, 0].set_title("boxenplot")
sns.pointplot(data=df, x="sex", y="age", color=CHART["primary_1"], errorbar=("ci", 95), ax=axes[1, 1])
axes[1, 1].set_title("pointplot (CI 95%)")
sns.barplot(data=df, x="sex", y="age", palette=[CHART["primary_1"], CHART["primary_2"]],
            errorbar=None, ax=axes[1, 2])
axes[1, 2].set_title("barplot (mean)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Quand choisir quoi** :

- **boxplot** : résumé compact (quartiles + outliers).
- **violinplot** : forme exacte de la distribution (KDE par groupe).
- **swarmplot** : tous les points (n < 500), montre les vraies valeurs.
- **boxenplot** : pour distributions à queues lourdes.
- **pointplot** : moyenne + intervalle de confiance — comparaison de moyennes.
- **barplot** : moyenne sans la dispersion — **à éviter en exploratoire** (cache l'information).
<!-- #endregion -->

<!-- #region -->
### 7.1 FacetGrid — un subplot par modalité
<!-- #endregion -->

<!-- #region -->
Pattern **small multiples** : même viz répétée pour chaque valeur d'une catégorielle. Permet la **comparaison structurée**. Alternative moderne : `sns.displot(col='sex')`.
<!-- #endregion -->

```python
g = sns.FacetGrid(df, col="sex", height=4, palette=PALETTE)
g.map_dataframe(sns.histplot, x="age", color=CHART["primary_1"], bins=20)
g.fig.suptitle("FacetGrid — histplot age par sexe", y=1.05)
plt.show()
```

<!-- #region -->
On voit que les hommes ont une distribution d'âge plus large (présence de très jeunes garçons + adultes), les femmes plus concentrées.
<!-- #endregion -->

<!-- #region -->
## 8. Bivariée — quali × quali
<!-- #endregion -->

<!-- #region -->
Deux options principales :

- **countplot avec `hue`** : effectifs absolus côte à côte.
- **pivot_table + heatmap** : vue compacte avec ratios ou comptes.

Ici on combine les deux pour le couple `sex × pclass` + on calcule le **taux de survie** par croisement.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.countplot(data=df, x="sex", hue="pclass", palette=PALETTE[:3], ax=axes[0])
axes[0].set_title("countplot — sex × pclass (effectifs)")
pivot = df.pivot_table(index="sex", columns="pclass", values="survived", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", ax=axes[1])
axes[1].set_title("Taux de survie — sex × pclass")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Lecture du heatmap** : les femmes de 1ère classe ont 97% de survie, les hommes de 3ème seulement 14%. La heatmap met en évidence ces contrastes en un coup d'œil — utile pour identifier des interactions.
<!-- #endregion -->

<!-- #region -->
## 9. Multivariée — 3 dimensions ou plus
<!-- #endregion -->

<!-- #region -->
Pour encoder 3+ dimensions dans une seule figure, on combine :

- **hue** : couleur (1 dim catégorielle).
- **style** : marker (1 dim catégorielle).
- **size** : taille du marker (1 dim numérique).
- **col / row** : facettes (1-2 dims catégorielles).

Limite cognitive : 4-5 dimensions max simultanément lisibles.
<!-- #endregion -->

<!-- #region -->
### 9.1 Pairplot — toutes les paires de variables
<!-- #endregion -->

<!-- #region -->
La matrice de tous les scatters bivariés + distributions univariées en diagonale. Avec `hue`, coloration par modalité catégorielle. **Vue d'ensemble** pour repérer les relations dignes d'investigation plus poussée.
<!-- #endregion -->

```python
g = sns.pairplot(df[["age", "fare", "sibsp", "survived"]].dropna(),
                 hue="survived", palette=[CHART["primary_2"], CHART["primary_1"]], height=2.2)
g.fig.suptitle("pairplot — age/fare/sibsp colorés par survived", y=1.02)
plt.show()
```

<!-- #region -->
**Diagonale** = distributions univariées (KDE par hue). **Hors-diagonale** = scatters bivariés. Crimson = mort, teal = survivant.
<!-- #endregion -->

<!-- #region -->
### 9.2 `sns.relplot` — encodage multidim direct
<!-- #endregion -->

<!-- #region -->
La fonction reine de seaborn pour les multivariées. Permet d'encoder facilement 5 dims simultanément (x, y, col, hue, style).
<!-- #endregion -->

```python
g = sns.relplot(data=df, x="age", y="fare", col="pclass", hue="sex", style="sex",
                kind="scatter", palette=[CHART["primary_1"], CHART["primary_2"]],
                height=4, alpha=0.6)
g.fig.suptitle("relplot — age vs fare, facetté par pclass, coloré par sex", y=1.03)
plt.show()
```

<!-- #region -->
On lit instantanément : les premières classes ont des fares élevés (jusqu'à 500), troisième classe écrasée en bas. La distinction sex est visible par couleur dans chaque facette.
<!-- #endregion -->

<!-- #region -->
### 9.3 Scatter encodage couleur + marker (helper)
<!-- #endregion -->

<!-- #region -->
Pattern utile en **monitoring industriel** : 2 dimensions catégorielles encodées simultanément. Le helper `scatter_by_category` factorise.
<!-- #endregion -->

```python
# Exemple 1 — Titanic : sex (couleur) × pclass (marker)
fig, ax = plt.subplots(figsize=(9, 5))
df_sample = df.dropna(subset=["age", "fare"]).sample(min(200, len(df)), random_state=0)
scatter_by_category(df_sample, x="age", y="fare", hue="sex", style="pclass", ax=ax,
                    title="Titanic — age vs fare (couleur=sex, marker=pclass)")
plt.tight_layout(); plt.show()
```

```python
# Exemple 2 — synthétique : capteurs avec status (couleur) × outlier (marker)
df_machines = pd.DataFrame({
    "sensor_00": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0],
    "sensor_01": [2.0, 3.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0],
    "machine_status": ["Normal", "Normal", "Normal", "Fault", "Maintenance",
                        "Normal", "Normal", "Fault"],
    "outliers": [False, True, False, True, False, True, False, False],
})
fig, ax = plt.subplots(figsize=(8, 5))
scatter_by_category(df_machines, x="sensor_00", y="sensor_01",
                    hue="machine_status", style="outliers", ax=ax,
                    title="Capteurs — status (couleur) × outlier (marker)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Usage type : monitoring industriel. Couleur = état machine, marker = flag outlier. On voit en un coup d'œil **deux dimensions** sans avoir à lire la légende.
<!-- #endregion -->

<!-- #region -->
## 10. Patterns avancés — barplots
<!-- #endregion -->

<!-- #region -->
On passe au dataset **diamonds** (~54 000 lignes, `cut` ordonné en 5 modalités, `color` à 7 modalités) pour les patterns plus riches.
<!-- #endregion -->

```python
diamonds = sns.load_dataset("diamonds")
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
diamonds["cut"] = pd.Categorical(diamonds["cut"], categories=cut_order, ordered=True)
print(f"Diamonds : {diamonds.shape}")
diamonds.head()
```

<!-- #region -->
On utilisera `cut` (5 modalités, ordre métier explicite), `color` (7 modalités), et `price`/`carat` (numériques).
<!-- #endregion -->

<!-- #region -->
### 10.1 Bar single — highlight modalité max
<!-- #endregion -->

<!-- #region -->
Pattern **highlight** : la valeur maximale est colorée en `accent_dark` (navy), les autres en primaire. Force le focus visuel.
<!-- #endregion -->

```python
counts = diamonds["cut"].value_counts().sort_index()
colors = [CHART["accent_dark"] if v == counts.max() else CHART["primary_1"]
          for v in counts.values]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(counts.index.astype(str), counts.values, color=colors)
ax.set_title("Nombre de diamants par 'cut' — max en couleur d'accent")
ax.set_ylabel("count")
plt.tight_layout(); plt.show()
```

<!-- #region -->
"Ideal" est la modalité la plus fréquente — révélée par le contraste avec les 4 autres uniformément teal.
<!-- #endregion -->

<!-- #region -->
### 10.2 Bar multiple subplots — 1 par catégorie
<!-- #endregion -->

<!-- #region -->
Pattern **small multiples** : une sous-figure par modalité. Permet de comparer la **même structure** entre modalités sans superposition.
<!-- #endregion -->

```python
fig, axes = plt.subplots(len(cut_order), 1, figsize=(12, 9), sharex=True)
for i, cut in enumerate(cut_order):
    counts_color = diamonds[diamonds["cut"] == cut]["color"].value_counts().sort_index()
    axes[i].bar(counts_color.index, counts_color.values,
                color=PALETTE[i % len(PALETTE)], label=cut)
    axes[i].legend(loc="upper right")
    axes[i].set_ylabel("count")
axes[-1].set_xlabel("color")
fig.suptitle("Bar multiple subplots — count(color) par cut", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
Chaque cut reçoit une couleur de la palette, garantissant l'unicité visuelle entre subplots.
<!-- #endregion -->

<!-- #region -->
### 10.3 Bar multiple overlap (alpha)
<!-- #endregion -->

<!-- #region -->
Toutes les modalités superposées sur le **même axe**, transparence `alpha=0.4`. Permet de voir les **zones communes**. Limite : illisible au-delà de 4-5 séries.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
for i, cut in enumerate(cut_order):
    counts_color = diamonds[diamonds["cut"] == cut]["color"].value_counts().sort_index()
    ax.bar(counts_color.index, counts_color.values, color=PALETTE[i % len(PALETTE)],
           label=cut, alpha=0.4, edgecolor=PALETTE[i % len(PALETTE)])
ax.legend()
ax.set_title("Bar multiple overlap (alpha=0.4)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 10.4 Bar stacked amount (effectifs cumulés)
<!-- #endregion -->

<!-- #region -->
Empile les modalités sur chaque barre. Somme verticale = **total par color**. Lecture : contributions absolues.
<!-- #endregion -->

```python
pivot = diamonds.groupby("color", observed=True)["cut"].value_counts().unstack(fill_value=0)[cut_order]
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(pivot))
for i, cut in enumerate(cut_order):
    ax.bar(pivot.index, pivot[cut].values, bottom=bottom,
           color=PALETTE[i % len(PALETTE)], label=cut)
    bottom += pivot[cut].values
ax.legend()
ax.set_title("Bar stacked amount — effectifs cumulés")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 10.5 Bar stacked ratio (normalisé en %)
<!-- #endregion -->

<!-- #region -->
Identique au stacked amount mais **normalisé à 100%**. Lecture : contributions **relatives**, met en évidence les **déséquilibres**.
<!-- #endregion -->

```python
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(pivot_pct))
for i, cut in enumerate(cut_order):
    ax.bar(pivot_pct.index, pivot_pct[cut].values, bottom=bottom,
           color=PALETTE[i % len(PALETTE)], label=cut)
    bottom += pivot_pct[cut].values
ax.legend()
ax.set_title("Bar stacked ratio — % par color")
ax.set_ylim(0, 100); ax.set_ylabel("% des diamants")
plt.tight_layout(); plt.show()
```

<!-- #region -->
On voit clairement la **proportion de cut Ideal** dans chaque color (D le moins, J le plus).
<!-- #endregion -->

<!-- #region -->
## 11. Patterns avancés — line plots
<!-- #endregion -->

<!-- #region -->
Pour les line plots non-temporels, on construit une **variable ordinale** : ici `carat-bins` (10 buckets) → `price` moyen. La "ligne" suit l'évolution du prix selon le poids.

8 variantes du plus simple au plus avancé.
<!-- #endregion -->

```python
diamonds["carat_bin"] = pd.cut(diamonds["carat"], bins=10)
diamonds["carat_bin_mid"] = diamonds["carat_bin"].apply(lambda x: x.mid).astype(float)
price_by_bin = diamonds.groupby("carat_bin_mid", observed=True)["price"].mean()
price_by_bin.head()
```

<!-- #region -->
### 11.1 Line single
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(price_by_bin.index, price_by_bin.values, color=CHART["accent_dark"], lw=2)
ax.set_title("Line single — prix moyen vs carat")
ax.set_xlabel("carat (midpoint bin)"); ax.set_ylabel("price moyen")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.2 Line single area (`fill_between`)
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(price_by_bin.index, price_by_bin.values, color=CHART["accent_dark"], lw=2)
ax.fill_between(price_by_bin.index, 0, price_by_bin.values, color=CHART["primary_1"], alpha=0.3)
ax.set_title("Line single area")
ax.set_xlabel("carat"); ax.set_ylabel("price moyen")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.3 Line multiple subplots (1 par cut)
<!-- #endregion -->

```python
fig, axes = plt.subplots(len(cut_order), 1, figsize=(12, 9), sharex=True)
for i, cut in enumerate(cut_order):
    sub = diamonds[diamonds["cut"] == cut].groupby("carat_bin_mid", observed=True)["price"].mean()
    axes[i].plot(sub.index, sub.values, color=PALETTE[i % len(PALETTE)], label=cut, lw=2)
    axes[i].legend(loc="upper left")
fig.suptitle("Line multiple subplots — price vs carat par cut", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.4 Line multiple horizon (subplots collés)
<!-- #endregion -->

<!-- #region -->
Subplots avec `hspace=0` et axes y sans labels — variante compacte du subplot, optimise l'espace tout en gardant la séparation des séries.
<!-- #endregion -->

```python
fig, axes = plt.subplots(len(cut_order), 1, figsize=(12, 6), sharex=True)
for i, cut in enumerate(cut_order):
    sub = diamonds[diamonds["cut"] == cut].groupby("carat_bin_mid", observed=True)["price"].mean()
    axes[i].fill_between(sub.index, 0, sub.values,
                          color=PALETTE[i % len(PALETTE)], alpha=0.8, label=cut)
    axes[i].legend(loc="upper left")
    axes[i].set_yticks([])
plt.subplots_adjust(hspace=0)
fig.suptitle("Line horizon chart — overlap minimal", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.5 Line multiple overlap
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
for i, cut in enumerate(cut_order):
    sub = diamonds[diamonds["cut"] == cut].groupby("carat_bin_mid", observed=True)["price"].mean()
    ax.plot(sub.index, sub.values, color=PALETTE[i % len(PALETTE)], label=cut, lw=2)
ax.legend()
ax.set_title("Line multiple overlap — toutes les cuts")
ax.set_xlabel("carat"); ax.set_ylabel("price moyen")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.6 Line multiple overlap area
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
for i, cut in enumerate(cut_order):
    sub = diamonds[diamonds["cut"] == cut].groupby("carat_bin_mid", observed=True)["price"].mean()
    ax.plot(sub.index, sub.values, color=PALETTE[i % len(PALETTE)], label=cut, lw=1.5)
    ax.fill_between(sub.index, 0, sub.values, color=PALETTE[i % len(PALETTE)], alpha=0.2)
ax.legend()
ax.set_title("Line overlap area")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.7 Line stacked amount
<!-- #endregion -->

```python
pivot_line = diamonds.groupby(["carat_bin_mid", "cut"], observed=True)["price"].mean().unstack(fill_value=0)[cut_order]
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(pivot_line))
for i, cut in enumerate(cut_order):
    ax.fill_between(pivot_line.index, bottom, bottom + pivot_line[cut].values,
                    color=PALETTE[i % len(PALETTE)], label=cut, alpha=0.8)
    bottom += pivot_line[cut].values
ax.legend()
ax.set_title("Line stacked amount")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.8 Stream graph (stacked centré sur 0)
<!-- #endregion -->

<!-- #region -->
Variation du stacked où chaque bande est centrée verticalement sur 0. Esthétique souvent associée au journalisme de données (NYT).
<!-- #endregion -->

```python
totals = pivot_line.sum(axis=1)
fig, ax = plt.subplots(figsize=(11, 5))
bottom = -totals.values / 2
for i, cut in enumerate(cut_order):
    ax.fill_between(pivot_line.index, bottom, bottom + pivot_line[cut].values,
                    color=PALETTE[i % len(PALETTE)], label=cut)
    bottom += pivot_line[cut].values
ax.legend(loc="lower left")
ax.set_yticks([])
for s in ["top", "right", "left", "bottom"]:
    ax.spines[s].set_visible(False)
ax.set_title("Stream graph — stacked centré sur 0")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Synthèse line plots** : 1 série → `single` · 2-4 séries → `overlap` ou `subplots` · 5+ séries → `horizon` ou `stream` pour ne pas saturer.
<!-- #endregion -->

<!-- #region -->
## 12. Patterns temporels et multi-séries (synthétique)
<!-- #endregion -->

<!-- #region -->
Pour les viz demandant **temps + état discret** (capteurs, machines), on génère une data synthétique courte au lieu de dépendre d'un dataset externe.
<!-- #endregion -->

```python
rng = np.random.RandomState(0)
n = 100
ts = pd.date_range("2026-01-01", periods=n, freq="h")
df_sensors = pd.DataFrame({
    "ts": ts,
    "sensor_a": np.sin(np.linspace(0, 6, n)) * 10 + rng.normal(0, 1, n) + 20,
    "sensor_b": np.cos(np.linspace(0, 4, n)) * 5 + rng.normal(0, 0.5, n) + 10,
    "sensor_c": np.linspace(0, 5, n) + rng.normal(0, 0.5, n),
    "state": rng.choice(["idle", "run", "fault"], size=n, p=[0.3, 0.6, 0.1]),
})
df_sensors.head()
```

<!-- #region -->
### 12.1 Multi-capteurs subplots
<!-- #endregion -->

```python
fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
for i, sens in enumerate(["sensor_a", "sensor_b", "sensor_c"]):
    axes[i].plot(df_sensors["ts"], df_sensors[sens],
                  color=PALETTE[i % len(PALETTE)], label=sens, lw=1.5)
    axes[i].legend(loc="upper left")
    axes[i].set_ylabel(sens)
axes[-1].set_xlabel("timestamp")
fig.suptitle("Capteurs — multi-subplots", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 12.2 Line plot événements discrets (catégoriel → ordinal)
<!-- #endregion -->

<!-- #region -->
Pour visualiser un état discret au cours du temps : mapper la catégorie → entier ordonné, puis ligne + coloration de fond par zone d'état.
<!-- #endregion -->

```python
state_order = ["idle", "run", "fault"]
state_map = {s: i for i, s in enumerate(state_order)}
state_colors = [CHART["neutral"], CHART["primary_1"], CHART["accent_warn"]]
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.plot(df_sensors["ts"], df_sensors["state"].map(state_map),
        color=CHART["accent_dark"], lw=1.5)
for i in range(len(df_sensors) - 1):
    ax.axvspan(df_sensors["ts"].iloc[i], df_sensors["ts"].iloc[i + 1],
               color=state_colors[state_map[df_sensors["state"].iloc[i]]], alpha=0.15)
ax.set_yticks(range(len(state_order)))
ax.set_yticklabels(state_order)
ax.set_title("État de la machine au cours du temps")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Pattern timeline d'événements** : utile pour log de capteurs, sessions utilisateur, états machines. La couleur de fond accentue la lecture (gris=idle, teal=run, rose=fault).
<!-- #endregion -->

<!-- #region -->
## 13. Visualisations expertes — multi-catégoriel
<!-- #endregion -->

<!-- #region -->
Dataset synthétique avec **5 dimensions catégorielles** (location/product/payment/gender/age). Permet de tester les viz dédiées aux cat × cat × cat × ...
<!-- #endregion -->

```python
n_synth = 1000
df_synth = pd.DataFrame({
    "location": rng.choice(["New York", "Boston", "Other"], size=n_synth),
    "product":  rng.choice(["food", "beverage"], size=n_synth),
    "payment":  rng.choice(["cash", "credit card"], size=n_synth),
    "gender":   rng.choice(["male", "female"], size=n_synth),
    "age":      rng.choice(["<25", "25-50", ">=50"], size=n_synth),
})
df_m = df_synth.groupby(list(df_synth.columns)).size().reset_index(name="freq")
df_m.head()
```

<!-- #region -->
### 13.1 Sunburst (plotly) — hiérarchie radiale
<!-- #endregion -->

```python
fig = px.sunburst(df_m, path=["location", "product", "payment", "gender", "age"],
                   values="freq", color="freq", color_continuous_scale="RdBu_r",
                   width=700, height=550,
                   title="Sunburst — location -> product -> payment -> gender -> age")
fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
fig.show()
```

<!-- #region -->
**Sunburst** : hiérarchie centrée. Lecture par anneau concentrique (du général au spécifique). Idéal quand on a une vraie hiérarchie naturelle.
<!-- #endregion -->

<!-- #region -->
### 13.2 Treemap (plotly)
<!-- #endregion -->

```python
fig = px.treemap(df_m, path=[px.Constant("all"), "location", "product",
                              "payment", "gender", "age"],
                  values="freq", color="freq", color_continuous_scale="viridis",
                  width=900, height=550,
                  title="Treemap — meme hierarchie en rectangulaire")
fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
fig.show()
```

<!-- #region -->
**Treemap** : même info que sunburst mais en **rectangulaire** — optimal pour de grandes hiérarchies (plus d'espace utilisable).
<!-- #endregion -->

<!-- #region -->
### 13.3 Multi-heatmap (grid 3×4)
<!-- #endregion -->

<!-- #region -->
Pattern **small multiples** appliqué aux heatmaps. Une heatmap par croisement de 3 facettes (location × product × gender).
<!-- #endregion -->

```python
import itertools
locations = df_synth["location"].unique()
products = df_synth["product"].unique()
genders = df_synth["gender"].unique()
combos = list(itertools.product(locations, products, genders))
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
for idx, (loc, prod, gen) in enumerate(combos[:rows * cols]):
    r, c = divmod(idx, cols)
    sub = df_m[(df_m["location"] == loc) & (df_m["product"] == prod)
               & (df_m["gender"] == gen)]
    ct = pd.crosstab(sub["payment"], sub["age"], values=sub["freq"], aggfunc="mean")
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="RdBu_r", cbar=False, ax=axes[r, c])
    axes[r, c].set_title(f"{loc} | {prod} | {gen}", fontsize=9)
    axes[r, c].set_ylabel(""); axes[r, c].set_xlabel("")
fig.suptitle("Multi-heatmap grid", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 13.4 Multi-barplot (grid 3×4)
<!-- #endregion -->

```python
fig, axes = plt.subplots(rows, cols, figsize=(16, 10), sharey=True)
for idx, (loc, prod, gen) in enumerate(combos[:rows * cols]):
    r, c = divmod(idx, cols)
    sub = df_m[(df_m["location"] == loc) & (df_m["product"] == prod)
               & (df_m["gender"] == gen)]
    sns.barplot(data=sub, x="age", y="freq", hue="payment",
                palette=[CHART["primary_1"], CHART["primary_2"]], ax=axes[r, c])
    axes[r, c].set_title(f"{loc} | {prod} | {gen}", fontsize=9)
    axes[r, c].set_xlabel(""); axes[r, c].set_ylabel("")
    leg = axes[r, c].get_legend()
    if leg is not None:
        leg.remove()
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99))
fig.suptitle("Multi-barplot grid", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 13.5 Stacked bar (grid 3×4)
<!-- #endregion -->

```python
fig, axes = plt.subplots(rows, cols, figsize=(16, 10), sharey=True)
for idx, (loc, prod, gen) in enumerate(combos[:rows * cols]):
    r, c = divmod(idx, cols)
    sub = df_m[(df_m["location"] == loc) & (df_m["product"] == prod)
               & (df_m["gender"] == gen)]
    ct = pd.crosstab(sub["payment"], sub["age"], values=sub["freq"], aggfunc="mean").fillna(0)
    ct.T.plot(kind="bar", stacked=True,
              color=[CHART["primary_1"], CHART["primary_2"]],
              ax=axes[r, c], legend=False)
    axes[r, c].set_title(f"{loc} | {prod} | {gen}", fontsize=9)
    axes[r, c].set_xlabel(""); axes[r, c].set_ylabel("")
    axes[r, c].tick_params(axis="x", labelrotation=0)
fig.suptitle("Stacked bar grid — payment x age", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 13.6 Parallel categories (plotly)
<!-- #endregion -->

<!-- #region -->
Visualise les **flux entre modalités** sur plusieurs dimensions. Chaque ligne = un flux observé, épaisseur ∝ fréquence.
<!-- #endregion -->

```python
fig = px.parallel_categories(df_m,
                              dimensions=["location", "product", "payment", "gender", "age"],
                              color="freq", color_continuous_scale="Tealrose",
                              title="Parallel categories")
fig.show()
```

<!-- #region -->
### 13.7 Mosaic plot (statsmodels)
<!-- #endregion -->

<!-- #region -->
Approximation des **fréquences conjointes par aires rectangulaires emboîtées**. Chaque rectangle a une surface ∝ fréquence du croisement. Hiérarchie lue de gauche à droite.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 8))
mosaic(df_synth, ["location", "gender", "payment"], ax=ax,
       properties=lambda key: {"color": PALETTE[hash("".join(key)) % len(PALETTE)]})
ax.set_title("Mosaic — location x gender x payment")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Lecture** : les rectangles les plus grands = combinaisons les plus fréquentes. Permet d'identifier visuellement les déséquilibres dans une table de contingence multi-dimensionnelle.
<!-- #endregion -->

<!-- #region -->
## 14. Bonnes pratiques et anti-patterns
<!-- #endregion -->

<!-- #region -->
### 14.1 Bonnes pratiques
<!-- #endregion -->

<!-- #region -->
- **Charte couleurs cohérente** sur tout le rapport (cf. section 2).
- **Légendes lisibles** : police ≥ 10pt, fond légèrement contrasté.
- **Axes labellisés** avec unités. Toujours.
- **Échelle 0-100%** pour les ratios (un stacked à 95% qu'on lit comme 100% trompe l'œil).
- **Couleurs d'accent réservées** aux highlights — pas de rouge partout.
- **Tri par fréquence** (sauf ordre métier explicite).
- **Small multiples** (subplots, FacetGrid) au lieu de superposer 6+ séries.
<!-- #endregion -->

<!-- #region -->
### 14.2 Anti-patterns
<!-- #endregion -->

<!-- #region -->
- **3D inutile** sur des données 2D (matplotlib `projection='3d'` pour épater) — illisible.
- **Pie chart > 5 modalités** — l'œil ne discrimine pas les angles.
- **Axes tronqués** qui exagèrent les différences (commencer à 95% au lieu de 0).
- **Couleurs aléatoires** ou cmap arc-en-ciel (`jet`) — pas perceptuellement uniforme.
- **Rouge + vert seul** — 8% des hommes daltoniens, ils ne distinguent pas.
- **Grilles épaisses ou trop denses** — bruit visuel inutile (Tufte : data-ink ratio).
<!-- #endregion -->

<!-- #region -->
### 14.3 Outils modernes alternatifs (2026)
<!-- #endregion -->

<!-- #region -->
- **[Plotly](https://plotly.com/python/)** + **[Dash](https://dash.plotly.com/)** : interactif, dashboards web.
- **[Altair](https://altair-viz.github.io/)** : approche déclarative basée sur Vega-Lite.
- **[Datashader](https://datashader.org/)** : pour très gros volumes (millions de points).
- **[hvplot](https://hvplot.holoviz.org/)** : interface haut niveau sur holoviews/bokeh, intégration pandas.
- **[plotnine](https://plotnine.org/)** : ggplot2 pythonisé pour les habitués de R.
<!-- #endregion -->

<!-- #region -->
## 15. Sources
<!-- #endregion -->

<!-- #region -->
- **Tufte, E.** (1983). *The Visual Display of Quantitative Information*. Concept fondateur du data-ink ratio.
- **Cleveland & McGill** (1984). *Graphical Perception*. Hiérarchie perceptive des encodages.
- **Munzner, T.** (2014). *Visualization Analysis and Design*. Référence académique moderne.
- **[seaborn documentation](https://seaborn.pydata.org/)** — API gallery exhaustive.
- **[matplotlib gallery](https://matplotlib.org/stable/gallery/index.html)** — exemples par catégorie.
- **[plotly express documentation](https://plotly.com/python/plotly-express/)**.
- **[Color Brewer 2](https://colorbrewer2.org/)** — palettes scientifiquement validées.
- **Notebooks frères du projet** :
  - `Structures_Preprocessing.ipynb` (preprocessing avancé)
  - `TS_Time_Series_Intro.ipynb` (séries temporelles, autocorrélation, lag)
  - `EDA_Analyse_Multivarie.ipynb` (PCA, CA, MCA, FAMD)
  - `EDA_Stats_Analyse_Desc_Visual.ipynb` (statistiques descriptives)
<!-- #endregion -->
