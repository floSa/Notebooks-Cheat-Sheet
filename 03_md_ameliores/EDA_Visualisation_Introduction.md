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
# 📊 Visualisation pour l'EDA — Cheat-sheet & Tutoriel
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Cheat-sheet** : référence visuelle complète pour préparer un rapport ou explorer un dataset.

**Particularité** : une **charte graphique unique** (8 couleurs nommées sémantiquement + 2 helpers) appliquée **partout** — démonstration d'une bonne pratique cruciale pour des rapports professionnels.

**Datasets utilisés** :

- `sns.load_dataset('titanic')` — utilisé partout (univariée → bivariée → multivariée → patterns bar/line avancés).
- Datasets synthétiques inline pour les patterns temporels (`df_sensors`) et multi-catégoriels (`df_synth`).

**Aucune dépendance externe** (pas de Kaggle, pas de google.colab, pas de fichier à télécharger).

**Renvois** :

- Preprocessing complet (1-hot, scalers, transformations) → `Structures_Preprocessing.ipynb`.
- Auto-corrélation, lag plots, séries temporelles complètes → `TS_Time_Series_Intro.ipynb`.
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
Versions attendues : `pandas >= 2.2`, `seaborn >= 0.13`. `plotly` est utilisé pour les visualisations interactives en section 13.
<!-- #endregion -->

<!-- #region -->
## 2. Charte graphique unique
<!-- #endregion -->

<!-- #region -->
**C'est LA bonne pratique fondamentale** de ce notebook. Pourquoi définir et respecter une charte couleurs ?

- **Lisibilité cross-figures** : un lecteur retrouve instantanément "la classe principale est toujours teal", "le mauvais est toujours rouge", sans relire chaque légende.
- **Identité visuelle du rapport** : 4 à 8 couleurs maximum, cohérent avec un branding éventuel.
- **Accessibilité daltonien (~8 % hommes)** : éviter les couples rouge/vert seuls. Privilégier teal/orange ou bleu/orange (sûrs pour la grande majorité).
- **Sémantique encodée dans le nom** : on n'utilise pas `primary_2` (cryptique), on utilise `mauvais` (parle). Le code se lit comme un texte.
- **Hiérarchie visuelle** : les couleurs d'**accent** (sage = `accent`, navy = `accent_dark`) sont réservées aux **highlights** (valeur max, classe d'intérêt). Les couleurs **claires** (beige, lavender, dusty_rose) habillent les séries secondaires sans attirer l'œil.

Référence essentielle : *The Visual Display of Quantitative Information* (Tufte, 1983) — concept de **data-ink ratio**.
<!-- #endregion -->

<!-- #region -->
### 2.1 Définition de la palette
<!-- #endregion -->

<!-- #region -->
Palette de **8 couleurs sémantiques** :

| Nom | Hex | Rôle sémantique |
|---|---|---|
| `primary_1`   | `#00798c` | Teal — info / catégorie principale |
| `mauvais`     | `#d1495b` | Crimson — bad / nul / critique |
| `moyen`       | `#edae49` | Saffron — moyen / warning |
| `accent`      | `#66a182` | Sage — accent / bon / highlight |
| `accent_dark` | `#2e4057` | Navy — texte fort, valeur max highlight |
| `lavender`    | `#9d83b8` | Violet pas trop foncé — catégorie secondaire |
| `dusty_rose`  | `#b8848e` | Rose terne — catégorie secondaire |
| `beige`       | `#c9b78b` | Beige — neutre / background / séries secondaires |
<!-- #endregion -->

```python
CHART: dict[str, str] = {
    "primary_1":   "#00798c",  # Teal     — info / catégorie principale
    "mauvais":     "#d1495b",  # Crimson  — bad / nul / critique
    "moyen":       "#edae49",  # Saffron  — moyen / warning
    "accent":      "#66a182",  # Sage     — accent / bon / highlight
    "accent_dark": "#2e4057",  # Navy     — texte fort, valeur max highlight
    "lavender":    "#9d83b8",  # Violet pas trop foncé — secondaire 1
    "dusty_rose":  "#b8848e",  # Rose terne            — secondaire 2
    "beige":       "#c9b78b",  # Beige                 — neutre / background
}
PALETTE: list[str] = list(CHART.values())

# Swatch visuel
fig, ax = plt.subplots(figsize=(11, 1.8))
for i, (name, hex_) in enumerate(CHART.items()):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=hex_))
    ax.text(i + 0.5, -0.35, name, ha="center", va="top", fontsize=8)
    txt_color = "white" if i in (0, 1, 3, 4) else "black"
    ax.text(i + 0.5, 0.5, hex_, ha="center", va="center", fontsize=9,
            color=txt_color, weight="bold")
ax.set_xlim(0, len(CHART)); ax.set_ylim(-0.6, 1)
ax.set_xticks([]); ax.set_yticks([])
for s in ["top", "right", "bottom", "left"]:
    ax.spines[s].set_visible(False)
ax.set_title("Charte graphique — 8 couleurs", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()
```

<!-- #region -->
Le swatch ci-dessus est la **référence visuelle** de toutes les figures du notebook. Les noms sont sémantiques : `mauvais` pour critique, `moyen` pour warning, `accent` pour highlight positif.
<!-- #endregion -->

<!-- #region -->
### 2.2 Helpers — `apply_chart_style()` et `scatter_by_category()`
<!-- #endregion -->

<!-- #region -->
Deux helpers pour **ne jamais répéter** le code de styling :

- `apply_chart_style()` — appliqué une fois en début de notebook, configure `seaborn` + `matplotlib` pour utiliser la charte par défaut partout.
- `scatter_by_category(df, x, y, hue, style=None)` — scatter encodant 2 dimensions catégorielles via couleur **et** marker. **Produit 2 légendes séparées** (une pour les couleurs, une pour les markers) — bien plus lisibles qu'une légende combinée du type "Normal — False, Normal — True, Fault — False...".
<!-- #endregion -->

```python
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


def scatter_by_category(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    style: str | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    hue_title: str | None = None,
    style_title: str | None = None,
) -> plt.Axes:
    """Scatter encodant 2 dimensions catégorielles via couleur ET marker.

    Produit DEUX légendes séparées (couleur + marker), bien plus lisibles
    qu'une légende combinée. La 1ère légende est conservée via `ax.add_artist`.

    Args:
        df: DataFrame source.
        x, y: colonnes numériques pour les axes.
        hue: colonne catégorielle encodée par couleur (mapping sur PALETTE).
        style: colonne catégorielle encodée par marker. Optionnel.
        ax: Axes existant. Si None, en crée un.
        title: titre du plot.
        hue_title, style_title: titres des deux légendes (par défaut = nom des colonnes).

    Returns:
        l'axe matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    hue_levels = list(df[hue].unique())
    color_map = {lvl: PALETTE[i % len(PALETTE)] for i, lvl in enumerate(hue_levels)}
    markers_list = ["o", "s", "^", "D", "v", "*", "P", "X"]

    if style is None:
        for lvl in hue_levels:
            d = df[df[hue] == lvl]
            ax.scatter(d[x], d[y], color=color_map[lvl], label=str(lvl),
                       alpha=0.75, edgecolor="white", s=80, linewidth=1)
        ax.legend(title=hue_title or hue, loc="best", frameon=True)
    else:
        style_levels = list(df[style].unique())
        marker_map = {lvl: markers_list[i % len(markers_list)] for i, lvl in enumerate(style_levels)}
        for h in hue_levels:
            for sty in style_levels:
                d = df[(df[hue] == h) & (df[style] == sty)]
                if len(d) == 0:
                    continue
                ax.scatter(d[x], d[y], color=color_map[h], marker=marker_map[sty],
                           alpha=0.75, edgecolor="white", s=90, linewidth=1)

        # 2 légendes séparées via proxy artists (Line2D)
        hue_handles = [plt.Line2D([], [], color=color_map[lvl], marker="s",
                                   linestyle="", markersize=10, label=str(lvl))
                       for lvl in hue_levels]
        leg1 = ax.legend(handles=hue_handles, title=hue_title or hue,
                          loc="upper left", frameon=True)
        ax.add_artist(leg1)  # CRUCIAL : sans ça la 2e légende écrase la 1ère

        style_handles = [plt.Line2D([], [], color="black", marker=marker_map[lvl],
                                     linestyle="", markersize=10, label=str(lvl))
                          for lvl in style_levels]
        ax.legend(handles=style_handles, title=style_title or style,
                  loc="upper right", frameon=True)

    ax.set_xlabel(x); ax.set_ylabel(y)
    if title:
        ax.set_title(title)
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

Colonnes principales utilisées dans ce notebook :

- `survived` (0/1) — target binaire.
- `pclass` (1/2/3) — classe du billet.
- `sex` (male/female), `age`, `fare`, `sibsp`, `parch`.
- `embarked` (C/Q/S) — port d'embarquement.
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
**Lecture** : `age` 20% manquant (à imputer), `deck` 77% manquant (à drop). Pattern classique d'EDA.
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
Le grand classique. Variante `seaborn.histplot` avec KDE superposée. Choix du nombre de bins critique (règle de Freedman-Diaconis ou Sturges en pratique).
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
`primary_1` (teal) = couleur de la distribution principale.
<!-- #endregion -->

<!-- #region -->
### 4.2 Boxplot et boxenplot
<!-- #endregion -->

<!-- #region -->
Le **boxplot** affiche médiane, quartiles, moustaches (1.5×IQR), outliers. Le **boxenplot** ajoute des quantiles supplémentaires — utile pour distributions à queues lourdes (ex : `fare`).

Couleur `mauvais` car `fare` Titanic est très déséquilibrée (queue droite = "anomalie tarifaire").
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(y=df["fare"], color=CHART["mauvais"], ax=axes[0])
axes[0].set_title("sns.boxplot — fare")
sns.boxenplot(y=df["fare"], color=CHART["mauvais"], ax=axes[1])
axes[1].set_title("sns.boxenplot — fare (queues lourdes)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Le boxenplot révèle la **structure des outliers** de manière plus informative qu'un boxplot classique.
<!-- #endregion -->

<!-- #region -->
### 4.3 KDE et rugplot
<!-- #endregion -->

<!-- #region -->
La **KDE** est l'alternative continue à l'histogramme (moins biaisée par le choix des bins). Le **rugplot** ajoute un tick par observation.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(df["fare"].dropna(), fill=True, color=CHART["moyen"], ax=ax, alpha=0.5)
sns.rugplot(df["fare"].dropna(), color=CHART["accent_dark"], ax=ax, height=0.05)
ax.set_title("KDE + rugplot — fare")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Pour petits échantillons (n < 50), préférer hist + rug ; pour grands, KDE seul suffit.
<!-- #endregion -->

<!-- #region -->
## 5. Univariée — variables qualitatives
<!-- #endregion -->

<!-- #region -->
Pour 1 catégorielle : **comptage par modalité**. Toujours **trier par fréquence** sauf si un ordre métier existe. Pattern utile : highlight de la modalité modale en couleur primaire, les autres en `beige` (neutre).
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order_emb = df["embarked"].value_counts().index.tolist()
sns.countplot(x="embarked", data=df, order=order_emb, ax=axes[0],
              palette=[CHART["primary_1"], CHART["beige"], CHART["beige"]])
axes[0].set_title("countplot — embarked (modale en primary_1, autres en beige)")
order_pcl = df["pclass"].value_counts().index.tolist()
sns.countplot(x="pclass", data=df, order=order_pcl, ax=axes[1],
              palette=[CHART["primary_1"], CHART["beige"], CHART["beige"]])
axes[1].set_title("countplot — pclass")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Pattern **highlight modale** : barre principale en `primary_1`, les autres en `beige` neutre. L'œil va directement à l'info principale.
<!-- #endregion -->

<!-- #region -->
## 6. Bivariée — quanti × quanti
<!-- #endregion -->

<!-- #region -->
### 6.1 Scatter et regplot
<!-- #endregion -->

<!-- #region -->
`scatter` montre les paires de points. `regplot` ajoute une régression linéaire + IC. Pour non-linéarités : `regplot(lowess=True)`. La droite de régression est en `mauvais` pour bien ressortir sur le scatter.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.scatterplot(data=df, x="age", y="fare", color=CHART["primary_1"], alpha=0.5, ax=axes[0])
axes[0].set_title("scatter — age vs fare")
sns.regplot(data=df, x="age", y="fare",
            scatter_kws={"color": CHART["primary_1"], "alpha": 0.3},
            line_kws={"color": CHART["mauvais"]}, ax=axes[1])
axes[1].set_title("regplot — fit linéaire")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 6.2 Jointplot — scatter + marginales
<!-- #endregion -->

<!-- #region -->
Le `jointplot` combine scatter (ou hex / kde) avec les **distributions marginales** de chaque axe.
<!-- #endregion -->

```python
g = sns.jointplot(data=df, x="age", y="fare", kind="hex", color=CHART["primary_1"], height=5)
g.fig.suptitle("jointplot kind='hex'", y=1.02)
plt.show()
```

<!-- #region -->
- `kind='hex'` : gros volumes (densité en hexagones).
- `kind='kde'` : densité 2D continue.
- `kind='scatter'` : petits volumes (par défaut).
<!-- #endregion -->

<!-- #region -->
### 6.3 Heatmap des corrélations
<!-- #endregion -->

<!-- #region -->
**Cmap divergente** (`RdBu_r`) centrée sur 0 : rouge = corr positive, bleu = négative. Toujours annoter.
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
À investiguer : |corr| > 0.5. Ici corr(pclass, fare) ≈ -0.55 (1ère classe = fare élevé).
<!-- #endregion -->

<!-- #region -->
## 7. Bivariée — quanti × quali
<!-- #endregion -->

<!-- #region -->
**6 alternatives principales** côte à côte sur `sex × age` : boxplot, violinplot, swarmplot, boxenplot, pointplot, barplot.
<!-- #endregion -->

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
two_palette = [CHART["primary_1"], CHART["mauvais"]]
sns.boxplot(data=df, x="sex", y="age", palette=two_palette, ax=axes[0, 0]);    axes[0, 0].set_title("boxplot")
sns.violinplot(data=df, x="sex", y="age", palette=two_palette, ax=axes[0, 1]); axes[0, 1].set_title("violinplot")
sns.swarmplot(data=df.sample(min(300, len(df)), random_state=0), x="sex", y="age",
              palette=two_palette, size=3, ax=axes[0, 2]); axes[0, 2].set_title("swarmplot (n=300)")
sns.boxenplot(data=df, x="sex", y="age", palette=two_palette, ax=axes[1, 0]); axes[1, 0].set_title("boxenplot")
sns.pointplot(data=df, x="sex", y="age", color=CHART["primary_1"], errorbar=("ci", 95), ax=axes[1, 1])
axes[1, 1].set_title("pointplot (CI 95%)")
sns.barplot(data=df, x="sex", y="age", palette=two_palette, errorbar=None, ax=axes[1, 2])
axes[1, 2].set_title("barplot (mean)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Quand choisir quoi** :

- **boxplot** : résumé compact (quartiles + outliers).
- **violinplot** : forme exacte de la distribution.
- **swarmplot** : tous les points (n < 500).
- **boxenplot** : distributions à queues lourdes.
- **pointplot** : moyenne + intervalle de confiance.
- **barplot** : moyenne seule — **à éviter en exploratoire** (cache la dispersion).
<!-- #endregion -->

<!-- #region -->
### 7.1 FacetGrid — un subplot par modalité
<!-- #endregion -->

<!-- #region -->
Pattern **small multiples** : même viz répétée pour chaque valeur d'une catégorielle.
<!-- #endregion -->

```python
g = sns.FacetGrid(df, col="sex", height=4)
g.map_dataframe(sns.histplot, x="age", color=CHART["primary_1"], bins=20)
g.fig.suptitle("FacetGrid — histplot age par sexe", y=1.05)
plt.show()
```

<!-- #region -->
## 8. Bivariée — quali × quali
<!-- #endregion -->

<!-- #region -->
- **countplot avec `hue`** : effectifs absolus côte à côte.
- **pivot_table + heatmap** : ratios ou comptes en vue compacte.

Combinés pour `sex × pclass` : effectifs + taux de survie par croisement.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.countplot(data=df, x="sex", hue="pclass",
              palette=[CHART["primary_1"], CHART["moyen"], CHART["mauvais"]], ax=axes[0])
axes[0].set_title("countplot — sex × pclass (effectifs)")
pivot = df.pivot_table(index="sex", columns="pclass", values="survived", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
            linewidths=0.5, linecolor="white", ax=axes[1])
axes[1].set_title("Taux de survie — sex × pclass")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Lecture** : femmes 1ère classe → 97% survie, hommes 3ème → 14%. La cmap divergente centrée sur 0.5 met en évidence les contrastes.

Encodage couleur : `primary_1` (1ère = principale), `moyen` (2ème = moyen), `mauvais` (3ème = catégorie la moins favorable historiquement).
<!-- #endregion -->

<!-- #region -->
## 9. Multivariée — 3 dimensions ou plus
<!-- #endregion -->

<!-- #region -->
Encoder 3+ dims via : `hue` (couleur), `style` (marker), `size` (taille), `col`/`row` (facettes). Limite cognitive : 4-5 dims max simultanément lisibles.
<!-- #endregion -->

<!-- #region -->
### 9.1 Pairplot
<!-- #endregion -->

<!-- #region -->
Matrice de scatters bivariés + distributions univariées en diagonale.
<!-- #endregion -->

```python
g = sns.pairplot(df[["age", "fare", "sibsp", "survived"]].dropna(),
                 hue="survived", palette=[CHART["mauvais"], CHART["primary_1"]], height=2.2)
g.fig.suptitle("pairplot — age/fare/sibsp colorés par survived", y=1.02)
plt.show()
```

<!-- #region -->
`mauvais` = mort, `primary_1` = survivant. L'**encodage sémantique** parle : rouge = "mauvaise issue" pour le passager.
<!-- #endregion -->

<!-- #region -->
### 9.2 `sns.relplot` — encodage multidim direct
<!-- #endregion -->

<!-- #region -->
La fonction reine de seaborn pour les multivariées : x, y, col, hue, style en 1 appel.
<!-- #endregion -->

```python
g = sns.relplot(data=df, x="age", y="fare", col="pclass", hue="sex", style="sex",
                kind="scatter", palette=[CHART["primary_1"], CHART["mauvais"]],
                height=4, alpha=0.6)
g.fig.suptitle("relplot — age vs fare, facetté par pclass, coloré par sex", y=1.03)
plt.show()
```

<!-- #region -->
### 9.3 Scatter encodage couleur + marker (helper avec 2 légendes)
<!-- #endregion -->

<!-- #region -->
Pattern critique en **monitoring industriel** : 2 dimensions catégorielles encodées simultanément. Le helper `scatter_by_category` produit **2 légendes distinctes** — couleur ET marker chacune dans sa boîte, infiniment plus lisible qu'une légende combinée.
<!-- #endregion -->

```python
# Exemple 1 — Titanic : sex (couleur) × pclass (marker)
fig, ax = plt.subplots(figsize=(9, 5))
df_sample = df.dropna(subset=["age", "fare"]).sample(min(200, len(df)), random_state=0)
scatter_by_category(df_sample, x="age", y="fare", hue="sex", style="pclass", ax=ax,
                    title="Titanic — age vs fare (couleur=sex, marker=pclass)",
                    hue_title="Sex", style_title="Pclass")
plt.tight_layout(); plt.show()
```

```python
# Exemple 2 — synthétique capteurs (format préservé de l'original)
data = [[1.0, 2.0, "Normal", False],
        [1.0, 3.0, "Normal", True],
        [1.0, 2.0, "Normal", False],
        [2.0, 1.0, "Fault", True],
        [2.0, 2.0, "Maintenance", False],
        [3.0, 1.0, "Normal", True],
        [3.0, 3.0, "Normal", False],
        [3.0, 2.0, "Fault", False]]
df_machines = pd.DataFrame(data, columns=["sensor_00", "sensor_01", "machine_status", "outliers"])
fig, ax = plt.subplots(figsize=(8, 5))
scatter_by_category(df_machines, x="sensor_00", y="sensor_01",
                    hue="machine_status", style="outliers", ax=ax,
                    title="Capteurs — status (couleur) × outlier (marker)",
                    hue_title="machine_status", style_title="outlier")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Note pédagogique** : un appel naïf `ax.legend()` deux fois ne marche pas — le 2e écrase le 1er. La technique correcte est `ax.add_artist(leg1)` après avoir créé la 1ère légende, puis créer la 2e normalement. Ce que fait `scatter_by_category` ci-dessus.
<!-- #endregion -->

<!-- #region -->
## 10. Patterns avancés — barplots
<!-- #endregion -->

<!-- #region -->
On reste sur **Titanic** : 5 patterns appliqués au croisement `pclass × embarked` (3 × 3 modalités, mais on commencera par `pclass` seul pour le pattern highlight).
<!-- #endregion -->

```python
df_clean = df.dropna(subset=["embarked"])
pclass_order = [1, 2, 3]
emb_order = ["S", "C", "Q"]
pivot_pe = df_clean.groupby(["pclass", "embarked"], observed=True).size().unstack(fill_value=0)[emb_order]
pivot_pe = pivot_pe.loc[pclass_order]
print("Pivot pclass x embarked :")
print(pivot_pe)
```

<!-- #region -->
### 10.1 Bar single — highlight modalité max
<!-- #endregion -->

<!-- #region -->
Pattern **highlight** : valeur maximale en `accent_dark`, les autres en `primary_1`. Force le focus.
<!-- #endregion -->

```python
counts_pcl = df_clean["pclass"].value_counts().sort_index()
colors = [CHART["accent_dark"] if v == counts_pcl.max() else CHART["primary_1"]
          for v in counts_pcl.values]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(counts_pcl.index.astype(str), counts_pcl.values, color=colors)
ax.set_title("Effectifs Titanic par pclass — max en accent_dark")
ax.set_xlabel("pclass"); ax.set_ylabel("count")
plt.tight_layout(); plt.show()
```

<!-- #region -->
La 3ème classe (la plus peuplée) ressort instantanément en navy.
<!-- #endregion -->

<!-- #region -->
### 10.2 Bar multiple subplots (1 par catégorie)
<!-- #endregion -->

<!-- #region -->
**Small multiples** : une sous-figure par modalité. Mêmes axes y → comparaison directe.

Encodage couleur sémantique : `primary_1` pour la 1ère classe, `moyen` pour la 2ème, `mauvais` pour la 3ème.
<!-- #endregion -->

```python
fig, axes = plt.subplots(len(pclass_order), 1, figsize=(11, 7), sharex=True)
subplot_colors = [CHART["primary_1"], CHART["moyen"], CHART["mauvais"]]
for i, pc in enumerate(pclass_order):
    sub = df_clean[df_clean["pclass"] == pc]["embarked"].value_counts().reindex(emb_order, fill_value=0)
    axes[i].bar(sub.index, sub.values, color=subplot_colors[i], label=f"pclass {pc}")
    axes[i].legend(loc="upper right")
    axes[i].set_ylabel("count")
axes[-1].set_xlabel("embarked")
fig.suptitle("Bar subplots — count(embarked) par pclass", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 10.3 Bar multiple overlap (alpha)
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
for i, pc in enumerate(pclass_order):
    sub = df_clean[df_clean["pclass"] == pc]["embarked"].value_counts().reindex(emb_order, fill_value=0)
    ax.bar(sub.index, sub.values, color=subplot_colors[i],
           label=f"pclass {pc}", alpha=0.45, edgecolor=subplot_colors[i])
ax.legend()
ax.set_title("Bar overlap (alpha=0.45)")
ax.set_xlabel("embarked"); ax.set_ylabel("count")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Limite : illisible au-delà de 4-5 séries. Avec 3 séries comme ici, encore lisible.
<!-- #endregion -->

<!-- #region -->
### 10.4 Bar stacked amount (effectifs cumulés)
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(emb_order))
for i, pc in enumerate(pclass_order):
    vals = pivot_pe.loc[pc].values
    ax.bar(emb_order, vals, bottom=bottom, color=subplot_colors[i], label=f"pclass {pc}")
    bottom += vals
ax.legend()
ax.set_title("Bar stacked amount — pclass × embarked (effectifs cumulés)")
ax.set_xlabel("embarked"); ax.set_ylabel("count")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Somme verticale = total par embarked. Lecture : contributions absolues.
<!-- #endregion -->

<!-- #region -->
### 10.5 Bar stacked ratio (normalisé en %)
<!-- #endregion -->

```python
pivot_pct = pivot_pe.div(pivot_pe.sum(axis=0), axis=1) * 100
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(emb_order))
for i, pc in enumerate(pclass_order):
    vals = pivot_pct.loc[pc].values
    ax.bar(emb_order, vals, bottom=bottom, color=subplot_colors[i], label=f"pclass {pc}")
    bottom += vals
ax.legend()
ax.set_title("Bar stacked ratio — % par embarked")
ax.set_ylim(0, 100); ax.set_ylabel("% des passagers"); ax.set_xlabel("embarked")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Lecture : contributions **relatives**, met en évidence les **déséquilibres**. On voit que Cherbourg (`C`) a une proportion bien plus élevée de 1ère classe que Southampton (`S`) ou Queenstown (`Q`).
<!-- #endregion -->

<!-- #region -->
## 11. Patterns avancés — line plots
<!-- #endregion -->

<!-- #region -->
Pour les line plots non-temporels, on construit une **variable ordinale** : `age` binné en 10 buckets → `fare` moyen. La "ligne" suit l'évolution du fare selon l'âge.

8 variantes du plus simple au plus avancé.
<!-- #endregion -->

```python
df_age = df_clean.dropna(subset=["age"]).copy()
df_age["age_bin"] = pd.cut(df_age["age"], bins=10)
df_age["age_bin_mid"] = df_age["age_bin"].apply(lambda x: x.mid).astype(float)
fare_by_age = df_age.groupby("age_bin_mid", observed=True)["fare"].mean()
fare_by_age.head()
```

<!-- #region -->
### 11.1 Line single
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(fare_by_age.index, fare_by_age.values, color=CHART["accent_dark"], lw=2)
ax.set_title("Line single — fare moyen par tranche d'âge")
ax.set_xlabel("age (midpoint bin)"); ax.set_ylabel("fare moyen")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.2 Line single area (`fill_between`)
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(fare_by_age.index, fare_by_age.values, color=CHART["accent_dark"], lw=2)
ax.fill_between(fare_by_age.index, 0, fare_by_age.values, color=CHART["primary_1"], alpha=0.3)
ax.set_title("Line single area")
ax.set_xlabel("age"); ax.set_ylabel("fare moyen")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.3 Line multiple subplots (1 par pclass)
<!-- #endregion -->

```python
fig, axes = plt.subplots(len(pclass_order), 1, figsize=(12, 7), sharex=True)
for i, pc in enumerate(pclass_order):
    sub = df_age[df_age["pclass"] == pc].groupby("age_bin_mid", observed=True)["fare"].mean()
    axes[i].plot(sub.index, sub.values, color=subplot_colors[i], label=f"pclass {pc}", lw=2)
    axes[i].legend(loc="upper right")
    axes[i].set_ylabel("fare moyen")
axes[-1].set_xlabel("age")
fig.suptitle("Line subplots — fare moyen par âge × pclass", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
Les échelles y diffèrent — la 1ère classe paie ~10× plus que la 3ème. Pour comparer les **formes**, utiliser `sharey=True`.
<!-- #endregion -->

<!-- #region -->
### 11.4 Line horizon (subplots collés `hspace=0`)
<!-- #endregion -->

```python
fig, axes = plt.subplots(len(pclass_order), 1, figsize=(12, 5), sharex=True)
for i, pc in enumerate(pclass_order):
    sub = df_age[df_age["pclass"] == pc].groupby("age_bin_mid", observed=True)["fare"].mean()
    axes[i].fill_between(sub.index, 0, sub.values, color=subplot_colors[i], alpha=0.8, label=f"pclass {pc}")
    axes[i].legend(loc="upper right")
    axes[i].set_yticks([])
plt.subplots_adjust(hspace=0)
fig.suptitle("Line horizon chart — overlap minimal", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.5 Line overlap
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
for i, pc in enumerate(pclass_order):
    sub = df_age[df_age["pclass"] == pc].groupby("age_bin_mid", observed=True)["fare"].mean()
    ax.plot(sub.index, sub.values, color=subplot_colors[i], label=f"pclass {pc}", lw=2)
ax.legend()
ax.set_title("Line overlap — toutes les pclass sur le même axe")
ax.set_xlabel("age"); ax.set_ylabel("fare moyen")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.6 Line overlap area
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 5))
for i, pc in enumerate(pclass_order):
    sub = df_age[df_age["pclass"] == pc].groupby("age_bin_mid", observed=True)["fare"].mean()
    ax.plot(sub.index, sub.values, color=subplot_colors[i], label=f"pclass {pc}", lw=1.5)
    ax.fill_between(sub.index, 0, sub.values, color=subplot_colors[i], alpha=0.2)
ax.legend()
ax.set_title("Line overlap area")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.7 Line stacked amount
<!-- #endregion -->

```python
pivot_line = df_age.groupby(["age_bin_mid", "pclass"], observed=True)["fare"].mean().unstack(fill_value=0)[pclass_order]
fig, ax = plt.subplots(figsize=(11, 5))
bottom = np.zeros(len(pivot_line))
for i, pc in enumerate(pclass_order):
    ax.fill_between(pivot_line.index, bottom, bottom + pivot_line[pc].values,
                    color=subplot_colors[i], label=f"pclass {pc}", alpha=0.8)
    bottom += pivot_line[pc].values
ax.legend()
ax.set_title("Line stacked amount — fare cumulé par pclass × âge")
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 11.8 Stream graph (stacked centré sur 0)
<!-- #endregion -->

```python
totals = pivot_line.sum(axis=1)
fig, ax = plt.subplots(figsize=(11, 5))
bottom = -totals.values / 2
for i, pc in enumerate(pclass_order):
    ax.fill_between(pivot_line.index, bottom, bottom + pivot_line[pc].values,
                    color=subplot_colors[i], label=f"pclass {pc}")
    bottom += pivot_line[pc].values
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
Pour les viz **temps + état discret** (capteurs, machines), on génère un dataset synthétique court. 3 capteurs continus + 1 colonne `state` discrète sur 100 timestamps.
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

<!-- #region -->
Chaque capteur reçoit une couleur de la palette (ici `primary_1`, `lavender`, `dusty_rose` — choisies pour bonne distinction).
<!-- #endregion -->

```python
fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
sensor_colors = [CHART["primary_1"], CHART["lavender"], CHART["dusty_rose"]]
for i, sens in enumerate(["sensor_a", "sensor_b", "sensor_c"]):
    axes[i].plot(df_sensors["ts"], df_sensors[sens], color=sensor_colors[i], label=sens, lw=1.5)
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
Pour visualiser un **état discret au cours du temps** : mapper la catégorie → entier ordonné, puis ligne + coloration de fond par zone d'état.

Encodage sémantique du fond : `idle`=`beige` (neutre), `run`=`primary_1` (normal), `fault`=`mauvais` (alerte). Le lecteur identifie instantanément les périodes de panne.
<!-- #endregion -->

```python
state_order = ["idle", "run", "fault"]
state_map = {s: i for i, s in enumerate(state_order)}
state_colors = [CHART["beige"], CHART["primary_1"], CHART["mauvais"]]
fig, ax = plt.subplots(figsize=(13, 3.5))
ax.plot(df_sensors["ts"], df_sensors["state"].map(state_map),
        color=CHART["accent_dark"], lw=1.5)
for i in range(len(df_sensors) - 1):
    ax.axvspan(df_sensors["ts"].iloc[i], df_sensors["ts"].iloc[i + 1],
               color=state_colors[state_map[df_sensors["state"].iloc[i]]], alpha=0.18)
ax.set_yticks(range(len(state_order))); ax.set_yticklabels(state_order)
ax.set_title("État de la machine au cours du temps (idle=beige, run=teal, fault=mauvais)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Pattern timeline d'événements** : utile pour log de capteurs, sessions utilisateur, états machines.
<!-- #endregion -->

<!-- #region -->
## 13. Visualisations expertes — multi-catégoriel
<!-- #endregion -->

<!-- #region -->
Dataset synthétique avec **5 dimensions catégorielles** (location/product/payment/gender/age) pour les viz dédiées au cat × cat × cat × ...
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
### 13.1 Sunburst (plotly)
<!-- #endregion -->

```python
fig = px.sunburst(df_m, path=["location", "product", "payment", "gender", "age"], values="freq",
                   color="freq", color_continuous_scale="RdBu_r", width=700, height=550,
                   title="Sunburst")
fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
fig.show()
```

<!-- #region -->
### 13.2 Treemap (plotly)
<!-- #endregion -->

```python
fig = px.treemap(df_m, path=[px.Constant("all"), "location", "product", "payment", "gender", "age"],
                  values="freq", color="freq", color_continuous_scale="viridis",
                  width=900, height=550, title="Treemap")
fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
fig.show()
```

<!-- #region -->
### 13.3 Multi-heatmap (grid 3×4)
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
    sub = df_m[(df_m["location"] == loc) & (df_m["product"] == prod) & (df_m["gender"] == gen)]
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
    sub = df_m[(df_m["location"] == loc) & (df_m["product"] == prod) & (df_m["gender"] == gen)]
    sns.barplot(data=sub, x="age", y="freq", hue="payment",
                palette=[CHART["primary_1"], CHART["mauvais"]], ax=axes[r, c])
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
    sub = df_m[(df_m["location"] == loc) & (df_m["product"] == prod) & (df_m["gender"] == gen)]
    ct = pd.crosstab(sub["payment"], sub["age"], values=sub["freq"], aggfunc="mean").fillna(0)
    ct.T.plot(kind="bar", stacked=True, color=[CHART["primary_1"], CHART["mauvais"]],
               ax=axes[r, c], legend=False)
    axes[r, c].set_title(f"{loc} | {prod} | {gen}", fontsize=9)
    axes[r, c].set_xlabel(""); axes[r, c].set_ylabel("")
    axes[r, c].tick_params(axis="x", labelrotation=0)
fig.suptitle("Stacked bar grid", y=1.00)
plt.tight_layout(); plt.show()
```

<!-- #region -->
### 13.6 Parallel categories (plotly)
<!-- #endregion -->

```python
fig = px.parallel_categories(df_m, dimensions=["location", "product", "payment", "gender", "age"],
                              color="freq", color_continuous_scale="Tealrose",
                              title="Parallel categories")
fig.show()
```

<!-- #region -->
### 13.7 Mosaic plot (statsmodels)
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(11, 8))
mosaic(df_synth, ["location", "gender", "payment"], ax=ax,
       properties=lambda key: {"color": PALETTE[hash("".join(key)) % len(PALETTE)]})
ax.set_title("Mosaic — location x gender x payment")
plt.tight_layout(); plt.show()
```

<!-- #region -->
## 14. Bonnes pratiques et anti-patterns
<!-- #endregion -->

<!-- #region -->
### 14.1 Bonnes pratiques
<!-- #endregion -->

<!-- #region -->
- **Charte couleurs cohérente** sur tout le rapport (cf. section 2).
- **Noms sémantiques** dans le code (`mauvais` plutôt que `primary_2`).
- **Légendes lisibles** : police ≥ 10pt, fond contrasté, **2 légendes séparées** quand on encode 2 dimensions catégorielles via couleur + marker.
- **Axes labellisés** avec unités.
- **Échelle 0-100 %** pour les ratios.
- **Couleurs d'accent réservées** aux highlights — pas de `accent_dark` partout.
- **Tri par fréquence** (sauf ordre métier explicite).
- **Small multiples** (subplots, FacetGrid) au lieu de superposer 6+ séries.
<!-- #endregion -->

<!-- #region -->
### 14.2 Anti-patterns
<!-- #endregion -->

<!-- #region -->
- **3D inutile** sur des données 2D — illisible.
- **Pie chart > 5 modalités** — l'œil ne discrimine pas les angles.
- **Axes tronqués** qui exagèrent les différences.
- **Couleurs aléatoires** ou cmap arc-en-ciel (`jet`) — pas perceptuellement uniforme.
- **Rouge + vert seul** — 8 % des hommes daltoniens.
- **Grilles épaisses ou trop denses** — bruit visuel inutile (Tufte : data-ink ratio).
- **Légende combinée pour deux dimensions** type "Normal — False, Normal — True, Fault — False..." — préférer 2 légendes séparées.
<!-- #endregion -->

<!-- #region -->
### 14.3 Outils modernes alternatifs (2026)
<!-- #endregion -->

<!-- #region -->
- **[Plotly](https://plotly.com/python/) + [Dash](https://dash.plotly.com/)** : interactif, dashboards web.
- **[Altair](https://altair-viz.github.io/)** : approche déclarative basée sur Vega-Lite.
- **[Datashader](https://datashader.org/)** : pour très gros volumes (millions de points).
- **[hvplot](https://hvplot.holoviz.org/)** : interface haut niveau sur holoviews/bokeh, intégration pandas.
- **[plotnine](https://plotnine.org/)** : ggplot2 pythonisé.
<!-- #endregion -->

<!-- #region -->
## 15. Sources
<!-- #endregion -->

<!-- #region -->
- **Tufte, E.** (1983). *The Visual Display of Quantitative Information*. Concept du data-ink ratio.
- **Cleveland & McGill** (1984). *Graphical Perception*. Hiérarchie perceptive des encodages.
- **Munzner, T.** (2014). *Visualization Analysis and Design*. Référence académique moderne.
- **[seaborn documentation](https://seaborn.pydata.org/)** — API gallery exhaustive.
- **[matplotlib gallery](https://matplotlib.org/stable/gallery/index.html)**.
- **[plotly express documentation](https://plotly.com/python/plotly-express/)**.
- **[Color Brewer 2](https://colorbrewer2.org/)** — palettes scientifiquement validées.
- **Notebooks frères** :
  - `Structures_Preprocessing.ipynb` (preprocessing avancé)
  - `TS_Time_Series_Intro.ipynb` (séries temporelles complètes)
  - `EDA_Analyse_Multivarie.ipynb` (PCA, CA, MCA, FAMD)
  - `EDA_Stats_Analyse_Desc_Visual.ipynb` (statistiques descriptives)
<!-- #endregion -->
