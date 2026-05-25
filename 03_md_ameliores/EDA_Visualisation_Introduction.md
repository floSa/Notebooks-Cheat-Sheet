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
# 🎨 Visualisation — Introduction
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** sur les **briques de visualisation** en Python pour l'EDA.

Couvre les **3 libs de base** :

1. **Matplotlib** — bas niveau, contrôle total, le couteau suisse.
2. **Seaborn** — surcouche statistique, syntaxe rapide.
3. **Plotly** — interactif, dashboards.

Pour chaque type de graphe (distribution, comparaison, corrélation, série temporelle), un exemple court par lib pour pouvoir copier-coller.

> Pour l'**EDA statistique** (descriptifs, distributions, corrélations), voir `EDA_Stats_Analyse_Desc_Visual`.
> Pour l'**analyse multivariée** (PCA, CA, MCA, FAMD), voir `EDA_Analyse_Multivarie`.
> Pour la **détection d'outliers**, voir `Detection_Outliers`.

Dataset : **Titanic** (mutualisé avec presque tous les notebooks ML).
<!-- #endregion -->

<!-- #region -->
## 1. Données : Titanic
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Titanic — disponible nativement dans seaborn (rapide, offline-ready)
df = sns.load_dataset("titanic")
print(df.head())
print(df.dtypes)
```

<!-- #region -->
## 2. Matplotlib — les bases
<!-- #endregion -->

<!-- #region -->
Matplotlib est la lib de référence. Toutes les autres (seaborn, pandas plotting, plotnine) s'appuient dessus. À maîtriser au minimum :

- `plt.subplots(nrows, ncols)` — figure + axes.
- Méthodes des axes : `ax.plot`, `ax.bar`, `ax.hist`, `ax.scatter`, `ax.boxplot`.
- Décoration : `set_title`, `set_xlabel`, `set_ylabel`, `legend`, `grid`, `tight_layout`.
- `savefig("name.png", dpi=150, bbox_inches='tight')` pour exporter propre.
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["age"].dropna(), bins=30, edgecolor="black")
axes[0].set_title("Histogramme — age")
axes[0].set_xlabel("age"); axes[0].set_ylabel("count")

axes[1].bar(df["class"].value_counts().index, df["class"].value_counts().values)
axes[1].set_title("Bar — class")

axes[2].scatter(df["age"], df["fare"], alpha=0.4)
axes[2].set_title("Scatter — age vs fare")
axes[2].set_xlabel("age"); axes[2].set_ylabel("fare")

plt.tight_layout()
```

<!-- #region -->
## 3. Seaborn — la surcouche statistique
<!-- #endregion -->

<!-- #region -->
Seaborn ajoute :

- Une **API par fonction** pour chaque type de graphe statistique (`histplot`, `kdeplot`, `boxplot`, `violinplot`, `heatmap`, ...).
- Une intégration native avec **pandas** (`x="col", y="col", hue="col", data=df`).
- Des **thèmes** par défaut esthétiques (`sns.set_theme()`).
- Des **figures multi-panneaux** statistiques avec `FacetGrid` / `pairplot`.

C'est le go-to pour 80 % des graphes EDA.
<!-- #endregion -->

```python
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df, x="age", hue="survived", kde=True, ax=axes[0])
axes[0].set_title("Distribution age par survie")

sns.boxplot(df, x="class", y="fare", hue="sex", ax=axes[1])
axes[1].set_title("Fare par classe et sexe")

sns.scatterplot(df, x="age", y="fare", hue="survived", alpha=0.6, ax=axes[2])
axes[2].set_title("Age vs fare colorié par survie")

plt.tight_layout()
```

<!-- #region -->
### 3.1 Catalogue rapide seaborn
<!-- #endregion -->

<!-- #region -->
| Fonction | Type de graphe | Quand l'utiliser |
|---|---|---|
| `histplot` | Histogramme | Distribution univariée |
| `kdeplot` | Densité (KDE) | Distribution univariée lissée |
| `displot` | Histogramme + KDE multi-facettes | Comparer distributions par groupe |
| `boxplot` / `violinplot` | Boîte à moustaches / Violon | Distribution par catégorie |
| `stripplot` / `swarmplot` | Points individuels | Petites tailles d'échantillon |
| `barplot` / `countplot` | Bars avec IC / counts | Comparer moyennes ou fréquences |
| `scatterplot` / `regplot` | Nuage de points | Relation 2 variables continues |
| `pairplot` | Matrice scatter+hist | Vue d'ensemble multivariée |
| `heatmap` | Matrice colorée | Corrélations, confusion matrix |
| `relplot` / `catplot` | Facets multi-panel | Comparer subgroups |
| `lineplot` | Lignes avec IC | Séries temporelles, courbes |
| `jointplot` | Scatter + marginales | Distribution conjointe |
<!-- #endregion -->

```python
# Pairplot : vue d'ensemble en 1 ligne
g = sns.pairplot(
    df[["age", "fare", "pclass", "survived"]].dropna(),
    hue="survived", diag_kind="kde", height=2.5,
)
g.fig.suptitle("Pairplot Titanic", y=1.02)
```

<!-- #region -->
## 4. Heatmap de corrélation
<!-- #endregion -->

```python
num_cols = df.select_dtypes(include="number").columns
corr = df[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Corrélations variables numériques")
plt.tight_layout()
```

<!-- #region -->
## 5. Plotly — interactif
<!-- #endregion -->

<!-- #region -->
Pour des graphes **interactifs** (zoom, hover, panel, dashboard), **Plotly** est le go-to en 2026. Deux APIs :

- **Plotly Express** (`px.scatter`, `px.histogram`, ...) — équivalent seaborn, syntaxe ultra-courte.
- **Plotly Graph Objects** (`go.Figure`, `go.Scatter`, ...) — bas niveau, contrôle total.

Pour notebooks Jupyter, les figures sont affichées inline en HTML/JS. Pour des apps web : **Dash** (Plotly) ou **Streamlit** (avec `st.plotly_chart`).

```python
# Décommenter pour exécuter (nécessite plotly installé)
"""
import plotly.express as px

fig = px.scatter(df, x="age", y="fare", color="survived", facet_col="class",
                 hover_data=["sex", "embarked"], title="Titanic — interactive")
fig.show()
"""
```
<!-- #endregion -->

<!-- #region -->
## 6. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 6.1 Choisir la bonne lib
<!-- #endregion -->

<!-- #region -->
| Besoin | Lib |
|---|---|
| Exploration rapide en notebook | **Seaborn** (1 ligne par graphe) |
| Contrôle pixel-parfait pour publication | **Matplotlib** |
| Dashboard interactif / hover | **Plotly** |
| Style "ggplot" / grammaire des graphiques | **plotnine** |
| Très grosse data (millions de points) | **Datashader** ou **Plotly + WebGL** |
| Graphes statistiques avancés | **seaborn objects API** (`sns.objects.Plot`, depuis seaborn 0.12) |
<!-- #endregion -->

<!-- #region -->
### 6.2 Conseils visuels
<!-- #endregion -->

<!-- #region -->
- **Aspect ratio** : ne pas étirer arbitrairement (un graphe 16:9 cache souvent des patterns).
- **Couleur** : utiliser des palettes **perceptuellement uniformes** (`viridis`, `cividis`) — pas `jet`. Penser daltoniens.
- **Échelle log** quand les données s'étendent sur plusieurs ordres de grandeur.
- **Annotation directe** > légende (quand peu de séries).
- **Toujours labelliser** les axes et donner un titre.
- **DPI 150+** pour exports.
- **Légende hors plot** quand elle est grosse : `legend(bbox_to_anchor=(1.05, 1))`.
<!-- #endregion -->

<!-- #region -->
### 6.3 Pièges classiques
<!-- #endregion -->

<!-- #region -->
- ❌ Pie chart à plus de 4-5 catégories (toujours préférer bar).
- ❌ Axes tronqués artificiellement (zoom Y qui exagère un effet).
- ❌ Trop de panels dans un `pairplot` au-delà de 8-10 colonnes → illisible.
- ❌ Pas de `tight_layout()` → labels qui se chevauchent.
- ❌ Réutiliser `plt.plot()` sur la même figure → l'ancien graphe traîne. Faire `plt.figure()` neuf à chaque cellule.
<!-- #endregion -->

<!-- #region -->
## 7. Sources
<!-- #endregion -->

<!-- #region -->
- [Matplotlib gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn gallery](https://seaborn.pydata.org/examples/)
- [Plotly Python](https://plotly.com/python/)
- [Seaborn objects API (depuis 0.12)](https://seaborn.pydata.org/tutorial/objects_interface.html)
- [From Data to Viz — choisir le bon graphe](https://www.data-to-viz.com/)
- Notebooks liés : `EDA_Stats_Analyse_Desc_Visual`, `EDA_Analyse_Multivarie`, `Detection_Outliers`.
<!-- #endregion -->
