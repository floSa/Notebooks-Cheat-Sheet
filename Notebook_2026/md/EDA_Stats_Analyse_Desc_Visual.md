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
# 📊 EDA — Statistiques descriptives & analyse visuelle
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Wiki** : catalogue de recettes de visualisation EDA — univariée numérique / catégorielle, bivariée croisée, table de contingence, corrélations, twin axes, encircling par enveloppe convexe, Pareto.

L'objectif est de pouvoir **copier-coller** une recette pour répondre à une question d'EDA précise, et d'avoir à côté de chaque graphique la **stat associée** (skew/kurtosis, chi²+Cramer's V, ANOVA, Kruskal-Wallis).

**Datasets** :

- `tips` (seaborn) — dataset principal, mix quanti / quali.
- `iris` (seaborn) — pour les pairplots multi-classes.
- DataFrame synthétique à 20 colonnes hétérogènes — pour montrer la stratégie *"regrouper par ordre de grandeur"*.

**Notebooks frères** : `EDA_Visualisation_Introduction` (intro plus pédagogique), `EDA_Analyse_Multivarie` (PCA / MCA / FAMD), `Detection_Outliers`.
<!-- #endregion -->

<!-- #region -->
## 0. Setup
<!-- #endregion -->

<!-- #region -->
On charge les libs, on définit la **palette CHART du projet** (8 hex, règle §9 du contrat), on règle `sns.set_theme(style="whitegrid", palette=PALETTE)`.
<!-- #endregion -->

```python
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.cm import ScalarMappable
from scipy import stats
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")

CHART: dict[str, str] = {
    "primary_1":   "#00798c",
    "mauvais":     "#d1495b",
    "moyen":       "#edae49",
    "accent":      "#66a182",
    "accent_dark": "#2e4057",
    "lavender":    "#9d83b8",
    "dusty_rose":  "#b8848e",
    "beige":       "#c9b78b",
}
PALETTE: list[str] = list(CHART.values())
sns.set_theme(style="whitegrid", palette=PALETTE)
```

<!-- #region -->
## 1. Dataset principal — `tips`
<!-- #endregion -->

<!-- #region -->
244 lignes, 7 colonnes : 4 qualitatives (`sex`, `smoker`, `day`, `time`), 2 quantitatives (`total_bill`, `tip`), 1 numérique discrète (`size`).
<!-- #endregion -->

```python
df_tips: pd.DataFrame = sns.load_dataset("tips")
print("shape :", df_tips.shape)
print("dtypes :")
print(df_tips.dtypes)
df_tips.head()
```

<!-- #region -->
## 2. Statistiques descriptives — univariées
<!-- #endregion -->

<!-- #region -->
### 2.1 Aperçu des types de variables
<!-- #endregion -->

<!-- #region -->
On visualise la répartition des `dtypes` (object / float / int / category) en **camembert + barplot côte à côte** — utile pour cadrer le travail à venir (combien de quali vs quanti, où mettre l'effort de preprocessing).

Règle couleur : multi-catégorie sans ordre → `PALETTE[:N]`.
<!-- #endregion -->

```python
type_counts = df_tips.dtypes.value_counts(normalize=True).reset_index()
type_counts.columns = ["type", "freq"]
type_counts["type"] = type_counts["type"].astype(str)
type_counts_g = type_counts.groupby("type", as_index=False)["freq"].sum()

n_types = len(type_counts_g)
couleurs = PALETTE[:n_types]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

# Sous-graphique 1 : camembert
ax1.pie(
    type_counts_g["freq"],
    labels=type_counts_g["type"],
    autopct="%1.1f%%",
    startangle=140,
    colors=couleurs,
)
ax1.axis("equal")
ax1.set_title("Camembert — % de chaque type de variable")

# Sous-graphique 2 : barplot annoté
for i, (t, f) in enumerate(zip(type_counts_g["type"], type_counts_g["freq"])):
    ax2.bar(i, f, color=couleurs[i], label=t)
    ax2.text(i, f, f"{f:.1%}", ha="center", va="bottom", fontsize=12)
ax2.set_xlabel("Type de variable")
ax2.set_ylabel("Fréquence")
ax2.set_title("Barplot — fréquence de chaque type")
ax2.set_xticks(range(n_types))
ax2.set_xticklabels(type_counts_g["type"])
ax2.legend()

plt.tight_layout()
plt.show()
```

<!-- #region -->
**Lecture** : sur `tips`, la majorité des colonnes sont `category` (4) + 2 `float` + 1 `int`. Bonne nouvelle : pas de typage à corriger en amont.
<!-- #endregion -->

<!-- #region -->
### 2.2 Variables qualitatives — `describe()`
<!-- #endregion -->

<!-- #region -->
`.describe(include=['object','category','bool'])` renvoie `count`, `unique`, `top` (mode), `freq` (effectif du mode). C'est le résumé minimal d'une colonne catégorielle.
<!-- #endregion -->

```python
df_quali = df_tips.select_dtypes(["object", "category", "bool"])
df_quali.describe()
```

<!-- #region -->
### 2.3 Variables quantitatives — `describe()` étendu
<!-- #endregion -->

<!-- #region -->
On enrichit le `.describe()` classique avec **skew**, **kurtosis** et **missing %** :

- **skew ≈ 0** : symétrique. **skew > 0** : queue à droite (revenus, montants). **skew < 0** : queue à gauche.
- **kurtosis** : excès vs normale. **> 0** : queues lourdes (outliers fréquents). **< 0** : plus aplatie.
- **missing_%** : indispensable pour décider imputation vs drop.
<!-- #endregion -->

```python
df_quanti = df_tips.select_dtypes(["float", "int64"])
desc = df_quanti.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
desc["skew"] = df_quanti.skew()
desc["kurtosis"] = df_quanti.kurt()
desc["missing_%"] = df_quanti.isna().mean() * 100
desc.round(2)
```

<!-- #region -->
### 2.4 Distribution des quanti — histogramme + KDE
<!-- #endregion -->

<!-- #region -->
1 subplot par variable quanti, `sns.histplot(..., kde=True, stat='density')` : on superpose l'**histogramme normalisé en densité** et le **KDE** (Kernel Density Estimate, lissage gaussien).

Couleur unique = `primary_1` (règle §9 : variable univariée → 1 seule couleur).
<!-- #endregion -->

```python
n_quanti = df_quanti.shape[1]
fig, axes = plt.subplots(ncols=n_quanti, nrows=1, figsize=(10 * n_quanti, 5))
if n_quanti == 1:
    axes = np.array([axes])

for col, ax in zip(df_quanti, axes.flat):
    sns.histplot(df_quanti[col], ax=ax, kde=True, stat="density",
                 linewidth=0, color=CHART["primary_1"])

plt.tight_layout()
plt.show()
```

<!-- #region -->
### 2.5 Distribution des quali — pies natifs
<!-- #endregion -->

<!-- #region -->
1 pie par variable quali avec **moins de 30 modalités** (au-delà, un pie devient illisible — préférer un barplot trié). Palette CHART.
<!-- #endregion -->

```python
ncols = max(df_quali.shape[1], 1)
fig, axes = plt.subplots(ncols=ncols, nrows=1, figsize=(6 * ncols, 6))
if ncols == 1:
    axes = np.array([axes])

for col, ax in zip(df_quali, axes.flat):
    if len(df_quali[col].value_counts()) < 30:
        n_mod = df_quali[col].nunique()
        df_quali[col].value_counts().plot.pie(
            ax=ax,
            autopct="%1.0f%%",
            colors=PALETTE[:n_mod],
        )
        ax.set_ylabel("")

plt.tight_layout()
plt.show()
```

<!-- #region -->
### 2.6 Distribution quanti sans outliers (par quantile)
<!-- #endregion -->

<!-- #region -->
Quand un outlier fait imploser l'échelle, on **coupe les `quantil`% extrêmes** de chaque côté avant de tracer. Réutilisable sur n'importe quel `df`.
<!-- #endregion -->

```python
def histo_quanti_sans_outliers(df: pd.DataFrame, quantil: float = 0.0) -> None:
    """Trace l'histogramme + KDE des variables quantitatives après retrait
    des `quantil`% extrêmes (gauche et droite).

    Args:
        df: DataFrame contenant au moins une colonne numérique.
        quantil: fraction à couper de chaque côté (0 = pas de coupe, 0.015 ≈ 1.5%).
    """
    df_quanti_local = df.select_dtypes(["float", "int64"])
    n = df_quanti_local.shape[1]
    fig, axes = plt.subplots(ncols=n, nrows=1, figsize=(8 * n, 5))
    if n == 1:
        axes = np.array([axes])
    for col, ax in zip(df_quanti_local, axes.flat):
        s = df_quanti_local[col].copy()
        s = s[s.between(s.quantile(quantil), s.quantile(1 - quantil))]
        sns.histplot(s, ax=ax, kde=True, stat="density",
                     linewidth=0, color=CHART["primary_1"])
        ax.set_xlabel(col)
    plt.tight_layout()
    plt.show()
```

<!-- #region -->
Appel : on coupe 1.5 % de chaque côté.
<!-- #endregion -->

```python
histo_quanti_sans_outliers(df_tips, quantil=0.015)
```

<!-- #region -->
### 2.7 Pies avec modalités rares regroupées
<!-- #endregion -->

<!-- #region -->
Pour les colonnes catégorielles à **long tail** (beaucoup de modalités à très faible fréquence), on regroupe toute modalité dont la fréquence relative est `< frek` dans une catégorie `other_(N)` (où `N` est le nombre de modalités absorbées). Print également les modalités rejetées pour traçabilité.
<!-- #endregion -->

```python
def pies_modalites_rares(df: pd.DataFrame, frek: float = 0.05) -> None:
    """Pour chaque colonne qualitative, regroupe les modalités de fréquence
    relative < `frek` dans "other_(N)" puis trace 1 pie par variable.

    Args:
        df: DataFrame ; les colonnes object/category/bool sont sélectionnées.
        frek: seuil de fréquence relative.
    """
    df_q = df.select_dtypes(["object", "category", "bool"])
    ncols = max(df_q.shape[1], 1)
    fig, axes = plt.subplots(ncols=ncols, nrows=1, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = np.array([axes])

    for col, ax in zip(df_q, axes.flat):
        ddf = pd.DataFrame({col: df_q[col].copy()})
        moda = ddf[col].unique()

        if ddf.dtypes[col] != "object":
            ddf = ddf.astype({col: "object"})

        ddf["freq"] = ddf.groupby([col])[col].transform("count") / ddf.shape[0]
        ddf[col] = ddf[col].where(ddf["freq"] > frek, "other")
        nb_absorbed = len(moda) - len(ddf[col].unique()) + 1
        autre = f"other_({nb_absorbed})"
        ddf[col] = ddf[col].replace("other", autre)
        n_mod = ddf[col].nunique()

        ddf[col].value_counts().plot(
            kind="pie", autopct="%1.0f%%",
            ax=ax, colors=PALETTE[:n_mod],
        )
        ax.set_ylabel("")
        label_removed = list(set(moda).symmetric_difference(set(ddf[col].unique())))
        if autre in label_removed:
            label_removed.remove(autre)
        if label_removed:
            print(f"{col} — modalités de freq < {frek} regroupées :",
                  label_removed[:5])

    plt.tight_layout()
    plt.show()
```

<!-- #region -->
Appel sur les quali de `tips` (rien à absorber ici puisque toutes les modalités passent 5 %).
<!-- #endregion -->

```python
pies_modalites_rares(df_quali, frek=0.05)
```

<!-- #region -->
### 2.8 Heatmap de corrélation
<!-- #endregion -->

<!-- #region -->
**Carte de fréquentation** : représentation graphique où l'intensité de couleur correspond à la valeur. Ici matrice de **Pearson** par défaut (linéaire) ; `RdBu_r` centré sur 0 (règle §9 : heatmap continue divergente).

`numeric_only=True` est obligatoire en pandas 2.x pour éviter l'erreur sur les colonnes non-numériques.
<!-- #endregion -->

```python
corr = df_tips.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap="RdBu_r",
    square=True,
    annot=True,
    fmt=".1g",
    ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.tight_layout()
plt.show()
```

<!-- #region -->
### 2.9 Cheat-sheet — `value_counts` avec / sans NaN
<!-- #endregion -->

<!-- #region -->
- `dropna=False` : compte les NaN comme une modalité (sinon ils sont silencieusement ignorés).
- `normalize=True` : renvoie les fréquences relatives au lieu des effectifs.
<!-- #endregion -->

```python
print(df_tips["size"].value_counts(dropna=False))
print()
print(df_tips["size"].value_counts(dropna=False, normalize=True))
```

<!-- #region -->
### 2.10 Cas — beaucoup de variables quanti hétérogènes
<!-- #endregion -->

<!-- #region -->
Quand on a 20+ colonnes numériques à **ordres de grandeur très différents** (capteurs, embeddings, mesures physiques mixtes), tout tracer sur un même axe écrase les petites variables. La stratégie : **regrouper les colonnes par ordre de grandeur similaire** (médiane × écart inter-quantile 0.1-0.9), puis 1 subplot par groupe.
<!-- #endregion -->

<!-- #region -->
**Étape 1** — générer un `df` synthétique de test avec des moyennes sur 6 ordres de grandeur (`10^Z`, `Z ∈ ±[1..3]`) et des écarts-types proportionnels.
<!-- #endregion -->

```python
def generate_dataframe(num_columns: int = 20, num_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Génère un DataFrame avec des colonnes normales d'ordres de grandeur variés.

    Args:
        num_columns: nombre de colonnes.
        num_rows: nombre de lignes.
        seed: graine reproductibilité.

    Returns:
        DataFrame de shape (num_rows, num_columns).
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    z_choices = np.array([-3, -2, -1, 1, 2, 3])
    for i in range(num_columns):
        z = int(rng.choice(z_choices))
        mean = float(rng.random()) * (10.0 ** z)
        sigma = 4.0 * abs(z)
        data[f"col_{i+1}"] = rng.normal(loc=mean, scale=sigma, size=num_rows)
    return pd.DataFrame(data)
```

<!-- #region -->
**Étape 2** — introduire des NaN aléatoires pour simuler des données réelles. Le bug du notebook d'origine (appel mort placé **après** le `return` dans la fonction) est corrigé : l'appel est maintenant fait à l'usage.
<!-- #endregion -->

```python
def introduce_missing_values(
    df: pd.DataFrame,
    min_frac: float = 0.05,
    max_frac: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Introduit aléatoirement des NaN dans chaque colonne.

    Args:
        df: DataFrame d'entrée.
        min_frac: taux min de NaN par colonne (0..1).
        max_frac: taux max de NaN par colonne (0..1).
        seed: graine reproductibilité.

    Returns:
        Copie du DataFrame avec NaN.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    for col in df.columns:
        frac = float(rng.uniform(min_frac, max_frac))
        n_missing = int(len(df) * frac)
        missing_idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan
    return df
```

<!-- #region -->
**Étape 3** — la fonction principale : trie les colonnes par médiane, regroupe celles dont les ratios médiane/q_diff sont compatibles, trace 1 lineplot par groupe.
<!-- #endregion -->

```python
def process_and_plot(df: pd.DataFrame, factor_threshold: int = 7) -> list[list[str]]:
    """Regroupe les colonnes numériques par ordre de grandeur similaire
    (médiane + écart inter-quantile 0.1/0.9) puis trace 1 lineplot par groupe.

    Args:
        df: DataFrame avec colonnes numériques.
        factor_threshold: taille max d'un groupe.

    Returns:
        Liste des groupes (chaque groupe = liste de noms de colonnes).
    """
    df_num = df.select_dtypes(include=np.number)

    stats_per_col: dict[str, dict[str, float]] = {}
    for col in df_num.columns:
        median = float(df_num[col].median())
        q1 = float(df_num[col].quantile(0.1))
        q9 = float(df_num[col].quantile(0.9))
        if not np.isfinite(median):
            continue
        stats_per_col[col] = {"median": median, "q_diff": abs(q9 - q1)}

    sorted_cols = sorted(stats_per_col.items(), key=lambda x: x[1]["median"])
    grouped: list[list[str]] = []

    while sorted_cols:
        current_group = [sorted_cols.pop(0)]
        current_medians = [current_group[0][1]["median"]]
        current_qdiffs = [current_group[0][1]["q_diff"]]

        i = 0
        while i < len(sorted_cols) and len(current_group) < factor_threshold:
            col, st = sorted_cols[i]
            med_c, diff_c = st["median"], st["q_diff"]
            med_g = float(np.mean(current_medians))
            diff_g = float(np.mean(current_qdiffs))

            denom_med = max(abs(med_c), abs(med_g)) or 1.0
            denom_q = max(abs(diff_c), abs(diff_g)) or 1.0
            ratio_med = min(abs(med_c), abs(med_g)) / denom_med
            ratio_q = min(abs(diff_c), abs(diff_g)) / denom_q

            if ratio_med < 0.1 or ratio_q < 0.1:
                i += 1
            else:
                current_group.append(sorted_cols.pop(i))
                current_medians.append(med_c)
                current_qdiffs.append(diff_c)

        grouped.append([c for c, _ in current_group])

    n_groups = len(grouped)
    fig, axes = plt.subplots(n_groups, 1, figsize=(15, 4 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]
    for i, group in enumerate(grouped):
        sns.lineplot(
            data=df_num[group],
            ax=axes[i], errorbar=None, dashes=False, markers=False,
            palette=PALETTE[:len(group)],
        )
        axes[i].set_title(f"Groupe {i+1} — variables : {', '.join(group)}", fontsize=12)
        axes[i].set_ylabel("Valeurs")
    axes[-1].set_xlabel("Index")
    plt.tight_layout()
    plt.show()
    return grouped
```

<!-- #region -->
**Étape 4** — démo end-to-end sur 20 colonnes synthétiques avec NaN modérés.
<!-- #endregion -->

```python
df_synth = generate_dataframe(num_columns=20, num_rows=100, seed=42)
df_synth = introduce_missing_values(df_synth, min_frac=0.05, max_frac=0.20, seed=42)
groups = process_and_plot(df_synth, factor_threshold=7)
print(f"{len(groups)} groupes formés")
```

<!-- #region -->
## 3. Statistiques bivariées
<!-- #endregion -->

<!-- #region -->
### 3.1 Qualitative × Qualitative
<!-- #endregion -->

<!-- #region -->
#### 3.1.a Boxplot groupé — pattern `melt`
<!-- #endregion -->

<!-- #region -->
Recette pandas/seaborn pour passer d'un DataFrame **wide** (1 colonne = 1 variable) à un **long** (1 colonne = nom variable, 1 colonne = valeur) avec `pd.melt`, puis l'utiliser dans un `sns.boxplot(x=..., y=..., hue=...)`.
<!-- #endregion -->

```python
rng = np.random.default_rng(0)
data1 = pd.DataFrame(rng.random((17, 3)), columns=["A", "B", "C"]).assign(Location=1)
data2 = pd.DataFrame(rng.random((17, 3)) + 0.2, columns=["A", "B", "C"]).assign(Location=2)
data3 = pd.DataFrame(rng.random((17, 3)) + 0.4, columns=["A", "B", "C"]).assign(Location=3)
cdf = pd.concat([data1, data2, data3])
print("cdf shape :", cdf.shape)
print(cdf.head(2))

mdf = pd.melt(cdf, id_vars=["Location"], var_name="Letter")
print("mdf shape :", mdf.shape)
print(mdf.head(2))

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="Location", y="value", hue="Letter", data=mdf,
            palette=PALETTE[:3], ax=ax)
plt.tight_layout()
plt.show()
```

<!-- #region -->
#### 3.1.b Table de contingence
<!-- #endregion -->

<!-- #region -->
3 variantes utiles de `pd.crosstab` :

1. `margins=True` : ajoute la somme par ligne / colonne.
2. `normalize=True` + `axis=1` : fréquences relatives **par ligne** (= profil de ligne).
3. `normalize='all'` × 100 : pourcentages **globaux** (somme à 100).
<!-- #endregion -->

```python
print("--- Effectifs avec marges ---")
print(pd.crosstab(df_tips["sex"], df_tips["time"], margins=True, dropna=True))

print("\n--- Profil par ligne (somme à 1 par ligne) ---")
print(pd.crosstab(df_tips["sex"], df_tips["time"], dropna=False)
      .apply(lambda r: r / r.sum(), axis=1))

print("\n--- % globaux (somme à 100) ---")
print(pd.crosstab(df_tips["sex"], df_tips["time"],
                  margins=True, dropna=False, normalize="all") * 100)
```

<!-- #region -->
#### 3.1.c Test du chi² + Cramer's V
<!-- #endregion -->

<!-- #region -->
Pour quantifier le lien entre deux variables catégorielles :

- **chi²** teste l'indépendance (H0). Si `p < 0.05` → on rejette → les variables sont **dépendantes**.
- **Cramer's V** mesure la **taille d'effet** (`0` = indépendance, `1` = association parfaite). Le chi² seul ne dit pas la force du lien, juste sa significativité. **Toujours regarder les deux.**

Formule : $V = \sqrt{\frac{\chi^2}{n \cdot (\min(r, k) - 1)}}$ où $n$ est l'effectif total, $r$ et $k$ sont les dimensions de la table.
<!-- #endregion -->

```python
ct = pd.crosstab(df_tips["sex"], df_tips["time"])
chi2_stat, p_value, dof, expected = stats.chi2_contingency(ct)
print(f"chi² stat = {chi2_stat:.3f}  dof = {dof}  p = {p_value:.3e}")
print(f"→ {'INDÉPENDANTES' if p_value > 0.05 else 'DÉPENDANTES (effet signif.)'}")


def cramers_v(confusion_matrix: np.ndarray) -> float:
    """Cramer's V — taille d'effet pour table de contingence cat×cat.

    Args:
        confusion_matrix: table de contingence 2D (sans marges).

    Returns:
        V ∈ [0, 1] : 0 = indépendance, 1 = association parfaite.
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


v = cramers_v(ct.values)
print(f"Cramer's V = {v:.3f}")
```

<!-- #region -->
#### 3.1.d Countplots avec annotations
<!-- #endregion -->

<!-- #region -->
Recette pour ajouter la valeur **au-dessus / sur** chaque barre d'un `sns.countplot`. On boucle sur `graph.patches` et on récupère `p.get_x()`, `p.get_width()`, `p.get_height()`.

Fonction réutilisable `draw()` ci-dessous, puis 2 variantes d'offset.
<!-- #endregion -->

```python
def draw(graph: plt.Axes) -> None:
    """Annote chaque barre d'un countplot/barplot avec sa hauteur,
    centrée horizontalement, juste au-dessus de la barre.

    Args:
        graph: l'`Axes` retourné par `sns.countplot` / `sns.barplot`.
    """
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x() + p.get_width() / 2.0,
                   height + 5, int(height), ha="center")
```

<!-- #region -->
**Style 1** — annotation positionnée **juste sous** le haut de la barre (`height - 6`). Plus discret, ne déborde pas du graphe.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
graph = sns.countplot(x="sex", hue="time", data=df_tips,
                      palette=PALETTE[:2], ax=ax)
for p in graph.patches:
    h = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2.0, h - 6, int(h), ha="center")
plt.show()
```

<!-- #region -->
**Style 2** — annotation **au-dessus** de la barre avec un offset `x = +0.2` et `y = +1`. Plus visible mais peut nécessiter d'augmenter `ylim`.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
g = sns.countplot(x="sex", hue="time", data=df_tips,
                  palette=PALETTE[:2], ax=ax)
for p in g.patches:
    h = round(p.get_height(), 2)
    g.text(p.get_x() + 0.2, h + 1, str(h), ha="center")
plt.show()
```

<!-- #region -->
#### 3.1.e Frequence plot (% par groupe)
<!-- #endregion -->

<!-- #region -->
Recette pour passer d'effectifs bruts à des **pourcentages par groupe** : `groupby(x)[y].value_counts(normalize=True)` → tableau long → `sns.barplot(y='percentage', hue=y)` avec annotations en %.
<!-- #endregion -->

```python
occupation_counts = (
    df_tips.groupby(["sex"])["time"]
    .value_counts(normalize=True)
    .rename("percentage")
    .mul(100)
    .reset_index()
    .sort_values("sex")
)

fig, ax = plt.subplots(figsize=(5, 5))
g = sns.barplot(x="sex", y="percentage", hue="time", data=occupation_counts,
                palette=PALETTE[:2], ax=ax)
g.set_ylim(0, 100)
for p in g.patches:
    txt = f"{round(p.get_height(), 2)}%"
    g.text(p.get_x() + 0.2, p.get_height() + 2, txt, ha="center")
plt.show()
```

<!-- #region -->
#### 3.1.f Autres ajustements possibles
<!-- #endregion -->

<!-- #region -->
4 façons alternatives d'obtenir un barplot de proportions. À garder dans la cheat-sheet — selon le contexte, l'une est plus expressive que l'autre.
<!-- #endregion -->

<!-- #region -->
**Variante 1** — style pipe pandas, `sns.barplot` direct (sans annotations).
<!-- #endregion -->

```python
x, y, hue = "sex", "proportion", "time"
plot_df = (
    df_tips[x]
    .groupby(df_tips[hue])
    .value_counts(normalize=True)
    .rename(y)
    .reset_index()
)
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=plot_df, x=x, y=y, hue=hue, palette=PALETTE[:2], ax=ax)
plt.show()
```

<!-- #region -->
**Variante 2** — `sns.catplot(kind='bar')`. Renvoie un `FacetGrid` ; pour annoter les barres on passe par `g.ax.patches`.
<!-- #endregion -->

```python
x_c, y_c = "sex", "time"
df_pct = df_tips.groupby(x_c)[y_c].value_counts(normalize=True).mul(100)
df_pct = df_pct.rename("percent").reset_index()

g = sns.catplot(x=x_c, y="percent", hue=y_c, kind="bar", data=df_pct,
                palette=PALETTE[:2], height=4, aspect=1.3)
g.ax.set_ylim(0, 100)
for p in g.ax.patches:
    txt = f"{round(p.get_height(), 2)}%"
    g.ax.text(p.get_x() + 0.2, p.get_height() + 2, txt, ha="center")
plt.show()
```

<!-- #region -->
**Variante 3** — `crosstab.plot(kind='bar')`. Le plus court syntaxiquement quand on a déjà une table de contingence.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(5, 5))
ct = pd.crosstab(df_tips["sex"], df_tips["time"], dropna=False).apply(
    lambda r: r / r.sum(), axis=1
)
ct.plot(kind="bar", ax=ax, color=PALETTE[:2])
for p in ax.patches:
    ax.annotate(
        f"{round(p.get_height() * 100, 2)}%",
        (p.get_x() + 0.02, p.get_height() + 0.005),
    )
plt.show()
```

<!-- #region -->
**Variante 4** — comptes simples pour vérifier la cohérence (sanity check).
<!-- #endregion -->

```python
print("Lunch :", (df_tips["time"].values == "Lunch").sum())
print("Male  :", (df_tips["sex"].values == "Male").sum())
```

<!-- #region -->
### 3.2 Quantitative × Quantitative
<!-- #endregion -->

<!-- #region -->
#### 3.2.a `lmplot` — OLS et LOWESS
<!-- #endregion -->

<!-- #region -->
`sns.lmplot` trace un **scatter + droite de régression OLS** et son IC à 95 %. Très pratique pour un coup d'œil rapide sur une relation linéaire.

- `ci=None` désactive l'enveloppe de confiance.
- `lowess=True` remplace l'OLS par une régression **non-paramétrique** (locally weighted) qui capture la non-linéarité.
<!-- #endregion -->

```python
g = sns.lmplot(x="total_bill", y="tip", data=df_tips, ci=None,
               height=5,
               scatter_kws={"color": CHART["primary_1"]},
               line_kws={"color": CHART["mauvais"]})
g.set_axis_labels("Total Bill", "Tip")
plt.show()
```

<!-- #region -->
Approche non-paramétrique LOWESS — plus robuste si la relation n'est pas linéaire.
<!-- #endregion -->

```python
g = sns.lmplot(x="total_bill", y="tip", data=df_tips, lowess=True,
               height=5,
               scatter_kws={"color": CHART["primary_1"]},
               line_kws={"color": CHART["mauvais"]})
g.set_axis_labels("Total Bill", "Tip")
plt.show()
```

<!-- #region -->
### 3.3 Qualitative(s) × Quantitative(s)
<!-- #endregion -->

<!-- #region -->
#### 3.3.a 2 quanti × 1 quali — `lmplot` avec `hue` / `col`
<!-- #endregion -->

<!-- #region -->
- `hue=` colore les points par modalité et trace **une droite par catégorie**.
- `col=` fait un **facet** : un sous-graphe séparé par modalité.
<!-- #endregion -->

```python
g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=df_tips,
               height=5, palette=PALETTE[:2])
g.set_axis_labels("Total Bill", "Tip")
plt.show()
```

<!-- #region -->
Facet `col` : 1 panneau par jour, droite indépendante.
<!-- #endregion -->

```python
g = sns.lmplot(x="total_bill", y="tip", hue="day", col="day", data=df_tips,
               height=4, palette=PALETTE[:4])
g.set_axis_labels("Total Bill", "Tip")
plt.show()
```

<!-- #region -->
#### 3.3.b Distributions par catégorie — swarm, box, combinaison
<!-- #endregion -->

<!-- #region -->
**Swarmplot** : chaque point reste visible (jitter ajusté pour ne pas se chevaucher). Bon pour des petits effectifs où on veut voir les valeurs individuelles. `hue=` ajoute une 2ᵉ dimension catégorielle.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(7, 5))
sns.swarmplot(data=df_tips, x="time", y="total_bill", hue="smoker",
              palette=PALETTE[:2], ax=ax)
plt.show()
```

<!-- #region -->
**Boxplot avec `whis=np.inf`** : les moustaches s'étendent au min/max → plus d'outliers signalés. Utile pour voir l'étendue brute sans masquer les valeurs extrêmes.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x="total_bill", y="day", data=df_tips, whis=np.inf,
            palette=PALETTE[:4], ax=ax)
plt.show()
```

<!-- #region -->
**Box + swarm superposés** : combine résumé statistique (box) et densité visuelle (swarm). Pattern classique en présentation de résultats biomédicaux.
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="total_bill", y="day", data=df_tips, whis=np.inf,
            palette=PALETTE[:4], ax=ax)
sns.swarmplot(x="total_bill", y="day", data=df_tips, color=".2",
              size=3, ax=ax)
plt.show()
```

<!-- #region -->
#### 3.3.c 3 quanti × 1 quali — `relplot` avec `size` encoding
<!-- #endregion -->

<!-- #region -->
`relplot` permet jusqu'à **4 dimensions visuelles** : `x`, `y`, `hue` (couleur) et `size` (taille des points). Pratique pour explorer un jeu multivarié sans matrice complète.

Ici on trace `sepal_length × petal_width`, couleur par espèce, taille proportionnelle à `sepal_width`. `sizes=(40, 400)` contrôle la plage de taille en pixels.
<!-- #endregion -->

```python
iris = sns.load_dataset("iris")
g = sns.relplot(
    x="sepal_length", y="petal_width",
    hue="species", size="sepal_width",
    sizes=(40, 400), alpha=0.5,
    palette=PALETTE[:3],
    height=6,
    data=iris,
)
plt.show()
```

<!-- #region -->
#### 3.3.d N quanti × 1 quali — `pairplot`
<!-- #endregion -->

<!-- #region -->
`pairplot` trace **toutes les combinaisons 2 à 2** des variables numériques. Argument `hue=` ajoute la couleur par catégorie ; `diag_kind=` contrôle le graphe diagonal (`hist`, `kde`, `auto`).
<!-- #endregion -->

```python
g = sns.pairplot(df_tips, x_vars=["total_bill", "size"], y_vars=["tip"],
                 hue="day", height=4, aspect=0.8, kind="reg",
                 palette=PALETTE[:4])
plt.show()
```

<!-- #region -->
Sur `iris` avec `diag_kind="hist"`.
<!-- #endregion -->

```python
g = sns.pairplot(iris, hue="species", diag_kind="hist", palette=PALETTE[:3])
plt.show()
```

<!-- #region -->
Variante : markers custom par classe (`"o"`, `"s"`, `"D"`). Utile pour print N&B.
<!-- #endregion -->

```python
g = sns.pairplot(iris, hue="species", markers=["o", "s", "D"],
                 palette=PALETTE[:3])
plt.show()
```

<!-- #region -->
#### 3.3.e Countplot avec double axe Y (twinx)
<!-- #endregion -->

<!-- #region -->
Pattern matplotlib avancé pour afficher **count à droite** et **fréquence relative à gauche** sur le même graphe. On utilise :

- `ax.twinx()` pour créer un 2ᵉ axe Y partageant le même X.
- `matplotlib.ticker.LinearLocator(11)` pour 11 ticks équidistants sur l'axe count.
- `matplotlib.ticker.MultipleLocator(10)` pour des ticks tous les 10 % sur l'axe freq.
- `ax2.grid(False)` pour ne pas superposer 2 grilles.

Palette `rocket_r` conservée (gradient sur **ordre numérique** = OK, règle §9 ne s'applique pas aux échelles numériques).
<!-- #endregion -->

```python
rng = np.random.default_rng(42)
dfWIM = pd.DataFrame({"AXLES": rng.normal(8, 2, 5000).astype(int)})
ncount = len(dfWIM)
order = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x="AXLES", data=dfWIM, order=order,
              hue="AXLES", hue_order=order, palette="rocket_r",
              legend=False, ax=ax)
ax.set_title("Distribution of Truck Configurations")
ax.set_xlabel("Number of Axles")

ax2 = ax.twinx()
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax2.yaxis.set_label_position("left")
ax2.set_ylabel("Frequency [%]")
ax.set_ylabel("Count")

for p in ax.patches:
    x_pts = p.get_bbox().get_points()[:, 0]
    y_pt = p.get_bbox().get_points()[1, 1]
    if y_pt > 0:
        ax.annotate(f"{100.0 * y_pt / ncount:.1f}%",
                    (x_pts.mean(), y_pt), ha="center", va="bottom")

ax.yaxis.set_major_locator(ticker.LinearLocator(11))
ax2.set_ylim(0, 100)
ax.set_ylim(0, ncount)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.grid(False)
plt.show()
```

<!-- #region -->
**Variante 2** — `ylim` automatique à `value_counts().max() × 1.1` (zoom plus serré sur les barres, l'axe freq est recalculé en conséquence).
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x="AXLES", data=dfWIM, order=order,
              hue="AXLES", hue_order=order, palette="rocket_r",
              legend=False, ax=ax)
ax.set_title("Distribution of Truck Configurations (ylim auto)")
ax.set_xlabel("Number of Axles")

ax2 = ax.twinx()
ax.yaxis.tick_right()
ax2.yaxis.tick_left()
ax.yaxis.set_label_position("right")
ax2.yaxis.set_label_position("left")
ax2.set_ylabel("Frequency [%]")
ax.set_ylabel("Count")

max_count = int(dfWIM["AXLES"].value_counts().max())
ax.yaxis.set_major_locator(ticker.LinearLocator(11))
ax.set_ylim(0, max_count * 1.1)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.set_ylim(0, (max_count * 1.1 / dfWIM["AXLES"].value_counts().sum()) * 100)

for p in ax.patches:
    x_pts = p.get_bbox().get_points()[:, 0]
    y_pt = p.get_bbox().get_points()[1, 1]
    if y_pt > 0:
        ax.annotate(f"{100.0 * y_pt / ncount:.1f}%",
                    (x_pts.mean(), y_pt), ha="center", va="bottom")
ax2.grid(False)
plt.show()
```

<!-- #region -->
#### 3.3.f Barplot avec intensité de couleur (`ScalarMappable`)
<!-- #endregion -->

<!-- #region -->
Pattern matplotlib pur pour **encoder une 3ᵉ variable continue** (la fréquence) en **intensité de couleur** des barres. On utilise :

- `plt.Normalize(min, max)` pour scaler la variable continue dans [0, 1].
- `plt.cm.<cmap>(norm(values))` pour récupérer les couleurs.
- `ScalarMappable + colorbar` pour ajouter une légende graduée.

Cmap `Blues` (gradient continu single-color) gardé tel quel — c'est un cas où la règle "1 couleur uniforme" du contrat ne s'applique pas (la couleur **encode** une autre variable).
<!-- #endregion -->

```python
data_demo = {
    "titre": ["A", "B", "C", "D", "E"],
    "count": [10, 15, 8, 12, 20],
    "frequence": [0.10, 0.30, 0.20, 0.15, 0.25],
}
df_demo = pd.DataFrame(data_demo).sort_values(by="count", ascending=False)

fig, ax = plt.subplots(figsize=(7, 5))
norm = plt.Normalize(0, 0.4)
colors = plt.cm.Blues(norm(df_demo["frequence"].values))
rects = ax.bar(df_demo["titre"], df_demo["count"], color=colors)

ax.set_xlabel("Titre")
ax.set_ylabel("Count")
ax.set_title("Barplot — count, intensité ∝ fréquence")

for i, rect in enumerate(rects):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, h - 1,
            f"{h:.0f}", ha="center", va="bottom")

sm = ScalarMappable(cmap=plt.cm.Blues, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_ylabel("Fréquence")
plt.show()
```

<!-- #region -->
**Variante 2** — au lieu d'afficher le count sur la barre, on affiche directement la **fréquence**. Utile quand la 3ᵉ variable est l'information principale et le count secondaire.
<!-- #endregion -->

```python
data_demo2 = {
    "Lettre": ["A", "B", "C", "D", "E"],
    "Count": [10, 15, 8, 12, 20],
    "Frequence": [0.10, 0.30, 0.20, 0.15, 0.25],
}
df_demo2 = pd.DataFrame(data_demo2).sort_values(by="Count", ascending=False)

fig, ax = plt.subplots(figsize=(7, 5))
norm = plt.Normalize(0, 0.4)
colors = plt.cm.Blues(norm(df_demo2["Frequence"].values))
rects = ax.bar(df_demo2["Lettre"], df_demo2["Count"], color=colors)

ax.set_xlabel("Lettre")
ax.set_ylabel("Count")
ax.set_title("Barplot — count, label = fréquence")

for i, rect in enumerate(rects):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, h - 1,
            f'{df_demo2["Frequence"].iloc[i]:.2f}', ha="center", va="bottom")

sm = ScalarMappable(cmap=plt.cm.Blues, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.ax.set_ylabel("Fréquence")
plt.show()
```

<!-- #region -->
#### 3.3.g Scatterplot custom par catégorie (boucle)
<!-- #endregion -->

<!-- #region -->
Pattern matplotlib pur (sans seaborn) : 1 `ax.scatter` par catégorie dans une boucle, couleurs récupérées depuis `PALETTE`. Permet **plus de contrôle** que `sns.scatterplot` (taille, marker, edgecolor par catégorie séparément).

Sanity check d'abord : on vérifie que les colonnes sont bien là.
<!-- #endregion -->

```python
df_tips.columns
```

<!-- #region -->
Scatter total_bill × tip, 1 série de points par jour.
<!-- #endregion -->

```python
categories = np.unique(df_tips["day"])
n_cat = len(categories)
colors = [PALETTE[i % len(PALETTE)] for i in range(n_cat)]

fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor="w", edgecolor="k")
for i, category in enumerate(categories):
    sub = df_tips.loc[df_tips.day == category, :]
    ax.scatter("total_bill", "tip", data=sub, s=20,
               c=colors[i], label=str(category))

ax.set_xlabel("total_bill", fontdict={"fontsize": 10})
ax.set_ylabel("tip", fontdict={"fontsize": 10})
ax.set_title("Scatterplot — total_bill vs tip (par jour)", fontsize=12)
ax.legend(fontsize=10)
plt.show()
```

<!-- #region -->
#### 3.3.h Scatterplot avec ConvexHull (encircling)
<!-- #endregion -->

<!-- #region -->
On trace l'**enveloppe convexe** de chaque groupe (le plus petit polygone englobant tous les points) pour montrer visuellement leur extension. Utilise `scipy.spatial.ConvexHull` et `matplotlib.patches.Polygon`.

Bon pour **2-3 classes seulement** : au-delà, les enveloppes se superposent et le graphe devient illisible. Ici on combine 2 enveloppes par groupe : une avec alpha (remplissage flou) et une avec linestyle pointillé pour bien marquer le contour.
<!-- #endregion -->

```python
def encircle(x: np.ndarray, y: np.ndarray, ax: plt.Axes | None = None, **kw) -> None:
    """Trace l'enveloppe convexe (ConvexHull) du nuage (x, y) en tant que
    polygone sur l'axe `ax`.

    Args:
        x, y: coordonnées 1D des points.
        ax: l'axe matplotlib (par défaut `plt.gca()`).
        **kw: passés à `matplotlib.patches.Polygon`.
    """
    if ax is None:
        ax = plt.gca()
    points = np.c_[x, y]
    hull = ConvexHull(points)
    poly = plt.Polygon(points[hull.vertices, :], **kw)
    ax.add_patch(poly)


categories_sex = list(np.unique(df_tips["sex"]))
colors_sex = [PALETTE[0], PALETTE[1]]

fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
for i, cat in enumerate(categories_sex):
    sub = df_tips[df_tips["sex"] == cat]
    ax.scatter(sub["total_bill"], sub["tip"], s=30,
               c=colors_sex[i], label=str(cat), edgecolors="black")

female = df_tips.loc[df_tips.sex == "Female", :]
encircle(female.total_bill, female.tip, ax=ax,
         ec=colors_sex[0], fc="none", alpha=0.3)
encircle(female.total_bill, female.tip, ax=ax,
         ec=colors_sex[0], fc="none", linestyle="--", linewidth=1)

male = df_tips.loc[df_tips.sex == "Male", :]
encircle(male.total_bill, male.tip, ax=ax,
         ec=colors_sex[1], fc="none", alpha=0.3)
encircle(male.total_bill, male.tip, ax=ax,
         ec=colors_sex[1], fc="none", linestyle="--", linewidth=1)

ax.set_xlabel("total_bill", fontdict={"fontsize": 14})
ax.set_ylabel("tip", fontdict={"fontsize": 14})
ax.set_title("Scatter + ConvexHull (encircling)", fontsize=14)
ax.legend(fontsize=10)
plt.show()
```

<!-- #region -->
#### 3.3.i `lmplot` stylisé
<!-- #endregion -->

<!-- #region -->
Recette pour **customiser** un `sns.lmplot` : `scatter_kws` pour le contour et la taille des points, `aspect` pour l'allure 16/9, palette CHART.
<!-- #endregion -->

```python
df_tips.sample(3, random_state=0)
```

```python
g = sns.lmplot(
    x="total_bill", y="tip", hue="time", data=df_tips,
    height=6, aspect=1.5,
    palette=PALETTE[:2],
    scatter_kws=dict(s=60, linewidths=0.7, edgecolors="black"),
)
g.figure.suptitle("Scatter + best-fit line par 'time'", y=1.02)
plt.show()
```

<!-- #region -->
#### 3.3.j Counts plot — taille ∝ overlap
<!-- #endregion -->

<!-- #region -->
Pattern pour visualiser le **nombre de points superposés** sur un scatter (couples (x, y) répétés) : on `groupby` les couples, on récupère le count, puis on passe `s = base + counts * k` à `plt.scatter`. Plus le point est gros, plus de (x, y) coïncident.

(Le notebook d'origine utilisait `sns.stripplot(sizes=...)`, qui n'accepte plus de tableau de tailles en seaborn ≥ 0.13 — on bascule sur `matplotlib.scatter` qui accepte un `s` array-like.)
<!-- #endregion -->

```python
df_counts = (
    df_tips.groupby(["total_bill", "tip"])
    .size()
    .reset_index(name="counts")
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    df_counts["total_bill"],
    df_counts["tip"],
    s=20 + df_counts["counts"] * 60,
    color=CHART["primary_1"],
    alpha=0.6,
    edgecolors="black",
    linewidths=0.4,
)
ax.set_title("Counts plot — size ∝ nombre de points superposés")
ax.set_xlabel("total_bill")
ax.set_ylabel("tip")
plt.show()
```

<!-- #region -->
## 4. Diagramme de Pareto (Plotly)
<!-- #endregion -->

<!-- #region -->
**Pareto** = bars de fréquence (axe gauche) + ligne de **% cumulé** (axe droite). Sert à matérialiser la **règle 80/20** : quelles modalités contribuent à 80 % des occurrences ?

C'est le seul graphe **Plotly** du notebook — choisi pour l'interactivité native du double-axe Y (qui est lourd à recréer en matplotlib).
<!-- #endregion -->

```python
def plot_pareto(df: pd.DataFrame, x_col: str) -> go.Figure:
    """Diagramme de Pareto (axes doubles : fréquence + % cumulé).

    Args:
        df: DataFrame contenant la colonne qualitative à analyser.
        x_col: nom de la colonne.

    Returns:
        Figure Plotly prête à `fig.show()`.
    """
    counts = df[x_col].value_counts().reset_index()
    counts.columns = [x_col, "Frequency"]
    counts["Cumulative"] = counts["Frequency"].cumsum()
    counts["Cumulative Percentage"] = (
        counts["Cumulative"] / counts["Frequency"].sum() * 100
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts[x_col], y=counts["Frequency"],
        name="Frequency", marker_color=CHART["primary_1"], yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=counts[x_col], y=counts["Cumulative Percentage"],
        name="Cumulative %", mode="lines+markers",
        line=dict(color=CHART["mauvais"]),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Pareto Chart",
        xaxis=dict(title=x_col),
        yaxis=dict(title=dict(text="Frequency",
                              font=dict(color=CHART["primary_1"])),
                   tickfont=dict(color=CHART["primary_1"])),
        yaxis2=dict(title=dict(text="Cumulative %",
                               font=dict(color=CHART["mauvais"])),
                    tickfont=dict(color=CHART["mauvais"]),
                    overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="center", x=0.5),
    )
    return fig


data_par = {"Category": ["A", "B", "A", "C", "B", "A", "D", "C", "B", "A"]}
df_par = pd.DataFrame(data_par)
fig_par = plot_pareto(df_par, "Category")
fig_par.show()
```

<!-- #region -->
## 5. AutoEDA — outils 2026
<!-- #endregion -->

<!-- #region -->
Plusieurs libs génèrent un **rapport HTML complet** en 1 appel, calculé directement depuis un DataFrame. Le rapport contient typiquement : aperçu / missing / distribution univariée / corrélations / interactions / alertes (duplicats, déséquilibre, outliers).

**Quand utiliser** : POC, prise en main d'un dataset inconnu, audit rapide en début de projet, comparaison train/test.

**Quand NE PAS** : pour publier — le rapport sent toujours le défaut de personnalisation. Préférer une viz manuelle ciblée (= les recettes des sections 2 et 3 de ce notebook).

Panorama des libs maintenues en 2026 :

| Lib | Forces | Faiblesses |
|---|---|---|
| **ydata-profiling** (ex `pandas-profiling`) | Très complet, alertes auto (skewed/missing/zeros), rapport HTML interactif | Lent sur >100k lignes. **Rename annoncé fin 2025 → `fg-data-profiling`** |
| **sweetviz** | Comparaison train/test out-of-the-box (`sv.compare`), target analysis | Moins de stats, pas d'interactions |
| **dataprep.eda** | Rapide grâce à Dask | Moins d'options, API moins stable |
| **autoviz** | Génère plein de graphes à la volée | Brut, pas de narratif |
| **lux** | Suggestions de viz directement dans le notebook | Maintenance ralentie |

Dans la suite on utilise les 2 outils les plus solides en 2026 : `ydata-profiling` (panorama) et `sweetviz` (panorama + comparaison train/test).
<!-- #endregion -->

<!-- #region -->
### 5.1 `ydata-profiling` — rapport HTML complet
<!-- #endregion -->

<!-- #region -->
`ProfileReport(df, minimal=True)` génère un rapport HTML autonome en ~1 seconde sur un petit dataset. Mode `minimal=True` désactive les analyses lourdes (interactions, correlations pairwise sur >X colonnes) — c'est l'option à utiliser par défaut.

**Note 2026** : la lib est en cours de rename vers `fg-data-profiling`. L'API reste identique pendant la phase de transition ; un `DeprecationWarning` apparaît à l'import. Pour migrer plus tard : `pip install fg-data-profiling` + `import data_profiling`.
<!-- #endregion -->

```python
from ydata_profiling import ProfileReport

profile = ProfileReport(
    df_tips,
    title="Tips — AutoEDA (ydata-profiling)",
    minimal=True,
    progress_bar=False,
)
profile.to_file("eda_tips_ydata.html")
```

<!-- #region -->
### 5.2 `sweetviz` — panorama autonome
<!-- #endregion -->

<!-- #region -->
`sv.analyze(df)` produit un rapport HTML autonome avec un layout 2 colonnes : statistiques + mini-graphes par variable. Cible 1 variable (`target_feat=`) si tu veux une analyse supervisée (corrélations avec la cible mises en évidence).

`pairwise_analysis="off"` évite la matrice de corrélation pairwise (lourde) — utile si on veut juste les distributions et stats univariées.
<!-- #endregion -->

```python
import sweetviz as sv

report_sv = sv.analyze(df_tips, pairwise_analysis="off")
report_sv.show_html(
    filepath="eda_tips_sweetviz.html",
    open_browser=False,
    layout="vertical",
)
```

<!-- #region -->
### 5.3 `sweetviz` — comparaison train / test
<!-- #endregion -->

<!-- #region -->
La fonction killer de sweetviz : `sv.compare([df1, label1], [df2, label2])` met en regard les distributions de **2 DataFrames** côte à côte. Cas d'usage typiques :

- **Train vs test** : détecter un data leak ou un décalage de distribution avant entraînement.
- **Avant / après preprocessing** : vérifier que l'imputation ou la transformation n'a pas tordu une distribution.
- **2 cohortes** : groupe contrôle vs traité, période A vs période B.

Ici split aléatoire 70 / 30 reproductible pour démo.
<!-- #endregion -->

```python
rng = np.random.default_rng(42)
mask = rng.random(len(df_tips)) < 0.7
df_train = df_tips[mask].reset_index(drop=True)
df_test = df_tips[~mask].reset_index(drop=True)
print(f"train n={len(df_train)} / test n={len(df_test)}")

report_cmp = sv.compare(
    [df_train, "Train"],
    [df_test, "Test"],
    pairwise_analysis="off",
)
report_cmp.show_html(
    filepath="eda_tips_sweetviz_compare.html",
    open_browser=False,
    layout="vertical",
)
```

<!-- #region -->
## 6. Quel graphique pour quelle question ?
<!-- #endregion -->

<!-- #region -->
Une fois l'AutoEDA digérée, on revient toujours à du **manuel ciblé** : "j'ai cette question précise, quel graphique + quelle stat ?". Cette section condense les recettes du notebook en une **matrice décisionnelle**, plus quelques pièges fréquents.
<!-- #endregion -->

<!-- #region -->
### 6.1 Matrice décisionnelle
<!-- #endregion -->

<!-- #region -->
| Question | Type variables | Graphe recommandé | Stat associée | Recette de ce notebook |
|---|---|---|---|---|
| Quelle est la distribution de cette variable ? | 1 quanti | Histogramme + KDE + boxplot | mean / median / std / **skew / kurtosis** | §2.4 — `sns.histplot(kde=True)` |
| Y a-t-il des outliers visibles ? | 1 quanti | Boxplot, histo log, QQ-plot | IQR, Z-score | §2.6 — quantile clip |
| Quelle est la fréquence des modalités ? | 1 quali | Pie (≤ 7 mod) ou barplot trié | mode, entropie | §2.5 — `value_counts().plot.pie` |
| Y a-t-il des modalités rares à regrouper ? | 1 quali long-tail | Pie après regroupement | seuil de fréquence | §2.7 — `pies_modalites_rares` |
| Y a-t-il une corrélation linéaire ? | 2 quanti | Heatmap | **Pearson** | §2.8 — `df.corr(numeric_only=True)` |
| Y a-t-il une relation monotone non-linéaire ? | 2 quanti | Scatter + LOWESS | **Spearman**, Kendall | §3.2 — `lmplot(lowess=True)` |
| Une 3ᵉ variable explique-t-elle la dispersion ? | 2 quanti + 1 quali | Scatter coloré, `lmplot(hue=)`, `relplot` | corr par groupe | §3.3.a / §3.3.c |
| Les groupes ont-ils des distributions différentes ? | 1 quali × 1 quanti | Box, violin, swarm | **ANOVA** (si normal), **Kruskal-Wallis** (sinon) | §3.3.b |
| Les 2 variables catégorielles sont-elles liées ? | 2 quali | Table de contingence, heatmap | **chi²** (signif.), **Cramer's V** (effet) | §3.1.b–c |
| Quelles modalités contribuent à 80 % du volume ? | 1 quali | **Pareto** (bars + ligne cumulée) | règle 80/20 | §4 — `plot_pareto` (Plotly) |
| Combien de points coïncident exactement ? | 2 quanti répétés | Counts plot (sizes ∝ overlap) | — | §3.3.j |
| Comparer 2 cohortes globalement | 1 df vs 1 df | AutoEDA `sv.compare` | tous tests par variable | §5.3 |
<!-- #endregion -->

<!-- #region -->
### 6.2 Ordre d'attaque conseillé
<!-- #endregion -->

<!-- #region -->
1. **`shape` / `dtypes` / `head`** : combien de lignes, combien de quanti vs quali, quel mix (§ 0–1).
2. **NaN** : `df.isna().sum().sort_values(ascending=False)` — décider drop vs impute *avant* de tracer quoi que ce soit.
3. **AutoEDA `minimal=True`** (§5) : vue d'ensemble en 30 s, surtout pour repérer les **alertes** (duplicats, déséquilibre, constance).
4. **Univariée numérique** (§2.3–2.4) : describe étendu + histo + box pour les variables-clés. Bascule en log si skew > 2.
5. **Univariée catégorielle** (§2.5–2.7) : value_counts + pie si peu de modalités, sinon regroupement.
6. **Matrice de corrélation** (§2.8) : Pearson + Spearman, on cherche les divergences (non-linéarités).
7. **Cat × Num** ciblé (§3.3.b) : box ou violin sur les vars d'intérêt + ANOVA/Kruskal pour quantifier.
8. **Cat × Cat** ciblé (§3.1.b–c) : crosstab + chi² + Cramer's V pour les paires hypothèse.
9. **Multivariée** (PCA / MCA / FAMD) : passer à `EDA_Analyse_Multivarie`.
10. **Outliers** dédié : passer à `Detection_Outliers`.
11. **Documenter** : 1 cellule MD courte par finding au fur et à mesure. Un notebook EDA sans commentaire est inutilisable 3 mois plus tard.
<!-- #endregion -->

<!-- #region -->
### 6.3 Pièges fréquents
<!-- #endregion -->

<!-- #region -->
- **Pie chart > 7 modalités** : illisible. Bascule en barplot trié (`value_counts().sort_values().plot.barh()`).
- **Heatmap sans `numeric_only=True`** : pandas 2.x lève une erreur ou (pire) renvoie des résultats partiels — toujours expliciter.
- **Heatmap divergente sans `center=0`** : le cmap divergent (`RdBu_r`, `coolwarm`) n'a aucun sens sans centre fixé sur 0 ou 0.5 (cf §2.8).
- **`sns.countplot(palette=…)` sans `hue=`** : `FutureWarning` en seaborn ≥ 0.13. Soit on passe un `hue=` (avec `legend=False` si redondant), soit on enlève la palette.
- **Annoter via `patches` sur un `FacetGrid`** : il faut passer par `g.ax.patches` (ou `g.axes.flat`), pas `g.patches` (cf §3.1.f variante 2).
- **Pearson sur des variables fortement asymétriques** : trompeur ; toujours regarder **Spearman en parallèle**, le désaccord = non-linéarité ou outliers (§3.2 et §2.8).
- **chi² sans Cramer's V** : on conclut à un effet "significatif" sans connaître sa force. Sur n grand, tout est significatif → l'effet `V` est ce qui compte (§3.1.c).
- **Réécrire `df` en silence** : très fréquent dans les notebooks de catalogue. Préférer `df_tips`, `df_synth`, `iris`, `dfWIM`... pour ne pas perdre le fil (le notebook d'origine ré-écrasait `df` 4 fois — corrigé ici).
<!-- #endregion -->

<!-- #region -->
## 7. Sources
<!-- #endregion -->

<!-- #region -->
- [scipy.stats — tests statistiques](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seaborn — statistical visualization](https://seaborn.pydata.org/tutorial/statistical_data.html)
- [Plotly — Pareto chart](https://plotly.com/python/bar-charts/)
- [ydata-profiling — docs](https://docs.profiling.ydata.ai/)
- [sweetviz](https://github.com/fbdesignpro/sweetviz)
- [SciPy — ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
- Notebooks frères dans ce repo : `EDA_Visualisation_Introduction`, `EDA_Analyse_Multivarie`, `Detection_Outliers`.
<!-- #endregion -->
