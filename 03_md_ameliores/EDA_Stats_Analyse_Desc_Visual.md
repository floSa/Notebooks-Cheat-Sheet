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
# 📊 EDA — Statistiques descriptives & visualisations
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki** sur l'**analyse exploratoire de données** (EDA) classique : statistiques descriptives, distributions, corrélations, croisements bivariés.

Suit le workflow standard :

1. **Vue d'ensemble** : info, missing, dtypes.
2. **Univariée numérique** : centrale (mean/median/mode), dispersion (std/IQR), forme (skew/kurtosis), distribution (hist, KDE, QQ).
3. **Univariée catégorielle** : freq, mode, entropie.
4. **Bivariée num × num** : corrélation Pearson/Spearman, scatter.
5. **Bivariée cat × cat** : table de contingence, chi², Cramer's V.
6. **Bivariée cat × num** : box, violin, ANOVA, Kruskal-Wallis.
7. **Multivariée** : voir `EDA_Analyse_Multivarie`.
8. **Outils 2026** d'**autoEDA** (ydata-profiling, sweetviz, autoviz).

Dataset : **Titanic** (mutualisé).
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid")

df = sns.load_dataset("titanic")
print(df.shape)
df.head()
```

<!-- #region -->
## 1. Vue d'ensemble
<!-- #endregion -->

```python
# Coup d'œil rapide
print("Shape :", df.shape)
print("\nDtypes :")
print(df.dtypes)
print("\nMissing :")
print(df.isna().sum().sort_values(ascending=False).head(10))
print("\nMemory :", df.memory_usage(deep=True).sum() / 1e6, "MB")
```

<!-- #region -->
## 2. Univariée — variables numériques
<!-- #endregion -->

<!-- #region -->
### 2.1 Statistiques résumées
<!-- #endregion -->

```python
num_cols = df.select_dtypes(include="number").columns.tolist()
desc = df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T

# Ajout skew + kurtosis (au-delà des stats classiques de .describe())
desc["skew"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurt()
desc["missing_%"] = df[num_cols].isna().mean() * 100
print(desc.round(2))
```

<!-- #region -->
**Lecture** :

- **skew** ≈ 0 : symétrique. > 0 : queue à droite (ex: revenus). < 0 : queue à gauche.
- **kurtosis** : excès vs normale. > 0 : queues lourdes (outliers fréquents). < 0 : plus aplatie.
- **missing_%** : indispensable pour décider imputation vs drop.
<!-- #endregion -->

<!-- #region -->
### 2.2 Distribution — histogramme + KDE + QQ-plot
<!-- #endregion -->

```python
col = "age"
data = df[col].dropna()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Histogramme + KDE
sns.histplot(data, bins=30, kde=True, ax=axes[0])
axes[0].set_title(f"Distribution {col}")

# Boxplot horizontal
sns.boxplot(x=data, ax=axes[1])
axes[1].set_title(f"Boxplot {col}")

# QQ plot vs Normale (pour vérifier la normalité visuellement)
stats.probplot(data, dist="norm", plot=axes[2])
axes[2].set_title(f"QQ-plot {col} vs Normale")
axes[2].grid(alpha=0.3)

plt.tight_layout()
```

<!-- #region -->
### 2.3 Tests de normalité
<!-- #endregion -->

```python
# Shapiro-Wilk : H0 = normalité. p<0.05 → rejet
shapiro_stat, shapiro_p = stats.shapiro(data)
# D'Agostino : autre test plus robuste sur grands échantillons
dag_stat, dag_p = stats.normaltest(data)
print(f"Shapiro-Wilk : stat={shapiro_stat:.3f}  p={shapiro_p:.4f}  → {'normal' if shapiro_p>0.05 else 'NON normal'}")
print(f"D'Agostino   : stat={dag_stat:.3f}  p={dag_p:.4f}  → {'normal' if dag_p>0.05 else 'NON normal'}")
```

<!-- #region -->
## 3. Univariée — variables catégorielles
<!-- #endregion -->

```python
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
for col in cat_cols[:3]:
    print(f"\n--- {col} ---")
    counts = df[col].value_counts(dropna=False)
    print(counts.to_frame("count").assign(pct=lambda x: x["count"] / len(df) * 100).round(1))

# Entropie de Shannon (mesure d'incertitude / diversité)
from scipy.stats import entropy
for col in cat_cols[:3]:
    p = df[col].value_counts(normalize=True)
    print(f"Entropie {col} = {entropy(p, base=2):.3f} bits  (max={np.log2(len(p)):.3f})")
```

<!-- #region -->
## 4. Bivariée — num × num : corrélations
<!-- #endregion -->

<!-- #region -->
- **Pearson** : corrélation linéaire. Sensible aux outliers, suppose normalité approximative.
- **Spearman** : rangs. Capture les monotonies non-linéaires. Robuste aux outliers.
- **Kendall** : autre coefficient de rang, plus robuste mais plus lent.

Règle : toujours regarder **Pearson + Spearman**. Si ils divergent franchement, c'est qu'il y a de la non-linéarité ou des outliers.
<!-- #endregion -->

```python
num_df = df[num_cols].dropna()

pearson = num_df.corr(method="pearson")
spearman = num_df.corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title("Pearson")
sns.heatmap(spearman, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title("Spearman")
plt.tight_layout()
```

<!-- #region -->
## 5. Bivariée — cat × cat : table de contingence + chi²
<!-- #endregion -->

```python
ct = pd.crosstab(df["sex"], df["survived"], margins=True)
print("Table de contingence :")
print(ct)

# Test du chi² : H0 = indépendance
chi2_stat, p_value, dof, expected = stats.chi2_contingency(pd.crosstab(df["sex"], df["survived"]))
print(f"\nChi² : stat={chi2_stat:.2f}  dof={dof}  p={p_value:.2e}")
print(f"→ {'INDÉPENDANTES' if p_value > 0.05 else 'DÉPENDANTES (effet significatif)'}")


# Cramer's V — taille d'effet (chi² ne dit pas la force, juste la signif)
def cramers_v(confusion_matrix: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


v = cramers_v(pd.crosstab(df["sex"], df["survived"]).values)
print(f"\nCramer's V = {v:.3f}  (0=indep, 1=parfait)")
```

<!-- #region -->
## 6. Bivariée — cat × num : box, ANOVA, Kruskal-Wallis
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(df, x="class", y="age", hue="survived", ax=axes[0])
axes[0].set_title("Age par classe & survie")
sns.violinplot(df, x="class", y="fare", hue="survived", split=True, ax=axes[1])
axes[1].set_title("Fare par classe & survie")
plt.tight_layout()


# ANOVA : H0 = mêmes moyennes entre groupes
groups = [df.loc[df["class"] == c, "age"].dropna() for c in df["class"].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"ANOVA age~class : F={f_stat:.2f}  p={p_anova:.4f}  → {'différence' if p_anova < 0.05 else 'pas de différence'}")

# Kruskal-Wallis : version non-paramétrique (pas d'hypothèse de normalité)
h_stat, p_kw = stats.kruskal(*groups)
print(f"Kruskal-Wallis : H={h_stat:.2f}  p={p_kw:.4f}")
```

<!-- #region -->
## 7. AutoEDA — gagner du temps en 2026
<!-- #endregion -->

<!-- #region -->
Plusieurs libs génèrent un **rapport HTML complet** en 1 ligne :

| Lib | Forces | Faiblesses |
|---|---|---|
| **ydata-profiling** (ex pandas-profiling) | Très complet, rapport HTML interactif | Lent sur grands datasets |
| **sweetviz** | Comparaison train/test out-of-the-box | Moins de stats |
| **dataprep.eda** | Très rapide grâce à Dask | Moins d'options |
| **autoviz** | Génère plein de graphes à la volée | Brut |
| **lux** | Suggestions de viz dans le notebook | Moins maintenu |

```python
# Pseudo-code ydata-profiling
# from ydata_profiling import ProfileReport
# profile = ProfileReport(df, title="Titanic EDA", minimal=True)
# profile.to_file("eda_titanic.html")
```

**Quand les utiliser** : POC ou découverte d'un nouveau dataset. **Quand NE PAS** : pour publier — préférer un travail manuel choisi avec soin.
<!-- #endregion -->

<!-- #region -->
## 8. Workflow recommandé 2026
<!-- #endregion -->

<!-- #region -->
1. **Charger** + `.shape`, `.dtypes`, `.head()`.
2. **Missing** : `df.isna().sum()` + heatmap des NaN si nombreux.
3. **AutoEDA** rapide (ydata-profiling) pour avoir une vue d'ensemble en 1 minute.
4. **Univariée** numérique : describe étendu (+ skew/kurtosis), histo + box + QQ pour les variables clés.
5. **Univariée** catégorielle : value_counts, freq, entropie.
6. **Bivariée** : matrice de corr (Pearson + Spearman), pairplot pour les vars clés.
7. **Cat × Num / Cat × Cat** : box / violin / heatmap contingence + tests stat.
8. **Multivariée** : PCA / MCA / FAMD → voir `EDA_Analyse_Multivarie`.
9. **Outliers** : voir `Detection_Outliers`.
10. **Documenter** les findings en markdown au fur et à mesure — un notebook EDA sans commentaire est inutilisable 3 mois plus tard.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [scipy.stats — tests statistiques](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seaborn statistical visualization](https://seaborn.pydata.org/tutorial/statistical_data.html)
- [ydata-profiling](https://docs.profiling.ydata.ai/)
- [sweetviz](https://github.com/fbdesignpro/sweetviz)
- Notebooks liés : `EDA_Visualisation_Introduction`, `EDA_Analyse_Multivarie`, `Detection_Outliers`.
<!-- #endregion -->
