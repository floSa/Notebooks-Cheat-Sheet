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
# 🐼 Pandas DataFrame — Cheat-sheet & Tutoriel
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** sur **pandas DataFrames** (v2.x, 2026). Tous les use cases courants en data work, avec un parti pris **idiomatique 2026**.

Couverture :

1. Création / chargement (CSV, Parquet, SQL, JSON).
2. Inspection : info, head, describe, dtypes, memory.
3. Indexation : `loc`, `iloc`, masking, multi-index.
4. **Méthodes vectorisées** vs boucles.
5. **GroupBy** : split-apply-combine.
6. **Merge / join** (= SQL).
7. **Pivot** / wide vs long (melt).
8. **Apply** : par cellule, par row, par col, par groupe.
9. **Missing data** (`isna`, `fillna`, `dropna`).
10. Types & **categorical** pour performance.
11. **Pandas 2.x** : Arrow backend, Copy-on-Write.
12. **Polars** (alternative 2026 plus rapide).

Dataset : **Titanic** + **California Housing** (mutualisés).
<!-- #endregion -->

<!-- #region -->
## 1. Création et chargement
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import seaborn as sns

# Création depuis dict
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "city": ["Paris", "Lyon", "Marseille"],
})
print(df)

# Chargement standard
# df = pd.read_csv("file.csv", parse_dates=["date_col"], dtype={"id": str})
# df = pd.read_parquet("file.parquet")              # plus rapide que CSV
# df = pd.read_excel("file.xlsx", sheet_name=0)
# df = pd.read_json("file.json", lines=True)
# df = pd.read_sql("SELECT * FROM t", connection)
```

```python
# Titanic pour la suite
titanic = sns.load_dataset("titanic")
print(titanic.shape)
```

<!-- #region -->
## 2. Inspection rapide
<!-- #endregion -->

```python
titanic.head(3)
titanic.tail(3)
titanic.info()                  # dtypes + non-null counts + memory
titanic.describe()              # stats numériques
titanic.describe(include="O")   # stats catégorielles
titanic.dtypes
titanic.shape
titanic.memory_usage(deep=True).sum() / 1e6   # MB (deep pour les strings)
```

<!-- #region -->
## 3. Indexation : `loc` vs `iloc` vs `[]`
<!-- #endregion -->

```python
# loc = label-based, iloc = position-based
titanic.loc[0]                   # ligne par label (= index 0 ici)
titanic.iloc[0]                  # ligne par position (= 1ère ligne)
titanic.iloc[-5:]                # 5 dernières
titanic.loc[:, "age":"fare"]     # cols entre labels
titanic.iloc[0:5, 0:3]           # bloc

# Boolean masking
titanic[titanic["age"] > 50]
titanic.query("age > 50 and sex == 'male'")  # syntaxe SQL-like
titanic[titanic["embarked"].isin(["S", "C"])]
titanic[~titanic["deck"].isna()]
```

```python
# Set index
ti = titanic.set_index("pclass")
ti.loc[1].head()                 # toutes les 1ères classes

# MultiIndex
mt = titanic.set_index(["pclass", "sex"]).sort_index()
mt.loc[(1, "female")].head()
mt.loc[1].loc["female"]          # équivalent
```

<!-- #region -->
## 4. Méthodes vectorisées (ne jamais itérer !)
<!-- #endregion -->

```python
# ❌ MAUVAIS — itération Python
def loop_double_age(df):
    df = df.copy()
    df["age2"] = 0.0
    for i in range(len(df)):
        df.loc[i, "age2"] = df.loc[i, "age"] * 2 if pd.notna(df.loc[i, "age"]) else 0
    return df


# ✅ BON — vectorisé (100-1000× plus rapide)
def vec_double_age(df):
    df = df.copy()
    df["age2"] = df["age"].fillna(0) * 2
    return df


import time
t = time.perf_counter(); _ = loop_double_age(titanic); print(f"loop : {time.perf_counter() - t:.3f}s")
t = time.perf_counter(); _ = vec_double_age(titanic); print(f"vec  : {time.perf_counter() - t:.4f}s")
```

<!-- #region -->
## 5. GroupBy — split-apply-combine
<!-- #endregion -->

```python
# Agrégations simples
titanic.groupby("class")["age"].mean()
titanic.groupby("class")[["age", "fare"]].agg(["mean", "median", "std"])

# Multi-groupes + multi-agrégations nommées
titanic.groupby(["class", "sex"]).agg(
    n=("age", "size"),
    age_mean=("age", "mean"),
    fare_max=("fare", "max"),
).round(2)

# Transform : renvoie une série de la même taille (broadcast)
titanic["age_dev_from_class_mean"] = titanic["age"] - titanic.groupby("class")["age"].transform("mean")
titanic[["age", "class", "age_dev_from_class_mean"]].head()

# Filter : garde uniquement les groupes qui satisfont une condition
titanic.groupby("class").filter(lambda g: g["age"].mean() > 25).shape
```

<!-- #region -->
## 6. Merge / Join (= SQL)
<!-- #endregion -->

```python
df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
df2 = pd.DataFrame({"id": [2, 3, 4], "score": [80, 90, 95]})

pd.merge(df1, df2, on="id", how="inner")
pd.merge(df1, df2, on="id", how="left")
pd.merge(df1, df2, on="id", how="outer", indicator=True)   # ajoute _merge col

# Merge sur colonnes nommées différemment
df3 = df2.rename(columns={"id": "user_id"})
pd.merge(df1, df3, left_on="id", right_on="user_id")

# Join par index
df1.set_index("id").join(df2.set_index("id"), how="outer")

# Concat (empilement)
pd.concat([df1, df1], axis=0)        # ajout de lignes
pd.concat([df1, df2.set_index("id")], axis=1)   # ajout de colonnes
```

<!-- #region -->
## 7. Pivot / wide vs long
<!-- #endregion -->

```python
# Pivot table (= SQL pivot)
pivot = titanic.pivot_table(
    index="class", columns="sex", values="survived",
    aggfunc="mean",
)
print(pivot)

# Wide → long (melt)
melted = pivot.reset_index().melt(id_vars="class", var_name="sex", value_name="survival_rate")
print(melted)

# Long → wide (pivot)
melted.pivot(index="class", columns="sex", values="survival_rate")

# crosstab (raccourci pour value_counts croisé)
pd.crosstab(titanic["class"], titanic["survived"], normalize="index")   # % par row
```

<!-- #region -->
## 8. Apply / Map
<!-- #endregion -->

```python
# Apply sur une Série (élément par élément)
titanic["name_len"] = titanic["who"].apply(len)

# Map sur une Série
titanic["gender_short"] = titanic["sex"].map({"male": "M", "female": "F"})

# Apply sur DataFrame avec axis=1 (par ligne) — souvent LENT, préférer vectorisé
# titanic["is_child"] = titanic.apply(lambda r: r["age"] < 18 if pd.notna(r["age"]) else False, axis=1)
# Préférer :
titanic["is_child"] = (titanic["age"] < 18).fillna(False)

# Apply par groupe (returning a Series)
def normalize(g):
    return (g - g.mean()) / g.std()

titanic["age_z"] = titanic.groupby("class")["age"].transform(normalize)
```

<!-- #region -->
## 9. Missing data
<!-- #endregion -->

```python
titanic.isna().sum().sort_values(ascending=False).head()    # missing par col

# Fill
titanic["age"].fillna(titanic["age"].median())               # médiane
titanic["embarked"].fillna(titanic["embarked"].mode()[0])    # mode
titanic.ffill()                                              # forward fill (séries)
titanic.bfill()                                              # backward fill
titanic["age"].interpolate(method="linear")                  # interpolation

# Drop
titanic.dropna(subset=["age", "embarked"])                   # drop si NaN dans subset
titanic.dropna(axis=1, thresh=len(titanic) * 0.5)            # drop cols avec >50% NaN

# Indicateur "était NaN" (utile en feature engineering)
titanic["age_was_nan"] = titanic["age"].isna().astype(int)
```

<!-- #region -->
## 10. Types & Categorical
<!-- #endregion -->

```python
# Conversion explicite
titanic["age"] = titanic["age"].astype("Float32")          # nullable float
titanic["pclass"] = titanic["pclass"].astype("category")   # économie mémoire + ordre

# Avant/après mémoire
print(f"Mémoire avant : {titanic.memory_usage(deep=True).sum() / 1e3:.0f} KB")

# Convertir TOUTES les object low-cardinality en category
for col in titanic.select_dtypes("object"):
    if titanic[col].nunique() / len(titanic) < 0.5:
        titanic[col] = titanic[col].astype("category")

print(f"Mémoire après : {titanic.memory_usage(deep=True).sum() / 1e3:.0f} KB")
```

<!-- #region -->
## 11. Pandas 2.x — Arrow backend & Copy-on-Write
<!-- #endregion -->

<!-- #region -->
Évolutions majeures de pandas 2.x :

- **PyArrow backend** : `pd.read_csv("f.csv", dtype_backend="pyarrow")` — strings columnaires zéro-copie, beaucoup plus rapide.
- **Copy-on-Write** (CoW, activé par défaut depuis 2.2) : finit les bugs "j'ai modifié une vue par accident".
- **Nullable dtypes** : `Int64` (vs int64), `Float64` (vs float64) supportent NaN sans cast en float.

```python
# Activer le backend pyarrow pour tout le notebook
# pd.options.future.infer_string = True
# df = pd.read_csv("file.csv", dtype_backend="pyarrow")
```
<!-- #endregion -->

<!-- #region -->
## 12. Polars — l'alternative 2026
<!-- #endregion -->

<!-- #region -->
**Polars** (Rust-based) est devenu en 2025-2026 une **alternative sérieuse** à pandas :

- **5-30× plus rapide** sur de gros datasets (Rust + lazy evaluation + multi-thread natif).
- **API plus stricte** (pas de surprises type "string column casted en object").
- **Lazy mode** : optimise tout le query plan avant d'exécuter (à la DuckDB).
- **Streaming** : peut process des fichiers plus gros que la RAM.

```python
"""
import polars as pl

df = pl.read_csv("file.csv")
result = (
    df.lazy()
    .filter(pl.col("age") > 18)
    .group_by("class")
    .agg([
        pl.col("fare").mean().alias("mean_fare"),
        pl.count().alias("n"),
    ])
    .sort("n", descending=True)
    .collect()           # déclenche l'exécution
)
"""
```

**Quand basculer en Polars** : fichiers > 100 MB, latence critique, pas besoin de l'écosystème pandas (matplotlib accepte polars depuis 2024).
<!-- #endregion -->

<!-- #region -->
## 13. Sources
<!-- #endregion -->

<!-- #region -->
- [pandas docs](https://pandas.pydata.org/docs/)
- [Effective Pandas — Matt Harrison (livre)](https://store.metasnake.com/effective-pandas-book)
- [Polars docs](https://docs.pola.rs/)
- [Modern Pandas series — Tom Augspurger](https://tomaugspurger.net/posts/modern-1-intro/)
- Notebooks liés : `Structure_Python`, `Structures_L_T_D_Cheat_Sheet`, `Structures_Preprocessing`, `Structure_BDD_DataFrame`.
<!-- #endregion -->
