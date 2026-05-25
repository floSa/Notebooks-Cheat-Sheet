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
# 🦆 DuckDB — Analytics SQL embarqué
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **DuckDB** (1.x, 2026) — le **"SQLite des analytics"** : DB columnaire OLAP embedded, devenu standard de fait pour le data wrangling local en 2024-2026.

Couverture :

1. **Pourquoi DuckDB** — vs pandas / vs Postgres / vs Spark.
2. **Installation & connexion** (in-memory, file, S3).
3. **Lecture directe** de CSV / Parquet / JSON / pandas / polars / arrow.
4. **SQL moderne** : window functions, CTEs, recursive, lateral joins.
5. **Performance** : tips, EXPLAIN, profiling.
6. **Integration** pandas / polars / Arrow zero-copy.
7. **Cas d'usage** : NYC Taxi parquet (dataset mutualisé).
8. **Extensions** : httpfs, spatial, FTS, vector search (vss).
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi DuckDB
<!-- #endregion -->

<!-- #region -->
**DuckDB** (2019, MIT license) — DB analytique **OLAP**, **embedded** (pas de serveur), **columnaire**, SQL ANSI complet, gratuit.

**Comparatif rapide 2026** :

| Outil | Type | Performance analytics | Échelle | Installation |
|---|---|---|---|---|
| **pandas** | DataFrame in-memory | OK (single-thread) | < 10 GB RAM | `pip install` |
| **Polars** | DataFrame Rust | **Excellente** (multi-thread) | < RAM machine | `pip install` |
| **DuckDB** | DB embedded OLAP | **Excellente** | < disque machine | `pip install` |
| **PostgreSQL** | DB OLTP serveur | Bonne | TB avec partitioning | Service à gérer |
| **ClickHouse** | DB OLAP serveur | Excellente | TB-PB | Service à gérer |
| **Spark** | Cluster compute | Excellente distribuée | PB | Cluster à gérer |
| **BigQuery / Snowflake** | DW cloud | Excellente | PB | SaaS, $$ |

**Quand utiliser DuckDB** :

- Analyses **single-machine** sur 100 MB - 100 GB.
- Lire des **parquet** sur S3 sans charger en RAM.
- **Remplacer pandas** quand pandas devient lent (>1M rows + aggregations).
- **Tester du SQL** sans avoir à monter un service Postgres.
- Glue entre formats hétérogènes (parquet + CSV + pandas).
<!-- #endregion -->

```python
import duckdb
import pandas as pd
import numpy as np

print(f"DuckDB version : {duckdb.__version__}")
```

<!-- #region -->
## 2. Connexion et premiers pas
<!-- #endregion -->

```python
# 3 modes :
# 1. In-memory (perd tout à la fermeture)
con = duckdb.connect(":memory:")

# 2. Fichier persistant
# con = duckdb.connect("mydb.duckdb")

# 3. Read-only (lecture concurrente)
# con = duckdb.connect("mydb.duckdb", read_only=True)

# Exécution simple
result = con.execute("SELECT 42 AS answer").fetchall()
print(result)

# Vers pandas direct
df = con.execute("SELECT 'hello' as msg, range AS i FROM range(5)").df()
print(df)
```

<!-- #region -->
## 3. Lecture directe (zero-config, zero-copy)
<!-- #endregion -->

<!-- #region -->
**Force majeure de DuckDB** : lire directement des fichiers et des DataFrames sans étape de chargement.
<!-- #endregion -->

```python
# Depuis un DataFrame pandas — zéro copie
df_pandas = pd.DataFrame({"x": [1,2,3], "y": [10,20,30]})
result = duckdb.sql("SELECT x, y, x*y AS xy FROM df_pandas").df()
print(result)

# Depuis NumPy array
arr = np.random.randn(100, 3)
df_np = pd.DataFrame(arr, columns=["a","b","c"])
duckdb.sql("SELECT AVG(a), STDDEV(b), MAX(c) FROM df_np").show()
```

```python
# Depuis CSV (sans le charger en RAM d'abord)
# duckdb.sql("SELECT * FROM read_csv_auto('data.csv') WHERE age > 30").df()

# Depuis Parquet (encore plus rapide grâce au pushdown)
# duckdb.sql("SELECT col1, col2 FROM read_parquet('data/*.parquet') WHERE date >= '2025-01-01'").df()

# Depuis S3 (avec extension httpfs)
# duckdb.sql("INSTALL httpfs; LOAD httpfs;")
# duckdb.sql("SELECT * FROM 's3://bucket/path/*.parquet' WHERE country = 'FR'").df()
```

<!-- #region -->
## 4. SQL moderne — features avancées
<!-- #endregion -->

```python
# Window functions
con.execute("""
WITH sales AS (
    SELECT * FROM (VALUES
        ('Alice', 'Q1', 100), ('Alice', 'Q2', 150),
        ('Bob', 'Q1', 200), ('Bob', 'Q2', 180)
    ) AS t(name, quarter, amount)
)
SELECT
    name, quarter, amount,
    SUM(amount) OVER (PARTITION BY name ORDER BY quarter) AS running_total,
    RANK() OVER (PARTITION BY quarter ORDER BY amount DESC) AS rank_in_quarter,
    LAG(amount, 1) OVER (PARTITION BY name ORDER BY quarter) AS prev_quarter
FROM sales
""").df()
```

```python
# CTE récursive — calcul de Fibonacci en SQL
con.execute("""
WITH RECURSIVE fib(n, a, b) AS (
    VALUES (1, 0, 1)
    UNION ALL
    SELECT n+1, b, a+b FROM fib WHERE n < 10
)
SELECT n, a FROM fib
""").df()
```

```python
# QUALIFY (alternative concise à where sur window function)
con.execute("""
WITH sales AS (
    SELECT * FROM (VALUES
        ('Alice', 100), ('Bob', 200), ('Charlie', 150), ('Dave', 300)
    ) AS t(name, amount)
)
SELECT *
FROM sales
QUALIFY RANK() OVER (ORDER BY amount DESC) <= 2  -- top 2 sans subquery
""").df()
```

<!-- #region -->
## 5. Performance
<!-- #endregion -->

```python
# EXPLAIN / EXPLAIN ANALYZE
plan = con.execute("EXPLAIN SELECT COUNT(*) FROM df_pandas WHERE x > 1").df()
print(plan)
```

<!-- #region -->
**Tips perf** :

- Utiliser **Parquet** plutôt que CSV — 5-20× plus rapide à lire.
- **Partitioner** par colonne fréquemment filtrée (Hive-style).
- **Multi-thread** : `PRAGMA threads=8;` (par défaut = cores).
- **Mémoire** : `PRAGMA memory_limit='8GB';` pour borner.
- **Persistance** : pour des intermédiaires lourds, créer une table physique au lieu de répéter la query.
- **Index** : DuckDB crée des **min-max indexes** automatiquement, rarement besoin de PRIMARY KEY explicite.
<!-- #endregion -->

<!-- #region -->
## 6. Integration pandas / polars / arrow zero-copy
<!-- #endregion -->

```python
# pandas → DuckDB → polars : zéro copie via Arrow
import pyarrow as pa

df_p = pd.DataFrame({"a": range(100), "b": [i*2 for i in range(100)]})

# Result → Arrow table
arrow_tbl = duckdb.sql("SELECT a, b, a+b AS sum FROM df_p").arrow()
print(f"Arrow table type : {type(arrow_tbl)}")

# Arrow → polars
# import polars as pl
# pl_df = pl.from_arrow(arrow_tbl)
```

<!-- #region -->
## 7. Cas d'usage — NYC Taxi parquet
<!-- #endregion -->

<!-- #region -->
Dataset mutualisé (cf `00_datasets.md`) : **NYC Taxi parquet** (1 mois ≈ 50 MB, 3M+ rides).

```bash
# Download dataset (manuel)
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet -O data/_shared/nyc_taxi/yellow_2024-01.parquet
```

```python
# Stats sur le dataset sans charger en RAM
"""
duckdb.sql('''
SELECT
    DATE_TRUNC('day', tpep_pickup_datetime) AS day,
    COUNT(*) AS n_rides,
    AVG(trip_distance) AS avg_distance_mi,
    AVG(fare_amount) AS avg_fare,
    SUM(tip_amount) AS total_tips
FROM 'data/_shared/nyc_taxi/yellow_2024-01.parquet'
WHERE passenger_count > 0 AND fare_amount BETWEEN 0 AND 200
GROUP BY 1
ORDER BY 1
LIMIT 10
''').df()
"""
```

Sur ~3M lignes, cette query tourne en quelques secondes sur un laptop. **pandas équivalent** prendrait beaucoup plus de temps et 5-10× plus de RAM.
<!-- #endregion -->

<!-- #region -->
## 8. Extensions notables
<!-- #endregion -->

<!-- #region -->
| Extension | Usage |
|---|---|
| **httpfs** | Lire/écrire S3, GCS, Azure Blob, HTTP |
| **postgres** | Lire/écrire vers une instance Postgres directement |
| **sqlite** | Idem SQLite |
| **excel** | Lire fichiers .xlsx |
| **json** | Parse / explode JSON avancé |
| **spatial** | Géospatial (PostGIS-like : ST_*) |
| **fts** | Full-text search BM25 (cf `NLP_Recherche_d_informations`) |
| **vss** (Vector Similarity Search) | Vector DB embedded (concurrence FAISS/LanceDB) |
| **autocomplete** | Suggestion SQL dans le CLI |
| **iceberg / delta** | Lire des tables Iceberg / Delta Lake |

```sql
INSTALL spatial;
LOAD spatial;
SELECT ST_Distance(ST_Point(2.35, 48.85), ST_Point(-74.00, 40.71));  -- Paris ↔ NYC en degrés
```
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [DuckDB — docs officielles](https://duckdb.org/docs/)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [DuckDB extensions](https://duckdb.org/docs/extensions/overview)
- [Modern Data Stack with DuckDB — blog](https://duckdb.org/2023/04/14/h2oai-1b.html)
- Notebooks liés : `Structures_DataFrame`, `Structure_BDD_DataFrame`, `BDD_Vectorielles`.
<!-- #endregion -->
