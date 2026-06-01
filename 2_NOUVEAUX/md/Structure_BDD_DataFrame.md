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
# 💾 DataFrame ↔ Bases de données
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet** : recettes pour lire/écrire un DataFrame depuis/vers une base de données (SQL et NoSQL).

Couvre :

1. **PostgreSQL / MySQL / SQLite** — SQLAlchemy 2.x + psycopg3.
2. **MongoDB** — PyMongo + adaptateurs DataFrame.
3. **DuckDB** — analytique embarqué (cf notebook dédié `BDD_DuckDB`).
4. **Parquet / Arrow** — formats colonnaires modernes.
5. **Bonnes pratiques 2026** : pooling, transactions, batch insert, retry.

> Pour une **base de données vectorielle** (FAISS/Qdrant/pgvector), voir `BDD_Vectorielles`.
> Pour **DuckDB en analytics**, voir `BDD_DuckDB`.
<!-- #endregion -->

<!-- #region -->
## 1. SQLAlchemy 2.x — l'ORM/SQL toolkit de référence
<!-- #endregion -->

<!-- #region -->
**SQLAlchemy 2.x** (2023) — refonte majeure. Deux modes :

- **Core** : SQL Expression Language, plus proche du SQL, le plus rapide.
- **ORM** : mapping Python classes ↔ tables.

Pour le **data work** (lecture / écriture DataFrame), on utilise rarement l'ORM — Core via `pd.read_sql` / `df.to_sql` suffit.
<!-- #endregion -->

```python
# Exemple SQLite en mémoire (auto-disponible, pas de setup)
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///:memory:")  # ou "sqlite:///mydb.db" pour fichier

# Insert depuis DataFrame
df = pd.DataFrame({"id": [1,2,3], "name": ["A","B","C"], "age": [25, 30, 35]})
df.to_sql("users", engine, if_exists="replace", index=False)

# Read depuis SQL
result = pd.read_sql("SELECT * FROM users WHERE age > 25", engine)
print(result)

# Avec paramètres (anti-SQL-injection)
threshold = 26
result = pd.read_sql(text("SELECT * FROM users WHERE age > :t"), engine, params={"t": threshold})
print(result)
```

<!-- #region -->
### 1.1 PostgreSQL avec psycopg3
<!-- #endregion -->

<!-- #region -->
**psycopg3** (2022) — successeur de psycopg2, async, copy intégrée.

```python
# pip install "psycopg[binary]" sqlalchemy
"""
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg://user:pw@host:5432/dbname",
    pool_size=5, max_overflow=10,
    pool_pre_ping=True,        # vérifie la conn avant use (évite stale conn)
    pool_recycle=3600,         # recycle conn toutes les heures
)

# Read large query streamé (évite OOM)
for chunk in pd.read_sql("SELECT * FROM big_table", engine, chunksize=10_000):
    process(chunk)

# Write fast : utiliser COPY (psycopg3 native)
from io import StringIO
buf = StringIO(); df.to_csv(buf, index=False, header=False); buf.seek(0)
with engine.raw_connection() as conn:
    with conn.cursor() as cur:
        cur.copy_expert("COPY users FROM STDIN WITH CSV", buf)
    conn.commit()
"""
```
<!-- #endregion -->

<!-- #region -->
### 1.2 Pattern d'écriture avec gestion d'erreurs
<!-- #endregion -->

```python
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager


@contextmanager
def transaction(engine):
    """Context manager qui commit ou rollback automatiquement."""
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except SQLAlchemyError:
        trans.rollback()
        raise
    finally:
        conn.close()


# Usage
df_demo = pd.DataFrame({"id": [10,11], "name": ["X","Y"], "age": [40, 50]})
try:
    with transaction(engine) as conn:
        df_demo.to_sql("users", conn, if_exists="append", index=False)
    print("Insert OK")
except SQLAlchemyError as e:
    print(f"Rollback : {e}")
```

<!-- #region -->
## 2. MongoDB — PyMongo + DataFrame
<!-- #endregion -->

<!-- #region -->
**MongoDB** : NoSQL document-store, parfait pour du semi-structuré (logs, profils utilisateurs, données hétérogènes).

```python
# pip install pymongo
"""
from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017")
db = client["my_db"]
coll = db["users"]

# Insert depuis DataFrame
df.to_dict(orient="records")
coll.insert_many(df.to_dict(orient="records"))

# Bulk insert avec batch_size + ordered=False (continue si une insert plante)
coll.insert_many(records, ordered=False)

# Read depuis MongoDB
cursor = coll.find({"age": {"$gt": 25}}, projection={"_id": 0})
df_loaded = pd.DataFrame(list(cursor))

# Aggregation pipeline pour les analytics
pipeline = [
    {"$match": {"age": {"$gt": 25}}},
    {"$group": {"_id": "$city", "avg_age": {"$avg": "$age"}, "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
]
df_agg = pd.DataFrame(list(coll.aggregate(pipeline)))
"""
```
<!-- #endregion -->

<!-- #region -->
## 3. Parquet / Arrow — formats columnaires modernes
<!-- #endregion -->

<!-- #region -->
**Parquet** : format columnaire compressé, devenu **standard de fait** pour les données analytiques (data lakes, S3, Snowflake, BigQuery). Beaucoup plus rapide et compact que CSV.

**Avantages vs CSV** :

- 5-20× plus petit (compression colonnaire + dictionnaires).
- 10-100× plus rapide à lire (lectures partielles par colonne).
- Schéma typé (pas de devinettes string vs int).
- Compatible Spark / Dask / Polars / DuckDB nativement.
<!-- #endregion -->

```python
import tempfile
from pathlib import Path

# Round-trip
tmp = Path(tempfile.gettempdir()) / "demo.parquet"
df.to_parquet(tmp, compression="snappy")   # ou "zstd" (meilleure compression)
df_back = pd.read_parquet(tmp, columns=["id", "name"])   # lecture partielle = rapide
print(df_back)
```

<!-- #region -->
### 3.1 Partitionnement
<!-- #endregion -->

<!-- #region -->
Pour de gros datasets, **partitionner** par colonne crée une arborescence qui permet le **pushdown filtering** :

```
data/
├── year=2024/
│   ├── month=01/file.parquet
│   └── month=02/file.parquet
└── year=2025/
    └── month=01/file.parquet
```

```python
# df.to_parquet("data/", partition_cols=["year", "month"])
# df = pd.read_parquet("data/", filters=[("year", "==", 2025)])  # ne lit que les fichiers concernés
```
<!-- #endregion -->

<!-- #region -->
## 4. Stack analytics 2026 — DuckDB en glue
<!-- #endregion -->

<!-- #region -->
**DuckDB** (voir notebook `BDD_DuckDB`) est devenu en 2026 le **couteau suisse** entre pandas/parquet/SQL :

- Lit/écrit pandas / polars / numpy / parquet / CSV / Arrow / JSON.
- SQL ANSI moderne (window funcs, CTE, recursive).
- Beaucoup plus rapide que pandas sur l'analytics.
- **Embedded** (pas de serveur), gratuit, MIT.

```python
"""
import duckdb

# Lit du SQL directement sur un DataFrame pandas
result = duckdb.sql("SELECT class, AVG(age) FROM titanic GROUP BY class").df()

# Ou directement sur un fichier parquet (sans le charger en RAM)
result = duckdb.sql("SELECT * FROM 'data/*.parquet' WHERE year = 2025").df()
"""
```
<!-- #endregion -->

<!-- #region -->
## 5. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 5.1 Connection pooling
<!-- #endregion -->

<!-- #region -->
**Ne jamais** ouvrir une nouvelle connexion par requête en prod. SQLAlchemy gère le pool automatiquement avec `create_engine(...)`. Ajuster `pool_size` et `max_overflow` selon le trafic.

`pool_pre_ping=True` est crucial pour éviter les `OperationalError: server closed connection`.
<!-- #endregion -->

<!-- #region -->
### 5.2 Batch insert
<!-- #endregion -->

<!-- #region -->
- **SQLAlchemy** : `df.to_sql(..., chunksize=10_000)` ou `method='multi'` pour des inserts batch.
- **PostgreSQL** : utiliser `COPY` (10-100× plus rapide que INSERT).
- **MongoDB** : `insert_many` avec `ordered=False`.
- **Toujours** wrap dans une transaction pour la cohérence.
<!-- #endregion -->

<!-- #region -->
### 5.3 Read en streaming
<!-- #endregion -->

<!-- #region -->
Pour les gros tables, ne pas tout charger en RAM :

- `pd.read_sql(query, engine, chunksize=10_000)` → itère sur des chunks DataFrame.
- `duckdb` lit en streaming nativement (pas besoin de chunksize).
- `polars.scan_*()` + `.collect(streaming=True)` pour les fichiers > RAM.
<!-- #endregion -->

<!-- #region -->
### 5.4 Retry transient errors
<!-- #endregion -->

<!-- #region -->
Réseaux instables → utiliser un décorateur retry avec backoff exponentiel :

```python
# pip install tenacity
"""
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.exc import OperationalError

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
       retry=lambda e: isinstance(e, OperationalError))
def query_db(engine, query):
    return pd.read_sql(query, engine)
"""
```
<!-- #endregion -->

<!-- #region -->
### 5.5 Schéma & migrations
<!-- #endregion -->

<!-- #region -->
- Versionner les schémas avec **Alembic** (SQLAlchemy) ou Liquibase.
- Pour les pipelines ML : **DVC**, **dbt** (transformation SQL versionnée) sont des incontournables 2026.
<!-- #endregion -->

<!-- #region -->
## 6. Sources
<!-- #endregion -->

<!-- #region -->
- [SQLAlchemy 2.x docs](https://docs.sqlalchemy.org/en/20/)
- [psycopg3 docs](https://www.psycopg.org/psycopg3/docs/)
- [PyMongo docs](https://pymongo.readthedocs.io/)
- [Apache Parquet — file format spec](https://parquet.apache.org/)
- [DuckDB docs](https://duckdb.org/docs/)
- [dbt docs](https://docs.getdbt.com/)
- Notebooks liés : `Structures_DataFrame`, `BDD_DuckDB`, `BDD_Vectorielles`.
<!-- #endregion -->
