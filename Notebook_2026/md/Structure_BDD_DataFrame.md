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
# 💾 DataFrame ↔ Bases de données (SQL & NoSQL)
<!-- #endregion -->

<!-- #region -->
Cheat-sheet + tutoriel pour **faire transiter des données entre un `pandas.DataFrame` et une base
de données**, dans les deux sens : **écrire** (DataFrame → base) et **lire** (base → DataFrame).

On couvre les backends relationnels (**PostgreSQL** via SQLAlchemy et psycopg 3) et **NoSQL**
(**MongoDB** via pymongo), puis les **outils colonnaires modernes 2026** (ADBC, Polars/connectorx,
DuckDB) qui transfèrent les données *zero-copy* via Apache Arrow. Enfin, les **patterns de
production** : upsert idempotent, bulk insert, streaming, retry.

Toutes les démos exécutables tournent sur **SQLite** (inclus dans Python), **DuckDB** et un
**MongoDB en mémoire** (`mongomock`) — aucun serveur externe requis. Le fil rouge est le dataset
**Titanic**. Les fonctions PostgreSQL sont des snippets réutilisables prêts à pointer vers un vrai
serveur.
<!-- #endregion -->

<!-- #region -->
## 0. Panorama : quel outil pour quel besoin
<!-- #endregion -->

<!-- #region -->
| Outil | Sens | Vitesse | Typage | Cas d'usage privilégié |
|---|---|---|---|---|
| **SQLAlchemy 2.0** (`to_sql`/`read_sql`) | ↔ | moyenne | NumPy (pertes possibles) | Référence portable, multi-dialecte, ORM |
| **psycopg 3** (`COPY`) | ↔ | très rapide (write) | manuel | Bulk insert PostgreSQL massif |
| **ADBC** (Arrow DB Connectivity) | ↔ | rapide | **Arrow fidèle** | Transfert colonnaire zero-copy, nullabilité propre |
| **connectorx** | → (read) | très rapide (parallèle) | Arrow | Lecture massive vers Polars/pandas |
| **Polars** (`read/write_database`) | ↔ | rapide | Arrow | Pipeline DataFrame moderne |
| **DuckDB** | ↔ | rapide | Arrow | SQL analytique *sur* le DataFrame / Parquet |
| **pymongo** | ↔ | moyenne | BSON/JSON | Documents semi-structurés (NoSQL) |

Règle générale 2026 : pour les **gros volumes typés**, préférer **ADBC + `dtype_backend="pyarrow"`**
(transfert colonnaire, pas d'aller-retour par objets Python). Pour la **portabilité** et l'ORM,
SQLAlchemy reste la valeur sûre. Pour l'**analytique locale**, DuckDB lit directement un DataFrame
ou un fichier Parquet sans le charger en RAM.
<!-- #endregion -->

<!-- #region -->
## 1. Setup : imports, dataset fil rouge, configuration
<!-- #endregion -->

<!-- #region -->
On importe les libs communes, on fixe la graine (reproductibilité, `seed=42`), on prépare un dossier
`data/` local pour les bases de démo, et on charge **Titanic** programmatiquement via seaborn
(rien à committer). `matplotlib` est forcé en backend `Agg` (pas de fenêtre interactive).
<!-- #endregion -->

```python
from __future__ import annotations

import json
import time
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend non-interactif : pas de plt.show
import matplotlib.pyplot as plt

# Reproductibilité
RNG_SEED = 42
np.random.seed(RNG_SEED)

# Dossier de travail pour les bases locales de démo
DATA_DIR = Path("data/Structure_BDD_DataFrame")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset fil rouge : Titanic (chargement programmatique)
import seaborn as sns

titanic: pd.DataFrame = sns.load_dataset("titanic")
print("Titanic shape:", titanic.shape)
print(titanic.dtypes)
titanic.head(3)
```

<!-- #region -->
### 1.1 Sécurité des connexions (jamais de mot de passe en dur)
<!-- #endregion -->

<!-- #region -->
**Anti-pattern** : écrire `pwd = "mon_pswd"` en clair dans le notebook (c'était le cas de l'original).
Les secrets doivent venir de **variables d'environnement** ou d'un gestionnaire de secrets.

`pydantic-settings` lit automatiquement la config depuis l'environnement (`$PG_HOST`, `$PG_PASSWORD`…),
valide les types et fournit des valeurs par défaut neutres. On masque toujours le mot de passe avant
de logger une URL de connexion. Les requêtes utilisent des **paramètres liés** (jamais de f-string
avec une valeur utilisateur) → protection contre l'**injection SQL**.
<!-- #endregion -->

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBConfig(BaseSettings):
    """Config de connexion lue depuis l'environnement (préfixe ``PG_``).

    Aucun secret en dur : ``host=$PG_HOST``, ``password=$PG_PASSWORD``, etc.
    Valeurs par défaut volontairement neutres pour la démo locale.
    """

    model_config = SettingsConfigDict(env_prefix="PG_", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    user: str = "app_user"
    password: str = "change-me"
    database: str = "app_db"
    db_schema: str = "public"


def build_pg_url(cfg: DBConfig, *, driver: str = "psycopg") -> str:
    """Construit une URL SQLAlchemy PostgreSQL (driver ``psycopg`` = psycopg 3)."""
    return (
        f"postgresql+{driver}://{cfg.user}:{cfg.password}"
        f"@{cfg.host}:{cfg.port}/{cfg.database}"
    )


def mask_url(url: str) -> str:
    """Masque le mot de passe d'une URL avant de la logger."""
    import re

    return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", url)


cfg = DBConfig()
print("URL (masquée) :", mask_url(build_pg_url(cfg)))
```

<!-- #region -->
La config est désormais externalisée et l'URL loggée ne révèle plus le secret. On peut passer à
l'écriture des données.
<!-- #endregion -->

<!-- #region -->
# 2. Écrire : DataFrame → Base de données
<!-- #endregion -->

<!-- #region -->
## 2.1 SQLAlchemy 2.0 — `to_sql` (la référence portable)
<!-- #endregion -->

<!-- #region -->
`pandas.to_sql` couplé à un `Engine` SQLAlchemy fonctionne sur **n'importe quel dialecte**
(SQLite, PostgreSQL, DuckDB, MySQL…). Points clés de l'API 2.0 :

- `engine.begin()` ouvre une **transaction atomique** (COMMIT automatique en sortie, ROLLBACK si
  exception) — plus sûr que `engine.connect()` seul.
- `method="multi"` regroupe plusieurs lignes par `INSERT` (souvent plus rapide ; sur PostgreSQL on
  préférera `COPY`, cf. 2.2).
- `chunksize` découpe l'écriture pour ne pas saturer la mémoire.
- `index=False` évite d'écrire l'index pandas comme colonne parasite.

> ⚠️ L'original avait un bug : `if_exists=strategy` suivi de `index=False` **sans virgule** →
> `SyntaxError`. Corrigé ici.
<!-- #endregion -->

```python
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine


def insert_dataframe_sqlalchemy(
    df: pd.DataFrame,
    engine: Engine,
    table_name: str,
    *,
    schema: str | None = None,
    if_exists: str = "append",
    chunksize: int = 1_000,
) -> int:
    """Insère un DataFrame via ``pandas.to_sql`` dans une transaction SQLAlchemy 2.0.

    Args:
        df: données à écrire.
        engine: moteur SQLAlchemy (dialecte quelconque : sqlite/postgres/duckdb…).
        table_name: table cible.
        schema: schéma cible (None = défaut du dialecte).
        if_exists: 'fail' | 'replace' | 'append'.
        chunksize: nombre de lignes par batch (insertion multi-lignes).

    Returns:
        Nombre de lignes écrites.
    """
    # Engine.begin() = transaction atomique (COMMIT auto en sortie, ROLLBACK si exception)
    with engine.begin() as conn:
        df.to_sql(
            name=table_name,
            con=conn,
            schema=schema,
            if_exists=if_exists,
            index=False,        # ne pas écrire l'index pandas comme colonne
            method="multi",      # INSERT multi-VALUES (plus rapide que ligne à ligne)
            chunksize=chunksize,
        )
    return len(df)
```

<!-- #region -->
On démontre la fonction sur **SQLite** (inclus dans Python, même API que PostgreSQL via SQLAlchemy) :
on écrit la table `passengers`, puis on relit le nombre de lignes pour vérifier.
<!-- #endregion -->

```python
sqlite_path = DATA_DIR / "titanic.db"
sqlite_path.unlink(missing_ok=True)
engine = create_engine(f"sqlite:///{sqlite_path}")

n_written = insert_dataframe_sqlalchemy(titanic, engine, "passengers", if_exists="replace")
with engine.connect() as conn:
    n_rows = conn.execute(text("SELECT COUNT(*) FROM passengers")).scalar_one()
print(f"SQLAlchemy write → {n_written} lignes ; relu en base : {n_rows}")
```

<!-- #region -->
## 2.2 PostgreSQL — psycopg 3 (création de table + COPY)
<!-- #endregion -->

<!-- #region -->
En 2026, **`psycopg` (version 3)** remplace `psycopg2` (legacy). Pour insérer un gros volume dans
PostgreSQL, **`COPY` est la méthode la plus rapide** : elle court-circuite le parsing ligne-à-ligne
d'`INSERT` (gain typique 10-100×).

On reconstruit ici la logique de l'original (création automatique de table en inférant les types SQL
depuis les dtypes pandas) en y ajoutant l'écriture par `COPY` et des transactions explicites. Le
mapping de types et le DDL sont **validés sur le DataFrame réel** ci-dessous ; l'appel réseau vers un
vrai PostgreSQL se fait hors sandbox.
<!-- #endregion -->

```python
def map_dtype_to_sql(dtype: Any) -> str:
    """Mappe un dtype pandas vers un type SQL PostgreSQL."""
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "TEXT"


def build_create_table_sql(df: pd.DataFrame, table: str, schema: str = "public") -> str:
    """Génère le DDL ``CREATE TABLE`` inféré depuis les dtypes du DataFrame."""
    cols = [f'"{col}" {map_dtype_to_sql(dt)}' for col, dt in df.dtypes.items()]
    return f'CREATE TABLE IF NOT EXISTS "{schema}"."{table}" (\n  ' + ",\n  ".join(cols) + "\n);"


def insert_dataframe_psycopg3_copy(
    df: pd.DataFrame,
    conn: "psycopg.Connection",
    table: str,
    *,
    schema: str = "public",
) -> int:
    """Insertion bulk PostgreSQL la plus rapide : ``COPY`` via psycopg 3.

    ``COPY`` court-circuite le parsing ligne-à-ligne d'``INSERT`` → 10-100× plus rapide
    sur de gros volumes. Requiert un vrai serveur PostgreSQL (hors sandbox).
    """
    cols = list(df.columns)
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    full = f'"{schema}"."{table}"'
    with conn.cursor() as cur:
        cur.execute(build_create_table_sql(df, table, schema))
        with cur.copy(f"COPY {full} ({cols_sql}) FROM STDIN") as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row(row)
    conn.commit()
    return len(df)


import psycopg  # psycopg 3, remplaçant moderne de psycopg2

# Validation de la logique de mapping/DDL sur le DataFrame réel (sans réseau) :
print(build_create_table_sql(titanic, "passengers", schema=cfg.db_schema))
```

<!-- #region -->
Le DDL généré couvre bien tous les types (BIGINT, DOUBLE PRECISION, BOOLEAN, TEXT). On peut le
brancher sur un vrai PostgreSQL en passant une connexion psycopg 3 à `insert_dataframe_psycopg3_copy`.
<!-- #endregion -->

<!-- #region -->
## 2.3 MongoDB — pymongo (`insert_many`)
<!-- #endregion -->

<!-- #region -->
MongoDB stocke des **documents** (proches de dicts JSON). On convertit le DataFrame en liste de
documents via un round-trip JSON qui **neutralise les types NumPy** et transforme `NaN` en `null`
(sinon pymongo refuse les types non-BSON). `insert_many(ordered=False)` continue d'insérer les autres
documents si l'un échoue (résilience + perf), et on insère par **lots** (`batch_size`).

La démo tourne réellement sur un **MongoDB en mémoire** via `mongomock` — aucun serveur requis.
<!-- #endregion -->

```python
from pymongo.collection import Collection


def _records_bson_safe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convertit un DataFrame en documents JSON/BSON-safe (NaN→None, numpy→natif)."""
    # Le round-trip JSON neutralise les types numpy et transforme NaN en null.
    return json.loads(df.to_json(orient="records", date_format="iso"))


def insert_dataframe_mongo(
    df: pd.DataFrame,
    collection: Collection,
    *,
    batch_size: int = 1_000,
) -> int:
    """Insère un DataFrame dans une collection MongoDB par lots.

    ``ordered=False`` : continue les autres documents si l'un échoue (perf + résilience).
    """
    records = _records_bson_safe(df)
    if not records:
        return 0
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.insert_many(batch, ordered=False)
        total += len(batch)
    return total
```

<!-- #region -->
Démo round-trip réellement exécutée via `mongomock` (serveur Mongo en mémoire) : on insère les 891
passagers puis on compte les documents.
<!-- #endregion -->

```python
import mongomock

mongo_client = mongomock.MongoClient()
mongo_db = mongo_client["app_db"]
passengers_coll = mongo_db["passengers"]

n_mongo = insert_dataframe_mongo(titanic, passengers_coll)
print(f"Mongo write → {n_mongo} documents ; count = {passengers_coll.count_documents({})}")
```

<!-- #region -->
# 3. Lire : Base de données → DataFrame
<!-- #endregion -->

<!-- #region -->
## 3.1 SQLAlchemy 2.0 — `read_sql_query` (params liés, streaming)
<!-- #endregion -->

<!-- #region -->
`pandas.read_sql_query` exécute une requête et renvoie un DataFrame. Bonnes pratiques :

- **Paramètres liés** via `text("... WHERE x = :p")` + `params={"p": ...}` → jamais de f-string avec
  une valeur externe (**anti-injection SQL**).
- `chunksize=N` renvoie un **itérateur** de DataFrames pour traiter de gros résultats sans saturer la
  RAM (agrégation incrémentale).
<!-- #endregion -->

```python
def query_to_dataframe_sqlalchemy(
    engine: Engine,
    query: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Exécute une requête paramétrée (anti-injection) et renvoie un DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params or {})


def stream_query_sqlalchemy(
    engine: Engine,
    query: str,
    *,
    chunksize: int = 200,
    params: dict[str, Any] | None = None,
) -> Iterator[pd.DataFrame]:
    """Itère le résultat par chunks (évite de charger toute la table en RAM)."""
    with engine.connect() as conn:
        yield from pd.read_sql_query(
            text(query), conn, params=params or {}, chunksize=chunksize
        )
```

<!-- #region -->
Démo : une requête **paramétrée** (passagers de 3e classe) puis un **agrégat** (taux de survie par
classe) sur la base SQLite remplie en 2.1.
<!-- #endregion -->

```python
df_third = query_to_dataframe_sqlalchemy(
    engine,
    "SELECT survived, fare FROM passengers WHERE pclass = :pc",
    params={"pc": 3},
)
survival_by_class = query_to_dataframe_sqlalchemy(
    engine,
    "SELECT pclass, AVG(survived) AS taux_survie, COUNT(*) AS n "
    "FROM passengers GROUP BY pclass ORDER BY pclass",
)
print("3e classe :", df_third.shape)
survival_by_class
```

<!-- #region -->
## 3.2 PostgreSQL — psycopg 3 (curseur, gros volumes)
<!-- #endregion -->

<!-- #region -->
Avec psycopg 3, on récupère les noms de colonnes via `cursor.description` puis on reconstruit le
DataFrame. Pour de **très gros résultats**, utiliser un **server-side cursor nommé**
(`conn.cursor(name='big')`) qui streame côté serveur sans tout charger côté client. Snippet
réutilisable ; l'exécution réseau est hors sandbox.
<!-- #endregion -->

```python
def query_to_dataframe_psycopg3(
    conn: "psycopg.Connection",
    query: str,
    params: tuple | dict | None = None,
) -> pd.DataFrame:
    """Exécute une requête psycopg 3 et reconstruit un DataFrame depuis le curseur.

    Pour de très gros résultats, utiliser un *server-side cursor* nommé :
    ``conn.cursor(name='big')`` qui streame côté serveur sans tout charger côté client.
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


print("query_to_dataframe_psycopg3 défini :", callable(query_to_dataframe_psycopg3))
```

<!-- #region -->
## 3.3 MongoDB — pymongo (`find` + aggregation)
<!-- #endregion -->

<!-- #region -->
Lecture MongoDB : `find(filter, projection)` filtre les documents et sélectionne les champs
(`{"_id": 0}` exclut l'identifiant interne). Pour de l'**analytique** (group by, moyennes…), le
**pipeline d'agrégation** (`aggregate([...])`) est l'équivalent NoSQL du `GROUP BY` SQL.
<!-- #endregion -->

```python
def query_to_dataframe_mongo(
    collection: Collection,
    filter_: dict[str, Any] | None = None,
    projection: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Lit des documents Mongo en DataFrame (``_id`` exclu par défaut)."""
    proj = projection or {"_id": 0}
    return pd.DataFrame(list(collection.find(filter_ or {}, proj)))


def aggregate_to_dataframe_mongo(
    collection: Collection,
    pipeline: list[dict[str, Any]],
) -> pd.DataFrame:
    """Exécute un pipeline d'agrégation Mongo et renvoie un DataFrame."""
    return pd.DataFrame(list(collection.aggregate(pipeline)))
```

<!-- #region -->
Démo exécutée sur la collection mongomock : un filtre (`sex == "female"`) puis une **agrégation**
(taux de survie par classe), équivalent du `GROUP BY` SQL vu en 3.1.
<!-- #endregion -->

```python
df_women = query_to_dataframe_mongo(
    passengers_coll, {"sex": "female"}, {"_id": 0, "survived": 1, "pclass": 1}
)
agg = aggregate_to_dataframe_mongo(
    passengers_coll,
    [
        {"$group": {"_id": "$pclass", "taux_survie": {"$avg": "$survived"}, "n": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ],
)
print("Femmes :", df_women.shape)
agg
```

<!-- #region -->
# 4. Backends modernes 2026 : Arrow, ADBC, Polars, DuckDB
<!-- #endregion -->

<!-- #region -->
## 4.1 ADBC — le standard colonnaire (zero-copy, typage fidèle)
<!-- #endregion -->

<!-- #region -->
**ADBC (Arrow Database Connectivity)** est le standard 2026 pour échanger des données tabulaires avec
une base. Contrairement aux drivers SQLAlchemy classiques (qui passent par des objets Python ligne à
ligne), ADBC transfère les données en **colonnaire Apache Arrow**, *zero-copy* : c'est plus rapide,
le **typage est préservé fidèlement**, et la **nullabilité** est propre.

En lisant avec `dtype_backend="pyarrow"`, le DataFrame obtenu est **backé par Arrow** (types
`int64[pyarrow]`, `double[pyarrow]`…) — pas de conversion destructive vers le système de types
NumPy. On démontre un round-trip write/read via le driver ADBC SQLite.
<!-- #endregion -->

```python
from adbc_driver_sqlite import dbapi as adbc_sqlite

adbc_path = DATA_DIR / "titanic_adbc.db"
adbc_path.unlink(missing_ok=True)

# Sous-ensemble numérique propre pour un round-trip Arrow bien typé
titanic_num = titanic[["survived", "pclass", "age", "fare"]].copy()

with adbc_sqlite.connect(str(adbc_path)) as adbc_conn:
    # WRITE via ADBC
    titanic_num.to_sql("passengers", adbc_conn, if_exists="replace", index=False)
    # READ via ADBC avec backend Arrow : typage fidèle + nullabilité propre, zero-copy
    df_arrow = pd.read_sql(
        "SELECT * FROM passengers", adbc_conn, dtype_backend="pyarrow"
    )

print("ADBC read dtypes (pyarrow-backed) :")
print(df_arrow.dtypes)
df_arrow.head(3)
```

<!-- #region -->
## 4.2 Polars — `read_database` / `write_database` (+ connectorx)
<!-- #endregion -->

<!-- #region -->
**Polars** est un DataFrame moderne (Rust, multi-thread, Arrow natif). `write_database` écrit via
SQLAlchemy ; `read_database_uri` lit via **connectorx**, un moteur de lecture **parallèle** très
rapide qui rapatrie le résultat directement en Arrow (idéal pour de gros `SELECT`).
<!-- #endregion -->

```python
import polars as pl

pl_path = DATA_DIR / "titanic_polars.db"
pl_path.unlink(missing_ok=True)
pl_df = pl.from_pandas(titanic_num)

# WRITE : Polars.write_database (s'appuie sur SQLAlchemy)
pl_df.write_database(
    "passengers",
    connection=f"sqlite:///{pl_path}",
    if_table_exists="replace",
)

# READ rapide via connectorx (lecture parallèle, zero-copy Arrow)
pl_back = pl.read_database_uri(
    "SELECT pclass, AVG(survived) AS taux FROM passengers GROUP BY pclass ORDER BY pclass",
    uri=f"sqlite://{pl_path}",
)
print("Polars read (connectorx) :")
pl_back
```

<!-- #region -->
## 4.3 DuckDB — SQL directement sur le DataFrame (zero-copy)
<!-- #endregion -->

<!-- #region -->
**DuckDB** est une base analytique *in-process* : elle exécute du **SQL directement sur un DataFrame
pandas** (référencé par son nom de variable, zero-copy via Arrow) sans étape d'import. Elle lit aussi
un fichier **Parquet** sans le charger entièrement en RAM (lecture colonnaire + prédicat pushdown).
C'est la « glue » analytique idéale entre pandas et le SQL.
<!-- #endregion -->

```python
import duckdb

# DuckDB lit un DataFrame pandas par son nom de variable (zero-copy via Arrow)
duck_res = duckdb.sql(
    """
    SELECT pclass,
           COUNT(*)        AS n,
           ROUND(AVG(fare), 2) AS fare_moyen,
           ROUND(AVG(survived), 3) AS taux_survie
    FROM titanic
    GROUP BY pclass
    ORDER BY pclass
    """
).df()
print("DuckDB sur DataFrame :")
print(duck_res)

# Round-trip Parquet (format colonnaire de référence)
parquet_path = DATA_DIR / "titanic.parquet"
titanic.to_parquet(parquet_path)
duck_parquet = duckdb.sql(
    f"SELECT COUNT(*) AS n, AVG(age) AS age_moyen FROM '{parquet_path}'"
).df()
print("DuckDB sur Parquet :")
duck_parquet
```

<!-- #region -->
# 5. Patterns de production (robustesse & performance)
<!-- #endregion -->

<!-- #region -->
## 5.1 Upsert — insertion idempotente (ON CONFLICT)
<!-- #endregion -->

<!-- #region -->
Un **upsert** insère les nouvelles lignes et **met à jour** celles déjà présentes (conflit sur une
clé). C'est la base de l'**idempotence** : rejouer le même chargement ne crée pas de doublons.
Syntaxe `INSERT … ON CONFLICT(clé) DO UPDATE` (SQLite & PostgreSQL ; ailleurs : `MERGE`).
<!-- #endregion -->

```python
def upsert_dataframe_sqlite(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    table: str,
    key_cols: list[str],
) -> int:
    """Upsert idempotent (INSERT … ON CONFLICT DO UPDATE) sur SQLite.

    Les lignes en conflit sur ``key_cols`` sont mises à jour ; les autres insérées.
    Même logique qu'``ON CONFLICT`` Postgres / ``MERGE`` ailleurs.
    """
    cols = list(df.columns)
    placeholders = ", ".join("?" for _ in cols)
    col_list = ", ".join(f'"{c}"' for c in cols)
    updates = ", ".join(f'"{c}"=excluded."{c}"' for c in cols if c not in key_cols)
    conflict = ", ".join(f'"{c}"' for c in key_cols)
    sql = (
        f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders}) '
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}"
    )
    conn.executemany(sql, df.itertuples(index=False, name=None))
    conn.commit()
    return len(df)


# Démo : on insère 3 tarifs, puis on upsert (pid 2 mis à jour, pid 4 ajouté)
raw = sqlite3.connect(DATA_DIR / "upsert_demo.db")
raw.execute("DROP TABLE IF EXISTS fares")
raw.execute('CREATE TABLE fares (pid INTEGER PRIMARY KEY, fare REAL)')
seed = pd.DataFrame({"pid": [1, 2, 3], "fare": [7.25, 71.28, 8.05]})
upsert_dataframe_sqlite(seed, raw, "fares", ["pid"])
update = pd.DataFrame({"pid": [2, 4], "fare": [99.99, 13.0]})
upsert_dataframe_sqlite(update, raw, "fares", ["pid"])
result_upsert = pd.read_sql_query("SELECT * FROM fares ORDER BY pid", raw)
raw.close()
result_upsert
```

<!-- #region -->
`pid=2` a bien été mis à jour (71.28 → 99.99) et `pid=4` inséré, sans doublon : l'opération est
idempotente.
<!-- #endregion -->

<!-- #region -->
## 5.2 Performance d'insertion — benchmark chiffré
<!-- #endregion -->

<!-- #region -->
La leçon vaut **pour tous les backends** : grouper les écritures dans **une seule transaction** et
utiliser `executemany` écrase l'insertion ligne-à-ligne avec un `COMMIT` par ligne (chaque commit
force un `fsync` disque). On chronomètre les deux approches sur Titanic répliqué et on trace le
résultat (palette CHART, couleurs neutres car ce sont des temps).
<!-- #endregion -->

```python
def _bench_row_by_row(rows: list[tuple]) -> float:
    """Pire cas : un INSERT + un COMMIT par ligne (aller-retours disque répétés)."""
    p = DATA_DIR / "bench_row.db"
    p.unlink(missing_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE t (survived INT, pclass INT, age REAL, fare REAL)")
    t0 = time.perf_counter()
    for r in rows:
        conn.execute("INSERT INTO t VALUES (?,?,?,?)", r)
        conn.commit()  # commit par ligne = fsync à chaque insert → très lent
    dt = time.perf_counter() - t0
    conn.close()
    p.unlink(missing_ok=True)
    return dt


def _bench_executemany(rows: list[tuple]) -> float:
    """Bonne pratique : executemany dans UNE seule transaction."""
    p = DATA_DIR / "bench_many.db"
    p.unlink(missing_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE t (survived INT, pclass INT, age REAL, fare REAL)")
    t0 = time.perf_counter()
    conn.executemany("INSERT INTO t VALUES (?,?,?,?)", rows)
    conn.commit()  # un seul commit pour tout le lot
    dt = time.perf_counter() - t0
    conn.close()
    p.unlink(missing_ok=True)
    return dt


# Volume mesurable mais court : Titanic répliqué (~1.8k lignes), valeurs natives SQLite
big = pd.concat([titanic_num] * 2, ignore_index=True)
rows = [tuple(None if pd.isna(v) else v for v in r)
        for r in big.itertuples(index=False, name=None)]
timings = {
    "ligne à ligne\n(commit/ligne)": _bench_row_by_row(rows),
    "executemany\n(1 transaction)": _bench_executemany(rows),
}
speedup = timings["ligne à ligne\n(commit/ligne)"] / timings["executemany\n(1 transaction)"]
print(f"Benchmark sur {len(rows):,} lignes :",
      {k.replace(chr(10), ' '): round(v, 4) for k, v in timings.items()},
      f"→ ×{speedup:.0f} plus rapide")

# Palette CHART (séries de temps neutres → couleurs neutres)
PRIMARY_1, LAVENDER = "#00798c", "#9d83b8"
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(list(timings.keys()), list(timings.values()), color=[PRIMARY_1, LAVENDER])
ax.set_ylabel("Temps (s)")
ax.set_title(f"Insertion SQLite ({len(rows):,} lignes) — batching vs ligne-à-ligne")
for i, v in enumerate(timings.values()):
    ax.text(i, v, f"{v:.3f}s", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(DATA_DIR / "benchmark_insert.png", dpi=110)
plt.show()
```

<!-- #region -->
L'écart est massif (plusieurs ordres de grandeur) : ne **jamais** committer ligne par ligne. Sur
PostgreSQL, pousser la logique encore plus loin avec `COPY` (cf. 2.2).
<!-- #endregion -->

<!-- #region -->
## 5.3 Streaming — lire/écrire sans saturer la RAM
<!-- #endregion -->

<!-- #region -->
Pour des tables qui ne tiennent pas en mémoire, on lit par **chunks** (`chunksize`) et on agrège
**incrémentalement** chunk après chunk. La fonction `stream_query_sqlalchemy` (3.1) renvoie un
itérateur ; ici on se contente de compter, mais le pattern s'étend à toute agrégation incrémentale.
<!-- #endregion -->

```python
def count_rows_streaming(engine: Engine, query: str, *, chunksize: int = 200) -> int:
    """Agrège un résultat SQL par chunks sans jamais tout charger en mémoire."""
    total = 0
    for chunk in stream_query_sqlalchemy(engine, query, chunksize=chunksize):
        total += len(chunk)  # ici simple comptage ; en réel : agrégation incrémentale
    return total


n_stream = count_rows_streaming(engine, "SELECT * FROM passengers")
print(f"Streaming : {n_stream} lignes parcourues par chunks de 200")
```

<!-- #region -->
## 5.4 Retry — encaisser les coupures réseau (tenacity)
<!-- #endregion -->

<!-- #region -->
Une connexion DB peut échouer transitoirement (timeout, réseau, basculement). `tenacity` ré-essaie
automatiquement avec un **backoff exponentiel**, en ne ciblant que les erreurs **transitoires** (pas
une erreur SQL définitive) et en respectant l'**idempotence** de l'opération. On simule ici une
connexion qui échoue deux fois puis réussit.
<!-- #endregion -->

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class TransientDBError(Exception):
    """Erreur réseau/DB transitoire (timeout, connexion refusée…)."""


@retry(
    retry=retry_if_exception_type(TransientDBError),
    wait=wait_exponential(multiplier=0.01, max=0.1),  # backoff court pour la démo
    stop=stop_after_attempt(5),
    reraise=True,
)
def fetch_with_retry(state: dict[str, int]) -> str:
    """Simule une connexion qui échoue 2 fois puis réussit (backoff exponentiel)."""
    state["calls"] += 1
    if state["calls"] < 3:
        raise TransientDBError(f"tentative {state['calls']} échouée")
    return "connecté"


state = {"calls": 0}
result = fetch_with_retry(state)
print(f"Retry : '{result}' après {state['calls']} tentatives")

# Nettoyage des connexions
engine.dispose()
mongo_client.close()
```

<!-- #region -->
## 6. Récapitulatif — quel outil quand
<!-- #endregion -->

<!-- #region -->
| Besoin | Outil recommandé |
|---|---|
| Portabilité multi-base, ORM | **SQLAlchemy 2.0** (`to_sql` / `read_sql_query`) |
| Insert massif PostgreSQL | **psycopg 3** + `COPY` |
| Transfert typé zero-copy (gros volumes) | **ADBC** + `dtype_backend="pyarrow"` |
| Lecture massive parallèle vers DataFrame | **connectorx** (via Polars) |
| Pipeline DataFrame moderne | **Polars** `read/write_database` |
| SQL analytique sur DataFrame / Parquet | **DuckDB** |
| Documents semi-structurés (NoSQL) | **pymongo** (`insert_many` / `aggregate`) |
| Chargement rejouable | **upsert** `ON CONFLICT` |
| Robustesse réseau | **tenacity** (retry + backoff) |
| Gros volumes en RAM limitée | **chunksize** (streaming) |

Réflexes transverses : secrets en **variables d'environnement**, requêtes **paramétrées**
(anti-injection), écritures **groupées en transaction**.
<!-- #endregion -->

<!-- #region -->
### Sources
<!-- #endregion -->

<!-- #region -->
- pandas — `read_sql` & ADBC : <https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html>
- Apache Arrow ADBC : <https://arrow.apache.org/adbc/>
- SQLAlchemy 2.0 : <https://docs.sqlalchemy.org/en/20/>
- psycopg 3 (`COPY`) : <https://www.psycopg.org/psycopg3/docs/basic/copy.html>
- Polars `read_database` : <https://docs.pola.rs/api/python/stable/reference/api/polars.read_database_uri.html>
- DuckDB + pandas/Parquet : <https://duckdb.org/docs/guides/python/import_pandas>
- tenacity : <https://tenacity.readthedocs.io/>
<!-- #endregion -->
