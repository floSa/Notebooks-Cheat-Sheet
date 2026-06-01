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
# 🦆 DuckDB — SQL analytique in-process
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Cheat-sheet + Wiki** sur **DuckDB**, le moteur SQL analytique *in-process*.

**Ce que couvre ce notebook** :

- Les trois façons d'exécuter du SQL en Python (`duckdb.sql()`, connexion fichier, DB-API 2.0).
- Requêter des DataFrames pandas / Polars et des fichiers Parquet / CSV **sans étape d'import**.
- Le **Friendly SQL** de DuckDB (`GROUP BY ALL`, `SELECT * EXCLUDE`, `COLUMNS()`, syntaxe FROM-first).
- L'**API relationnelle** paresseuse et l'**interopérabilité** pandas / Polars / Arrow / NumPy.
- Un **benchmark** pandas vs DuckDB vs SQLite.
- **JupySQL** : écrire du SQL natif dans une cellule via les magics `%sql` / `%%sql`.

**Dataset** : NYC Yellow Taxi (janvier 2024, Parquet ~50 MB, téléchargé programmatiquement et mis en cache dans `data/BDD_DuckDB/`) + table de correspondance des zones. Un DataFrame synthétique sert au benchmark.

**Version** : DuckDB 1.5.x (2026).
<!-- #endregion -->

<!-- #region -->
## 1. À propos : OLAP in-process
<!-- #endregion -->

<!-- #region -->
**DuckDB** est à l'analytique (OLAP) ce que **SQLite** est au transactionnel (OLTP) : une base **in-process**, sans serveur, embarquée dans le processus Python — mais conçue pour les **requêtes analytiques** sur de gros volumes.

Ses caractéristiques clés :

- **Moteur colonnaire vectorisé** : les données sont traitées par blocs de colonnes, ce qui accélère drastiquement les agrégations, filtres et jointures par rapport à un moteur ligne à ligne.
- **Zéro déploiement** : `pip install duckdb`, aucune configuration de serveur.
- **Lecture directe** de fichiers Parquet / CSV / JSON, locaux ou distants (S3), sans étape de chargement.
- **Intégration native** avec pandas, Polars et PyArrow (zéro copie via Arrow).

Lancé en open-source en 2019, DuckDB a atteint sa version **1.0 en juin 2024** et est en **1.5.x** en 2026 (extension DuckLake en production, support Polars LazyFrame avec pushdown, types VARIANT).
<!-- #endregion -->

<!-- #region -->
| | OLTP (SQLite, PostgreSQL) | OLAP (DuckDB) |
|---|---|---|
| Charge typique | Beaucoup de petites transactions (INSERT/UPDATE) | Quelques grosses requêtes analytiques (SELECT, agrégats) |
| Stockage | Orienté **ligne** | Orienté **colonne** |
| Cas d'usage | Application, backend web | Analyse, reporting, data science |
| Concurrence écriture | Forte | Faible (1 écrivain) |
<!-- #endregion -->

<!-- #region -->
## 2. Installation et imports
<!-- #endregion -->

<!-- #region -->
Installation via uv : `uv add duckdb pyarrow polars`. On centralise ici les imports et la palette graphique réutilisée pour le benchmark. Le premier import de `duckdb` ne télécharge rien (binaire embarqué).
<!-- #endregion -->

```python
from __future__ import annotations

import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import duckdb
import numpy as np
import pandas as pd

# Palette graphique du projet (réutilisée pour le benchmark)
PALETTE = ["#00798c", "#d1495b", "#edae49", "#66a182", "#2e4057",
           "#9d83b8", "#b8848e", "#c9b78b"]
DATA_DIR = Path("data/BDD_DuckDB")

print("DuckDB version:", duckdb.__version__)
```

<!-- #region -->
## 3. Trois façons d'exécuter du SQL
<!-- #endregion -->

<!-- #region -->
DuckDB expose trois entrées complémentaires :

- `duckdb.sql(...)` : exécute sur une **connexion in-memory globale** implicite et renvoie une *relation* paresseuse. Pratique pour l'exploration.
- `duckdb.connect(chemin)` : ouvre une **base fichier persistante** (les tables survivent au programme).
- Le **curseur DB-API 2.0** (PEP 249) : API bas niveau `execute` / `fetchone` / `fetchall`, identique à `sqlite3`.

Commençons par `duckdb.sql()` : la relation renvoyée s'affiche directement.
<!-- #endregion -->

```python
answer = duckdb.sql("SELECT 42 AS answer, 'duckdb' AS engine")
answer
```

<!-- #region -->
**Connexion persistante** : `duckdb.connect(path)` crée (ou rouvre) un fichier `.duckdb`. Les objets créés y sont conservés sur disque.
<!-- #endregion -->

```python
DATA_DIR.mkdir(parents=True, exist_ok=True)
demo_con = duckdb.connect(str(DATA_DIR / "demo.duckdb"))
demo_con.execute("CREATE OR REPLACE TABLE t AS SELECT * FROM range(5) AS r(n)")
print("Table persistante t:", demo_con.execute("SELECT * FROM t").fetchall())
demo_con.close()
```

<!-- #region -->
**Curseur DB-API 2.0** : même interface que `sqlite3` (utile pour du code générique compatible PEP 249). Ici sur une base in-memory (`connect()` sans argument).
<!-- #endregion -->

```python
dbapi_con = duckdb.connect()  # in-memory
cur = dbapi_con.cursor()
cur.execute("SELECT n, n * n AS square FROM range(4) AS r(n)")
print("DB-API fetchall:", cur.fetchall())
dbapi_con.close()
```

<!-- #region -->
## 4. Le jeu de données NYC Yellow Taxi
<!-- #endregion -->

<!-- #region -->
On utilise les trajets de taxis jaunes de New York (janvier 2024) au format Parquet. Un helper télécharge le fichier **une seule fois** (cache local) ainsi que la table de correspondance des zones (`LocationID` → `Borough` / `Zone`).
<!-- #endregion -->

```python
def download_if_missing(url: str, dest: Path) -> Path:
    """Télécharge `url` vers `dest` si le fichier n'existe pas déjà.

    Args:
        url: URL source (Parquet / CSV).
        dest: chemin local de destination.

    Returns:
        Le chemin local `dest` (téléchargé ou déjà présent).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Téléchargement {url} -> {dest} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

taxi_path = download_if_missing(TAXI_URL, DATA_DIR / "yellow_tripdata_2024-01.parquet")
zones_path = download_if_missing(ZONES_URL, DATA_DIR / "taxi_zone_lookup.csv")
print("Parquet:", taxi_path, f"{taxi_path.stat().st_size / 1e6:.1f} MB")
```

<!-- #region -->
On crée deux **vues** sur la connexion in-memory par défaut (`trips` = le Parquet, `zones` = le CSV). Elles seront réutilisées dans toutes les cellules suivantes. L'aperçu confirme la lecture directe du Parquet, sans import préalable.
<!-- #endregion -->

```python
preview = duckdb.sql(f"SELECT * FROM read_parquet('{taxi_path}') LIMIT 5").df()

# Vues partagées sur la connexion in-memory par défaut (réutilisées partout)
duckdb.sql(f"CREATE OR REPLACE VIEW trips AS SELECT * FROM read_parquet('{taxi_path}')")
duckdb.sql(f"CREATE OR REPLACE VIEW zones AS SELECT * FROM read_csv('{zones_path}')")
print("Nb trajets:", duckdb.sql("SELECT count(*) FROM trips").fetchone()[0])
preview
```

<!-- #region -->
## 5. Requêter sans importer
<!-- #endregion -->

<!-- #region -->
La fonctionnalité signature de DuckDB : interroger un **DataFrame pandas par son nom de variable** ou un **fichier sur disque par son chemin**, sans jamais charger les données dans une table. DuckDB voit les variables Python du scope courant comme des tables.
<!-- #endregion -->

```python
df = duckdb.sql("SELECT * FROM trips USING SAMPLE 50000 ROWS").df()  # échantillon pandas

duckdb.sql("""
    SELECT passenger_count, COUNT(*) AS n
    FROM df                       -- 'df' (variable Python) vue comme une table
    WHERE passenger_count IS NOT NULL
    GROUP BY ALL
    ORDER BY passenger_count
""").df()
```

<!-- #region -->
Même chose directement sur le fichier Parquet, sans aucune variable intermédiaire : `read_parquet(chemin)` s'utilise comme une table.
<!-- #endregion -->

```python
duckdb.sql(f"""
    SELECT PULocationID, COUNT(*) AS pickups
    FROM read_parquet('{taxi_path}')
    GROUP BY ALL
    ORDER BY pickups DESC
    LIMIT 5
""").df()
```

<!-- #region -->
## 6. Bases persistantes et tables
<!-- #endregion -->

<!-- #region -->
Pour matérialiser les données (plutôt qu'une vue recalculée à chaque requête), on crée une **table** dans une base fichier. On en profite pour ne garder que les colonnes utiles et filtrer les lignes aberrantes.
<!-- #endregion -->

```python
con = duckdb.connect(str(DATA_DIR / "taxi.duckdb"))
con.execute(f"""
    CREATE OR REPLACE TABLE trips AS
    SELECT VendorID, tpep_pickup_datetime, passenger_count, trip_distance,
           PULocationID, DOLocationID, payment_type, fare_amount,
           tip_amount, total_amount
    FROM read_parquet('{taxi_path}')
    WHERE trip_distance > 0 AND total_amount > 0
""")
con.execute(f"CREATE OR REPLACE TABLE zones AS SELECT * FROM read_csv('{zones_path}')")
print("Tables créées dans taxi.duckdb")
```

<!-- #region -->
`SHOW TABLES` liste les tables ; `SUMMARIZE` donne un profil statistique rapide (min, max, cardinalité approchée) — très pratique pour une première inspection.
<!-- #endregion -->

```python
print(con.sql("SHOW TABLES").df().to_string())
con.sql("SUMMARIZE trips").df()[["column_name", "min", "max", "approx_unique"]]
```

<!-- #region -->
## 7. Requêtes : du simple à l'agrégat
<!-- #endregion -->

<!-- #region -->
On déroule une progression classique : **SELECT + WHERE + ORDER**, puis agrégat, puis jointure, puis sous-requête. D'abord les trajets au montant le plus élevé (en écartant les distances aberrantes).
<!-- #endregion -->

```python
con.sql("""
    SELECT trip_distance, fare_amount, tip_amount, total_amount
    FROM trips
    WHERE trip_distance < 100          -- on écarte les distances aberrantes
    ORDER BY total_amount DESC
    LIMIT 5
""").df()
```

<!-- #region -->
**Agrégat** : nombre de trajets et tarif moyen par nombre de passagers. `GROUP BY ALL` regroupe automatiquement sur toutes les colonnes non agrégées du SELECT.
<!-- #endregion -->

```python
con.sql("""
    SELECT passenger_count,
           COUNT(*)            AS n,
           ROUND(AVG(fare_amount), 2) AS fare_moyen
    FROM trips
    WHERE passenger_count BETWEEN 1 AND 6
    GROUP BY ALL
    ORDER BY passenger_count
""").df()
```

<!-- #region -->
**Jointure** avec la table des zones pour traduire `PULocationID` en *borough*, et classer les zones de prise en charge par volume.
<!-- #endregion -->

```python
con.sql("""
    SELECT z.Borough,
           COUNT(*)                    AS pickups,
           ROUND(AVG(t.total_amount), 2) AS panier_moyen
    FROM trips t
    JOIN zones z ON t.PULocationID = z.LocationID
    GROUP BY ALL
    ORDER BY pickups DESC
    LIMIT 6
""").df()
```

<!-- #region -->
**Sous-requête / pourcentage** : part des trajets « longs » (> 5 miles) par mode de paiement. On calcule total et compte conditionnel dans une sous-requête, puis le ratio à l'extérieur. On ferme la connexion fichier ensuite pour libérer le verrou.
<!-- #endregion -->

```python
pct_longs = con.sql("""
    SELECT payment_type,
           total_trajets                            AS n_trajets,
           ROUND(longs / total_trajets * 100, 1)    AS pct_longs
    FROM (
        SELECT payment_type,
               1.0 * COUNT(*)                                         AS total_trajets,
               1.0 * SUM(CASE WHEN trip_distance > 5 THEN 1 ELSE 0 END) AS longs
        FROM trips
        GROUP BY payment_type
    )
    ORDER BY pct_longs DESC
""").df()
con.close()  # libère le fichier taxi.duckdb
pct_longs
```

<!-- #region -->
## 8. Friendly SQL
<!-- #endregion -->

<!-- #region -->
DuckDB ajoute du sucre syntaxique qui rend le SQL plus concis (et que d'autres moteurs ont depuis adopté) :

- `GROUP BY ALL` : regroupe sur toutes les colonnes non agrégées.
- `SELECT * EXCLUDE (col, ...)` : tout sauf certaines colonnes ; `REPLACE (expr AS col)` pour en remplacer.
- `COLUMNS('regex')` : applique une expression à toutes les colonnes matchant un motif.
- **Syntaxe FROM-first** : `FROM t SELECT ...`, dans l'ordre logique d'exécution.

Ici on sélectionne tout sauf deux colonnes, puis on moyenne d'un coup toutes les colonnes `*amount`.
<!-- #endregion -->

```python
exclude_demo = duckdb.sql("""
    SELECT * EXCLUDE (VendorID, store_and_fwd_flag)
    FROM trips
    LIMIT 3
""").df()
print("Colonnes après EXCLUDE:", list(exclude_demo.columns))

duckdb.sql("""
    SELECT ROUND(AVG(COLUMNS('.*amount')), 2)   -- s'applique à fare/tip/tolls/total_amount
    FROM trips
    WHERE total_amount > 0
""").df()
```

<!-- #region -->
La syntaxe **FROM-first** met la source en premier : pratique pour écrire les requêtes dans l'ordre où elles sont évaluées.
<!-- #endregion -->

```python
duckdb.sql("""
    FROM trips
    SELECT payment_type, COUNT(*) AS n
    GROUP BY ALL
    ORDER BY n DESC
""").df()
```

<!-- #region -->
## 9. API relationnelle (lazy)
<!-- #endregion -->

<!-- #region -->
Plutôt que des chaînes SQL, on peut construire une requête par **chaînage de méthodes Python**. Les opérations sont **paresseuses** : rien n'est calculé tant qu'une méthode terminale (`.df()`, `.fetchall()`...) n'est pas appelée, ce qui laisse DuckDB optimiser le plan global.
<!-- #endregion -->

```python
rel = duckdb.read_parquet(str(taxi_path))            # relation paresseuse
(
    rel.filter("trip_distance > 5 AND total_amount > 0")
       .aggregate("payment_type, ROUND(avg(total_amount), 2) AS panier_moyen")
       .order("panier_moyen DESC")
).df()                                                # .df() déclenche l'exécution
```

<!-- #region -->
## 10. Parquet et pushdown
<!-- #endregion -->

<!-- #region -->
DuckDB ne lit que ce dont il a besoin dans un Parquet : **projection pushdown** (seulement les colonnes demandées) et **predicate pushdown** (filtres appliqués au plus tôt, row-groups ignorés). `EXPLAIN` montre le plan : on y voit les `Projections` et `Filters` poussés jusqu'au scan, et une estimation du nombre de lignes après filtre.
<!-- #endregion -->

```python
plan = duckdb.sql(f"""
    EXPLAIN
    SELECT VendorID, total_amount
    FROM read_parquet('{taxi_path}')
    WHERE trip_distance > 10
""").fetchone()[1]
print(plan)
```

<!-- #region -->
## 11. Interopérabilité : pandas, Polars, Arrow, NumPy
<!-- #endregion -->

<!-- #region -->
Un résultat DuckDB se matérialise dans le format voulu : `.df()` (pandas), `.pl()` (Polars), `.fetch_arrow_table()` (PyArrow), `.fetchnumpy()` (dict NumPy). L'échange passe par Apache Arrow, souvent sans copie.
<!-- #endregion -->

```python
q = duckdb.sql("SELECT payment_type, COUNT(*) AS n FROM trips GROUP BY ALL ORDER BY 1")
as_pandas = q.df()
as_polars = q.pl()
as_arrow = q.fetch_arrow_table()      # pyarrow.Table
as_numpy = q.fetchnumpy()             # dict de colonnes NumPy
print("pandas :", type(as_pandas).__name__, as_pandas.shape)
print("polars :", type(as_polars).__name__, as_polars.shape)
print("arrow  :", type(as_arrow).__name__, as_arrow.num_rows, "rows")
print("numpy  :", {k: v.dtype.name for k, v in as_numpy.items()})
```

<!-- #region -->
L'inverse fonctionne aussi : DuckDB peut requêter un **LazyFrame Polars** directement, avec pushdown des filtres et projections vers Polars.
<!-- #endregion -->

```python
import polars as pl

lf = pl.scan_parquet(str(taxi_path))            # LazyFrame (non matérialisé)
n_via_polars = duckdb.sql("SELECT COUNT(*) AS n FROM lf").fetchone()[0]
print("Lignes vues via le LazyFrame Polars:", n_via_polars)
```

<!-- #region -->
## 12. Benchmark : pandas vs DuckDB vs SQLite
<!-- #endregion -->

<!-- #region -->
On reprend l'expérience historique du notebook (agrégation filtrée sur un gros DataFrame) avec un harnais propre : un context manager `timer` et une fonction `benchmark` qui prend le **temps médian** sur plusieurs exécutions.
<!-- #endregion -->

```python
import matplotlib.pyplot as plt


@contextmanager
def timer() -> Iterator[Callable[[], float]]:
    """Context manager mesurant le temps écoulé (s) du bloc englobé.

    Usage:
        with timer() as elapsed:
            ...
        print(elapsed())
    """
    start = time.perf_counter()
    end: list[float] = []
    yield lambda: (end[0] if end else time.perf_counter() - start)
    end.append(time.perf_counter() - start)


def benchmark(fn: Callable[[], object], repeats: int = 3) -> float:
    """Renvoie le temps médian (s) sur `repeats` exécutions de `fn`."""
    times: list[float] = []
    for _ in range(repeats):
        with timer() as elapsed:
            fn()
        times.append(elapsed())
    return float(np.median(times))
```

<!-- #region -->
On génère un DataFrame synthétique (5M lignes, seed fixe pour la reproductibilité). Les bornes de longitude sont correctes (`low < high`) — le notebook d'origine les avait inversées.
<!-- #endregion -->

```python
rng = np.random.default_rng(42)
num_rows = 5_000_000
synth = pd.DataFrame({
    "pickup_longitude": rng.uniform(low=-74.05, high=-73.75, size=num_rows),
    "trip_duration": rng.normal(loc=10.0, scale=5.0, size=num_rows),
})
print("Synthétique:", synth.shape)
```

<!-- #region -->
Trois implémentations de la **même** agrégation (moyenne d'une métrique sous condition) : pandas (masque booléen), DuckDB (requête directe sur le DataFrame), SQLite (table in-memory).
<!-- #endregion -->

```python
import sqlite3

THRESHOLD = -73.95


def agg_pandas() -> float:
    return synth.loc[synth["pickup_longitude"] < THRESHOLD, "trip_duration"].mean()


def agg_duckdb() -> float:
    return duckdb.sql(
        f"SELECT AVG(trip_duration) FROM synth WHERE pickup_longitude < {THRESHOLD}"
    ).fetchone()[0]


_sqlite = sqlite3.connect(":memory:")
synth.to_sql("synth", _sqlite, index=False, if_exists="replace")


def agg_sqlite() -> float:
    cur = _sqlite.execute(
        f"SELECT AVG(trip_duration) FROM synth WHERE pickup_longitude < {THRESHOLD}"
    )
    return cur.fetchone()[0]
```

<!-- #region -->
On mesure les trois et on récapitule dans un DataFrame trié par temps croissant.
<!-- #endregion -->

```python
results = pd.DataFrame({
    "moteur": ["pandas", "DuckDB", "SQLite"],
    "temps_s": [benchmark(agg_pandas), benchmark(agg_duckdb), benchmark(agg_sqlite)],
}).sort_values("temps_s")
_sqlite.close()
results
```

<!-- #region -->
Visualisation en barres horizontales (palette projet, une couleur par moteur).
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh(results["moteur"], results["temps_s"], color=PALETTE[:3])
ax.set_xlabel("Temps médian (s) — plus court = mieux")
ax.set_title("Agrégation filtrée sur 5M lignes : pandas vs DuckDB vs SQLite")
ax.invert_yaxis()
for y, v in enumerate(results["temps_s"]):
    ax.text(v, y, f" {v:.3f}s", va="center")
fig.tight_layout()
plt.show()
```

<!-- #region -->
## 13. JupySQL : du SQL natif en cellule
<!-- #endregion -->

<!-- #region -->
**JupySQL** (maintenu par Ploomber, successeur d'`ipython-sql`) ajoute les magics `%sql` (une ligne) et `%%sql` (cellule entière) pour écrire du SQL directement, se brancher sur DuckDB (ou Postgres/MySQL) et même tracer des graphiques.

Installation : `uv add jupysql duckdb-engine`.

Dans un notebook, on écrit normalement `%load_ext sql`, `%sql duckdb://`, etc. Ici les cellules utilisent la forme programmatique équivalente `get_ipython().run_line_magic(...)` / `run_cell_magic(...)` afin de rester du Python valide et portable — l'effet est identique.
<!-- #endregion -->

```python
get_ipython().run_line_magic("load_ext", "sql")           # équivaut à : %load_ext sql
get_ipython().run_line_magic("config", "SqlMagic.displaylimit = 10")
get_ipython().run_line_magic("config", "SqlMagic.feedback = 0")
```

<!-- #region -->
On ouvre une connexion DuckDB in-memory (`%sql duckdb://`) puis on recrée la vue `trips` à partir du Parquet dans cette connexion JupySQL (indépendante de la connexion `duckdb.sql` plus haut).
<!-- #endregion -->

```python
get_ipython().run_line_magic("sql", "duckdb://")          # équivaut à : %sql duckdb://
get_ipython().run_cell_magic(
    "sql", "",
    f"CREATE OR REPLACE VIEW trips AS SELECT * FROM read_parquet('{taxi_path}')",
)
```

<!-- #region -->
Requête sur une ligne avec `%sql` (le notebook d'origine avait ici une requête cassée, avec un `FROM` vide — on la corrige).
<!-- #endregion -->

```python
get_ipython().run_line_magic(
    "sql", "SELECT payment_type, count(*) AS n FROM trips GROUP BY ALL ORDER BY n DESC"
)
```

<!-- #region -->
Requête multi-lignes avec `%%sql` (cellule entière) : le résultat est un `ResultSet` convertible en DataFrame via `.DataFrame()`.
<!-- #endregion -->

```python
get_ipython().run_cell_magic(
    "sql", "",
    "SELECT passenger_count, round(avg(total_amount), 2) AS panier\n"
    "FROM trips WHERE total_amount > 0 GROUP BY ALL ORDER BY passenger_count",
)
```

<!-- #region -->
## 14. Quand choisir DuckDB ?
<!-- #endregion -->

<!-- #region -->
DuckDB n'efface pas pandas / Polars / un SGBD serveur : il les complète. Repères de décision :

| Besoin | Outil conseillé |
|---|---|
| Manipulation interactive de données petites/moyennes, écosystème Python riche | **pandas** |
| Même chose mais gros volumes / pipelines lazy, en mémoire | **Polars** |
| Analytique SQL sur Parquet/CSV, gros volumes, sans serveur | **DuckDB** |
| Transactions concurrentes, écritures intensives, multi-utilisateurs | **PostgreSQL / MySQL** |
| Persistance légère, embarqué, OLTP simple | **SQLite** |

DuckDB brille quand on veut **du SQL analytique performant directement sur des fichiers**, sans infrastructure. Il se combine très bien avec pandas/Polars (requêter un DataFrame) et avec le cloud via l'extension `httpfs` (lecture S3). L'extension **DuckLake** (2026) ajoute un format de *lakehouse* transactionnel par-dessus le stockage objet.
<!-- #endregion -->
