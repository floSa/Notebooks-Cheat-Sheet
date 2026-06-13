---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="iW99VNGYfauE" -->
# DataFrame To Database
<!-- #endregion -->

<!-- #region id="eo6oFgIogF0_" -->
## Postgres by Sqlalchemy
<!-- #endregion -->

```python id="A9BZLHf7g_14"
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
```

```python id="dzN27Xm6hP2Z"
db_params = {
    "host" : "10.20.120.24" ,
    "port" : 5434,
    "user" : "admin",
    "pwd" : "mon_pswd",
    "db" : "ma_bdd",
    "schema": 'mon_schema'
    }
```

```python id="EHbEBX1QgFRR"
def insert_dataframe_to_db(df: pd.DataFrame, db_params: dict, table_name: str):
    """
    Insère un DataFrame dans une table PostgreSQL en utilisant pandas.to_sql.
    Args:
    df (pd.DataFrame): Le DataFrame à insérer.
    db_params (dict): Dictionnaire contenant les informations de connexion à la base de données avec les clés:
        - host : adresse de la base de données
        - port : port de connexion
        - user : utilisateur
        - pwd  : mot de passe de l'utilisateur
        - db   : nom de la base de données
        - schema : schéma cible où la table sera créée
    Raises:
    Exception: Si une erreur survient lors de la connexion ou de l'insertion.
    """

    try:
        # Construire l'URL de connexion pour SQLAlchemy
        db_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['pwd']}@{db_params['host']}:{db_params['port']}/{db_params['db']}"
        engine = create_engine(db_url)

        # Insérer les données dans la base
        try:
            # Vérifier si la table existe
            inspector = inspect(engine)
            table_exists = inspector.has_table(table_name)
            # Insérer les données ou créer la table si elle n'existe pas
            if table_exists:
                strategy ='append'
            else:
                strategy ='fail'

            df.to_sql(
                name = table_name,  # Nom de la table dans PostgreSQL
                con = engine,
                schema = db_params['schema'],  # Schéma cible
                if_exists = strategy # 'replace', 'fail' , 'append'
                index=False,  # Ne pas insérer l'index du DataFrame comme colonne
                method = 'multi' # 'multi' Optimisation des insertions multiples
            )
        except SQLAlchemyError as sql_err:
            raise Exception(f"Erreur lors de l'insertion des données dans la table : {sql_err}")
        except Exception as e:
            raise Exception(f"Erreur inattendue lors de l'insertion : {e}")

    except Exception as e:
        raise Exception(f"Erreur lors de la connexion à la base de données : {e}")

    finally:
        # Fermer proprement la connexion à la base
        if 'engine' in locals():
            engine.dispose()
```

<!-- #region id="sok0WDkIlhvT" -->
## Postgres by Psycopg2
<!-- #endregion -->

```python id="xSBTW-mvmH2G"
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
```

```python id="JvFkaXt9mLz-"
db_params = {
    "host" : "10.20.120.24" ,
    "port" : 5434,
    "user" : "admin",
    "pwd" : "mon_pswd",
    "db" : "ma_bdd",
    "schema": 'mon_schema'
    }
```

```python id="LJwBqum0fJKE"
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

def insert_dataframe_to_db(df: pd.DataFrame, db_params: dict, table_name: str):
    """
    Insérer un DataFrame dans une table PostgreSQL en utilisant uniquement psycopg2.
    Si la table n'existe pas, la créer automatiquement à partir de la structure du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame à insérer.
        db_params (dict): Paramètres de connexion à la base de données contenant :
            - host   : adresse de la base de données.
            - port   : port de connexion.
            - user   : nom d'utilisateur.
            - pwd    : mot de passe.
            - db     : nom de la base de données.
            - schema : schéma cible dans la base de données.
        table_name (str): Nom de la table dans laquelle insérer les données.

    Raises:
        Exception: En cas d'erreur lors de la connexion, de la création de la table ou de l'insertion.
    """
    try:
        # Établir la connexion à la base de données
        conn = psycopg2.connect(
            host=db_params["host"],
            port=db_params["port"],
            user=db_params["user"],
            password=db_params["pwd"],
            dbname=db_params["db"]
        )
        conn.autocommit = False  # Gérer les transactions manuellement
    except Exception as e:
        raise Exception(f"Erreur lors de la connexion à la base de données : {e}")

    try:
        cursor = conn.cursor()
        # Construire le nom complet de la table avec schéma
        full_table_name = f'"{db_params["schema"]}"."{table_name}"'

        # Vérifier si la table existe dans le schéma spécifié
        query_check = """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_name = %s
            )
        """
        cursor.execute(query_check, (db_params["schema"], table_name))
        table_exists = cursor.fetchone()[0]

        # Si la table n'existe pas, la créer en inférant les types SQL depuis le DataFrame
        if not table_exists:
            columns_definitions = []
            for col, dtype in df.dtypes.items():
                # Mapper les types de données pandas vers des types SQL
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "DOUBLE PRECISION"
                elif pd.api.types.is_bool_dtype(dtype):
                    sql_type = "BOOLEAN"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "TIMESTAMP"
                else:
                    sql_type = "TEXT"
                # Encadrer le nom de colonne par des guillemets pour préserver la casse et les caractères spéciaux
                columns_definitions.append(f'"{col}" {sql_type}')

            create_query = f"CREATE TABLE {full_table_name} ({', '.join(columns_definitions)});"
            cursor.execute(create_query)

        # Préparer l'insertion des données
        cols = list(df.columns)
        # Encadrer chaque nom de colonne par des guillemets pour éviter les problèmes d'identificateurs
        cols_quoted = ', '.join([f'"{col}"' for col in cols])
        insert_query = f"INSERT INTO {full_table_name} ({cols_quoted}) VALUES %s"

        # Convertir le DataFrame en liste de tuples pour l'insertion en masse
        data_tuples = [tuple(row) for row in df.to_numpy()]

        # Insérer les données en utilisant execute_values pour optimiser l'insertion multiple
        execute_values(cursor, insert_query, data_tuples)

        # Valider la transaction
        conn.commit()
    except Exception as e:
        conn.rollback()  # Annuler la transaction en cas d'erreur
        raise Exception(f"Erreur lors de l'insertion des données : {e}")
    finally:
        # Fermer le curseur et la connexion
        cursor.close()
        conn.close()

```

<!-- #region id="16Urq0NjFXe7" -->
## MongoDB by pymongo
<!-- #endregion -->

```python id="rrATM2yGffNn"
from pymongo import MongoClient
import pandas as pd

def insert_dataframe_to_db(df: pd.DataFrame, db_params: dict, collection_name: str):
    """
    Insérer un DataFrame dans une collection MongoDB en utilisant uniquement PyMongo.
    Si la collection n'existe pas, la créer automatiquement.

    Args:
        df (pd.DataFrame): DataFrame à insérer.
        db_params (dict): Paramètres de connexion à MongoDB contenant :
            - host : adresse du serveur MongoDB.
            - port : port de connexion.
            - user : nom d'utilisateur (optionnel si non requis).
            - pwd  : mot de passe (optionnel).
            - db   : nom de la base de données.
        collection_name (str): Nom de la collection dans laquelle insérer les données.

    Raises:
        Exception: En cas d'erreur lors de la connexion ou de l'insertion.
    """
    try:
        # Construire l'URI de connexion pour MongoDB
        if "user" in db_params and "pwd" in db_params:
            mongo_uri = f"mongodb://{db_params['user']}:{db_params['pwd']}@{db_params['host']}:{db_params['port']}/"
        else:
            mongo_uri = f"mongodb://{db_params['host']}:{db_params['port']}/"

        # Établir la connexion au serveur MongoDB
        client = MongoClient(mongo_uri)
        # Accéder à la base de données spécifiée
        db = client[db_params["db"]]
    except Exception as e:
        raise Exception(f"Erreur lors de la connexion à MongoDB : {e}")

    try:
        # Vérifier si la collection existe dans la base de données
        if collection_name not in db.list_collection_names():
            # Créer la collection explicitement (facultatif, car MongoDB la crée automatiquement à l'insertion)
            collection = db.create_collection(collection_name)
        else:
            collection = db[collection_name]

        # Convertir le DataFrame en liste de documents (dictionnaires)
        data_records = df.to_dict(orient="records")
        if not data_records:
            raise Exception("Le DataFrame est vide, aucune donnée à insérer.")

        # Insérer les documents en masse dans la collection
        collection.insert_many(data_records)
    except Exception as e:
        raise Exception(f"Erreur lors de l'insertion des données dans MongoDB : {e}")
    finally:
        # Fermer la connexion au serveur MongoDB
        client.close()

```

<!-- #region id="WFBEKzpUIyMJ" -->
# Database TO Dataframe
<!-- #endregion -->

<!-- #region id="F1g7KnZBI4GT" -->
## Postgres by Sqlalchemy
<!-- #endregion -->

```python id="0gts79-xI3In"
import pandas as pd
from sqlalchemy import create_engine

def query_to_dataframe_sqlalchemy(db_params: dict, query: str) -> pd.DataFrame:
    """
    Exécuter une requête SQL via SQLAlchemy et renvoyer le résultat sous forme de DataFrame.

    Args:
        db_params (dict): Paramètres de connexion PostgreSQL avec les clés :
            - host : adresse du serveur
            - port : port de connexion
            - user : nom d'utilisateur
            - pwd  : mot de passe
            - db   : nom de la base de données
        query (str): Requête SQL à exécuter.

    Returns:
        pd.DataFrame: DataFrame contenant le résultat de la requête.

    Raises:
        Exception: En cas d'erreur de connexion ou d'exécution de la requête.
    """
    engine = None
    try:
        # Construire l'URL de connexion pour SQLAlchemy
        db_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['pwd']}@" \
                 f"{db_params['host']}:{db_params['port']}/{db_params['db']}"
        engine = create_engine(db_url)
        # Exécuter la requête et charger le résultat dans un DataFrame
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        raise Exception(f"Erreur lors de l'exécution de la requête via SQLAlchemy : {e}")
    finally:
        if engine is not None:
            engine.dispose()

```

<!-- #region id="qYJU9x8PLLT4" -->
## Postgres by psycopg2
<!-- #endregion -->

```python id="b5tFVq9ULLc1"
# --- Fonction avec psycopg2 ---
import psycopg2

def query_to_dataframe_psycopg2(db_params: dict, query: str) -> pd.DataFrame:
    """
    Exécuter une requête SQL via psycopg2 et renvoyer le résultat sous forme de DataFrame.

    Args:
        db_params (dict): Paramètres de connexion PostgreSQL avec les clés :
            - host : adresse du serveur
            - port : port de connexion
            - user : nom d'utilisateur
            - pwd  : mot de passe
            - db   : nom de la base de données
        query (str): Requête SQL à exécuter.

    Returns:
        pd.DataFrame: DataFrame contenant le résultat de la requête.

    Raises:
        Exception: En cas d'erreur de connexion ou d'exécution de la requête.
    """
    conn = None
    cursor = None
    try:
        # Établir la connexion avec psycopg2
        conn = psycopg2.connect(
            host=db_params["host"],
            port=db_params["port"],
            user=db_params["user"],
            password=db_params["pwd"],
            dbname=db_params["db"]
        )
        cursor = conn.cursor()
        # Exécuter la requête
        cursor.execute(query)
        data = cursor.fetchall()
        # Récupérer les noms de colonnes depuis la description du curseur
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        return df
    except Exception as e:
        raise Exception(f"Erreur lors de l'exécution de la requête via psycopg2 : {e}")
    finally:
        # Fermer le curseur et la connexion
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
```

<!-- #region id="s_nb9SjkLV3O" -->
## MogoDB by pymongo
<!-- #endregion -->

```python id="0nMXfH01LV_9"
# --- Fonction pour MongoDB ---
from pymongo import MongoClient

def query_to_dataframe_mongo(db_params: dict, query: dict, collection_name: str) -> pd.DataFrame:
    """
    Exécuter une requête sur une collection MongoDB et renvoyer le résultat sous forme de DataFrame.

    Args:
        db_params (dict): Paramètres de connexion MongoDB avec les clés :
            - host : adresse du serveur MongoDB
            - port : port de connexion
            - user : (optionnel) nom d'utilisateur
            - pwd  : (optionnel) mot de passe
            - db   : nom de la base de données
        query (dict): Dictionnaire représentant le filtre MongoDB.
        collection_name (str): Nom de la collection à interroger.

    Returns:
        pd.DataFrame: DataFrame contenant les documents récupérés.

    Raises:
        Exception: En cas d'erreur de connexion ou d'exécution de la requête.
    """
    client = None
    try:
        # Construire l'URI de connexion pour MongoDB
        if "user" in db_params and "pwd" in db_params:
            mongo_uri = f"mongodb://{db_params['user']}:{db_params['pwd']}@" \
                        f"{db_params['host']}:{db_params['port']}/"
        else:
            mongo_uri = f"mongodb://{db_params['host']}:{db_params['port']}/"

        # Établir la connexion au serveur MongoDB
        client = MongoClient(mongo_uri)
        db = client[db_params["db"]]
        collection = db[collection_name]
        # Exécuter la requête et convertir le curseur en liste
        data_list = list(collection.find(query))
        df = pd.DataFrame(data_list)
        return df
    except Exception as e:
        raise Exception(f"Erreur lors de l'exécution de la requête sur MongoDB : {e}")
    finally:
        if client is not None:
            client.close()
```
