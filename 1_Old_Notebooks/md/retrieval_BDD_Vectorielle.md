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
    language: python
    name: python3
---

<!-- #region id="i_35qdQQUeTJ" -->
# Recherche de similarité sur des vecteurs (Base de Données Vectorielles)
<!-- #endregion -->

<!-- #region id="SWfGe_7MUeTK" -->
FAISS est une bibliothèque open-source de Meta, conçue pour effectuer des recherches
efficaces et rapides sur des vecteurs de grande dimension.

1. **Encodage** : SentenceTransformer encode des phrases en vecteurs (embeddings) qui résument leur sens.  
2. **Indexation** : Ces vecteurs sont Indexées
3. **Recherche** : Recherche de similarité sur la base des plus proches voisins à partir du calcule les distances entre les vecteurs
<!-- #endregion -->

<!-- #region id="LaMB4Yk1UeTM" -->
## **1. Vectorisation des phrases**
<!-- #endregion -->

<!-- #region id="MQXFNy4qUeTN" -->
### **SentenceTransformer : Générer des vecteurs sémantiques à partir de phrases**  
Lorsqu'on travaille avec du texte, les ordinateurs ne comprennent pas les mots comme les humains. Pour rendre le texte exploitable pour des modèles, on doit le convertir en **vecteurs** numériques, aussi appelés **embeddings**. Ces vecteurs traduisent la signification d'une phrase dans un espace mathématique.

#### Comment SentenceTransformer fonctionne-t-il ?  
SentenceTransformer est basé sur des modèles de type **Transformers**, comme BERT, mais il est optimisé pour produire des vecteurs qui capturent la **sémantique globale** d'une phrase (et pas seulement le sens mot par mot). Voici les étapes simplifiées du processus :  
1. **Encodage du texte** : Le texte est découpé en sous-unités appelées **tokens**, puis transformé en vecteurs initiaux grâce à un modèle pré-entraîné.  
2. **Traitement avec des couches de Transformer** : Ces vecteurs passent par plusieurs couches Transformer, où des mécanismes comme l'**attention** leur permettent de capter les relations contextuelles entre les mots.  
3. **Production d’un vecteur global** : SentenceTransformer combine les informations pour produire un vecteur unique pour la phrase entière. Ce vecteur est souvent une réduction dans un espace de 384 ou 768 dimensions.

Ce vecteur contient des informations sur le **sens global** de la phrase dans un espace vectoriel :  
- Deux phrases similaires dans leur contenu auront des vecteurs proches.  
- Deux phrases très différentes auront des vecteurs éloignés.  

L’objectif est de créer une représentation qui permet de **comparer efficacement les phrases**, indépendamment de leur formulation exacte.
<!-- #endregion -->

<!-- #region id="5GOpJ9OCUeTN" -->
### Tableau récapitulatif des modèles d'encodage de phrases

| **Modèle**                   | **Cas d'utilisation**                                     | **Avantages**                                                                 | **Inconvénients**                                                              |
|-------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| **all-MiniLM-L6-v2**          | Recherche sémantique, applications légères               | Rapide, léger, haute précision pour la recherche sémantique                   | Moins précis pour des textes longs ou complexes                                |
| **mpnet-base-v2**             | Recommandé pour une précision maximale                  | Très performant pour des tâches sémantiques complexes                         | Plus lent que MiniLM, plus lourd                                              |
| **distiluse-base-multilingual-cased-v2** | Multilingue : recherche dans plusieurs langues         | Gère plus de 50 langues, performant sur des textes courts et multilingues     | Moins précis que les modèles monolingues pour une langue unique                |
| **bert-base-nli-mean-tokens** | Classification, appariement de textes, recherche         | Solide sur des tâches classiques de NLI (Natural Language Inference)          | Modèle plus ancien, moins optimisé pour la recherche pure                     |
| **roberta-base-nli-stsb-mean-tokens** | Tâches de similarité, appariements complexes          | Très performant pour des textes techniques et des contextes précis            | Plus lourd et plus lent que MiniLM                                             |
| **sentence-t5-base**          | Traduction vers un vecteur universel, génération de texte | Basé sur T5, performant sur des tâches complexes comme la génération de texte | Gourmand en ressources, plus lent                                              |
<!-- #endregion -->

```python id="boHRZHCeUeTO"
!pip install sentence_transformers -q
```

```python id="nRCvw5UpUeTO"
# Importation des bibliothèques nécessaires
import numpy as np
from sentence_transformers import SentenceTransformer
```

```python id="Vb5pDD9hUeTP" outputId="08524e85-9319-4ecd-e923-b635cd2f42d2"
# Nous allons utiliser quelques phrases comme exemples.
phrases = [
    "Le chat dort sur le tapis.",
    "Le chien joue dans le jardin.",
    "Un oiseau vole dans le ciel.",
    "Le soleil brille aujourd'hui.",
    "Il pleut dans la ville."
]

# Utilisation d'un modèle pré-entraîné pour convertir les phrases en vecteurs.
# SentenceTransformer est une bibliothèque qui facilite la création de telles embeddings.
model = SentenceTransformer('all-MiniLM-L6-v2')  # Chargement d'un modèle compact pour les embeddings

# Conversion des phrases en vecteurs
data = np.array(model.encode(phrases), dtype='float32')
print(f"Dimensions des vecteurs : {data.shape}")  # Doit être (nombre de phrases, dimension du vecteur)
```

<!-- #region id="a0pCCe4PUeTQ" -->
## **2. Indexation et recherche**
<!-- #endregion -->

<!-- #region id="aIGPUJriUeTR" -->
### Comparaison des solutions

| Base de données | API          | Métadonnées   | Recherche textuelle | Graphe sémantique | Filtrage | Particularités                        |
|------------------|--------------|---------------|----------------------|-------------------|----------|---------------------------------------|
| **FAISS**        | Non (librairie) | Limité        | Non                  | Non               | Non      | Haute performance, faible latence. |
| **Chroma**       | Oui (Python)  | Oui           | Oui                  | Non               | Oui      | Spécifique aux workflows de machine learning. |
| **Weaviate**     | Oui (REST, GraphQL) | Oui           | Oui                  | Oui               | Oui      | Modèles pré-entraînés, connecteurs aux LLM. |
| **Milvus**       | Oui (REST, Python, Go) | Oui           | Non                  | Non               | Oui      | Architecture distribuée, évolutive. |
| **Qdrant**       | Oui (REST, gRPC, Python) | Oui           | Oui (via intégrations) | Non               | Oui      | Optimisé pour le filtrage et la mise à jour des données. |


<!-- #endregion -->

<!-- #region id="TeCZ_fGtUeTR" -->
### **A - FAISS**
<!-- #endregion -->

<!-- #region id="DDTtAixYUeTS" -->
#### Comment fonctionne FAISS ?
FAISS (Facebook AI Similarity Search) est conçu pour gérer la **recherche rapide** dans des bases contenant des millions (voire des milliards) de vecteurs. L’algorithme sous-jacent varie selon l’index utilisé.

L'index FAISS qui repose sur la **distance euclidienne**
$\mathbf{u}$ et $\mathbf{v}$ sont définis comme :
$$
d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}
$$

Ce calcul mesure à quel point deux vecteurs sont éloignés dans l’espace. Plus la distance est proche de **0**, plus les vecteurs (et donc les phrases) sont similaires.  
<!-- #endregion -->

<!-- #region id="uksQkpOYUeTS" -->
#### Comparaison des trois approches

| Méthode          | Avantages                                      | Inconvénients                              | Cas d'utilisation                           |
|-------------------|-----------------------------------------------|--------------------------------------------|---------------------------------------------|
| **IndexFlatL2**   | Résultats exacts, simple, pas d'entraînement   | Lent pour les grandes bases                | Petites bases ou besoin d'une précision maximale. |
| **IndexIVFFlat**  | Rapide pour des bases volumineuses, ajustable | Nécessite un entraînement, moins précis    | Grandes bases avec possibilité d'ajuster précision/vitesse. |
| **HNSW**          | Très rapide, pas d'entraînement, dynamique    | Gourmand en mémoire, graphe coûteux à construire | Bases très grandes nécessitant des mises à jour fréquentes. |

<!-- #endregion -->

```python id="ue0VYZUUUeTS"
!pip install faiss-cpu -q
```

```python id="H7QFdG18UeTT"
import faiss
```

```python id="ZvAHgKEPUeTT" outputId="53e6abda-76b2-415f-9a53-6bfc8c68d06f"
dimension = data.shape[1]  # Dimension des vecteurs
print("dimension" , dimension)
```

<!-- #region id="saaA1PLjUeTT" -->
#### **IndexFlatL2**
**Description :**  
IndexFlatL2 est l'index de recherche le plus simple dans FAISS. Il effectue une recherche exhaustive en calculant la distance euclidienne entre le vecteur de requête et tous les vecteurs de la base.

- **Avantages :**
  - Garantit des résultats exacts (pas d'approximation).
  - Simple à implémenter.
  - Pas besoin d'entraînement préalable.

- **Inconvénients :**
  - Lente pour de grandes bases, car elle compare chaque vecteur un par un.
  - Pas optimisée pour des bases de plusieurs millions de vecteurs.
<!-- #endregion -->

```python id="5nxFi9eaUeTU" outputId="1580910f-b7dc-4cce-8a84-7de81d1b4649"
#Création de l'index
index = faiss.IndexFlatL2(dimension)  # Création de l'index pour des vecteurs de dimension d
# Ajout des vecteurs à l'index
index.add(data)
print(f"Nombre de vecteurs dans l'index : {index.ntotal}")
```

```python id="IyuIo9aAUeTU" outputId="4a30a9aa-d53f-4137-ddb9-fe69f221e358"
# Ajoutons une nouvelle phrase et recherchons les phrases les plus similaires dans l'index.
new_phrases = [
    "Un oiseau chante dans le ciel clair."
]

# Vectorisation de la nouvelle phrase
queries = np.array(model.encode(new_phrases), dtype='float32')

# Recherche des k voisins les plus proches
k = 3  # Nombre de voisins à retourner

distances, indices = index.search(queries, k)

# Affichage des résultats
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. Phrase similaire : {phrases[idx]} (Distance : {distances[0][i]})")

```

<!-- #region id="ObrVFFVlUeTV" -->
#### **IndexIVFFlat**
**Description :**  
IndexIVFFlat (Inverted File Flat) utilise un clustering préalable pour diviser l'espace vectoriel en plusieurs régions (clusters). Lors de la recherche, seuls certains clusters sont explorés, ce qui réduit considérablement les calculs.

- **Calcul de la distance :**  
La distance euclidienne (L2) est utilisée, mais le calcul est limité aux vecteurs présents dans les clusters les plus pertinents.

- **Avantages :**
  - Recherche rapide pour des bases volumineuses grâce à l'exploration ciblée.
  - Paramètre ajustable (`nprobe`) permettant de contrôler le nombre de clusters explorés et donc la précision/vitesse.

- **Inconvénients :**
  - Nécessite un entraînement préalable pour effectuer le clustering.
  - Moins précis si `nprobe` est trop faible ou si le clustering est mal réalisé.

<!-- #endregion -->

```python id="lZ0lGNTOUeTV"
n_clusters = 3
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, n_clusters)

# Entraînement et ajout des vecteurs
index_ivf.train(data)
index_ivf.add(data)
```

```python id="2jKfyhvPUeTV" outputId="626900c6-2f02-4a2d-886e-3773d114a312"
# Recherche
# Ajoutons une nouvelle phrase et recherchons les phrases les plus similaires dans l'index.
new_phrases = [
    "Un oiseau chante dans le ciel clair."
]

query = np.array(model.encode(new_phrases), dtype='float32')

# Nombre de clusters explorés
index_ivf.nprobe = 2
# Nombre de voisins à retourner
k = 3
# Recherche des k voisins les plus proches
distances, indices = index_ivf.search(query, k=k)

# Affichage des résultats
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. Phrase similaire : {phrases[idx]} (Distance : {distances[0][i]:.4f})")
```

<!-- #region id="pm-Jg6N5UeTW" -->
#### **HNSW (Hierarchical Navigable Small World)**
**Description :**  
HNSW est basé sur une structure de graphe hiérarchique. Les vecteurs sont représentés comme des nœuds connectés à leurs voisins proches. Lors de la recherche, l’algorithme navigue dans ce graphe pour trouver les voisins les plus proches.

- **Calcul de la distance :**  
La distance euclidienne (L2) est utilisée pour mesurer la proximité entre les vecteurs, mais le graphe limite les comparaisons à un sous-ensemble de la base.

- **Avantages :**
  - Très rapide pour des recherches sur des bases volumineuses.
  - Pas besoin d'entraînement préalable.
  - Prise en charge facile de l'ajout et de la suppression de vecteurs dans l'index.

- **Inconvénients :**
  - La construction initiale du graphe peut être coûteuse en temps.
  - Consomme plus de mémoire par rapport à IndexFlatL2 ou IndexIVFFlat.
<!-- #endregion -->

```python id="ijssdwJ3UeTW"
# Création d'IndexHNSW
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 est le nombre de voisins par nœud
index_hnsw.add(data)
```

```python id="gIOebAYiUeTW" outputId="f0b4b58b-e8fa-4621-c210-945aafeb40ff"
# Nombre de voisins à retourner
k = 3
# Recherche
distances_hnsw, indices_hnsw = index_hnsw.search(query, k=k)

# Affichage des résultats
for i, idx in enumerate(indices_hnsw[0]):
    print(f"Phrase similaire : {phrases[idx]} (Distance : {distances_hnsw[0][i]:.4f})")
```

<!-- #region id="c4nNo-kKUeTW" -->
### **B - Weaviate** (No ok)
<!-- #endregion -->

<!-- #region id="gHa4ifIEUeTX" -->
##### **Étape 1 : Installation de Weaviate via Docker Compose : Créez un fichier docker-compose.yml**

**Fichier Docker compose .Yaml**

```yaml
version: "3.4"
services:
  weaviate:
    image: semitechnologies/weaviate:1.20.0
    container_name: weaviate
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=20
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER=none  # Utilisation d'un vectoriseur externe
```

**Commande bash**

```bash
docker-compose up -d
```
<!-- #endregion -->

<!-- #region id="wewW_4yXUeTX" -->
##### **Étape 2 : Configuration de l'environnement Python**
<!-- #endregion -->

<!-- #region id="bFzAKDH7UeTX" -->
Installez les bibliothèques nécessaires :
<!-- #endregion -->

```python id="xlwX0FNaUeTX"
!pip uninstall weaviate-client
!pip install weaviate-client
```

<!-- #region id="5CvgoKgzUeTX" -->
Charger les bibliothèques nécessaires :
<!-- #endregion -->

```python id="EqKrD1qRUeTY"
import weaviate
print(weaviate.__version__)
```

```python id="DcfzimcGUeTY"
import weaviate-client
from sentence_transformers import SentenceTransformer
```

<!-- #region id="l_FqO430UeTY" -->
##### **Étape 3 : Connexion à Weaviate et création d'un schéma**
<!-- #endregion -->

<!-- #region id="nTLHAVKPUeTY" -->
Connectez-vous à l'instance Weaviate :
<!-- #endregion -->

```python id="LryhTvoVUeTa"
client = weaviate.Client("http://localhost:8080")
```

```python id="kEZU7rf7UeTa"
schema = {
    "class": "SentenceVectors",
    "vectorizer": "none",  # Les vecteurs sont gérés manuellement
    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
            "description": "Phrase originale"
        }
    ]
}
# Ajout du schéma à Weaviate
client.schema.create_class(schema)

```

<!-- #region id="K7vvn_d4UeTa" -->
##### **Étape 4 : Ajout de vecteurs dans Weaviate**
<!-- #endregion -->

```python id="8cWp7sVkUeTa"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Ceci est une phrase.", "Une autre phrase pour tester."]
embeddings = model.encode(sentences).tolist()
```

```python id="GHls54mzUeTa"
for text, vector in zip(sentences, embeddings):
    client.data_object.create(
        data_object={"text": text},
        class_name="SentenceVectors",
        vector=vector
    )
```

<!-- #region id="c9MubXAcUeTb" -->
##### **Étape 5 : Recherche de vecteurs similaires**
<!-- #endregion -->

<!-- #region id="mll6Q_MyUeTb" -->
Encoder une nouvelle phrase pour la recherche :
<!-- #endregion -->

```python id="rkAwzBAAUeTb"
query_sentence = "Testez cette phrase."
query_vector = model.encode([query_sentence]).tolist()[0]
```

<!-- #region id="8IMzss53UeTb" -->
Effectuer une recherche vectorielle dans Weaviate :
<!-- #endregion -->

```python id="uAC9LpNYUeTb"
result = client.query.get("SentenceVectors", ["text"]).with_near_vector({
    "vector": query_vector
}).with_limit(5).do()

# Affichez les résultats
for item in result["data"]["Get"]["SentenceVectors"]:
    print(f"Text: {item['text']}")
```

<!-- #region id="BKSbpT4sUeTb" -->
##### **Étape 6 : Gestion des collections**
<!-- #endregion -->

<!-- #region id="fIJKduC8UeTb" -->
Lister les classes existantes :
<!-- #endregion -->

```python id="_BUDcfP9UeTc"
schema = client.schema.get()
print(schema)
```

<!-- #region id="THo_1RS0UeTc" -->
Supprimer une classe :
<!-- #endregion -->

```python id="YsX0kl7fUeTc"
client.schema.delete_class("SentenceVectors")
```

<!-- #region id="2C2vdgyPUeTc" -->
### **C - Milvus** (No ok)
<!-- #endregion -->

<!-- #region id="2VxkiGo8UeTd" -->
##### **Étape 1 : Installation de Milvus via Docker Compose : Créez un fichier docker-compose.yml**

**Fichier Docker compose .Yaml**

```yaml
version: '3.5'
services:
  milvus:
    image: milvusdb/milvus:v2.2.10
    container_name: milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - ./volumes/db:/var/lib/milvus
      - ./volumes/conf:/etc/milvus
    environment:
      - "TZ=UTC"
```

**Commande bash**

```bash
docker-compose up -d
```
<!-- #endregion -->

<!-- #region id="iNpjp0y-UeTd" -->
##### **Étape 2 : Configuration de l'environnement Python**
<!-- #endregion -->

```python id="G3WqWnJwUeTd"
!pip install pymilvus sentence-transformers numpy
```

```python id="jGZvWkUhUeTd"
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import numpy as np
```

<!-- #region id="WrWFunT2UeTe" -->
##### **Étape 3 : Connexion à Milvus et création d'une collection**
<!-- #endregion -->

<!-- #region id="2zGd1wwgUeTe" -->
Connection à la base Milvus :
<!-- #endregion -->

```python id="ubEsaLhhUeTe"
connections.connect(host="127.0.0.1", port="19530")
```

<!-- #region id="BfYz-_rpUeTf" -->
Créez une collection pour stocker vos vecteurs :
<!-- #endregion -->

```python id="jQ1Q8l-8UeTf"
# Schéma de la collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="Collection de vecteurs d'embeddings")
collection = Collection(name="sentence_vectors", schema=schema)

```

<!-- #region id="z-sQL7gsUeTf" -->
##### **Étape 4 : Ajout de vecteurs à la collection**
<!-- #endregion -->

<!-- #region id="YgRgU1LnUeTf" -->
Génération des vecteurs (embeddings) :
<!-- #endregion -->

```python id="oqU0ttw_UeTf"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Ceci est une phrase.", "Une autre phrase pour tester."]
embeddings = model.encode(sentences).tolist()
```

<!-- #region id="l3onBVKXUeTg" -->
Insertion des vecteurs dans Milvus :
<!-- #endregion -->

```python id="LG7ZQnaCUeTg"
# Préparation des données
data = [
    [None] * len(embeddings),  # Les IDs sont générés automatiquement
    embeddings
]
collection.insert(data)

```

<!-- #region id="nUJaCPR5UeTg" -->
Création de l'index pour optimiser les recherches
<!-- #endregion -->

```python id="YeUwfytfUeTg"
collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})
```

<!-- #region id="jRGyj8PNUeTg" -->
Chargez la collection en mémoire :
<!-- #endregion -->

```python id="yWqySow4UeTh"
collection.load()
```

<!-- #region id="jT-Ql2HWUeTh" -->
##### **Étape 5 : Recherche de vecteurs similaires**
<!-- #endregion -->

```python id="ny0to3YHUeTh"
query_sentence = "Testez cette phrase."
query_vector = model.encode([query_sentence]).tolist()
```

```python id="_aG0nVYsUeTh"
search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["id"]
)

# Affichez les résultats
for result in results:
    print(f"ID: {result.id}, Distance: {result.distance}")

```

<!-- #region id="1QL5AC3lUeTh" -->
##### **Étape 6 : Nettoyage des ressources**
<!-- #endregion -->

<!-- #region id="WG56lwS-UeTh" -->
Supprimez la collection:
<!-- #endregion -->

```python id="4GNRZw9yUeTi"
collection.drop()
```

<!-- #region id="O6qj63VRUeTi" -->
Déconnection de Milvus
<!-- #endregion -->

```python id="rgOgQKcJUeTi"
connections.disconnect()
```

<!-- #region id="7G6iOx6gUeTi" -->
### **D - Qdrant**
<!-- #endregion -->

<!-- #region id="VsijqecQUeTi" -->
#### **Étape 1 : Installation de Qdrant**

1. **Pré-requis :**
   - Docker installé et configuré.

2. **Installation de Qdrant via Docker :**
   Exécutez la commande suivante pour lancer un conteneur Qdrant :
   ```bash
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
   ```
   ```bash
   docker run -p 6333:6333 -d -v $(pwd)/path/to/data:/qdrant/storage     qdrant/qdrant
   ```
   Qdrant est maintenant accessible sur `http://localhost:6333`.
<!-- #endregion -->

<!-- #region id="gq9XXgaqUeTj" -->
#### Étape 2 : Configuration de l'environnement Python
<!-- #endregion -->

<!-- #region id="vdM74of2UeTk" -->
1. Installez les bibliothèques nécessaires :
<!-- #endregion -->

```python id="rSVmzEVRUeTk"
!pip install --upgrade ydata-profiling -q
```

```python id="PBbkPcw9UeTk"
!pip install qdrant-client -q
```

<!-- #region id="JUqgaoKMUeTk" -->
2. Charger les library en mémoire
<!-- #endregion -->

```python id="FwEbIzh6UeTl"
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
```

<!-- #region id="DkeVkf9PUeTl" -->
#### **Étape 3 : Connexion à Qdrant et création d'une collection**

Connection-vous à l'instance Qdrant :
<!-- #endregion -->

```python id="rEfWgLphUeTl"
client = QdrantClient(host="192.168.10.113", port=6333)
```

<!-- #region id="zS1W1hmuUeTl" -->
Création d'une collection pour stocker les vecteurs
<!-- #endregion -->

```python id="ea-QCgDhUeTl" outputId="6f626593-39f6-4250-b977-0006409b019a"
client.recreate_collection(
    collection_name="sentence_vectors",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Taille des vecteurs et type de distance
)
```

<!-- #region id="TNnlb3zgUeTm" -->
#### **Étape 4 : Ajout de vecteurs à la collection**

1. Utiliser un modèle SentenceTransformer pour générer des embeddings :
<!-- #endregion -->

```python id="EQpGbdaeUeTm"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Ceci est une phrase.", "Une autre phrase pour tester."]
embeddings = model.encode(sentences).tolist()
```

<!-- #region id="knafqlSYUeTm" -->
2. Ajouter les vecteurs et les phrases dans la collection :
<!-- #endregion -->

```python id="lJqrx6HrUeTm" outputId="10db9ffb-6ccf-4314-a514-9bd2e8d06218"
from qdrant_client.models import PointStruct

# Création des points dans le format attendu
points = [
    PointStruct(
        id=i,  # ID unique pour chaque vecteur
        vector=embedding,  # Le vecteur généré
        payload={"text": sentence}  # Métadonnée associée
    )
    for i, (embedding, sentence) in enumerate(zip(embeddings, sentences))
]

# Ajout des points dans la collection
client.upsert(
    collection_name="sentence_vectors",
    points=points
)
print("Vecteurs et phrases insérés avec succès dans la collection.")
```

<!-- #region id="E2up-5dSUeTm" -->
#### **Étape 5 : Recherche de vecteurs similaires**
<!-- #endregion -->

<!-- #region id="m0YgmRDDUeTm" -->
1. Encodez une nouvelle phrase pour la recherche :
<!-- #endregion -->

```python id="ApSOvolFUeTm"
query_sentence = "Testez cette phrase."
query_vector = model.encode([query_sentence]).tolist()[0]
```

<!-- #region id="oEQZXi8JUeTm" -->
2. Effectuez une recherche vectorielle dans Qdrant :
<!-- #endregion -->

```python id="DPUqFKu_UeTn" outputId="f38205d8-1135-43d5-b736-d01725881d19"
results = client.search(
    collection_name="sentence_vectors",
    query_vector=query_vector,
    limit=5  # Nombre de résultats souhaités
)

# Affichez les résultats
for result in results:
    print(f"Text: {result.payload['text']}, Score: {result.score}")
```

<!-- #region id="IsEyFPEBUeTn" -->
#### **Étape 6 : Gestion des collections**
<!-- #endregion -->

<!-- #region id="ZKvo0IFaUeTo" -->
1. Pour lister toutes les collections existantes :
<!-- #endregion -->

```python id="Y533VUydUeTo"
collections = client.get_collections()
for collection in collections.collections:
    print(f"Collection: {collection.name}")
```

<!-- #region id="X4BlGoDYUeTp" -->
2. Pour supprimer une collection :
<!-- #endregion -->

```python id="yjcvEUASUeTq"
client.delete_collection("sentence_vectors")
```

<!-- #region id="QaGntTDpUeTq" -->
### **E - Chroma**
<!-- #endregion -->

<!-- #region id="W6ez-FgRUeTq" -->
##### **Étape 1 : Lancement du serice sur Docker**
<!-- #endregion -->

<!-- #region id="R_yWAwxuUeTq" -->
```bash  
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```
```bash  Z
docker run -d --name chromadb_service -v $(pwd)/data/chromadb:/data  -p 8000:8000 chromadb/chromadb:latest
```
<!-- #endregion -->

<!-- #region id="Lt-tyXCeUeTq" -->
##### **Étape 2 : Installation des package python et chargement des library**
<!-- #endregion -->

```python id="v3auouaSUeTr"
!pip install chromadb -q
```

```python id="iI5iAfsUUeTr"
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
```

<!-- #region id="t_pQ2Xm-UeTr" -->
##### **Étape 3 : Configurez et initialisez une instance locale de Chroma**
<!-- #endregion -->

```python id="oAajJUq9UeTr"
import chromadb
from chromadb.config import Settings
```

```python id="uDEitJ08UeTr"
# Étape 1 : Connexion au serveur Chroma
client = chromadb.Client(
    Settings(
        chroma_server_host="192.168.10.113",  # Remplacez par l'adresse IP de votre serveur si nécessaire
        chroma_server_http_port=8000
    )
)
# chroma_api_impl="rest",
# persist_directory="./chroma_data",  # Dossier pour sauvegarder les données
```

```python id="hZ7YaG-UUeTr"
client = chromadb.Client(Settings(
    persist_directory="./chroma_data",  # Dossier pour sauvegarder les données
    chroma_db_impl="duckdb+parquet",   # Implémentation de la base
))
```

<!-- #region id="eQKT_HYCUeTr" -->
##### **Étape 4 : Création d'une collection pour les vecteurs**
<!-- #endregion -->

<!-- #region id="IvSuJ19WUeTr" -->
Création de la collection
<!-- #endregion -->

```python id="-kZh2MaNUeTs"
collection = client.get_or_create_collection("sentence_vectors")
```

<!-- #region id="8U5rm5IfUeTs" -->
Ajout de vecteurs dans la collection
<!-- #endregion -->

```python id="xq1P4I1PUeTs"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Ceci est une phrase.", "Une autre phrase pour tester."]
embeddings = model.encode(sentences).tolist()
```

<!-- #region id="rWd8BNilUeTs" -->
##### **Étape 5 : Recherche de vecteurs similaires**
<!-- #endregion -->

<!-- #region id="JylC58JHUeTs" -->
Encodez une nouvelle phrase pour la recherche :
<!-- #endregion -->

```python id="zRda-mFYUeTs"
query_sentence = "Testez cette phrase."
query_vector = model.encode([query_sentence]).tolist()[0]
```

<!-- #region id="koeGHeheUeTt" -->
Effectuez une recherche dans la collection :
<!-- #endregion -->

```python id="R9ZRzn0BUeTt"
results = collection.query(
    query_embeddings=[query_vector],  # Vecteur à rechercher
    n_results=5  # Nombre de résultats souhaités
)

# Affichez les résultats
for i, doc in enumerate(results["documents"][0]):
    print(f"Document: {doc}, ID: {results['ids'][0][i]}, Distance: {results['distances'][0][i]}")

```

<!-- #region id="WA7mM_sqUeTt" -->
##### **Étape 6 : Gestion des collections**
<!-- #endregion -->

<!-- #region id="JIvYa_0mUeTt" -->
Lister toutes les collections :
<!-- #endregion -->

```python id="6YRtQGJtUeTt"
collections = client.list_collections()
for col in collections:
    print(f"Collection: {col.name}")

```

<!-- #region id="3l7nO8L4UeTu" -->
Supprimer une collection :
<!-- #endregion -->

```python id="4e6jbGz3UeTu"
client.delete_collection("sentence_vectors")
```

<!-- #region id="RhkQmeKhUeTu" -->
Sauvegarde et récupération des données (persist_directory)
<!-- #endregion -->

```python id="UWEGTV3PUeTu"
client = chromadb.Client(Settings(
    persist_directory="./chroma_data",
    chroma_db_impl="duckdb+parquet",
))
collection = client.get_collection("sentence_vectors")

```

<!-- #region id="a3NUt-S9UeTu" -->
# Stockage, Lecture et découpage de PDF
<!-- #endregion -->

<!-- #region id="kqrmim9aUeTu" -->
## Base documentaire
<!-- #endregion -->

<!-- #region id="5Dt9vTbqUeTu" -->

<!-- #endregion -->

<!-- #region id="i-ZZfxmMUeTu" -->
### **MINIO**
<!-- #endregion -->

<!-- #region id="pjWExuwHUeTu" -->
#### **Étape 1 : Installation du Service MINIO**

1. **Installation de Minio via Docker :**  
   
   Exécutez la commande suivante pour lancer un conteneur MINIO :
   ```yaml
    version: '3.8'
    services:
    minio:
        image: minio/minio:latest
        container_name: minio
        environment:
        - MINIO_ACCESS_KEY=minioaccesskey
        - MINIO_SECRET_KEY=miniosecretkey
        - MINIO_ROOT_USER=admin
        - MINIO_ROOT_PASSWORD=admin123
        ports:
        - "9002:9000"
        volumes:
        - ./minio/data:/data
        command: server /data
        restart: always
    volumes:
    minio_data:   
    ```

On le lance avec la commande
   ```bash
    docker-compose up -d
   ```
<!-- #endregion -->

<!-- #region id="PotQSJ9iUeTv" -->
#### **Étape 2 : Décharger un fichier PDF dans MinIO**
<!-- #endregion -->

```python id="3cbJ7cgsUeTv"
!pip install minio -q
```

```python id="VmD30Q-tUeTv" outputId="8ae622cb-33ac-4c97-c792-8d2c9d9fa53a"
from minio import Minio
from minio.error import S3Error

# Connexion à MinIO
client = Minio(
    "192.168.10.113:9002",  # L'adresse de votre serveur MinIO
    access_key="admin",  # Clé d'accès minioaccesskey
    secret_key="admin123",  # Clé secrète
    secure=False  # Désactiver HTTPS si nécessaire
)

# Créer un bucket (si ce n'est pas déjà fait)
bucket_name = "pdf-bucket"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name) #bucket_name

# Télécharger le fichier PDF
with open("C:/Users/horellou.florian/Downloads/test.pdf", "rb") as file_data:
    client.put_object(bucket_name, "mon_pdf.pdf", file_data, length=-1, part_size=10*1024*1024)

print("Fichier PDF téléchargé avec succès")
```

<!-- #region id="vHyN-WhVUeTw" -->
#### **Étape 3 : Télécharger un fichier PDF depuis MinIO**
<!-- #endregion -->

```python id="iteF_GQrUeTw" outputId="4e28c54b-9de7-4d99-ef04-95dc2fafc26b"
# Télécharger le fichier PDF
response = client.get_object(bucket_name, "mon_pdf.pdf")
with open("downloaded_pdf.pdf", "wb") as file_data:
    for data in response.stream(32*1024):
        file_data.write(data)

print("Fichier PDF téléchargé avec succès")
```

<!-- #region id="MPlVFIIZUeTw" -->
### **MongoDB (GridFS)**
<!-- #endregion -->

<!-- #region id="umHi-SfVUeTw" -->
#### **Étape 1 : Installation du Service MongoDB**

1. **Installation de MongoDB via Docker :**
   
   Exécutez la commande suivante pour lancer un conteneur MongoDB :
   ```yaml
    version: '3.8'
    services:
    mongodb:
        image: mongo:latest
        container_name: mongo-pdf
        ports:
        - "27017:27017"
        volumes:
        - mongodb_data:/data/db
    volumes:
    mongodb_data:
    ```

On le lance avec la commande

```bash
    docker-compose up -d
```
<!-- #endregion -->

<!-- #region id="VQti2mFyUeTw" -->
#### **Étape 2 : GridFS dans MongoDB :**
<!-- #endregion -->

```python id="jlWPWkDXUeTw"
!pip install pymongo -q
```

```python id="96Ou8vSEUeTx"
from pymongo import MongoClient
import gridfs

client = MongoClient('mongodb://192.168.10.113:27017/')
db = client['pdf_database']
fs = gridfs.GridFS(db)

# Lire et Stocker le fichier PDF
with open('C:/Users/horellou.florian/Downloads/test.pdf', 'rb') as f:
    file_id = fs.put(f, filename='mon_pdf.pdf')

```

```python id="-qK7QozSUeTx"
# Lire le fichier stocké
pdf_file = fs.get(file_id)
with open('downloaded_pdf.pdf', 'wb') as f:
    f.write(pdf_file.read())
```

<!-- #region id="hVtnhfHiUeTx" -->
### **PostgreSQL (pg_partman)**
<!-- #endregion -->

<!-- #region id="m9Zo6urtUeTx" -->
#### **Étape 1 : Installation du Service PostgreSQL**

1. **Installation de PostgreSQL via Docker :**
   
   Exécutez la commande suivante pour lancer un conteneur PostgreSQL :
   ```yaml
    version: '3.8'
    services:
    postgres:
        image: postgres:latest
        container_name: postgres-pdf
        environment:
        POSTGRES_PASSWORD: exemple
        ports:
        - "5436:5432"
        volumes:
        - postgres_data:/var/lib/postgresql/data
    volumes:
    postgres_data:
    ```

On le lance avec la commande

```bash
    docker-compose up -d
```
<!-- #endregion -->

<!-- #region id="8WxGci-SUeTy" -->
#### **Étape 2 : pg_partman avec postreSQL**
<!-- #endregion -->

```python id="K5PVDzhkUeTy"
!pip install psycopg2 -q
```

```python id="xv0Mr7NRUeTy"
import psycopg2

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(dbname="postgres", user="postgres",   password="example", host="192.168.10.113" , port=5436 )
cursor = conn.cursor()

# Création de la table pour stocker les PDF
cursor.execute('''CREATE TABLE IF NOT EXISTS pdf_files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255),
    file_data BYTEA
)''')

# Lire le fichier PDF et l'insérer
with open('C:/Users/horellou.florian/Downloads/test.pdf', 'rb') as f:
    file_data = f.read()
    cursor.execute("INSERT INTO pdf_files (filename, file_data) VALUES (%s, %s)", ('mon_pdf.pdf', file_data))

# Commit et fermer la connexion
conn.commit()

# Récupérer le fichier PDF
cursor.execute("SELECT file_data FROM pdf_files WHERE filename = %s", ('mon_pdf.pdf',))
pdf_data = cursor.fetchone()[0]
with open('downloaded_pdf.pdf', 'wb') as f:
    f.write(pdf_data)

cursor.close()
conn.close()
```

<!-- #region id="hDggWjSZUeTy" -->
## Lecture et extraction d'informations à partir de pdf
<!-- #endregion -->

<!-- #region id="vEmA-U9yUeTy" -->
### Version A (fitz / PyMuPDF)
<!-- #endregion -->

```python id="IOQHu5rHUeTy"
!pip install fitz -q
```

```python id="Yfinq6fQUeTz"
!pip install pymupdf -q
```

```python id="Qvg_FbJnUeTz"
#!pip install pypdfium2 -q
```

```python id="t2l5FtmWUeTz"
!pip install PyWavelets -q
```

```python id="xz6nn9S5UeTz"
!pip install pdf2image -q
```

```python id="xmrUHXhWUeTz" outputId="30c81a16-86cc-4fa3-d1de-281e54f98b8c"
# !pip uninstall camelot -q
# !pip uninstall camelot-py -q
!pip install camelot-py[cv] -q
```

```python id="nYqSXdn0UeTz"
#!pip install camelot -q
```

```python id="8_guTtTBUeT0" outputId="12ed4800-87f7-4fa6-9436-3242ef042578"
import fitz  # PyMuPDF pour extraire le texte
from pdf2image import convert_from_path  # Pour extraire les images
import camelot  # Pour extraire les tableaux
```

```python id="DLJ8mivZUeT0"
def extract_text_blocks_from_pdf(pdf_path):
    """
    Extrait le texte d'un PDF en segments logiques (pages et paragraphes).
    """
    document = fitz.open(pdf_path)
    text_blocks = []

    for page_num in range(len(document)):
        page = document[page_num]
        blocks = page.get_text("blocks")  # Récupération par blocs
        for block in blocks:
            if block[4].strip():  # Exclut les blocs vides
                text_blocks.append({
                    "page": page_num + 1,
                    "text": block[4].strip()
                })
    return text_blocks

def extract_images_from_pdf(pdf_path, output_folder="output_images"):
    document = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(document)):
        page = document[page_num]
        pix = page.get_pixmap(dpi=300)  # Conversion en image avec résolution 300 DPI
        image_path = f"{output_folder}/page_{page_num + 1}.png"
        pix.save(image_path)
        image_paths.append(image_path)

    """
    Extrait les images d'un fichier PDF et les enregistre dans un dossier.

    :param pdf_path: Chemin vers le fichier PDF
    :param output_folder: Chemin vers le dossier de sortie
    """
    # Ouvrir le fichier PDF
    pdf_document = fitz.open(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir les pages et extraire les images
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # Type de fichier (jpg, png, etc.)

            # Nom du fichier image
            image_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
            output_path = os.path.join(output_folder, image_filename)

            # Sauvegarder l'image
            with open(output_path, "wb") as image_file:
                image_file.write(image_bytes)

            print(f"Image extraite : {output_path}")

    pdf_document.close()


    return image_paths

def extract_tables_from_pdf(pdf_path, output_folder="output_tables"):
    """
    Extrait les tableaux d'un PDF et les sauvegarde en CSV.
    """
    tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")  # Stream ou lattice selon le PDF
    table_paths = []

    for i, table in enumerate(tables):
        table_path = f"{output_folder}/table_{i + 1}.csv"
        table.to_csv(table_path)
        table_paths.append(table_path)

    return table_paths

def process_pdf(pdf_path, output_image_folder="output_images", output_table_folder="output_tables"):
    """
    Pipeline complet pour traiter un PDF : texte, images, tableaux.
    """
    # Extraction du texte
    text_blocks = extract_text_blocks_from_pdf(pdf_path)

    # Extraction des images
    images = extract_images_from_pdf(pdf_path, output_image_folder)

    # Extraction des tableaux
    tables = extract_tables_from_pdf(pdf_path, output_table_folder)

    return {
        "text_blocks": text_blocks,
        "images": images,
        "tables": tables
    }
```

```python id="fpLmkL-6UeT0"
# Exemple d'utilisation
if __name__ == "__main__":
    pdf_path = "C:/Users/horellou.florian/Downloads/test.pdf"  # Chemin du PDF
    output_path = "C:/Users/horellou.florian/Downloads/extract"
    result = process_pdf(pdf_path, output_path , output_path)

    # Afficher les résultats
    print("\n--- Texte extrait ---")
    for block in result["text_blocks"][:5]:  # Afficher quelques blocs
        print(f"Page {block['page']}:\n{block['text']}\n")

    print("\n--- Images extraites ---")
    print("\n".join(result["images"]))

    print("\n--- Tableaux extraits ---")
    print("\n".join(result["tables"]))

```

```python id="OpMGOk10UeT0"
def extract_images_filtered(pdf_path, output_folder, min_width=1000, min_height=1000, min_size_bytes=1000):
    """
    Extrait les images d'un fichier PDF, avec un filtrage sur les dimensions et la taille du fichier.

    :param pdf_path: Chemin vers le fichier PDF
    :param output_folder: Chemin vers le dossier de sortie
    :param min_width: Largeur minimale pour conserver une image
    :param min_height: Hauteur minimale pour conserver une image
    :param min_size_bytes: Taille minimale en octets pour conserver une image
    """
    # Ouvrir le fichier PDF
    pdf_document = fitz.open(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir les pages et extraire les images
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # Type de fichier (jpg, png, etc.)
            width = base_image["width"]
            height = base_image["height"]

            # Vérifier les dimensions minimales
            if width < min_width or height < min_height:
                print(f"Ignoré (dimensions trop petites) : Page {page_number+1}, Image {img_index+1}, {width}x{height}")
                continue

            # Vérifier la taille minimale du fichier
            if len(image_bytes) < min_size_bytes:
                print(f"Ignoré (taille de fichier trop petite) : Page {page_number+1}, Image {img_index+1}, {len(image_bytes)} octets")
                continue

            # Si l'image respecte les critères, elle est sauvegardée
            image_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
            output_path = os.path.join(output_folder, image_filename)

            with open(output_path, "wb") as image_file:
                image_file.write(image_bytes)

            print(f"Image extraite : {output_path} (Dimensions : {width}x{height}, Taille : {len(image_bytes)} octets)")

    pdf_document.close()
```

```python id="OFslNOJnUeT1"
# Exemple d'utilisation

pdf_path =  "C:/Users/horellou.florian/Downloads/test.pdf"
output_folder = "C:/Users/horellou.florian/Downloads/extract"

extract_images_filtered(pdf_path, output_folder, min_width=100, min_height=100, min_size_bytes=1000) # 10000 = 10ko

```

<!-- #region id="SuSMMUKzUeT1" -->
### Version B (pdfplumber)
<!-- #endregion -->

```python id="qmqw0FQ3UeT1"
!pip install pdfplumber -q
```

```python id="45UHY3fcUeT1"
!pip install pillow -q
```

```python id="EteWQ6Q9UeT1"
!pip install pytesseract -q
```

```python id="qNvtkhjzUeT2" outputId="8dfd3a02-470e-42b5-d740-a88b9341809f"
import pytesseract

# Spécifiez le chemin de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Tesseract est prêt !")
```

```python id="jP49hkgZUeT2"
import pdfplumber
import os
from PIL import Image
import pytesseract


def extract_images_with_pdfplumber(pdf_path, output_folder, min_width=150, min_height=150, ocr_filter=True):
    """
    Extrait des images d'un fichier PDF à l'aide de pdfplumber.
    Les images peuvent être filtrées par dimensions et analyse OCR.

    :param pdf_path: Chemin vers le fichier PDF
    :param output_folder: Chemin vers le dossier de sortie
    :param min_width: Largeur minimale pour considérer une image comme un schéma
    :param min_height: Hauteur minimale pour considérer une image comme un schéma
    :param ocr_filter: Si True, applique un OCR pour filtrer les images contenant du texte
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            print(f"Traitement de la page {page_number}...")

            # Extraire toutes les images de la page
            for img_index, image in enumerate(page.images):
                # Extraire les métadonnées de l'image
                x0, y0, x1, y1 = image["x0"], image["y0"], image["x1"], image["y1"]
                width, height = int(x1 - x0), int(y1 - y0)

                # Filtrer par dimensions
                if width < min_width or height < min_height:
                    print(f"Ignoré : Image trop petite ({width}x{height}) à la page {page_number}")
                    continue

                # Extraire l'image
                image_data = page.within_bbox((x0, y0, x1, y1)).to_image(resolution=300)
                image_path = os.path.join(output_folder, f"page_{page_number}_img_{img_index + 1}.png")
                image_data.save(image_path, format="PNG")
                print(f"Image extraite : {image_path} ({width}x{height})")

                # Optionnel : appliquer OCR pour filtrer les images contenant du texte
                if ocr_filter:
                    if contains_text(image_path):
                        print(f"Schéma détecté (via OCR) : {image_path}")
                    else:
                        print(f"Ignoré : Pas de texte détecté dans {image_path}")
                        os.remove(image_path)

    print(f"Extraction terminée. Les schémas pertinents sont enregistrés dans {output_folder}")


def contains_text(image_path):
    """
    Vérifie si une image contient du texte en utilisant Tesseract OCR.

    :param image_path: Chemin vers l'image
    :return: True si du texte est détecté, False sinon
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return len(text.strip()) > 0  # Retourne True si du texte est détecté
    except Exception as e:
        print(f"Erreur OCR pour {image_path}: {e}")
        return False


```

```python id="_vD3wdnXUeT3"
# Exemple d'utilisation

pdf_path =  "C:/Users/horellou.florian/Downloads/test.pdf"
output_folder = "C:/Users/horellou.florian/Downloads/extract"

extract_images_with_pdfplumber(pdf_path, output_folder, min_width=5, min_height=5, ocr_filter=True)
```

```python id="NsK2ZRZ0UeT3"
import pdfplumber
import os
from PIL import Image


def extract_all_images(pdf_path, output_folder):
    """
    Extrait toutes les images, y compris celles du fond, d'un fichier PDF à l'aide de pdfplumber.

    :param pdf_path: Chemin vers le fichier PDF
    :param output_folder: Chemin vers le dossier de sortie
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            print(f"Traitement de la page {page_number}...")

            # Extraire toutes les images (y compris celles en arrière-plan)
            for img_index, image in enumerate(page.images):
                # Coordonnées de l'image
                x0, y0, x1, y1 = image["x0"], image["y0"], image["x1"], image["y1"]

                # Extraire l'image
                cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image(resolution=300)
                image_path = os.path.join(output_folder, f"page_{page_number}_img_{img_index + 1}.png")
                cropped_image.save(image_path, format="PNG")
                print(f"Image extraite : {image_path}")

            # Vérifier s'il y a des objets graphiques dans le fond
            vector_graphics = page.objects["vector"]
            if vector_graphics:
                print(f"Objets vectoriels détectés à la page {page_number} : {len(vector_graphics)}")
                # Vous pourriez ajouter un traitement supplémentaire ici pour les graphiques vectoriels

    print(f"Extraction terminée. Images et graphiques enregistrés dans {output_folder}")

```

```python id="p53nVlAgUeT3"
# Exemple d'utilisation

pdf_path =  "C:/Users/horellou.florian/Downloads/test.pdf"
output_folder = "C:/Users/horellou.florian/Downloads/extract"

extract_all_images(pdf_path, output_folder)
```

```python id="HRyKNmfBUeT3"
import re
from PyPDF2 import PdfReader
import os


def sanitize_filename(name):
    """
    Nettoie un nom de fichier en remplaçant les caractères non autorisés.
    """
    return re.sub(r"[^\w\-_\.]", "_", name)


def extract_objects_from_pdf(pdf_path, output_folder):
    """
    Extrait toutes les images et autres ressources d'un fichier PDF à l'aide de PyPDF2.

    :param pdf_path: Chemin vers le fichier PDF
    :param output_folder: Dossier où les objets extraits seront enregistrés
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        print(f"Traitement de la page {page_number}...")

        # Vérification des ressources dans la page
        if "/XObject" in page["/Resources"]:
            xobjects = page["/Resources"]["/XObject"].get_object()

            for obj_name, obj in xobjects.items():
                # Déréférencer l'objet pour accéder à ses propriétés
                xobj = obj.get_object()

                # Vérifier le type de l'objet
                if "/Subtype" in xobj and xobj["/Subtype"] == "/Image":
                    # Extraire les métadonnées de l'image
                    width = xobj["/Width"]
                    height = xobj["/Height"]
                    color_space = xobj.get("/ColorSpace", "Unknown")
                    print(f"Image détectée : {obj_name} (w={width}, h={height}, color_space={color_space})")

                    # Extraire les données binaires de l'image
                    img_data = xobj._data

                    # Nettoyer le nom de l'objet pour éviter les caractères invalides
                    safe_name = sanitize_filename(obj_name)

                    # Construire le chemin de sauvegarde
                    img_path = os.path.join(output_folder, f"page_{page_number}_{safe_name}.jpg")

                    # Sauvegarder l'image
                    with open(img_path, "wb") as img_file:
                        img_file.write(img_data)
                    print(f"Image enregistrée : {img_path}")

                else:
                    print(f"Objet {obj_name} de type {xobj.get('/Subtype', 'Inconnu')} ignoré.")

    print(f"Extraction terminée. Les objets extraits sont dans {output_folder}")

```

```python id="rIubABKaUeT3" outputId="a5d32db5-f546-4ea8-f02b-1c057316b15a"
# Exemple d'utilisation

pdf_path =  "C:/Users/horellou.florian/Downloads/test.pdf"
output_folder = "C:/Users/horellou.florian/Downloads/extract"

extract_objects_from_pdf(pdf_path, output_folder)

```

<!-- #region id="6Wofm9-kUeT4" -->
### C - pdfplumber + fitz
<!-- #endregion -->

```python id="meiFJD_MUeT4"
!pip install reportlab -q
```

```python id="J0VohwmGUeT4"
import os
import fitz
import pdfplumber
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Preformatted



def extract_header_fontsize_from_pdf(pdf_path):
    font_size_counter = Counter()

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(len(pdf.pages)):
            # lines1 = pdf.pages[i].extract_text().split('\n')
            words = pdf.pages[i].extract_words(extra_attrs=['fontname', 'size'])
            lines = {}

            for word in words:
                line_num = word['top']
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(word)

            for line_words in lines.values():
                font_size_counter[line_words[0]['size']] += 1

    # Find the font sizes that were used more than once
    repeated_sizes = [size for size, count in font_size_counter.items() if count > 1]
    print(repeated_sizes)
    # Return the highest font size among the repeated sizes
    if repeated_sizes:
        return max(repeated_sizes)
    else:
        return None


def extract_lines_with_font_size(pdf_path, target_font_size):
    lines_with_target_font_size = []

    with pdfplumber.open(pdf_path) as pdf:
        for i in range(len(pdf.pages)):
            words = pdf.pages[i].extract_words(extra_attrs=['fontname', 'size'])
            lines = {}

            for word in words:
                line_num = word['top']
                if line_num not in lines:
                    lines[line_num] = []
                lines[line_num].append(word)

            for line_num, line_words in lines.items():
                line_font_sizes = [word['size'] for word in line_words]
                if target_font_size in line_font_sizes:
                    line_text = ' '.join([word['text'] for word in line_words])
                    lines_with_target_font_size.append(line_text)

    return lines_with_target_font_size

def write_chunks_to_pdf(chunks, output_pdf_path):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    story = []
    for chunk in chunks:
        preformatted = Preformatted(chunk, styles["Normal"])
        story.append(preformatted)

    doc.build(story)

def save_chunks_as_pdfs(chunks, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, chunk in enumerate(chunks, start=1):
        output_pdf_path = os.path.join(output_folder, f"output_pdf_part{i}.pdf")
        write_chunks_to_pdf([chunk], output_pdf_path)
        print(f"PDF saved at: {output_pdf_path}")

def extract_chunks_from_pdf(pdf_path, markers):
    chunks = []
    current_chunk = []
    current_marker_index = 0

    pdf_document = fitz.open(pdf_path)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text = page.get_text("text")

        lines = text.split('\n')
        for line in lines:
            if current_marker_index < len(markers) and markers[current_marker_index] in line:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                current_marker_index += 1

            current_chunk.append(line)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    pdf_document.close()

    output_folder = "C:/Users/horellou.florian/Downloads/extract/"
    save_chunks_as_pdfs(chunks, output_folder)

    return chunks


```

```python id="hlFTWahpUeT4" outputId="0f4e4061-717d-450d-9aa0-83b8cfdd3a1a"
pdf_path =  "C:/Users/horellou.florian/Downloads/test.pdf"
output_folder = "C:/Users/horellou.florian/Downloads/extract"

def main():
    pdf1 = 'C:/Users/horellou.florian/Downloads/test.pdf'
    extracted_font_size = extract_header_fontsize_from_pdf(pdf1)
    extracted_headers = extract_lines_with_font_size(pdf1,extracted_font_size)
    print(extracted_headers)
    chunks = extract_chunks_from_pdf(pdf1, extracted_headers)

if __name__ == "__main__":
    main()
```

<!-- #region id="K677bu2VUeT4" -->
### D - ON EST OKAYYYYYYYYYYYYYYYYYYYYYYYY !
<!-- #endregion -->

```python id="D_Q1Q1sZUeT5"
!pip install opencv-python -q
```

```python id="JshuGjcOUeT5"
!pip install pymupdf -q
```

```python id="ectSRvYEUeT5"
import os
import cv2
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image

# Répertoire où enregistrer les figures extraites
OUTPUT_DIR = "C:/Users/horellou.florian/Downloads/extract/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chemin du PDF source
PDF_PATH = "C:/Users/horellou.florian/Downloads/test.pdf"

# Indique où se trouve tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def pdf_page_to_image(pdf_path, page_number, zoom=2.0):
    """
    Convertit une page PDF en image PIL avec un zoom pour plus de détails.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"Le PDF compte {doc.page_count} pages, page demandée : {page_number}")
        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pil_img
    finally:
        doc.close()

def merge_overlapping_boxes(boxes):
    """
    Fusionne les boîtes si l'aire commune est significative.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged_boxes = []
    current_box = boxes[0]

    for box in boxes[1:]:
        x0, y0, x1, y1 = current_box
        nx0, ny0, nx1, ny1 = box

        ix0 = max(x0, nx0)
        iy0 = max(y0, ny0)
        ix1 = min(x1, nx1)
        iy1 = min(y1, ny1)

        if ix0 < ix1 and iy0 < iy1:
            intersection_area = (ix1 - ix0) * (iy1 - iy0)
            area_current = (x1 - x0) * (y1 - y0)
            area_next = (nx1 - nx0) * (ny1 - ny0)

            if intersection_area > min(area_current, area_next) - intersection_area:
                current_box = (
                    min(x0, nx0),
                    min(y0, ny0),
                    max(x1, nx1),
                    max(y1, ny1)
                )
            else:
                merged_boxes.append(current_box)
                current_box = box
        else:
            merged_boxes.append(current_box)
            current_box = box

    merged_boxes.append(current_box)

    return merged_boxes


def detect_figures_in_page(pil_img):
    """
    Détecte les zones non textuelles dans une page image et retourne les bounding boxes.
    """
    np_img = np.array(pil_img)
    data = pytesseract.image_to_data(np_img, output_type=Output.DICT, lang="eng")
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    mask_img = gray_img.copy()

    for i in range(len(data["text"])):
        if data["text"][i].strip():
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            cv2.rectangle(mask_img, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)

    _, bin_img = cv2.threshold(mask_img, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    figure_bboxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40 and h > 40:
            figure_bboxes.append((x, y, x + w, y + h))

    merged_bboxes = merge_overlapping_boxes(figure_bboxes)
    return merged_bboxes

def get_text_bboxes(page, zoom=2.0):
    """
    Extrait les coordonnées des blocs de texte d'une page PDF.
    """
    text_bboxes = []
    try:
        for block in page.get_text("dict")["blocks"]:
            if block['type'] == 0:
                x0, y0, x1, y1 = block['bbox']
                x0 *= zoom
                y0 *= zoom
                x1 *= zoom
                y1 *= zoom
                text_bboxes.append((int(x0), int(y0), int(x1), int(y1)))
        return text_bboxes
    except KeyError:
        return text_bboxes

def adjust_image_boxes_with_text_boxes(image_boxes, text_boxes):
    """
    Ajuste chaque boîte d'image en fonction des boîtes de texte qui se chevauchent.

    Args:
        image_boxes (list): Liste des boîtes d'image sous forme de tuples (x0, y0, x1, y1).
        text_boxes (list): Liste des boîtes de texte sous forme de tuples (x0, y0, x1, y1).

    Returns:
        list: Liste des boîtes ajustées.
    """
    adjusted_boxes = []

    for image_box in image_boxes:
        x0, y0, x1, y1 = image_box
        new_coords = [x0, y0, x1, y1]

        for text_box in text_boxes:
            tx0, ty0, tx1, ty1 = text_box

            # Vérifie si les boîtes se chevauchent
            if not (tx1 < x0 or tx0 > x1 or ty1 < y0 or ty0 > y1):
                # Ajuste les coordonnées si la boîte de texte est plus "large"
                new_coords[0] = min(new_coords[0], tx0)
                new_coords[1] = min(new_coords[1], ty0)
                new_coords[2] = max(new_coords[2], tx1)
                new_coords[3] = max(new_coords[3], ty1)

        # Ajoute la boîte ajustée à la liste
        adjusted_boxes.append(tuple(new_coords))

    return adjusted_boxes

def save_cropped_images(pdf_path, output_dir, margin=5):
    """
    Parcourt le PDF et enregistre chaque figure détectée comme une image individuelle.
    """
    doc = fitz.open(pdf_path)
    nb_pages = doc.page_count

    for page_num in range(1, nb_pages + 1):
        print(f"[INFO] Traitement de la page {page_num}/{nb_pages}...")
        page = doc.load_page(page_num - 1)

        pil_page_img = pdf_page_to_image(pdf_path, page_num, zoom=2.0)
        images_boxes = detect_figures_in_page(pil_page_img)
        texts_boxes = get_text_bboxes(page)

        # Ajuste les boîtes d'image en fonction des boîtes de texte
        adjusted_boxes = adjust_image_boxes_with_text_boxes(images_boxes, texts_boxes)

        # Supprime les doublons pour éviter les enregistrements multiples
        unique_boxes = list(set(adjusted_boxes))

        for idx, (x0, y0, x1, y1) in enumerate(unique_boxes, start=1):
            # Ajoute une marge autour de la boîte
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(pil_page_img.width, x1 + margin)
            y1 = min(pil_page_img.height, y1 + margin)

            cropped = pil_page_img.crop((x0, y0, x1, y1))


            output_filename = f"figure_page{page_num}_{idx}.png"
            output_path = os.path.join(output_dir, output_filename)
            cropped.save(output_path)
            print(f"[INFO] Enregistré : {output_filename}")

    doc.close()
    print("[FIN] Extraction des figures terminée.")

```

```python id="DtEk2SeCUeT5" outputId="11030988-338c-4ce4-c71e-eeebe6439495"
save_cropped_images(PDF_PATH, OUTPUT_DIR)
```

<!-- #region id="jhedlScxUeT5" -->
## Suite
<!-- #endregion -->

```python id="9Cv2rErlUeT6"
import os
import cv2
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
import numpy as np
from PIL import Image

# Répertoire où enregistrer les figures extraites
OUTPUT_DIR = "C:/Users/horellou.florian/Downloads/extract/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chemin du PDF source
PDF_PATH = "C:/Users/horellou.florian/Downloads/test.pdf"

# Indique où se trouve tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def pdf_page_to_image(pdf_path, page_number, zoom=2.0):
    """
    Convertit une page PDF en image PIL avec un zoom pour plus de détails.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"Le PDF compte {doc.page_count} pages, page demandée : {page_number}")
        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pil_img
    finally:
        doc.close()

def merge_overlapping_boxes(boxes):
    """
    Fusionne les boîtes si l'aire commune est significative.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged_boxes = []
    current_box = boxes[0]

    for box in boxes[1:]:
        x0, y0, x1, y1 = current_box
        nx0, ny0, nx1, ny1 = box

        ix0 = max(x0, nx0)
        iy0 = max(y0, ny0)
        ix1 = min(x1, nx1)
        iy1 = min(y1, ny1)

        if ix0 < ix1 and iy0 < iy1:
            intersection_area = (ix1 - ix0) * (iy1 - iy0)
            area_current = (x1 - x0) * (y1 - y0)
            area_next = (nx1 - nx0) * (ny1 - ny0)

            if intersection_area > min(area_current, area_next) - intersection_area:
                current_box = (
                    min(x0, nx0),
                    min(y0, ny0),
                    max(x1, nx1),
                    max(y1, ny1)
                )
            else:
                merged_boxes.append(current_box)
                current_box = box
        else:
            merged_boxes.append(current_box)
            current_box = box

    merged_boxes.append(current_box)

    return merged_boxes


def detect_figures_in_page(pil_img):
    """
    Détecte les zones non textuelles dans une page image et retourne les bounding boxes.
    """
    np_img = np.array(pil_img)
    data = pytesseract.image_to_data(np_img, output_type=Output.DICT, lang="eng")
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    mask_img = gray_img.copy()

    for i in range(len(data["text"])):
        if data["text"][i].strip():
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            cv2.rectangle(mask_img, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)

    _, bin_img = cv2.threshold(mask_img, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    figure_bboxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40 and h > 40:
            figure_bboxes.append((x, y, x + w, y + h))

    merged_bboxes = merge_overlapping_boxes(figure_bboxes)
    return merged_bboxes

def get_text_bboxes(page, zoom=2.0):
    """
    Extrait les coordonnées des blocs de texte d'une page PDF.
    """
    text_bboxes = []
    try:
        for block in page.get_text("dict")["blocks"]:
            if block['type'] == 0:
                x0, y0, x1, y1 = block['bbox']
                x0 *= zoom
                y0 *= zoom
                x1 *= zoom
                y1 *= zoom
                text_bboxes.append((int(x0), int(y0), int(x1), int(y1)))
        return text_bboxes
    except KeyError:
        return text_bboxes

def adjust_image_boxes_with_text_boxes(image_boxes, text_boxes):
    """
    Ajuste chaque boîte d'image en fonction des boîtes de texte qui se chevauchent.

    Args:
        image_boxes (list): Liste des boîtes d'image sous forme de tuples (x0, y0, x1, y1).
        text_boxes (list): Liste des boîtes de texte sous forme de tuples (x0, y0, x1, y1).

    Returns:
        list: Liste des boîtes ajustées.
    """
    adjusted_boxes = []

    for image_box in image_boxes:
        x0, y0, x1, y1 = image_box
        new_coords = [x0, y0, x1, y1]

        for text_box in text_boxes:
            tx0, ty0, tx1, ty1 = text_box

            # Vérifie si les boîtes se chevauchent
            if not (tx1 < x0 or tx0 > x1 or ty1 < y0 or ty0 > y1):
                # Ajuste les coordonnées si la boîte de texte est plus "large"
                new_coords[0] = min(new_coords[0], tx0)
                new_coords[1] = min(new_coords[1], ty0)
                new_coords[2] = max(new_coords[2], tx1)
                new_coords[3] = max(new_coords[3], ty1)

        # Ajoute la boîte ajustée à la liste
        adjusted_boxes.append(tuple(new_coords))

    return adjusted_boxes


#### FONCTION RAJOUTE #####

def is_grayscale(np_img, sat_threshold=15, sat_std_threshold=10):
    """
    Détermine si une image (sous forme NumPy) est majoritairement en niveaux de gris
    en se basant sur la saturation (canal S en HSV).
    """
    hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    sat = hsv_img[:, :, 1]
    mean_sat = np.mean(sat)
    std_sat = np.std(sat)
    return (mean_sat < sat_threshold) and (std_sat < sat_std_threshold)


def is_table_heuristic(np_img, line_threshold=5):
    """
    Détermine si l'image extraite correspond potentiellement à un tableau
    en détectant un certain nombre de lignes horizontales/verticales.
    """
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    # Binarisation (inverse)
    _, binarized = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Détection lignes horizontales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, horizontal_kernel)

    # Détection lignes verticales
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_vertical = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, vertical_kernel)

    contours_h, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_v, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nb_horizontal = len(contours_h)
    nb_vertical = len(contours_v)

    return (nb_horizontal >= line_threshold and nb_vertical >= line_threshold)


def extract_text_outside_figures(pdf_path, page, figure_boxes, overlap_tolerance=0.1):
    """
    Extrait le texte se trouvant en dehors des zones 'figure_boxes' pour la page donnée.
    Utilise la même échelle (zoom) que celle appliquée lors de la conversion PDF->image.
    """
    blocks = page.get_text("blocks")  # liste de blocs [x0,y0,x1,y1, "texte", ...]
    text_extracted = []

    for block in blocks:
        x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
        content = block[4].strip()
        if not content:
            continue

        block_area = (x1 - x0) * (y1 - y0)
        is_outside = True

        for (fx0, fy0, fx1, fy1) in figure_boxes:
            ix0 = max(x0, fx0)
            iy0 = max(y0, fy0)
            ix1 = min(x1, fx1)
            iy1 = min(y1, fy1)
            if ix0 < ix1 and iy0 < iy1:
                intersection_area = (ix1 - ix0) * (iy1 - iy0)
                # s'il y a un fort recouvrement, on rejette
                if intersection_area / float(block_area) > overlap_tolerance:
                    is_outside = False
                    break

        if is_outside:
            text_extracted.append(content)

    return text_extracted

#### FONCTION PRINCIPALE #####

def save_cropped_images2(pdf_path, output_dir, margin=5):
    """
    Parcourt le PDF et enregistre chaque figure détectée comme une image individuelle.
    """
    doc = fitz.open(pdf_path)
    nb_pages = doc.page_count

    for page_num in range(1, nb_pages + 1):
        print(f"[INFO] Traitement de la page {page_num}/{nb_pages}...")
        page = doc.load_page(page_num - 1)

        pil_page_img = pdf_page_to_image(pdf_path, page_num, zoom=2.0)
        images_boxes = detect_figures_in_page(pil_page_img)
        texts_boxes = get_text_bboxes(page)

        # Ajuste les boîtes d'image en fonction des boîtes de texte
        adjusted_boxes = adjust_image_boxes_with_text_boxes(images_boxes, texts_boxes)

        # Supprime les doublons pour éviter les enregistrements multiples
        unique_boxes = list(set(adjusted_boxes))

        # On pourrait prendre 'unique_boxes' pour recouvrir toutes les zones
        text_outside = extract_text_outside_figures(pdf_path, page, unique_boxes, overlap_tolerance=0.1)
        print("[INFO] Texte hors figures :")
        for t in text_outside:
            print(" -", t)

        for idx, (x0, y0, x1, y1) in enumerate(unique_boxes, start=1):
            # Ajoute une marge autour de la boîte
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(pil_page_img.width, x1 + margin)
            y1 = min(pil_page_img.height, y1 + margin)

            cropped = pil_page_img.crop((x0, y0, x1, y1))

            #### NOUVEL AJOUT ####
            cropped_np = np.array(cropped)

            if is_table_heuristic(cropped_np):
                figure_type = "table"
            elif is_grayscale(cropped_np):
                figure_type = "diagram_grayscale"
            else:
                figure_type = "photo_color"

            print(f"[INFO] Type de figure détecté : {figure_type}")


            #### FIN DE L AJOUT ####

            output_filename = f"figure_page{page_num}_{idx}_{figure_type}.png"
            output_path = os.path.join(output_dir, figure_type, output_filename)
            # Pensez à créer le dossier figure_type si besoin
            os.makedirs(os.path.join(output_dir, figure_type), exist_ok=True)
            cropped.save(output_path)
            print(f"[INFO] Enregistré : {output_filename}")

    doc.close()
    print("[FIN] Extraction des figures terminée.")

```

```python id="uezmEAGJUeT6" outputId="15244a7d-9833-44b0-f2bf-215611e339a7"
save_cropped_images2(PDF_PATH, OUTPUT_DIR)
```

<!-- #region id="XP1wlyQgUeT6" -->
## Test pour les tableau
<!-- #endregion -->

```python id="9qAptDS4UeT6"
import fitz  # PyMuPDF
import os

def is_table(block):
    """
    Vérifie si un bloc ressemble à un tableau en vérifiant la présence de 'lines' et en analysant sa structure.
    """
    # Vérifiez si le bloc contient des lignes
    if "lines" not in block:
        return False

    # Exemple de logique pour identifier un tableau
    # (Vous pouvez ajuster cette logique en fonction de votre PDF)
    return len(block["lines"]) > 1  # Considère un bloc comme un tableau s'il a plus d'une ligne

def extract_tables_as_images(input_file, output_dir):
    # Ouvrir le fichier PDF
    doc = fitz.open(input_file)

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcourir chaque page du PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Récupérer les zones de texte
        text_blocks = page.get_text("dict")["blocks"]

        # Filtrer les blocs de texte pour identifier les tableaux
        tables = [block for block in text_blocks if is_table(block)]

        # Enregistrer chaque tableau comme une image
        for i, table in enumerate(tables):
            # Récupérer les coordonnées du tableau
            bbox = fitz.Rect(table["bbox"])
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=bbox)  # Récupérer l'image du tableau
            pix.save(os.path.join(output_dir, f"page_{page_num}_table_{i}.png"))


# Utilisation de la fonction
# Définir les chemins
input_file = 'C:/Users/horellou.florian/Downloads/test.pdf'
output_dir = 'C:/Users/horellou.florian/Downloads/extract/tableaux'
extract_tables_as_images(input_file, output_dir)

```

```python id="OC28T8JWUeT7" outputId="9267e929-434d-4d37-e293-9c0e6e59cef6"
import os
from transformers import pipeline
from PIL import Image

# Configuration
CHEMIN_DOSSIER = "C:/Users/horellou.florian/Downloads/extract"  # Remplacez 'nom' par votre nom d'utilisateur si nécessaire
CATEGORIES = ['schemas', 'images', 'tableaux']
EXTENSIONS_IMAGES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

def charger_classificateur():
    """
    Charge le pipeline de classification d'images zéro-shot de Hugging Face utilisant CLIP.
    """
    try:
        classifier = pipeline("zero-shot-image-classification",
                              model="openai/clip-vit-base-patch32")
        return classifier
    except Exception as e:
        print(f"Erreur lors du chargement du classificateur : {e}")
        exit(1)

def predire_categorie(classifier, image_path, categories):
    """
    Prédit la catégorie d'une image en utilisant le classificateur chargé.
    """
    try:
        # Ouvrir l'image pour s'assurer qu'elle est valide
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erreur lors de l'ouverture de l'image {image_path} : {e}")
        return None

    try:
        result = classifier(image, candidate_labels=categories)
        # 'result' est une liste contenant un dictionnaire
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
            categorie_predite = result['labels'][0]
            score = result['scores'][0]
            print(f"Image: {os.path.basename(image_path)} => Catégorie: {categorie_predite} (Confiance: {score:.2f})")
            return categorie_predite
        else:
            print(f"Format inattendu du résultat pour l'image {image_path}: {result}")
            return None
    except Exception as e:
        print(f"Erreur lors de la classification de l'image {image_path} : {e}")
        return None

def renommer_images(chemin_dossier, classifier, categories):
    """
    Parcourt les images dans le dossier spécifié, les classe et les renomme en fonction de leur catégorie.
    """
    for nom_fichier in os.listdir(chemin_dossier):
        chemin_complet = os.path.join(chemin_dossier, nom_fichier)

        # Vérifier si c'est un fichier et une image
        if os.path.isfile(chemin_complet) and nom_fichier.lower().endswith(EXTENSIONS_IMAGES):
            categorie = predire_categorie(classifier, chemin_complet, categories)

            if categorie:
                # Déterminer le nouveau nom
                base_nom, extension = os.path.splitext(nom_fichier)
                nouveau_nom = f"{categorie}_{base_nom}{extension}"
                chemin_nouveau = os.path.join(chemin_dossier, nouveau_nom)

                try:
                    # Vérifier si le nouveau nom existe déjà pour éviter les collisions
                    if not os.path.exists(chemin_nouveau):
                        os.rename(chemin_complet, chemin_nouveau)
                        print(f"Renommé : {nom_fichier} -> {nouveau_nom}\n")
                    else:
                        print(f"Le fichier {nouveau_nom} existe déjà. Renommage ignoré.\n")
                except Exception as e:
                    print(f"Erreur lors du renommage de {nom_fichier} : {e}\n")
            else:
                print(f"Catégorie non déterminée pour {nom_fichier}. Fichier non renommé.\n")
        else:
            print(f"Fichier ignoré (non-image ou non valide) : {nom_fichier}\n")

def main():
    """
    Fonction principale qui orchestre le chargement du classificateur et le renommage des images.
    """
    # Vérifier si le dossier existe
    if not os.path.isdir(CHEMIN_DOSSIER):
        print(f"Le dossier spécifié n'existe pas : {CHEMIN_DOSSIER}")
        exit(1)

    print("Chargement du classificateur, cela peut prendre un moment...")
    classifier = charger_classificateur()
    print("Classificateur chargé avec succès.\n")

    print("Commencement du processus de renommage des images...\n")
    renommer_images(CHEMIN_DOSSIER, classifier, CATEGORIES)
    print("Processus terminé.")

if __name__ == "__main__":
    main()

```

<!-- #region id="LMMxF-3yUeT7" -->
#### CV2
<!-- #endregion -->

```python id="zCbxK8CtUeT7" outputId="109d3c13-b067-4c9f-e693-11ddfac744d8"

import cv2
import numpy as np
import os
import shutil

# Définir les chemins
input_dir = 'C:/Users/horellou.florian/Downloads/extract'
output_dir = 'C:/Users/horellou.florian/Downloads/extract/cv2'

# Créer le sous-dossier pour cette solution
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

def is_table(image_path):
    image = cv2.imread(image_path, 0)  # Charger en niveaux de gris
    if image is None:
        return False
    # Appliquer une binarisation adaptative
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Inverser les couleurs
    thresh = 255 - thresh

    # Détecter les lignes horizontales et verticales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combiner les lignes détectées
    table_structure = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)

    # Trouver les contours
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Ajuster le seuil selon la taille des tableaux
            return True
    return False

# Parcourir les images et identifier les tableaux
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(input_dir, filename)
        if is_table(image_path):
            # Copier l'image dans le dossier de sortie
            shutil.copy(image_path, os.path.join(output_dir, filename))

print("Détection terminée avec succès. Les tableaux identifiés sont dans le dossier 'OpenCV'.")


```

<!-- #region id="232U4w26UeT8" -->
#### camelot
<!-- #endregion -->

```python id="y06YBzCYUeT8" outputId="47b25633-d9c4-426f-8224-33b83faa5ff7"
# !pip uninstall camelot-py -q
!pip install camelot-py[cv] -q
```

```python id="PAJ-dePnUeT8" outputId="691aba9d-cc20-434e-95bf-130fdb93e9e2"
import camelot

# Specify the path to the PDF file

file_path = 'C:/Users/horellou.florian/Downloads/test.pdf'

# Read the tables from the PDF
tables = camelot.read_pdf(file_path)

# Print the number of tables found
print(f"Total tables found: {tables.n}")

# Print the content of the first table
print(tables[0].df)
```

```python id="XXAvmUvwUeT8"

```

<!-- #region id="I51JNfWRUeT8" -->
#### Pyteceract
<!-- #endregion -->

```python id="ESyBvMnOUeT9" outputId="ffe03976-168b-475c-81de-e9c4b86d7fbe"
import pytesseract

# Spécifiez le chemin de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Tesseract est prêt !")
```

```python id="XVelCAZFUeT9" outputId="1765de20-cbdc-46e3-ad84-e33edf3bd11d"
import cv2
import pytesseract
from PIL import Image
import os
import shutil

# Définir les chemins
input_dir = 'C:/Users/horellou.florian/Downloads/extract'
output_dir = 'C:/Users/horellou.florian/Downloads/extract/pytesseract'

# Créer le sous-dossier pour cette solution
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Configurer le chemin de Tesseract si nécessaire
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def is_table(image_path):
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(ocr_result['level'])
    lines = {}
    # Compter le nombre de mots par ligne
    for i in range(n_boxes):
        line_num = ocr_result['line_num'][i]
        if line_num in lines:
            lines[line_num] += 1
        else:
            lines[line_num] = 1
    # Si il y a plusieurs lignes avec plusieurs mots, cela peut indiquer un tableau
    if len(lines) > 3:  # Ajuster le seuil en fonction
        return True
    return False

# Parcourir les images et identifier les tableaux
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        image_path = os.path.join(input_dir, filename)
        if is_table(image_path):
            # Copier l'image dans le dossier de sortie
            shutil.copy(image_path, os.path.join(output_dir, filename))

print("Détection terminée avec succès. Les tableaux identifiés sont dans le dossier 'Tesseract'.")
```

<!-- #region id="6toXfX3HUeT9" -->
# Autres Tests
<!-- #endregion -->

<!-- #region id="2K9mE7ozUeT9" -->
#### Reconnaissance d'entité sur les images
<!-- #endregion -->

```python id="_jd6hnxgUeT9" outputId="36c98de8-c35e-4bac-b60d-03d3c9b9d61a"
!pip install pytesseract -q
```

```python id="CWPYJrajUeT-" outputId="17205aad-6be7-4dc7-f3a0-47ecd82db94d"
import pytesseract

# Spécifiez le chemin de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("Tesseract est prêt !")
```

```python id="LrkvLIoCUeT-"
import cv2
import pytesseract
import numpy as np

# Détection de texte avec Tesseract
def detect_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Appliquer une petite pré-traitement
    gray = cv2.medianBlur(gray, 3)

    # Extraire le texte avec Tesseract
    text = pytesseract.image_to_string(gray, lang='eng')

    # Vérifier la densité de texte (nombre de caractères significatifs)
    return len(text.strip()) > 10  # Texte significatif si plus de 10 caractères

# Détection de tableaux (présence de grilles)
def detect_table(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Détecter les lignes horizontales et verticales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combiner les lignes détectées
    grid = cv2.add(horizontal_lines, vertical_lines)

    # Détecter les contours des grilles
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 5  # Tableau probable si plusieurs grilles détectées

# Classification finale de l'image
def classify_image(image_path):
    if detect_text(image_path):
        return "Texte"
    elif detect_table(image_path):
        return "Tableau"
    else:
        return "Schéma/Image"

```

```python id="qF9SNcv2UeT-"
!pip install --upgrade tensorflow numpy
```

```python id="SMvDLkNTUeT-"
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Charger le modèle pré-entraîné
model = MobileNetV2(weights='imagenet')

image = 'figure_page6_5.png'  # tableau + schema
# image = "figure_page6_4.png" # schema

# image  = "figure_page8_5.png" #  schema
# image  = "figure_page4_2.png" # Tableau
img_path = classify_image(OUTPUT_DIR+image)

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Prédire la classe
preds = model.predict(x)
print('Prédictions:', decode_predictions(preds, top=3)[0])
```

```python id="LJbR54gxUeT-" outputId="0715dea1-fa17-451e-d1e2-9e46cc24df92"
# Exemple d'utilisation

print(f"L'image est classée comme : {category}")
```

<!-- #region id="qyW1670bUeT-" -->
#### **Box text**
<!-- #endregion -->

```python id="MVUWhTBOUeT_" outputId="d0908fd1-45d4-41f2-da71-2c494eca7e99"
import os
import fitz  # PyMuPDF
from PIL import Image

# Répertoire où enregistrer les zones de texte extraites
OUTPUT_DIR = "C:/Users/horellou.florian/Downloads/text_blocks/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chemin du PDF source
PDF_PATH = "C:/Users/horellou.florian/Downloads/test.pdf"

def pdf_page_to_image(pdf_path, page_number, zoom=2.0):
    """
    Convertit une page PDF en image PIL avec un zoom pour une meilleure résolution.
    """
    doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > doc.page_count:
            raise ValueError(f"Le PDF contient {doc.page_count} pages, page demandée : {page_number}")
        page = doc.load_page(page_number - 1)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pil_img
    finally:
        doc.close()

def extract_text_blocks_as_images(pdf_path, output_dir, margin=5, zoom=2.0):
    """
    Extrait les zones de texte (blocs ou paragraphes) d'un PDF et les enregistre comme images.
    """
    doc = fitz.open(pdf_path)
    nb_pages = doc.page_count

    for page_num in range(1, nb_pages + 1):
        print(f"[INFO] Traitement de la page {page_num}/{nb_pages}...")

        # Charger la page et l'image associée
        page = doc.load_page(page_num - 1)
        pil_page_img = pdf_page_to_image(pdf_path, page_num, zoom=zoom)

        # Extraire les blocs de texte
        text_blocks = []
        for block in page.get_text("dict")["blocks"]:
            if block['type'] == 0:  # Type 0 correspond aux blocs de texte
                x0, y0, x1, y1 = block['bbox']
                text_blocks.append((x0, y0, x1, y1))

        # Enregistrer chaque bloc comme image
        for idx, (x0, y0, x1, y1) in enumerate(text_blocks, start=1):
            # Appliquer une marge autour du bloc
            x0 = max(0, int(x0 * zoom) - margin)
            y0 = max(0, int(y0 * zoom) - margin)
            x1 = min(pil_page_img.width, int(x1 * zoom) + margin)
            y1 = min(pil_page_img.height, int(y1 * zoom) + margin)

            # Découper et enregistrer le bloc
            cropped_img = pil_page_img.crop((x0, y0, x1, y1))
            output_filename = f"text_block_page{page_num}_{idx}.png"
            output_path = os.path.join(output_dir, output_filename)
            cropped_img.save(output_path)
            print(f"[INFO] Enregistré : {output_filename}")

    doc.close()
    print("[FIN] Extraction des blocs de texte terminée.")

```

```python id="UWgLBG8WUeT_" outputId="aecf0e8c-7b1e-4ee6-9483-acb342f84a74"
extract_text_blocks_as_images(PDF_PATH, OUTPUT_DIR, margin=10, zoom=2.0)
```

```python id="5_pnj_piUeT_" outputId="75a0b2f4-5569-41e4-dc8e-e848b81a9e9d"
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Charger le modèle pré-entraîné
model = MobileNetV2(weights='imagenet')

# Charger une image PNG
img_path = 'image.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Prédire la classe
preds = model.predict(x)
print('Prédictions:', decode_predictions(preds, top=3)[0])
```

<!-- #region id="TAk9sy5CUeT_" -->
recupération des texts box
<!-- #endregion -->

```python id="T6JFH3CiUeUA" outputId="251ca1ae-2ae2-4266-e1b5-d919c96fc3fb"
!pip install matplotlib -q
```

```python id="IDVjxObSUeUA" outputId="1ee2a761-6d2c-46d1-e792-d1074f6df8d0"
import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from PIL import Image

import math
```

```python id="jC5X1CqRUeUA" outputId="f59dc4d1-8ba9-4e26-bf95-76c58e1cd365"
def get_text_bboxes(page_number, zoom=2.0):
    """
    Extrait les coordonnées des blocs de texte d'une page PDF et les retourne sous forme de liste de tuples.
    Les coordonnées sont multipliées par le facteur de zoom.

    Args:
        page: L'objet page de PyMuPDF.
        zoom: Le facteur de zoom appliqué à l'image.

    Returns:
        Une liste de tuples (x0, y0, x1, y1) représentant les coordonnées des blocs de texte, ou une liste vide si aucun bloc n'est trouvé.
        Retourne None en cas d'erreur avec l'objet page.
    """
    if not isinstance(page_number, fitz.Page):
      print("Erreur : l'argument 'page_number' n'est pas un objet fitz.Page valide.")
      return None

    text_bboxes = []
    try:
        for block in page_number.get_text("dict")["blocks"]:
            if block['type'] == 0:  # Type 0 correspond aux blocs de texte
                x0, y0, x1, y1 = block['bbox']
                x0 *= zoom
                y0 *= zoom
                x1 *= zoom
                y1 *= zoom
                text_bboxes.append((int(x0), int(y0), int(x1), int(y1)))
        return text_bboxes
    except KeyError: #Gere le cas ou la page ne contient pas de block
        return text_bboxes


def extract_and_display_boxes(pdf_path, page_number, text_bboxes, zoom=2.0):
    """
    Extrait et affiche le contenu de chaque boîte de texte sous forme de sous-graphiques (3 colonnes par ligne).

    Args:
        pdf_path (str): Le chemin vers le fichier PDF.
        page_number (int): Le numéro de la page (indexé à 0).
        text_bboxes (list): Une liste de tuples (x0, y0, x1, y1) représentant les coordonnées des blocs de texte.
        zoom (float): Le facteur de zoom appliqué à l'image.
    """
    # Charger le document PDF
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # Générer une image de la page avec zoom
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    doc.close()

    # Convertir l'image de la page en un objet PIL
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    # Calculer le nombre de lignes nécessaires pour 3 colonnes
    num_boxes = len(text_bboxes)
    num_columns = 3
    num_rows = math.ceil(num_boxes / num_columns)

    # Créer les sous-graphiques
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Applatir les axes pour une indexation plus facile

    # Parcourir chaque boîte et afficher les zones correspondantes
    for i, (x0, y0, x1, y1) in enumerate(text_bboxes):
        # Redimensionner les coordonnées selon le zoom
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Découper la zone correspondante de l'image
        cropped_img = img.crop((x0, y0, x1, y1))

        # Afficher la zone découpée dans le sous-graphe correspondant
        axes[i].imshow(cropped_img)
        axes[i].axis("off")
        axes[i].set_title(f"Box {i + 1}")

    # Désactiver les axes inutilisés si le nombre de boîtes est inférieur au nombre de sous-graphiques
    for j in range(num_boxes, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


```

```python id="DWMJJ1E0UeUA" outputId="2b9a4385-27aa-4614-9976-e42b80604543"
# Exemple d'utilisation
PDF_PATH = "C:/Users/horellou.florian/Downloads/test.pdf" # Remplacez par le chemin de votre PDF
PAGE_NUMBER = 0  # Numéro de la page à afficher (indexé à 0)
ZOOM = 2.0  # Facteur de zoom


try:
    # Charger le document PDF et obtenir les boîtes de texte
    doc = fitz.open(PDF_PATH)
    page = doc[PAGE_NUMBER]
    bboxes = get_text_bboxes(page, zoom=ZOOM)
    doc.close()

    if bboxes:
        # Extraire et afficher le contenu des boîtes
        extract_and_display_boxes(PDF_PATH, PAGE_NUMBER, bboxes, zoom=ZOOM)
    else:
        print("Aucune boîte de texte trouvée sur la page.")
except Exception as e:
    print(f"Erreur lors du traitement : {e}")
```

<!-- #region id="wbbwpLIMUeUB" -->
#### Ajout des correspondance entre les box images et texte
<!-- #endregion -->

<!-- #region id="O_xHUWEdUeUB" -->
# Autre à Vérifier
<!-- #endregion -->

<!-- #region id="SzOblW0IUeUB" -->

<!-- #endregion -->

```python id="jW5ZUUM8UeUB" outputId="d31ddf9a-6b6e-4bfc-bf7d-2a9df0b79a8a"
import fitz  # PyMuPDF
import os
import pytesseract
from PIL import Image
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Fonction pour extraire les images d'un PDF
def extract_images(pdf_path, temp_folder):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    pdf_document = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # jpg, png, etc.
            image_path = os.path.join(temp_folder, f"page_{page_number+1}_img_{img_index+1}.{image_ext}")

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            image_paths.append(image_path)

    pdf_document.close()
    return image_paths

# Fonction pour utiliser OCR et vérifier si une image contient du texte
def contains_text(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return len(text.strip()) > 0
    except Exception as e:
        print(f"Erreur OCR pour {image_path}: {e}")
        return False

# Fonction pour classifier les images (schéma ou non) avec un modèle ML
def classify_image(image_path, model, target_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    return prediction[0][0] > 0.5  # Retourne True si c'est un schéma

# Pipeline principal
def extract_and_classify_schemas(pdf_path, output_folder, temp_folder="temp_images", model_path="schema_model.h5"):
    # Étape 1 : Extraire les images
    image_paths = extract_images(pdf_path, temp_folder)

    # Charger le modèle ML
    model = load_model(model_path)

    # Créer le dossier de sortie
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Étape 2 : Filtrer les images
    for image_path in image_paths:
        try:
            # Vérification par OCR
            if contains_text(image_path):
                print(f"Schéma détecté (via OCR) : {image_path}")
                save_path = os.path.join(output_folder, os.path.basename(image_path))
                os.rename(image_path, save_path)
                continue

            # Vérification par ML
            if classify_image(image_path, model):
                print(f"Schéma détecté (via ML) : {image_path}")
                save_path = os.path.join(output_folder, os.path.basename(image_path))
                os.rename(image_path, save_path)
            else:
                print(f"Ignoré : {image_path}")
                os.remove(image_path)  # Supprime les images non pertinentes
        except Exception as e:
            print(f"Erreur lors du traitement de {image_path} : {e}")
            continue

    print(f"Extraction et classification terminées. Schémas enregistrés dans {output_folder}")

```

```python id="HozjhBW4UeUB"
pdf_path =  "C:/Users/horellou.florian/Downloads/test.pdf"
output_folder = "C:/Users/horellou.florian/Downloads/extract"


extract_and_classify_schemas(pdf_path, output_folder)
```

```python id="zwXcUx33UeUB" outputId="88871b69-eebd-4030-f906-4dec6bc73ac1"
import numpy as np
```

```python id="Cu-vBGe3UeUB"

```
