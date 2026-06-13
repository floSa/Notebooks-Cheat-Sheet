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

<!-- #region id="VPydbUN4pPRY" -->
# Recherche de similarité sur des vecteurs (Base de Données Vectorielles)
<!-- #endregion -->

<!-- #region id="UByu0NTFpTeW" -->
## **1. Vectorisation des phrases**
<!-- #endregion -->

<!-- #region id="v0Eb9gGrpZ4d" -->
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

<!-- #region id="Ay37i-2VpcjV" -->
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

```python id="pimoQg_bpSpp"
!pip install sentence_transformers -q
```

```python id="i4soQyhXpRzM"
# Importation des bibliothèques nécessaires
import numpy as np
from sentence_transformers import SentenceTransformer
```

```python id="DQzk8CxSplhm"
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

<!-- #region id="lN7fk7egpnRZ" -->
## **2. Indexation et recherche**
<!-- #endregion -->

<!-- #region id="hXINyN4cpqS4" -->
### Comparaison des solutions

| Base de données | API          | Métadonnées   | Recherche textuelle | Graphe sémantique | Filtrage | Particularités                        |
|------------------|--------------|---------------|----------------------|-------------------|----------|---------------------------------------|
| **FAISS**        | Non (librairie) | Limité        | Non                  | Non               | Non      | Haute performance, faible latence. |
| **Chroma**       | Oui (Python)  | Oui           | Oui                  | Non               | Oui      | Spécifique aux workflows de machine learning. |
| **Weaviate**     | Oui (REST, GraphQL) | Oui           | Oui                  | Oui               | Oui      | Modèles pré-entraînés, connecteurs aux LLM. |
| **Milvus**       | Oui (REST, Python, Go) | Oui           | Non                  | Non               | Oui      | Architecture distribuée, évolutive. |
| **Qdrant**       | Oui (REST, gRPC, Python) | Oui           | Oui (via intégrations) | Non               | Oui      | Optimisé pour le filtrage et la mise à jour des données. |


<!-- #endregion -->

<!-- #region id="wVA9dQXFpsxZ" -->
### **A - FAISS**
<!-- #endregion -->

<!-- #region id="05aWOd0apuTe" -->
#### Comment fonctionne FAISS ?
FAISS (Facebook AI Similarity Search) est conçu pour gérer la **recherche rapide** dans des bases contenant des millions (voire des milliards) de vecteurs. L’algorithme sous-jacent varie selon l’index utilisé.

L'index FAISS qui repose sur la **distance euclidienne**
$\mathbf{u}$ et $\mathbf{v}$ sont définis comme :
$$
d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}
$$

Ce calcul mesure à quel point deux vecteurs sont éloignés dans l’espace. Plus la distance est proche de **0**, plus les vecteurs (et donc les phrases) sont similaires.  
<!-- #endregion -->

<!-- #region id="7TkxqxAtpyRE" -->
#### Comparaison des trois approches

| Méthode          | Avantages                                      | Inconvénients                              | Cas d'utilisation                           |
|-------------------|-----------------------------------------------|--------------------------------------------|---------------------------------------------|
| **IndexFlatL2**   | Résultats exacts, simple, pas d'entraînement   | Lent pour les grandes bases                | Petites bases ou besoin d'une précision maximale. |
| **IndexIVFFlat**  | Rapide pour des bases volumineuses, ajustable | Nécessite un entraînement, moins précis    | Grandes bases avec possibilité d'ajuster précision/vitesse. |
| **HNSW**          | Très rapide, pas d'entraînement, dynamique    | Gourmand en mémoire, graphe coûteux à construire | Bases très grandes nécessitant des mises à jour fréquentes. |

<!-- #endregion -->

```python id="uCFbV7bnp0o4"
!pip install faiss-cpu -q
```

```python id="GiSQ0Xwhp1pY"
import faiss
```

```python id="0UVtOqPjp3gY"
dimension = data.shape[1]  # Dimension des vecteurs
print("dimension" , dimension)
```

<!-- #region id="QOufcyU8p6kl" -->
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

```python id="O5vpDtUFp5bN"
#Création de l'index
index = faiss.IndexFlatL2(dimension)  # Création de l'index pour des vecteurs de dimension d
# Ajout des vecteurs à l'index
index.add(data)
print(f"Nombre de vecteurs dans l'index : {index.ntotal}")
```

```python id="h0HDUT5hqbOc"
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

<!-- #region id="N3DRRM1_qGpu" -->

<!-- #endregion -->

<!-- #region id="dqA4UpRSqX5I" -->
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

```python id="c2qPjkr7qYW8"
n_clusters = 3
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, n_clusters)

# Entraînement et ajout des vecteurs
index_ivf.train(data)
index_ivf.add(data)
```

```python id="-1pspbo7qhb5"
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

<!-- #region id="1RpikSyBqkJG" -->
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

```python id="WvaBWrXkqkwj"
# Création d'IndexHNSW
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 est le nombre de voisins par nœud
index_hnsw.add(data)
```

```python id="yeaDcPVBqmiW"
# Nombre de voisins à retourner
k = 3
# Recherche
distances_hnsw, indices_hnsw = index_hnsw.search(query, k=k)

# Affichage des résultats
for i, idx in enumerate(indices_hnsw[0]):
    print(f"Phrase similaire : {phrases[idx]} (Distance : {distances_hnsw[0][i]:.4f})")
```

<!-- #region id="oB97XQ0uqq8e" -->
### **B - Weaviate** (No ok)
<!-- #endregion -->

<!-- #region id="J42L6X_iqshL" -->
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

<!-- #region id="h7bHchX3rGJG" -->
##### **Étape 2 : Configuration de l'environnement Python**
<!-- #endregion -->

<!-- #region id="XmqFziWarOyY" -->
Installez les bibliothèques nécessaires :
<!-- #endregion -->

```python id="Zi9KxOHBqraL"
!pip uninstall weaviate-client
!pip install weaviate-client
```

<!-- #region id="2j9hyEmurTpA" -->
Charger les bibliothèques nécessaires :
<!-- #endregion -->

```python id="8CYWrIF5rU0o"
import weaviate
print(weaviate.__version__)
```

```python id="JmuBmpWMrWXA"
import weaviate-client
from sentence_transformers import SentenceTransformer
```

<!-- #region id="XduQbTIDrYaO" -->
##### **Étape 3 : Connexion à Weaviate et création d'un schéma**
<!-- #endregion -->

<!-- #region id="wDnV3xkcrZy9" -->
Connectez-vous à l'instance Weaviate :
<!-- #endregion -->

```python id="_4Y8SkX7rZBE"
client = weaviate.Client("http://localhost:8080")
```

```python id="ONGlQd1Jrdso"
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

<!-- #region id="z1Jwx-IIrgwG" -->
##### **Étape 4 : Ajout de vecteurs dans Weaviate**
<!-- #endregion -->

```python id="aD11MV0frhE7"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Ceci est une phrase.", "Une autre phrase pour tester."]
embeddings = model.encode(sentences).tolist()
```

```python id="nS8QP7hVriVE"
for text, vector in zip(sentences, embeddings):
    client.data_object.create(
        data_object={"text": text},
        class_name="SentenceVectors",
        vector=vector
    )
```
