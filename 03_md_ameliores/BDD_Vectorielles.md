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
# 🧊 Bases de données vectorielles
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur les **bases de données vectorielles** — comment stocker, indexer et **chercher rapidement** dans des collections de vecteurs (embeddings).

C'est la **brique de stockage** sous tout système moderne de recherche sémantique, RAG, recommandation, déduplication, anomaly detection.

> **Ce notebook est une fusion** de `BDD_Vectorielles` et `retrieval_BDD_Vectorielle`.
> Pour le pipeline RAG complet (retrieval + LLM, hybrid search, reranking, eval), voir **`NLP_Recherche_d_informations`**.
> Pour le parsing PDF / Docling (qui était mélangé dans le notebook d'origine), voir **`DE_Docling`**.

Couverture :

1. **Théorie** — kNN, ANN, métriques (cosine / L2 / dot).
2. **Paysage 2026** — matrice de décision (FAISS / Qdrant / LanceDB / pgvector / Weaviate / Milvus / Chroma / Pinecone).
3. **FAISS** — la lib de référence : indices Flat, IVF, HNSW, PQ. Maths + code.
4. **LanceDB** — embedded, columnar, idéal pour prototypage local + prod légère.
5. **Qdrant** — le standard open-source serveur en 2026 (avec filtres).
6. **pgvector** — quand on a déjà Postgres.
7. **Bonnes pratiques** : choix d'index, dimensionnalité, métadonnées, sharding.
<!-- #endregion -->

<!-- #region -->
## 1. Théorie : kNN exact vs ANN
<!-- #endregion -->

<!-- #region -->
### 1.1 Le problème
<!-- #endregion -->

<!-- #region -->
On a `N` vecteurs `v_i ∈ ℝ^d` (typiquement `N = 10^4` à `10^9`, `d = 256` à `1536`). Pour une requête `q ∈ ℝ^d`, on veut les **k plus proches** au sens d'une métrique de similarité.

**Recherche exacte (brute force)** : calcule la similarité avec **tous** les vecteurs → `O(N·d)`. Pour `N = 10M, d = 768`, ça fait `7.6 G` opérations par requête. Impraticable au-delà de quelques 100k.

**Recherche approximative (ANN — Approximate Nearest Neighbors)** : sacrifie 0.5-5 % de recall pour gagner **100×-1000×** en vitesse. C'est ce que font toutes les vraies vector DBs.
<!-- #endregion -->

<!-- #region -->
### 1.2 Métriques de similarité
<!-- #endregion -->

<!-- #region -->
| Métrique | Formule | Quand |
|---|---|---|
| **Cosine** | `(v · q) / (‖v‖ ‖q‖)` | Embeddings de texte (la norme du vecteur n'a pas de sens) |
| **Dot product** | `v · q` | Embeddings normalisés (= cosine) ou modèles entraînés avec dot-product loss |
| **L2 (euclidien)** | `‖v - q‖²` | Embeddings d'image, ANN générique |
| **Hamming** | nombre de bits différents | Vecteurs binaires (LSH, binary embeddings) |

**Pro tip 2026** : si tu utilises des embeddings de texte (BGE, E5, MiniLM, ...) → **toujours normaliser à L2=1**, puis utiliser le **dot product**. C'est mathématiquement équivalent au cosine et **plus rapide** (1 multiplication-addition par dim, pas de norme à calculer).
<!-- #endregion -->

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Mini corpus pour les démos suivantes
corpus = [
    "FAISS is a library for efficient similarity search of dense vectors.",
    "Qdrant is an open-source vector database written in Rust.",
    "LanceDB is an embedded columnar vector database based on Apache Arrow.",
    "pgvector adds vector similarity search to PostgreSQL.",
    "Milvus is a cloud-native vector database for billion-scale workloads.",
    "Cosine similarity measures the angle between two vectors.",
    "Hierarchical Navigable Small World is a popular ANN index.",
    "Product Quantization compresses vectors to reduce memory footprint.",
]

# Encoder léger (90 MB, CPU) — pour les démos suivantes
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = encoder.encode(corpus, normalize_embeddings=True).astype("float32")
print(f"Shape : {embeddings.shape}  dtype : {embeddings.dtype}")
```

<!-- #region -->
## 2. Paysage 2026 — matrice de décision
<!-- #endregion -->

<!-- #region -->
| Solution | Type | Scaling | Idéal pour |
|---|---|---|---|
| **FAISS** | Lib in-process (C++/Python) | 10M+ vecteurs single machine | Recherche + ML / lib bas niveau, contrôle total |
| **LanceDB** | Embedded (columnar Arrow) | 100M sur disque, single machine | Prototypage + prod légère, intégration pandas/arrow |
| **Chroma** | Embedded ou serveur | 1-10M | Démos RAG, prototypes rapides |
| **Qdrant** | Serveur (Rust) | 100M-1B+ avec sharding | **Le standard open-source 2026** : filtres puissants, scalable, RBAC |
| **Weaviate** | Serveur (Go) | 100M+ avec sharding | Modules natifs (transformers, generative-x), GraphQL |
| **Milvus** | Serveur (Go/C++) | 1B-10B+ avec sharding | Très large échelle, K8s natif |
| **pgvector** | Extension PostgreSQL | 1-50M, scale via pgvectorscale | **Si vous avez déjà Postgres** (transactions, joins) |
| **Pinecone** | SaaS managé | Quasi illimité | Pas envie de gérer l'infra, budget OK |
| **Vespa** | Serveur (Java) | Très large échelle, recherche structurée + vecteurs | Stack hybride moteur de recherche complet |

### Règle de choix simplifiée (2026)

```
└─ Tu as déjà Postgres ?         → pgvector (+ pgvectorscale si > 10M)
└─ Single machine, prototypage ?  → LanceDB (embedded) ou FAISS (lib pure)
└─ Production open-source ?       → Qdrant (90 % des cas)
└─ Très large échelle (1B+) ?     → Milvus
└─ Budget OK, pas d'infra ?       → Pinecone (managé)
```

**Bench typique 2026 (à 50M vecteurs, 99 % recall)** :
- pgvectorscale : ~470 QPS
- Qdrant : ~40 QPS (filtrées) à 200 QPS (non filtrées)
- LanceDB : ~150 QPS

**À retenir** : toujours benchmarker sur **TES** données et **TES** queries — les chiffres publiés utilisent de la data synthétique.
<!-- #endregion -->

<!-- #region -->
## 3. FAISS — la lib de référence
<!-- #endregion -->

<!-- #region -->
**FAISS** (Facebook AI Similarity Search) est la lib de référence pour la recherche de similarité dense. C'est une **lib in-process** (pas un serveur) — on l'utilise quand on veut le contrôle total et qu'on n'a pas besoin de stockage persistant complexe.

Trois familles d'indices couvrent 95 % des cas :

| Index | Idée | Recall | Vitesse | RAM |
|---|---|---|---|---|
| **IndexFlatL2** / **IndexFlatIP** | Recherche exhaustive (brute force) | 100 % | Lent (`O(N·d)`) | `N·d·4` octets |
| **IndexIVFFlat** | Clustering + recherche dans les clusters proches | ~95-99 % | 10-100× plus rapide | Idem Flat + clustering |
| **IndexHNSW** | Graphe hiérarchique de voisins | ~95-99 % | Très rapide en query | 1.5-2× la taille des vecteurs |
| **IndexPQ** / **IVF+PQ** | Quantization produit (compression) | ~85-95 % | Très rapide + faible RAM | 4-32 octets par vecteur |
<!-- #endregion -->

<!-- #region -->
### 3.1 IndexFlat (exact, baseline)
<!-- #endregion -->

```python
import faiss

d = embeddings.shape[1]

# Index Flat (exact). IP = Inner Product = dot product = cosine pour vecteurs normalisés.
index_flat = faiss.IndexFlatIP(d)
index_flat.add(embeddings)
print(f"IndexFlatIP : {index_flat.ntotal} vecteurs, d={d}")

# Search
query = encoder.encode(["fast similarity search library"], normalize_embeddings=True).astype("float32")
D, I = index_flat.search(query, k=3)
print(f"Top-3 indices : {I[0]}")
print(f"Scores        : {D[0]}")
for idx in I[0]:
    print(f"  → {corpus[idx]}")
```

<!-- #region -->
### 3.2 IndexIVFFlat (clustering)
<!-- #endregion -->

<!-- #region -->
**Idée** : partitionne l'espace en `nlist` clusters (k-means). À la query, on cherche uniquement dans les `nprobe` clusters les plus proches du query. Trade-off classique :

- `nlist` grand → plus de clusters, chacun plus petit → search rapide mais peut rater des voisins.
- `nprobe` grand → meilleure recall mais plus lent.

**Bonne valeur de départ** : `nlist = 4·√N`, `nprobe = nlist // 10`.

Important : **il faut entraîner l'index** avant d'ajouter (apprend les centres de clusters).
<!-- #endregion -->

```python
nlist = 4   # peu de clusters parce que peu de data en démo
quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# Entraînement (k-means sur les vecteurs) — sur de la vraie data, prendre un sample
index_ivf.train(embeddings)
index_ivf.add(embeddings)

# Augmenter nprobe = plus précis (au pire = full search)
index_ivf.nprobe = 2
D, I = index_ivf.search(query, k=3)
print(f"IVF top-3 (nprobe={index_ivf.nprobe}) : {I[0]}")
```

<!-- #region -->
### 3.3 IndexHNSW (Hierarchical Navigable Small World)
<!-- #endregion -->

<!-- #region -->
**Idée** : construit un graphe multi-couches où chaque vecteur est connecté à ses voisins proches. La recherche descend la hiérarchie en suivant les arêtes — `O(log N)` au lieu de `O(N)`.

**L'index ANN dominant en 2026** (utilisé par défaut par Qdrant, Weaviate, Milvus, LanceDB sous le capot).

Hyperparamètres principaux :

- `M` : nombre de voisins par nœud (16-32 typique). Plus grand = meilleur recall, plus de mémoire.
- `efConstruction` : qualité de construction (40-200 typique).
- `efSearch` : qualité de search (40-200 typique). Modifiable au runtime.
<!-- #endregion -->

```python
M = 16  # nombre de voisins (typique 16-32)
index_hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
index_hnsw.hnsw.efConstruction = 40
index_hnsw.add(embeddings)

# efSearch contrôle la qualité (40-200 typique)
index_hnsw.hnsw.efSearch = 16
D, I = index_hnsw.search(query, k=3)
print(f"HNSW top-3 : {I[0]}")
```

<!-- #region -->
### 3.4 Persistance
<!-- #endregion -->

```python
# Save / load
faiss.write_index(index_hnsw, "/tmp/demo_hnsw.faiss")
index_loaded = faiss.read_index("/tmp/demo_hnsw.faiss")
print(f"Reloaded index : {index_loaded.ntotal} vecteurs")
```

<!-- #region -->
## 4. LanceDB — embedded columnar
<!-- #endregion -->

<!-- #region -->
**LanceDB** est une **base vectorielle embedded** (pas de serveur à lancer) basée sur le format **Lance** (équivalent Parquet optimisé pour la similarity search). Stockage colonnaire Apache Arrow → intégration native avec pandas, polars, DuckDB.

**Quand l'utiliser** :

- Prototypage rapide (pas d'infra).
- Production single-machine avec besoin de filtres SQL-like.
- Quand on veut une vraie DB (snapshot, versioning, time travel) sans serveur.
<!-- #endregion -->

```python
import lancedb
import pandas as pd

# Crée une DB locale dans /tmp/lancedb
db = lancedb.connect("/tmp/lancedb_demo")

# Données : texte + vecteur + metadata
data = pd.DataFrame({
    "id": list(range(len(corpus))),
    "text": corpus,
    "vector": embeddings.tolist(),
    "domain": ["lib", "db", "db", "db", "db", "theory", "theory", "theory"],
})

# Drop si existe (idempotence du re-run)
if "demo" in db.list_tables():
    db.drop_table("demo")
table = db.create_table("demo", data=data)
print(f"LanceDB table : {len(table)} rows")

# Search vector + filtre SQL-like
q_vec = encoder.encode("fast similarity search library", normalize_embeddings=True).astype("float32")
results = (
    table.search(q_vec)
         .where("domain = 'db'")     # ← filtre métadonnées, comme un SQL
         .limit(3)
         .to_pandas()
)
print(results[["id", "text", "domain", "_distance"]])
```

<!-- #region -->
## 5. Qdrant — le standard open-source 2026
<!-- #endregion -->

<!-- #region -->
**Qdrant** est devenu le choix par défaut pour la vector DB serveur open-source en 2026. Écrit en Rust, très rapide, **filtres puissants** (combine vector search + boolean filters efficacement, contrairement à beaucoup d'autres).

Lancement local (Docker) :

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

Puis depuis Python :

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

# Création d'une collection
client.recreate_collection(
    collection_name="demo",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Ajout des points (vecteur + payload)
points = [
    PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": corpus[i]})
    for i in range(len(corpus))
]
client.upsert(collection_name="demo", points=points)

# Search avec filtre
hits = client.search(
    collection_name="demo",
    query_vector=q_vec.tolist(),
    query_filter=Filter(
        must=[FieldCondition(key="domain", match=MatchValue(value="db"))]
    ),
    limit=3,
)
for h in hits:
    print(f"  {h.score:.3f}  {h.payload['text']}")
```

> Le code ci-dessus n'est pas exécuté dans ce notebook (nécessite Qdrant running). À copier-coller dans ton projet.
<!-- #endregion -->

<!-- #region -->
## 6. pgvector — quand on a déjà Postgres
<!-- #endregion -->

<!-- #region -->
**pgvector** est une extension PostgreSQL qui ajoute un type `vector` + opérateurs de distance + index HNSW/IVFFlat. Avec **pgvectorscale** (Timescale), on monte facilement à 50M+ vecteurs.

**Avantages immenses** :

- **Transactions ACID** entre vecteurs et données structurées.
- **Joins SQL** avec d'autres tables (users, products, ...).
- **Backup, replication, RBAC** : tout l'écosystème Postgres.
- **Un seul système** à maintenir au lieu de Postgres + vector DB séparée.

Setup :

```sql
CREATE EXTENSION vector;

CREATE TABLE items (
    id BIGSERIAL PRIMARY KEY,
    text TEXT,
    embedding vector(384)
);

-- Index HNSW (rapide, recommandé en 2026)
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Insert + search
INSERT INTO items (text, embedding) VALUES ('foo', '[0.1, ...]');
SELECT text, 1 - (embedding <=> '[0.2, ...]') AS cosine_sim
FROM items
ORDER BY embedding <=> '[0.2, ...]'
LIMIT 5;
```

```python
# Côté Python avec psycopg + pgvector :
# from pgvector.psycopg import register_vector
# import psycopg
# conn = psycopg.connect("postgresql://...")
# register_vector(conn)
# conn.execute("INSERT INTO items (text, embedding) VALUES (%s, %s)", (text, embedding))
```
<!-- #endregion -->

<!-- #region -->
## 7. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 7.1 Choix d'index
<!-- #endregion -->

<!-- #region -->
- `N < 10k` → **Flat** (exact). Pas de raison d'approximate.
- `10k ≤ N ≤ 1M` → **HNSW** (la valeur par défaut en 2026).
- `1M ≤ N ≤ 10M` → HNSW ou IVF+PQ si RAM contrainte.
- `N > 10M` → IVF+PQ obligatoire, sharding probable.

> Toujours **mesurer le recall** sur ton dataset (calcule ground truth avec un Flat sur un échantillon, compare à ton index ANN).
<!-- #endregion -->

<!-- #region -->
### 7.2 Dimensionnalité
<!-- #endregion -->

<!-- #region -->
- 384d (MiniLM, bge-small) : excellent ratio perf/coût, marche bien à très grande échelle.
- 768d (BGE-base, GTE-base) : sweet spot 2026 pour la plupart des cas.
- 1024d (BGE-large, multilingual-e5-large) : pour multilingue ou perf max.
- 1536d (text-embedding-3-large) : très bon mais 4× plus de RAM que 384d.
- **Matryoshka embeddings** (BGE-m3) : tu peux **tronquer** un vecteur 1024d à 256d et garder ~95 % de la perf. Utilisé en prod : index sur 256d → re-rank top-100 sur 1024d.
<!-- #endregion -->

<!-- #region -->
### 7.3 Métadonnées et filtres
<!-- #endregion -->

<!-- #region -->
- **Toujours stocker** : `source`, `doc_id`, `chunk_idx`, `date`, `version`, `tags`.
- Permet le **filtrage** (par utilisateur, par projet, par date) avant ou après search.
- **Pré-filtre vs post-filtre** : la plupart des DBs en 2026 (Qdrant, Weaviate) font du **filtered ANN** (filtre pendant la traversée du graphe HNSW) — c'est exact ET rapide. À l'inverse, post-filtrer après ANN peut perdre des résultats.
<!-- #endregion -->

<!-- #region -->
### 7.4 Versioning et MAJ
<!-- #endregion -->

<!-- #region -->
- Quand on **change de modèle d'embedding**, **tous** les vecteurs doivent être re-générés.
- Stratégie : index versionné (`v1`, `v2`), basculement progressif, comparaison A/B.
- LanceDB et Pinecone permettent du **time travel** ; sinon, manage à la main avec un champ `model_version`.
<!-- #endregion -->

<!-- #region -->
### 7.5 Quantization
<!-- #endregion -->

<!-- #region -->
À grande échelle, la RAM devient le bottleneck. Solutions 2026 :

- **Scalar quantization** (int8) : ÷4 la RAM, perte de recall <1 %.
- **Product Quantization** (PQ) : ÷16 à ÷64, perte de recall 1-3 %.
- **Binary embeddings** (1 bit/dim) : ÷32, perte plus forte mais possible pour le rough retrieval + re-rank.

Toutes ces options sont natives dans FAISS, Qdrant, Milvus.
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- [FAISS — docs officielles + wiki](https://github.com/facebookresearch/faiss/wiki)
- [Qdrant — docs officielles](https://qdrant.tech/documentation/)
- [LanceDB — docs](https://lancedb.github.io/lancedb/)
- [pgvector — GitHub](https://github.com/pgvector/pgvector) + [pgvectorscale](https://github.com/timescale/pgvectorscale)
- [Weaviate](https://weaviate.io/developers/weaviate) · [Milvus](https://milvus.io/docs) · [Chroma](https://docs.trychroma.com/) · [Pinecone](https://docs.pinecone.io/)
- [Vector DB benchmarks 2026 — CallSphere](https://callsphere.ai/blog/vector-database-benchmarks-2026-pgvector-qdrant-weaviate-milvus-lancedb)
- [Best Vector Databases 2026 — MarkTechPost](https://www.marktechpost.com/2026/05/10/best-vector-databases-in-2026-pricing-scale-limits-and-architecture-tradeoffs-across-nine-leading-systems/)
- [Top 15 vector databases — Medium 2026](https://medium.com/@pratik-rupareliya/top-15-vector-databases-in-2026-a-production-decision-guide-from-100-enterprise-deployments-dd58a04f51a5)
- [Pipeline RAG complet → notebook NLP_Recherche_d_informations](NLP_Recherche_d_informations.ipynb)
<!-- #endregion -->
