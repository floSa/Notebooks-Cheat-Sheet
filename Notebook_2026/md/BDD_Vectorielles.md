---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python (notebooks-refonte)
    name: notebooks-refonte
---

<!-- #region -->
# Bases de données vectorielles & recherche de similarité
<!-- #endregion -->

<!-- #region -->
Ce notebook est une **cheat-sheet + tutoriel** sur la brique « index » d'un système de recherche sémantique : transformer du texte en **vecteurs** (embeddings), puis retrouver rapidement les éléments les plus **proches** d'une requête.

On couvre, dans l'ordre :

1. **Vectorisation** des phrases avec `sentence-transformers`.
2. **Mesures de similarité** : distance L2, produit scalaire, cosinus (et pourquoi normaliser).
3. **FAISS** — la librairie d'indexation de référence (Flat exact, IVF, HNSW).
4. **Exact vs approché (ANN)** : mesurer le compromis *recall@k / latence*.
5. **Panorama 2026** des bases vectorielles.
6-11. Démos concrètes : **Chroma, Qdrant, LanceDB, Milvus** (embarqués) puis **Weaviate, pgvector** (mode serveur).
12. **Persistance** des index.
13. **Guide de choix** 2026.

> 🔎 **Périmètre** : ce notebook couvre la couche *index / recherche k-NN* et le panorama des bases vectorielles (il fusionne les deux anciens notebooks de similarité vectorielle). Le **parsing de documents** (PDF, OCR, stockage objet) relève du notebook `DE_Docling` ; le **pipeline RAG complet** (génération, reranking) est l'extension naturelle évoquée en conclusion.
<!-- #endregion -->

<!-- #region -->
## Dépendances
<!-- #endregion -->

<!-- #region -->
L'environnement est géré par **`uv`** (pas de `!pip install` dans les cellules). Paquets utilisés :

```bash
uv add sentence-transformers faiss-cpu chromadb qdrant-client lancedb \
       "pymilvus[milvus-lite]" weaviate-client "psycopg[binary]" scikit-learn
```

- `sentence-transformers` : génération d'embeddings.
- `faiss-cpu` : indexation/recherche vectorielle (librairie).
- `chromadb`, `qdrant-client`, `lancedb`, `pymilvus` : bases vectorielles embarquées.
- `weaviate-client`, `psycopg` : clients pour les bases en mode serveur (Weaviate, PostgreSQL/pgvector).
- `scikit-learn` : jeu de données 20 Newsgroups.
<!-- #endregion -->

<!-- #region -->
On importe les bibliothèques communes et on définit deux helpers réutilisés partout : `embed()` (texte → vecteurs) et `show_neighbors()` (affichage des résultats). `WORKDIR` est un dossier temporaire pour les bases de démo éphémères.
<!-- #endregion -->

```python
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

RNG = np.random.default_rng(42)
WORKDIR = Path(tempfile.mkdtemp(prefix="vdb_"))  # bases de démo éphémères

# Modèle d'embedding partagé (384 dimensions, rapide).
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts: list[str], normalize: bool = False) -> np.ndarray:
    """Encode une liste de textes en vecteurs float32.

    Args:
        texts: phrases / documents à vectoriser.
        normalize: si True, normalise L2 chaque vecteur (utile pour le cosinus).

    Returns:
        Matrice (len(texts), dim) en float32.
    """
    vecs = EMBED_MODEL.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def show_neighbors(
    query: str, texts: list[str], indices: np.ndarray, scores: np.ndarray, score_name: str = "score"
) -> None:
    """Affiche joliment les voisins retournés par une recherche."""
    print(f"  Requête : {query!r}")
    for rank, (idx, sc) in enumerate(zip(indices, scores), start=1):
        snippet = texts[idx].replace("\n", " ")[:70]
        print(f"    {rank}. ({score_name}={sc:.4f}) {snippet}")
```

<!-- #region -->
## 1. Vectorisation des phrases
<!-- #endregion -->

<!-- #region -->
Les ordinateurs ne « comprennent » pas les mots : on convertit chaque phrase en un **vecteur** numérique (un *embedding*) qui encode son **sens** dans un espace de plusieurs centaines de dimensions. Deux phrases au contenu proche auront des vecteurs proches ; deux phrases éloignées, des vecteurs éloignés. C'est ce qui rend la recherche **sémantique** (par le sens) possible, indépendamment de la formulation exacte.
<!-- #endregion -->

<!-- #region -->
### Comment fonctionne SentenceTransformer ?
<!-- #endregion -->

<!-- #region -->
`SentenceTransformer` s'appuie sur des modèles de type **Transformer** (famille BERT), mais optimisés pour produire **un seul vecteur par phrase** capturant sa sémantique globale :

1. **Tokenisation** : le texte est découpé en sous-unités (*tokens*) puis projeté en vecteurs initiaux.
2. **Couches Transformer** : le mécanisme d'**attention** met chaque token en relation avec son contexte.
3. **Pooling** : les vecteurs de tokens sont agrégés (souvent une moyenne) en **un vecteur global** de 384 ou 768 dimensions.
<!-- #endregion -->

<!-- #region -->
### Quel modèle d'embedding choisir (2026) ?
<!-- #endregion -->

<!-- #region -->
On garde ici **`all-MiniLM-L6-v2`** : léger (384-dim), rapide, idéal pour apprendre et prototyper. Pour de la production en 2026, plusieurs modèles le surpassent nettement :

| Modèle | Dim | Points forts | Licence |
|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | Très rapide, léger, baseline solide | Apache 2.0 |
| **all-mpnet-base-v2** | 768 | Meilleure qualité mono-lingue (EN) | Apache 2.0 |
| **BGE-M3** | 1024 | Multilingue (100+ langues), dense + sparse + multi-vecteur | MIT |
| **Qwen3-Embedding (0.6B→8B)** | 1024→4096 | SOTA MTEB, excellent français | Apache 2.0 |
| **Jina-embeddings-v3** | 1024 | Long contexte (8k), task-specific LoRA | CC-BY-NC (commercial : API) |

> ⚠️ Le **vecteur dépend du modèle** : on ne peut pas mélanger dans un même index des embeddings produits par deux modèles différents. Versionnez toujours l'index avec le nom du modèle.
<!-- #endregion -->

<!-- #region -->
On vectorise 5 phrases françaises jouets. La sortie `(5, 384)` confirme : 5 phrases, 384 dimensions chacune.
<!-- #endregion -->

```python
phrases = [
    "Le chat dort sur le tapis.",
    "Le chien joue dans le jardin.",
    "Un oiseau vole dans le ciel.",
    "Le soleil brille aujourd'hui.",
    "Il pleut dans la ville.",
]

data = embed(phrases)
print(f"Dimensions des vecteurs : {data.shape}")  # (5, 384)
DIM = data.shape[1]
```

<!-- #region -->
## 2. Mesurer la similarité : L2, produit scalaire, cosinus
<!-- #endregion -->

<!-- #region -->
Trois façons de comparer deux vecteurs $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ :

- **Distance euclidienne (L2)** — *plus c'est petit, plus c'est proche* :
$$ d_{L2}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2} $$

- **Produit scalaire (inner product, IP)** — *plus c'est grand, plus c'est proche* :
$$ \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^{d} u_i v_i $$

- **Similarité cosinus** — angle entre les vecteurs, insensible à la norme :
$$ \cos(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \, \|\mathbf{v}\|} \in [-1, 1] $$

**Bonne pratique 2026** : les modèles d'embedding modernes encodent le sens par la **direction** du vecteur, pas par sa norme → on privilégie le **cosinus**. Astuce clé : si on **normalise** les vecteurs (norme = 1), alors *produit scalaire = cosinus*. C'est pourquoi on couplera souvent `normalize` + un index « inner product ».
<!-- #endregion -->

<!-- #region -->
Calculons la matrice de similarité cosinus entre les 5 phrases et trouvons la paire la plus proche.
<!-- #endregion -->

```python
def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    """Matrice de similarité cosinus (vecteurs normalisés puis produit scalaire)."""
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return norm @ norm.T


sim = cosine_sim_matrix(data)
masked = sim - np.eye(len(phrases)) * 2  # ignore la diagonale (auto-similarité = 1)
i, j = np.unravel_index(np.argmax(masked), sim.shape)
print(f"Paire la plus proche (cos={sim[i, j]:.3f}) :")
print(f"  - {phrases[i]}")
print(f"  - {phrases[j]}")
```

<!-- #region -->
La paire retenue est sémantiquement cohérente : la similarité cosinus capture bien la proximité de sens.
<!-- #endregion -->

<!-- #region -->
## 3. FAISS — la librairie d'indexation
<!-- #endregion -->

<!-- #region -->
**FAISS** (*Facebook AI Similarity Search*) est **la** librairie de référence pour la recherche de plus proches voisins à grande échelle (millions à milliards de vecteurs). Ce n'est **pas une base de données** : pas de persistance automatique, pas de CRUD ni de filtrage par métadonnée — juste des **index** ultra-optimisés. On choisit le type d'index selon le compromis recherché entre **exactitude**, **vitesse** et **mémoire**.

| Index | Avantages | Inconvénients | Cas d'usage |
|---|---|---|---|
| **IndexFlatL2 / FlatIP** | Exact, simple, aucun entraînement | Lent à grande échelle (O(N)) | Petites bases, précision maximale |
| **IndexIVFFlat** | Rapide sur grandes bases, ajustable | Entraînement requis, approché | Grandes bases, compromis réglable |
| **IndexHNSWFlat** | Très rapide, dynamique, sans entraînement | Gourmand en mémoire, construction coûteuse | Très grandes bases, mises à jour fréquentes |
<!-- #endregion -->

<!-- #region -->
On importe FAISS et on prépare la dimension des vecteurs, ainsi qu'une requête de test.
<!-- #endregion -->

```python
import faiss

print("dimension :", DIM)

new_phrases = ["Un oiseau chante dans le ciel clair."]
query_vec = embed(new_phrases)  # (1, 384), non normalisé
K = 3
```

<!-- #region -->
### 3.1 IndexFlatL2 (recherche exacte)
<!-- #endregion -->

<!-- #region -->
`IndexFlatL2` effectue une recherche **exhaustive** : il calcule la distance L2 entre la requête et **tous** les vecteurs. Résultats **exacts**, aucune approximation, aucun entraînement — mais coût linéaire en nombre de vecteurs.
<!-- #endregion -->

```python
def build_flat_l2(vectors: np.ndarray) -> faiss.Index:
    """Index FAISS exact basé sur la distance euclidienne (L2)."""
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


flat_l2 = build_flat_l2(data)
print(f"[FlatL2] vecteurs indexés : {flat_l2.ntotal}")
dist_l2, idx_l2 = flat_l2.search(query_vec, K)
show_neighbors(new_phrases[0], phrases, idx_l2[0], dist_l2[0], score_name="distL2")
```

<!-- #region -->
La phrase sur l'oiseau ressort en tête (distance la plus faible) — résultat attendu.
<!-- #endregion -->

<!-- #region -->
### 3.2 IndexFlatIP (cosinus via normalisation)
<!-- #endregion -->

<!-- #region -->
FAISS ne propose pas de métrique « cosinus » native. On l'obtient en **normalisant** les vecteurs (`faiss.normalize_L2`) puis en utilisant un index **produit scalaire** `IndexFlatIP` : sur des vecteurs unitaires, *produit scalaire = cosinus*. Les scores sont alors dans $[-1, 1]$ (*1 = identique*).
<!-- #endregion -->

```python
def build_flat_ip(vectors: np.ndarray) -> faiss.Index:
    """Index produit scalaire ; sur vecteurs normalisés L2 → similarité cosinus."""
    vectors = vectors.copy()
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


flat_ip = build_flat_ip(data)
q_norm = query_vec.copy()
faiss.normalize_L2(q_norm)
sim_ip, idx_ip = flat_ip.search(q_norm, K)
show_neighbors(new_phrases[0], phrases, idx_ip[0], sim_ip[0], score_name="cos")
```

<!-- #region -->
Même classement que L2, mais les scores sont désormais des **similarités cosinus** (plus interprétables : proches de 1 = très similaires).
<!-- #endregion -->

<!-- #region -->
### Un corpus réaliste : 20 Newsgroups
<!-- #endregion -->

<!-- #region -->
Cinq phrases ne suffisent pas pour illustrer les index approchés (IVF, HNSW) ni un *benchmark*. On charge ~2 000 documents de **20 Newsgroups** (4 catégories) via scikit-learn, puis on les vectorise. On réserve 100 documents comme **requêtes** pour mesurer le *recall* plus loin.

> Premier appel : téléchargement dans `~/scikit_learn_data/` (cache).
<!-- #endregion -->

```python
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
news = fetch_20newsgroups(
    subset="train",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=42,
)
# Nettoyage léger + garde des documents de taille raisonnable
docs_all = [d.strip().replace("\n", " ") for d in news.data]
labels_all = [news.target_names[t] for t in news.target]
keep = [k for k, d in enumerate(docs_all) if 40 < len(d) < 2000][:2000]
docs = [docs_all[k] for k in keep]
cats = [labels_all[k] for k in keep]
print(f"Documents retenus : {len(docs)}")

corpus_emb = embed(docs)  # (N, 384), non normalisé (pour L2)
print(f"Embeddings corpus : {corpus_emb.shape}")

# Split base / requêtes pour le benchmark
N_QUERIES = 100
db_emb = corpus_emb[:-N_QUERIES]
q_emb = corpus_emb[-N_QUERIES:]
db_docs = docs[:-N_QUERIES]
print(f"Base : {len(db_emb)} vecteurs — Requêtes : {len(q_emb)}")
```

<!-- #region -->
### 3.3 IndexIVFFlat (ANN par clustering)
<!-- #endregion -->

<!-- #region -->
`IndexIVFFlat` (*Inverted File*) **partitionne** l'espace en `nlist` cellules (clustering k-means). À la recherche, seules les `nprobe` cellules les plus proches de la requête sont explorées → beaucoup moins de calculs. Un **entraînement** (`train`) est nécessaire pour apprendre les centroïdes. Le paramètre `nprobe` règle le compromis vitesse/précision (plus grand = plus précis mais plus lent).
<!-- #endregion -->

```python
def build_ivf(vectors: np.ndarray, nlist: int = 50, nprobe: int = 10) -> faiss.Index:
    """Index IVF : clustering en `nlist` cellules, recherche dans `nprobe` cellules."""
    quantizer = faiss.IndexFlatL2(vectors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index


ivf = build_ivf(db_emb)
print(f"[IVFFlat] indexés : {ivf.ntotal}, nlist=50, nprobe={ivf.nprobe}")
d_ivf, i_ivf = ivf.search(q_emb[:1], K)
show_neighbors("(doc requête 0)", db_docs, i_ivf[0], d_ivf[0], score_name="distL2")
```

<!-- #region -->
### 3.4 IndexHNSWFlat (ANN par graphe)
<!-- #endregion -->

<!-- #region -->
**HNSW** (*Hierarchical Navigable Small World*) construit un **graphe** multi-niveaux où chaque vecteur est un nœud relié à ses voisins proches. La recherche *navigue* dans le graphe de proche en proche. Très rapide, pas d'entraînement, ajout/suppression dynamiques — au prix d'une consommation mémoire plus élevée. Le paramètre `M` (voisins par nœud) influe sur la qualité et la mémoire.
<!-- #endregion -->

```python
def build_hnsw(vectors: np.ndarray, m: int = 32) -> faiss.Index:
    """Index HNSW : graphe navigable hiérarchique, `m` voisins par nœud."""
    index = faiss.IndexHNSWFlat(vectors.shape[1], m)
    index.add(vectors)
    return index


hnsw = build_hnsw(db_emb)
print(f"[HNSW] indexés : {hnsw.ntotal}")
d_hnsw, i_hnsw = hnsw.search(q_emb[:1], K)
show_neighbors("(doc requête 0)", db_docs, i_hnsw[0], d_hnsw[0], score_name="distL2")
```

<!-- #region -->
## 4. Exact vs approché : recall@k & latence
<!-- #endregion -->

<!-- #region -->
La recherche exacte (`Flat`) coûte $O(N)$ par requête. Les index **approchés** (ANN : IVF, HNSW) sacrifient un peu de précision pour gagner énormément en vitesse. On mesure ce compromis avec le **recall@k** :

$$ \text{recall@k} = \frac{|\,\text{voisins approchés} \cap \text{voisins exacts}\,|}{k} $$

La **vérité terrain** est donnée par `IndexFlatL2` (exact). On compare IVF et HNSW dessus, sur les 100 requêtes réservées.
<!-- #endregion -->

```python
def recall_at_k(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    """Fraction moyenne des k voisins exacts retrouvés par la recherche approchée."""
    hits = [len(set(a) & set(e)) / e.shape[0] for a, e in zip(approx_idx, exact_idx)]
    return float(np.mean(hits))


def timed_search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Lance une recherche et renvoie (indices, temps_total_ms)."""
    t0 = time.perf_counter()
    _, idx = index.search(queries, k)
    return idx, (time.perf_counter() - t0) * 1000.0


K_BENCH = 10
exact = build_flat_l2(db_emb)
gt_idx, t_exact = timed_search(exact, q_emb, K_BENCH)  # vérité terrain
ivf_idx, t_ivf = timed_search(ivf, q_emb, K_BENCH)
hnsw_idx, t_hnsw = timed_search(hnsw, q_emb, K_BENCH)

bench = {
    "FlatL2 (exact)": (1.0, t_exact),
    "IVFFlat": (recall_at_k(ivf_idx, gt_idx), t_ivf),
    "HNSW": (recall_at_k(hnsw_idx, gt_idx), t_hnsw),
}
for name, (rec, ms) in bench.items():
    print(f"  {name:16s} recall@{K_BENCH}={rec:.3f}  temps={ms:.2f} ms")
```

<!-- #region -->
Visualisons le compromis : à gauche le *recall* (qualité), à droite la latence totale sur les 100 requêtes.
<!-- #endregion -->

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
names = list(bench)
ax1.bar(names, [bench[n][0] for n in names], color="#00798c")
ax1.set_title(f"Recall@{K_BENCH} (vs FlatL2 exact)")
ax1.set_ylim(0, 1.05)
ax2.bar(names, [bench[n][1] for n in names], color="#d1495b")
ax2.set_title(f"Latence — {len(q_emb)} requêtes (ms)")
for ax in (ax1, ax2):
    ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
plt.show()
```

<!-- #region -->
Lecture typique : **HNSW** atteint un recall quasi exact pour une latence bien moindre ; **IVF** est réglable via `nprobe`. Sur de petits corpus, l'exact reste compétitif — l'ANN paie surtout à grande échelle.
<!-- #endregion -->

<!-- #region -->
## 5. Panorama des bases vectorielles (2026)
<!-- #endregion -->

<!-- #region -->
FAISS est une *librairie* : pour la production on veut souvent une **base de données vectorielle** offrant CRUD, persistance, métadonnées, filtrage, scaling, et parfois recherche *hybride* (vecteur + mots-clés). Comparatif des solutions courantes en 2026 :

| Base | Déploiement | API | Métadonnées + filtrage | Hybride | Particularités |
|---|---|---|---|---|---|
| **FAISS** | Librairie (in-process) | Python/C++ | ❌ (à gérer soi-même) | ❌ | Le plus rapide, brique de bas niveau |
| **Chroma** | Embarqué / serveur | Python | ✅ | ✅ | Simple, idéal proto & petites prods |
| **LanceDB** | Embarqué (in-process) | Python/JS/Rust | ✅ | ✅ | Format columnar Lance, zéro serveur |
| **Qdrant** | Embarqué / serveur | REST, gRPC, Python | ✅ (filtrage riche) | ✅ | Écrit en Rust, équilibré, populaire |
| **Milvus** | Lite (fichier) / distribué | REST, gRPC, Python | ✅ | ✅ | Conçu pour le très grand volume |
| **Weaviate** | Serveur (Docker/cloud) | REST, GraphQL, gRPC | ✅ | ✅ | Modules vectorizer, schéma riche |
| **pgvector** | Extension PostgreSQL | SQL | ✅ (via SQL) | ✅ (FTS) | Reste dans Postgres, joins relationnels |
| **sqlite-vec** | Extension SQLite (embarqué) | SQL | ✅ | ⚠️ | Ultra-léger, embarqué dans SQLite |
| **Pinecone** | Managé (cloud, serverless) | REST, Python | ✅ | ✅ | Zéro ops, scaling automatique, payant |
| **Vespa** | Serveur / cloud | HTTP, YQL | ✅ | ✅ | Recherche + ranking ML avancé, gros volumes |

Les sections suivantes montrent les **mêmes** opérations (créer une collection, insérer, rechercher, filtrer) sur plusieurs de ces moteurs, avec le **même** sous-corpus normalisé (cosinus).
<!-- #endregion -->

<!-- #region -->
On prépare un sous-corpus de 500 documents **normalisés** (pour le cosinus) et une requête, partagés par toutes les démos qui suivent.
<!-- #endregion -->

```python
N_DEMO = 500
demo_docs = db_docs[:N_DEMO]
demo_cats = cats[:N_DEMO]
demo_emb = db_emb[:N_DEMO].copy()
faiss.normalize_L2(demo_emb)  # normalisés → cosinus pour toutes les bases

q_demo = q_emb[:1].copy()  # une requête (doc 20NG)
faiss.normalize_L2(q_demo)
DIM_DEMO = demo_emb.shape[1]
K_DEMO = 3
```

<!-- #region -->
## 6. ChromaDB (embarqué)
<!-- #endregion -->

<!-- #region -->
**Chroma** est une base vectorielle pensée pour le développement : un `EphemeralClient()` tourne **en mémoire** (idéal tests), un `PersistentClient(path=...)` écrit sur disque. On crée une *collection*, on ajoute des vecteurs avec leurs **documents** et **métadonnées**, puis on requête — éventuellement avec un **filtre** `where` sur une métadonnée. On force l'espace `cosine` à la création.
<!-- #endregion -->

```python
import chromadb

chroma_client = chromadb.EphemeralClient()
coll = chroma_client.create_collection(name="news", metadata={"hnsw:space": "cosine"})
coll.add(
    ids=[str(i) for i in range(N_DEMO)],
    embeddings=demo_emb.tolist(),
    documents=demo_docs,
    metadatas=[{"category": c} for c in demo_cats],
)

res = coll.query(query_embeddings=q_demo.tolist(), n_results=K_DEMO)
print(f"Top-{K_DEMO} (sans filtre) :")
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"    (dist={dist:.4f}) {doc[:70]}")

res_f = coll.query(query_embeddings=q_demo.tolist(), n_results=K_DEMO, where={"category": "sci.space"})
print(f"Top-{K_DEMO} filtré category=sci.space : {len(res_f['ids'][0])} résultats")
```

<!-- #region -->
Le filtre `where` restreint la recherche à une catégorie — combinaison typique recherche sémantique + filtrage structurel.
<!-- #endregion -->

<!-- #region -->
## 7. Qdrant (local en mémoire)
<!-- #endregion -->

<!-- #region -->
**Qdrant** (écrit en Rust) offre un **mode local** sans serveur : `QdrantClient(":memory:")` (ou `path=...` pour persister), parfait pour développer et tester. On déclare la métrique (`COSINE`) à la création de la collection, on `upsert` des *points* (vecteur + `payload`), puis on interroge avec `query_points`, éventuellement filtré sur le payload via `Filter`.
<!-- #endregion -->

```python
from qdrant_client import QdrantClient, models

qclient = QdrantClient(":memory:")
qclient.create_collection(
    collection_name="news",
    vectors_config=models.VectorParams(size=DIM_DEMO, distance=models.Distance.COSINE),
)
qclient.upsert(
    collection_name="news",
    points=[
        models.PointStruct(
            id=i, vector=demo_emb[i].tolist(), payload={"text": demo_docs[i], "category": demo_cats[i]}
        )
        for i in range(N_DEMO)
    ],
)

hits = qclient.query_points(collection_name="news", query=q_demo[0].tolist(), limit=K_DEMO).points
print(f"Top-{K_DEMO} (sans filtre) :")
for h in hits:
    print(f"    (cos={h.score:.4f}) {h.payload['text'][:70]}")

hits_f = qclient.query_points(
    collection_name="news",
    query=q_demo[0].tolist(),
    limit=K_DEMO,
    query_filter=models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value="sci.space"))]
    ),
).points
print(f"Top-{K_DEMO} filtré category=sci.space : {len(hits_f)} résultats")
```

<!-- #region -->
## 8. LanceDB (embarqué, format Lance)
<!-- #endregion -->

<!-- #region -->
**LanceDB** est une base **embarquée** (in-process, zéro serveur) bâtie sur le format columnar **Lance**. On `connect` à un dossier (persistance sur disque), on `create_table` à partir d'une liste de dicts (vecteur + champs), puis `.search(...).limit(k)` retourne un DataFrame avec la colonne `_distance`.
<!-- #endregion -->

```python
import lancedb

ldb = lancedb.connect(str(WORKDIR / "lancedb"))
ltable = ldb.create_table(
    "news",
    data=[
        {"vector": demo_emb[i].tolist(), "text": demo_docs[i], "category": demo_cats[i]}
        for i in range(N_DEMO)
    ],
    mode="overwrite",
)

lres = ltable.search(q_demo[0].tolist()).limit(K_DEMO).to_pandas()
print(f"Top-{K_DEMO} :")
for _, row in lres.iterrows():
    print(f"    (dist={row['_distance']:.4f}) {row['text'][:70]}")
```

<!-- #region -->
## 9. Milvus (Milvus Lite, fichier local)
<!-- #endregion -->

<!-- #region -->
**Milvus** est une base conçue pour le **très grand volume** (architecture distribuée). Pour prototyper, **Milvus Lite** offre la même API dans un simple **fichier local** (`MilvusClient("xxx.db")`, disponible sous Linux/macOS). On crée une collection avec sa dimension et sa métrique, on `insert`, puis on `search`.
<!-- #endregion -->

```python
from pymilvus import MilvusClient

mclient = MilvusClient(str(WORKDIR / "milvus_demo.db"))
mclient.create_collection(collection_name="news", dimension=DIM_DEMO, metric_type="COSINE")
mclient.insert(
    collection_name="news",
    data=[
        {"id": i, "vector": demo_emb[i].tolist(), "text": demo_docs[i], "category": demo_cats[i]}
        for i in range(N_DEMO)
    ],
)

mres = mclient.search(collection_name="news", data=[q_demo[0].tolist()], limit=K_DEMO, output_fields=["text"])
print(f"Top-{K_DEMO} :")
for hit in mres[0]:
    print(f"    (cos={hit['distance']:.4f}) {hit['entity']['text'][:70]}")
mclient.close()
```

<!-- #region -->
Pour passer à l'échelle (mode serveur distribué), on lance Milvus **standalone** via Docker, puis on pointe le client sur `http://localhost:19530` au lieu d'un fichier :

```yaml
# docker-compose.yml (Milvus standalone, extrait)
services:
  milvus:
    image: milvusdb/milvus:v2.5.4
    command: ["milvus", "run", "standalone"]
    ports:
      - "19530:19530"   # gRPC
      - "9091:9091"     # metrics
    environment:
      ETCD_USE_EMBED: "true"
      COMMON_STORAGETYPE: local
```

```bash
docker compose up -d
# puis : MilvusClient("http://localhost:19530")
```
<!-- #endregion -->

<!-- #region -->
## 10. Weaviate (v4, mode serveur)
<!-- #endregion -->

<!-- #region -->
**Weaviate** fonctionne en **serveur** (Docker ou cloud). On le démarre via Docker Compose :

```yaml
# docker-compose.yml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.27.0
    ports:
      - "8080:8080"   # REST
      - "50051:50051" # gRPC
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
      DEFAULT_VECTORIZER_MODULE: none   # on fournit nos propres vecteurs
```

```bash
docker compose up -d
```

> ⚠️ **API v4** : le client a été entièrement repensé fin 2023. L'ancien code v3 (`weaviate.Client(...)`, `client.schema.create_class`, `client.data_object.create`) **ne fonctionne plus**. En v4 on utilise `weaviate.connect_to_local()`, `client.collections.create(...)`, `collection.data.insert(...)` / `batch`, et `collection.query.near_vector(...)`.
<!-- #endregion -->

<!-- #region -->
Code client **v4** ci-dessous. La cellule est **gardée** (`try/except`) : sans serveur Weaviate sur `:8080`, elle affiche un message et continue — le notebook reste exécutable de bout en bout.
<!-- #endregion -->

```python
try:
    import weaviate
    from weaviate.classes.config import DataType, Property

    wclient = weaviate.connect_to_local()  # nécessite un serveur sur :8080
    if wclient.collections.exists("News"):
        wclient.collections.delete("News")
    news_coll = wclient.collections.create(
        name="News",
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
        ],
    )
    with news_coll.batch.dynamic() as batch:
        for i in range(N_DEMO):
            batch.add_object(
                properties={"text": demo_docs[i], "category": demo_cats[i]}, vector=demo_emb[i].tolist()
            )
    wres = news_coll.query.near_vector(near_vector=q_demo[0].tolist(), limit=K_DEMO)
    for obj in wres.objects:
        print(f"    {obj.properties['text'][:70]}")
    wclient.close()
except Exception as exc:  # ImportError ou pas de serveur
    print(f"⚠️  Weaviate non disponible (serveur requis, voir docker-compose) — skip : {type(exc).__name__}")
```

<!-- #region -->
## 11. pgvector (PostgreSQL + extension)
<!-- #endregion -->

<!-- #region -->
**pgvector** ajoute un type `vector` à **PostgreSQL** : les embeddings vivent dans une colonne SQL, aux côtés de vos données relationnelles. Idéal si vos données sont **déjà** en Postgres (pas de base supplémentaire à opérer). On lance un Postgres avec l'extension :

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg17
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: postgres
```

```sql
-- dans la base : activer l'extension une fois
CREATE EXTENSION IF NOT EXISTS vector;
```

Opérateurs de distance : `<->` (L2), `<#>` (produit scalaire négatif), `<=>` (**distance cosinus**). On indexe ensuite avec `USING hnsw` ou `ivfflat` pour passer à l'échelle.
<!-- #endregion -->

<!-- #region -->
Code client `psycopg` ci-dessous, **gardé** (`try/except`, `connect_timeout=3`) : sans serveur Postgres, la cellule skippe proprement.
<!-- #endregion -->

```python
try:
    import psycopg

    conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/postgres", connect_timeout=3)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE TABLE IF NOT EXISTS news (id int PRIMARY KEY, text text, embedding vector({DIM_DEMO}))")
        for i in range(N_DEMO):
            cur.execute(
                "INSERT INTO news (id, text, embedding) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (i, demo_docs[i], demo_emb[i].tolist()),
            )
        # opérateur <=> = distance cosinus
        cur.execute("SELECT text FROM news ORDER BY embedding <=> %s::vector LIMIT %s", (q_demo[0].tolist(), K_DEMO))
        for (text,) in cur.fetchall():
            print(f"    {text[:70]}")
    conn.commit()
    conn.close()
except Exception as exc:  # ImportError ou pas de serveur
    print(f"⚠️  pgvector non disponible (Postgres requis, voir docker-compose) — skip : {type(exc).__name__}")
```

<!-- #region -->
## 12. Persistance des index
<!-- #endregion -->

<!-- #region -->
Ré-encoder et ré-indexer à chaque démarrage est coûteux. FAISS sérialise un index avec `write_index` / `read_index`. Les bases (Chroma `PersistentClient`, LanceDB, Milvus Lite) persistent nativement dans leur dossier/fichier.

> 💡 **Règle d'or** : versionnez l'index **avec le modèle d'embedding** qui l'a produit — un index n'est valide qu'avec le modèle exact ayant généré ses vecteurs.
<!-- #endregion -->

```python
index_path = WORKDIR / "faiss_news.index"
faiss.write_index(exact, str(index_path))
reloaded = faiss.read_index(str(index_path))
print(f"Index sauvé puis rechargé : ntotal={reloaded.ntotal} (avant={exact.ntotal})")
assert reloaded.ntotal == exact.ntotal
```

<!-- #region -->
## 13. Quel moteur choisir ? (2026)
<!-- #endregion -->

<!-- #region -->
Petit arbre de décision :

- **Prototype / app locale, peu d'ops** → **Chroma** ou **LanceDB** (embarqués, zéro serveur).
- **Choix équilibré, filtrage riche, montée en charge progressive** → **Qdrant**.
- **Données déjà dans PostgreSQL, scale modéré, joins SQL** → **pgvector**.
- **Très grand volume, débit élevé, multi-tenant** → **Milvus** ou **Weaviate**.
- **Moteur de recherche sur-mesure, contrôle total des index** → **FAISS** (en brique de bas niveau).

Quelle que soit la base : **normalisez** vos vecteurs et utilisez le **cosinus** avec les embeddings modernes, et indexez en **HNSW** dès que le corpus grossit. Pour réduire les coûts mémoire à grande échelle, pensez à la **quantization** (PQ, scalar/binary) et aux embeddings **Matryoshka** (tronquer la dimension — ex. 1024 → 256 — sans ré-entraîner).
<!-- #endregion -->

<!-- #region -->
### Pour aller plus loin
<!-- #endregion -->

<!-- #region -->
- **Quantization** (PQ, scalar quantization) pour réduire l'empreinte mémoire des gros index.
- **Recherche hybride** : combiner similarité vectorielle et BM25 (mots-clés), puis **reranking** d'un cross-encoder.
- **RAG** : brancher ce *retriever* sur un LLM (génération augmentée par récupération). L'ingestion de documents en amont (PDF, OCR, découpage) est traitée dans le notebook `DE_Docling`.
<!-- #endregion -->
