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
# 🔍 Recherche d'informations et RAG
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur la **recherche d'informations** (IR) appliquée au NLP moderne, et son extension naturelle en 2026 : **RAG** (Retrieval-Augmented Generation).

Couvre tout le spectre :

1. **Sparse retrieval** classique — TF-IDF, **BM25** (le baseline qui résiste à tout).
2. **Dense retrieval** moderne — embeddings denses via Sentence Transformers, BGE, E5.
3. **Hybrid search** — fusion BM25 + dense (Reciprocal Rank Fusion).
4. **Re-ranking** — ColBERT, cross-encoders.
5. **Pipeline RAG complet** — retrieval + LLM pour générer une réponse contextualisée.
6. **Frameworks haut niveau** — LangChain, LlamaIndex (panorama).
7. **Évaluation** RAG — Recall@k, MRR, RAGAS.

> Pour la partie **stockage** des embeddings (vector DBs : FAISS, Qdrant, LanceDB, pgvector), voir le notebook dédié `BDD_Vectorielles`.
<!-- #endregion -->

<!-- #region -->
## 1. Le problème — pourquoi pas un LLM seul
<!-- #endregion -->

<!-- #region -->
Un LLM (Llama, Mistral, GPT) connaît son corpus d'entraînement mais :

- **Knowledge cutoff** — il ignore tout ce qui s'est passé après sa date d'entraînement.
- **Hallucinations** — sur un fait précis qu'il n'a pas vu, il invente plausiblement.
- **Coût de fine-tuning** — incorporer 1 GB de docs internes en fine-tuning : long, cher, à refaire à chaque mise à jour.
- **Traçabilité** — un LLM ne peut pas citer ses sources.

**Solution : RAG.** À chaque question, on **retrouve** les passages pertinents dans une base documentaire, on les **passe en contexte** au LLM, qui génère une réponse **ancrée** sur ces passages (et peut citer les sources).

```
question  →  retrieval (top-k passages)  →  prompt augmenté  →  LLM  →  réponse + sources
```

Le **retrieval** est le maillon critique : un bon LLM avec un mauvais retrieval donnera de mauvaises réponses. D'où l'importance de bien comprendre les algos de recherche.
<!-- #endregion -->

<!-- #region -->
## 2. Vue d'ensemble du paysage 2026
<!-- #endregion -->

<!-- #region -->
Trois familles d'approches, à combiner :

| Famille | Idée | Forces | Faiblesses |
|---|---|---|---|
| **Sparse** (TF-IDF, BM25, SPLADE) | Match exact de mots / tokens | Excellent sur **vocabulaire spécifique** (jargon, codes, noms propres). Pas d'entraînement. Interprétable. | Échoue sur reformulations ("voiture" vs "automobile"). |
| **Dense** (sentence-transformers, BGE, E5, GTE) | Embeddings vectoriels, similarité cosine | Comprend **paraphrase et sémantique**. Multilingue. | Peut rater le mot-clé exact. Demande GPU à grosse échelle. Modèle à choisir/entraîner. |
| **Hybrid** | BM25 + dense, fusion des rankings (RRF) | Best of both worlds — **le baseline de prod en 2026**. | Plus de pièces à orchestrer. |

Au-dessus, on ajoute du **re-ranking** (ColBERT, cross-encoder) sur le top-50 du retrieval pour booster la précision à top-5.

**Constat 2026** (benchmark BEIR + études sectorielles) :
- BM25 reste **étonnamment fort** sur des corpus à vocabulaire précis (juridique, finance, code).
- Dense gagne sur les corpus conversationnels et la paraphrase.
- **Hybrid + reranking** = quasi systématiquement le meilleur (Recall@5 ≈ 0.82, MRR ≈ 0.60 sur benchmarks docs+tables).
<!-- #endregion -->

<!-- #region -->
## 3. Sparse retrieval : TF-IDF et BM25
<!-- #endregion -->

<!-- #region -->
### 3.1 TF-IDF — le baseline classique
<!-- #endregion -->

<!-- #region -->
Pour chaque terme `t` dans le document `d` :

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{|\{d' : t \in d'\}|}
$$

- **TF** : fréquence du terme dans le doc (souvent log-normalisée).
- **IDF** : inverse de la fréquence documentaire — pénalise les termes communs ("le", "et").

Score d'une query `q` vs doc `d` : produit scalaire / cosine similarity entre vecteurs TF-IDF.

**Limites** : pondération brute, pas de normalisation par longueur du doc, sensible aux mots fréquents.
<!-- #endregion -->

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Mini corpus pour la démo
corpus = [
    "Python is a high-level general-purpose programming language.",
    "FAISS is a library for efficient similarity search of dense vectors.",
    "Transformers are neural network architectures based on self-attention.",
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Cosine similarity measures the angle between two vectors.",
    "Hybrid search combines lexical and semantic retrieval.",
]

vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(corpus)


def search_tfidf(query: str, k: int = 3) -> list[tuple[int, float, str]]:
    """Renvoie le top-k (idx, score, doc) pour la query."""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, doc_vectors).ravel()
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), corpus[i]) for i in top_idx]


for idx, score, doc in search_tfidf("vector similarity search", k=3):
    print(f"  [{score:.3f}] {doc}")
```

<!-- #region -->
### 3.2 BM25 — le baseline qui résiste à tout
<!-- #endregion -->

<!-- #region -->
**BM25** (Best Match 25) est une amélioration probabiliste de TF-IDF qui domine encore en 2026 sur de nombreux corpus. Il normalise par longueur du document et sature la contribution de la fréquence.

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{tf}(t, d) \cdot (k_1 + 1)}{\text{tf}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

- `k_1 ≈ 1.2-2.0` : saturation de la TF.
- `b ≈ 0.75` : importance de la normalisation par longueur.
- `avgdl` : longueur moyenne des docs du corpus.

**Pourquoi ça marche encore en 2026** : BM25 capture parfaitement les **matchs exacts** (noms propres, codes, jargon) là où les embeddings denses peuvent "lisser" l'information.

On utilise `bm25s`, une implémentation pure Python ultra rapide (faster than Elasticsearch sur petit-moyen corpus).
<!-- #endregion -->

```python
import bm25s

# Tokenization simple + indexation
corpus_tokens = bm25s.tokenize(corpus, stopwords="english")
retriever = bm25s.BM25()
retriever.index(corpus_tokens)


def search_bm25(query: str, k: int = 3) -> list[tuple[int, float, str]]:
    """Top-k via BM25."""
    q_tokens = bm25s.tokenize([query], stopwords="english")
    results, scores = retriever.retrieve(q_tokens, k=k)
    out = []
    for idx, score in zip(results[0], scores[0]):
        out.append((int(idx), float(score), corpus[int(idx)]))
    return out


for idx, score, doc in search_bm25("vector similarity search", k=3):
    print(f"  [{score:.3f}] {doc}")
```

<!-- #region -->
## 4. Dense retrieval : Sentence Transformers, BGE, E5
<!-- #endregion -->

<!-- #region -->
### 4.1 Idée
<!-- #endregion -->

<!-- #region -->
On encode chaque document en un **vecteur dense** `v ∈ ℝ^d` (typiquement `d=384, 768, 1024`). La query est encodée pareil. La similarité cosine entre vecteurs reflète la similarité sémantique.

Différence-clé avec BERT brut :
- Un BERT pré-entraîné renvoie des embeddings non alignés pour la similarité — il faut un **fine-tuning contrastif** (positifs proches, négatifs éloignés).
- C'est ce que font **Sentence-Transformers** (SBERT), et les familles modernes **BGE** (BAAI), **E5** (Microsoft), **GTE** (Alibaba), **Nomic Embed**, **MixedBread mxbai**.

**Modèles 2026 recommandés** :

| Modèle | Dim | Langues | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | EN | Léger et excellent, top du MTEB pour sa taille |
| `BAAI/bge-m3` | 1024 | 100+ | Multilingue + sparse + dense hybride natif |
| `intfloat/multilingual-e5-large-instruct` | 1024 | 100+ | Instruction-following à l'encodage |
| `Alibaba-NLP/gte-modernbert-base` | 768 | EN | ModernBERT pour la vitesse + contexte 8k |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | EN | Le classique léger, rapide |

> Référence du leaderboard : **MTEB** (Massive Text Embedding Benchmark) sur HF Spaces.
<!-- #endregion -->

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Modèle léger pour la démo (90 MB, tourne CPU)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

doc_embeddings = encoder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
print(f"Shape embeddings : {doc_embeddings.shape}")


def search_dense(query: str, k: int = 3) -> list[tuple[int, float, str]]:
    """Top-k par cosine similarity."""
    q_emb = encoder.encode([query], normalize_embeddings=True)
    # Vecteurs normalisés → cosine = produit scalaire
    scores = (doc_embeddings @ q_emb.T).ravel()
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), corpus[i]) for i in top_idx]


for idx, score, doc in search_dense("vector similarity search", k=3):
    print(f"  [{score:.3f}] {doc}")
```

<!-- #region -->
> **Note paraphrase** : essaie une query qui n'a aucun mot en commun avec les docs (ex: `"finding similar objects in a database"`) → dense retrieval s'en sort, BM25 retourne potentiellement du bruit. C'est le superpouvoir du dense.
<!-- #endregion -->

<!-- #region -->
## 5. Hybrid search : Reciprocal Rank Fusion (RRF)
<!-- #endregion -->

<!-- #region -->
Combiner BM25 et dense, **sans normaliser les scores** (qui sont sur des échelles différentes). La solution standard est **RRF** (Reciprocal Rank Fusion) :

$$
\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}
$$

avec `k = 60` (constante usuelle, peu sensible). Chaque retriever vote pour les docs en fonction de leur **rang** (1er, 2ème, ...) — peu importe que les scores soient calibrés.

**C'est le pattern n°1 en prod RAG en 2026** : run BM25 + dense en parallèle, fusion RRF, top-k final.
<!-- #endregion -->

```python
def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fusion RRF de plusieurs rankings (liste d'indices triés).

    Args:
        rankings: liste de listes d'indices, dans l'ordre décroissant de score.
        k: constante RRF (60 par défaut).
    Returns:
        liste (idx, score_rrf) triée par score décroissant.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_hybrid(query: str, k: int = 3) -> list[tuple[int, float, str]]:
    """Top-k hybrid : BM25 + dense, fusion RRF."""
    bm25_hits = [idx for idx, _, _ in search_bm25(query, k=10)]
    dense_hits = [idx for idx, _, _ in search_dense(query, k=10)]
    fused = rrf_fuse([bm25_hits, dense_hits])[:k]
    return [(idx, score, corpus[idx]) for idx, score in fused]


for idx, score, doc in search_hybrid("how to find similar vectors fast", k=3):
    print(f"  [{score:.4f}] {doc}")
```

<!-- #region -->
## 6. Re-ranking : ColBERT et cross-encoders
<!-- #endregion -->

<!-- #region -->
**Idée du pipeline 2-stage** :

1. **Retrieval** rapide (BM25 + dense + RRF) → top-100 candidats.
2. **Re-ranking** précis sur ces 100 candidats avec un modèle plus lourd → top-5/10 final.

Le retriever travaille en `O(corpus)` (rapide mais moins précis), le reranker en `O(100)` (lent par doc mais sur peu de candidats).

**Deux familles de rerankers** :

- **Cross-encoders** — concatène `[CLS] query [SEP] doc [SEP]`, passe dans un BERT-like, sortie = score scalaire. Le plus précis, plus lent. Ex : `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`.
- **ColBERT** — late interaction : encode query et doc séparément en suite de token-embeddings, puis pour chaque token query prend le max sim avec les tokens du doc, somme. Très précis et plus rapide qu'un cross-encoder full.

**Modèles 2026 phares** :

| Modèle | Type | Notes |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | Cross-encoder | Multilingue, top du MTEB reranking |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | Léger, baseline en/web |
| `colbert-ir/colbertv2.0` | ColBERT | Pipeline indexé (RAGatouille) |
| `mixedbread-ai/mxbai-rerank-large-v1` | Cross-encoder | Open-source, top quality |
<!-- #endregion -->

```python
from sentence_transformers import CrossEncoder

# Reranker léger pour démo
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def search_two_stage(query: str, k_retrieve: int = 10, k_final: int = 3) -> list[tuple[int, float, str]]:
    """Pipeline 2-stage : retrieval hybride + re-ranking cross-encoder."""
    # Stage 1 : retrieval hybride
    candidates = search_hybrid(query, k=k_retrieve)
    cand_idx = [c[0] for c in candidates]
    cand_docs = [corpus[i] for i in cand_idx]

    # Stage 2 : reranking
    pairs = [[query, doc] for doc in cand_docs]
    rerank_scores = reranker.predict(pairs)

    # Tri final
    reranked = sorted(zip(cand_idx, rerank_scores, cand_docs), key=lambda x: x[1], reverse=True)
    return [(int(i), float(s), d) for i, s, d in reranked[:k_final]]


for idx, score, doc in search_two_stage("how to find similar vectors fast", k_retrieve=6, k_final=3):
    print(f"  [{score:+.3f}] {doc}")
```

<!-- #region -->
## 7. Pipeline RAG complet : retrieval + LLM generation
<!-- #endregion -->

<!-- #region -->
Le retriever donne **les passages**. Le LLM **synthétise** une réponse en langage naturel ancrée sur ces passages.

Le prompt suit toujours la même structure :

```
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".
Cite the source numbers [1], [2] for each claim.

Context:
[1] {passage_1}
[2] {passage_2}
...

Question: {question}
Answer:
```

Cette discipline du **"context-only"** réduit drastiquement les hallucinations.
<!-- #endregion -->

```python
from transformers import pipeline as hf_pipeline

# LLM léger pour démo CPU
llm = hf_pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    device=-1,
)


def build_rag_prompt(question: str, passages: list[str]) -> list[dict]:
    """Construit les messages chat avec contexte injecté."""
    context_block = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    system = (
        "You are a precise assistant. Answer using ONLY the context. "
        "If the answer is not in the context, say 'I don't know'. "
        "Cite source numbers like [1], [2]."
    )
    user = f"Context:\n{context_block}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rag_answer(question: str, k: int = 3) -> str:
    """Pipeline RAG complet : retrieve top-k + generate."""
    top = search_two_stage(question, k_retrieve=10, k_final=k)
    passages = [doc for _, _, doc in top]
    messages = build_rag_prompt(question, passages)
    out = llm(messages, max_new_tokens=120, do_sample=False)
    return out[0]["generated_text"][-1]["content"]


# Démo
question = "What is BM25 used for?"
print(rag_answer(question))
```

<!-- #region -->
> **Note** : avec un SmolLM 135M la qualité de génération est limitée — c'est juste pour valider le pipeline. En prod, utilise Llama 3.x, Mistral, Qwen 3 (via vLLM/TGI) ou une API.
<!-- #endregion -->

<!-- #region -->
## 8. Frameworks haut niveau : LangChain et LlamaIndex
<!-- #endregion -->

<!-- #region -->
Au lieu d'écrire la plomberie soi-même, deux frameworks dominent en 2026 :

| Framework | Forces | Faiblesses |
|---|---|---|
| **LangChain** | Très large couverture (chains, agents, tools, memory), modulaire | API en mouvement, parfois trop d'abstractions |
| **LlamaIndex** | Centré sur le RAG / les indexes, très bon retriever store, query engines avancés | Moins extensif côté agents |

**Mon conseil pratique en 2026** :

- Pour un **RAG simple bien fait** : LlamaIndex (ou directement sentence-transformers + Qdrant).
- Pour des **workflows multi-step avec agents et tools** : LangChain ou LangGraph.
- Pour **prototyper vite** : DSPy (déclaratif, signatures, optimisation auto des prompts).
- Pour la **prod custom** : code pur (transformers + sentence-transformers + vector DB) — moins de surprises.

```python
# Exemple LlamaIndex en 5 lignes (pseudo-code) :
# from llama_index.core import VectorStoreIndex, Document
# documents = [Document(text=t) for t in corpus]
# index = VectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine()
# response = query_engine.query("What is BM25?")
```
<!-- #endregion -->

<!-- #region -->
## 9. Évaluation RAG : Recall@k, MRR, RAGAS
<!-- #endregion -->

<!-- #region -->
### 9.1 Métriques retrieval (mesure si on retrouve les bons docs)
<!-- #endregion -->

<!-- #region -->
- **Recall@k** — proportion des docs pertinents qui apparaissent dans le top-k.
- **MRR** (Mean Reciprocal Rank) — moyenne de `1 / rang du premier doc pertinent`.
- **nDCG@k** — Normalized Discounted Cumulative Gain, pondère par la position.
- **MAP** (Mean Average Precision).

Datasets de référence : **BEIR** (18 datasets standardisés), **MS MARCO** (passage retrieval), **MTEB** (embeddings).
<!-- #endregion -->

```python
def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Proportion de docs pertinents dans le top-k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved: list[int], relevant: set[int]) -> float:
    """Reciprocal rank du 1er doc pertinent (0 si aucun)."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


# Mini éval : pour la query "vector similarity", les docs pertinents sont 1 (FAISS) et 4 (cosine)
retrieved = [i for i, _, _ in search_two_stage("vector similarity search", k_retrieve=6, k_final=6)]
relevant_set = {1, 4}
print(f"Top retrieved : {retrieved}")
print(f"Recall@3 = {recall_at_k(retrieved, relevant_set, k=3):.3f}")
print(f"Recall@5 = {recall_at_k(retrieved, relevant_set, k=5):.3f}")
print(f"MRR      = {mrr(retrieved, relevant_set):.3f}")
```

<!-- #region -->
### 9.2 Métriques RAG end-to-end (mesure la qualité de la réponse)
<!-- #endregion -->

<!-- #region -->
**RAGAS** est le framework de référence 2026. Métriques principales :

- **Faithfulness** — la réponse est-elle ancrée sur le contexte ? (anti-hallucination)
- **Answer Relevancy** — la réponse répond-elle à la question ?
- **Context Precision** — les passages retournés sont-ils pertinents (vs bruit) ?
- **Context Recall** — couvre-t-on bien l'info nécessaire ?
- **Answer Correctness** — comparaison vs ground truth (si dispo).

Ces métriques utilisent un **LLM-as-judge** : un LLM (souvent GPT-4 ou Claude) note chaque axe.

```python
# Exemple usage RAGAS (pseudo-code) :
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy
# results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
```
<!-- #endregion -->

<!-- #region -->
## 10. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 10.1 Préparation des documents
<!-- #endregion -->

<!-- #region -->
- **Chunking intelligent** : pas de découpe brute à 512 tokens. Préférer un chunking par **structure logique** (paragraphe, section, jusqu'à 800 tokens) + **overlap** de 100-200 tokens. Voir **Docling** pour les PDFs / docs complexes (notebook `DE_Docling`).
- **Métadonnées** : conserver `source`, `page`, `date`, `auteur`, `chapitre` pour pouvoir filtrer et citer.
- **Pré-traitement** : enlever boilerplate (en-têtes / pieds de page), normaliser unicode, dédupliquer.
<!-- #endregion -->

<!-- #region -->
### 10.2 Choix du retriever
<!-- #endregion -->

<!-- #region -->
- **Démarrer en hybrid** dès le début. Évite de regretter plus tard.
- **Vocabulaire technique / noms propres** → BM25 a un poids fort, parfois 0.7.
- **Conversationnel / paraphrase** → dense a un poids fort, parfois 0.7.
- **Toujours ajouter un reranker** au bout, ça coûte peu et gagne beaucoup.
<!-- #endregion -->

<!-- #region -->
### 10.3 Engineering de prompt
<!-- #endregion -->

<!-- #region -->
- **Contexte AVANT la question** (recence bias des LLMs).
- **Instruction "use ONLY the context"** explicite.
- **Cite tes sources** dans la consigne.
- Pour les **réponses longues** : demande un format (bullet, JSON) pour réduire la divagation.
- **Compression** du contexte si nécessaire : `LongLLMLingua`, `ContextCompressor` de LlamaIndex.
<!-- #endregion -->

<!-- #region -->
### 10.4 Évaluation et monitoring
<!-- #endregion -->

<!-- #region -->
- Constituer un **dataset eval** dès le début (50-200 paires question/réponse de référence).
- Tracker en CI : Recall@5, MRR, et au moins 1-2 métriques RAGAS.
- En prod : logger les queries, retrievals, réponses → permet du **fine-tuning du retriever** sur des hard negatives quelques mois plus tard.
<!-- #endregion -->

<!-- #region -->
### 10.5 Variantes avancées
<!-- #endregion -->

<!-- #region -->
- **HyDE** (Hypothetical Document Embeddings) — le LLM génère une "fausse" réponse à la query, on encode cette réponse pour faire le retrieval (souvent meilleur que d'encoder la query nue).
- **Multi-query** — le LLM génère 3-5 reformulations de la query, on retrieve sur chacune et on fusionne.
- **Corrective RAG (CRAG)** — un classifieur évalue la pertinence des passages retournés ; si insuffisante, fallback web search.
- **Self-RAG** — le LLM apprend à décider quand retriever est nécessaire et à critiquer ses propres réponses.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [BM25 vs Sparse vs Hybrid Search in RAG — Medium 2026](https://medium.com/@dewasheesh.rana/bm25-vs-sparse-vs-hybrid-search-in-rag-from-layman-to-pro-e34ff21c4ada)
- [Hybrid RAG Search: BM25 + Embeddings — Deep Dive 2026](https://techbytes.app/posts/hybrid-rag-search-bm25-embeddings-deep-dive-2026/)
- [Production RAG : Hybrid + ReRanking — Machine-Mind ML](https://machine-mind-ml.medium.com/production-rag-that-works-hybrid-search-re-ranking-colbert-splade-e5-bge-624e9703fa2b)
- [From BM25 to Corrective RAG — arXiv 2604.01733](https://arxiv.org/abs/2604.01733)
- [BEIR benchmark](https://github.com/beir-cellar/beir)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence-Transformers docs](https://www.sbert.net/)
- [bm25s — fast pure-Python BM25](https://github.com/xhluca/bm25s)
- [RAGAS framework](https://docs.ragas.io/)
- [LlamaIndex docs](https://docs.llamaindex.ai/)
- [LangChain docs](https://python.langchain.com/)
<!-- #endregion -->
