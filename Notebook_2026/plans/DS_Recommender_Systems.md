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
# 🎬 Recommender Systems
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Scientist + AI Engineer · Wiki + Tutoriel

**Dataset(s)** : MovieLens 1M / 25M, Amazon reviews

Systèmes de recommandation : du baseline collaborative filtering aux modèles 2026 (transformer / LLM-based).

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre
<!-- #endregion -->

<!-- #region -->
Explicit (ratings 1-5) vs implicit (clicks, views, time spent). Sparse matrix users × items. Métriques en ligne (CTR) vs offline.
<!-- #endregion -->

<!-- #region -->
## 2. Baselines
<!-- #endregion -->

<!-- #region -->
Popularité (most-watched), item-item top similar (cosine over items), user-user. Souvent baselines très fortes.
<!-- #endregion -->

<!-- #region -->
## 3. Collaborative Filtering KNN
<!-- #endregion -->

<!-- #region -->
User-KNN, Item-KNN. Memory-based vs model-based. Lib : `surprise`.
<!-- #endregion -->

<!-- #region -->
## 4. Matrix Factorization — SVD, ALS, NMF
<!-- #endregion -->

<!-- #region -->
Décomposition U×V^T avec rangs latents. SGD vs ALS. Implicit-feedback ALS (Hu 2008).
<!-- #endregion -->

<!-- #region -->
## 5. Deep Learning
<!-- #endregion -->

<!-- #region -->
Neural CF (NCF, He 2017), Wide & Deep, DeepFM. Embeddings users/items + MLP.
<!-- #endregion -->

<!-- #region -->
## 6. Two-Tower models
<!-- #endregion -->

<!-- #region -->
YouTube DNN, Facebook EBR. User tower + item tower, similarity au runtime. Sert le retrieval ANN.
<!-- #endregion -->

<!-- #region -->
## 7. Sequential recommenders
<!-- #endregion -->

<!-- #region -->
SASRec (Self-Attention), BERT4Rec, Transformer4Rec (NVIDIA). Capturer l'ordre des actions.
<!-- #endregion -->

<!-- #region -->
## 8. LLM-based recommenders (2024-2026)
<!-- #endregion -->

<!-- #region -->
Approche : retrieve top-N via embeddings + reranking via LLM. P5 (Google), LLaRA, RecLM. Cold start beaucoup plus facile.
<!-- #endregion -->

<!-- #region -->
## 9. Évaluation offline
<!-- #endregion -->

<!-- #region -->
Precision@k, Recall@k, NDCG, MRR, Hit Rate, Coverage, Novelty, Diversity. Leave-one-out vs random split.
<!-- #endregion -->

<!-- #region -->
## 10. Cold start et exploration/exploitation
<!-- #endregion -->

<!-- #region -->
Bandits contextuels (LinUCB, Thompson sampling). Hybrides content + collaborative.
<!-- #endregion -->

<!-- #region -->
## 11. Stack 2026
<!-- #endregion -->

<!-- #region -->
Libs : `implicit`, `lightfm`, `surprise`, `recbole`, `nvidia/merlin`, `recommenders` (Microsoft).
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Microsoft Recommenders — repos](https://github.com/microsoft/recommenders)
- [RecBole](https://recbole.io/)
- [Merlin — NVIDIA](https://github.com/NVIDIA-Merlin/Merlin)
- [Awesome RecSys](https://github.com/jihoo-kim/awesome-RecSys)
<!-- #endregion -->
