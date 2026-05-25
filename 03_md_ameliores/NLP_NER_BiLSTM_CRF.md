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
# 🕰️ NER avec BiLSTM-CRF (Wiki historique)
<!-- #endregion -->

<!-- #region -->
**⚠️ Méthode historique** (état de l'art 2016-2019, dépassée par les transformers depuis 2019).

Ce notebook est conservé comme **référence pédagogique et historique** sur l'architecture qui a dominé la NER pendant 4 ans avant l'avènement de BERT. Pour la NER en production en 2026, voir le notebook **`NLP_NER`** (transformers + GLiNER).

**Pourquoi continuer à comprendre BiLSTM-CRF en 2026 ?**

1. **Pédagogie** — comprendre la **CRF** (Conditional Random Field) reste utile : on retrouve l'idée dans d'autres structures (parsing, segmentation, alignement).
2. **Légèreté** — un BiLSTM-CRF tourne sans GPU, fait <50 MB, peut être déployé sur edge.
3. **Domaines low-resource** — sur des langues/domaines très spécifiques avec peu de data, un BiLSTM-CRF custom peut encore rivaliser avec un transformer fine-tuné mal.
4. **Comprendre la littérature** — la moitié des papers NER pré-2020 utilisent cette archi.

Couverture :
1. Rappel : le problème NER (renvoi au notebook principal).
2. **Embeddings de tokens** : word embeddings vs character-level.
3. **BiLSTM** : architecture, intuition.
4. **CRF** : la formulation mathématique et pourquoi c'est mieux qu'une softmax token-par-token.
5. **BiLSTM-CRF** : l'assemblage.
6. Implémentation PyTorch minimale.
7. Bilan : forces, limites, alternatives 2026.
<!-- #endregion -->

<!-- #region -->
## 1. Rappel — NER en 30 secondes
<!-- #endregion -->

<!-- #region -->
Voir `NLP_NER.ipynb` pour les bases (formats IOB, datasets, métriques). En résumé :

- Tâche de **token classification** : chaque token reçoit un tag (`O`, `B-PER`, `I-PER`, ...).
- Difficulté : les tags ne sont **pas indépendants** entre eux. `I-PER` ne peut pas suivre `O` (illégal en IOB2). C'est cette dépendance que la CRF va modéliser.
<!-- #endregion -->

<!-- #region -->
## 2. Embeddings de tokens
<!-- #endregion -->

<!-- #region -->
Pour passer un mot à un réseau, il faut un **vecteur**. Trois options historiques :

| Embedding | Idée | Forces | Faiblesses |
|---|---|---|---|
| **Random init** (lookup table) | Embed appris from scratch | Simple, marche bien si beaucoup de data | Pas de généralisation aux mots rares |
| **Word2Vec / GloVe / FastText** | Pré-entraîné sur grand corpus | Sémantique pré-acquise | Pas contextuel ("bank" = banque ou rive ?) |
| **Character-level CNN/BiLSTM** | Embed le mot à partir de ses caractères | Robuste aux fautes / out-of-vocabulary | Plus lourd |

Le pattern qui a gagné en 2016-2018 : **concat(GloVe + char-BiLSTM)** — gère le vocabulaire ouvert ET garde la sémantique.

Les transformers ont rendu tout ça obsolète en 2019 (les sub-word tokenizers + attention bidirectionnelle font mieux que GloVe + char-BiLSTM).
<!-- #endregion -->

<!-- #region -->
## 3. BiLSTM — Bidirectional LSTM
<!-- #endregion -->

<!-- #region -->
Un **LSTM** lit la séquence de gauche à droite et produit un hidden state `h_t` par token, qui résume "tout ce qu'il a vu jusqu'à `t`".

Problème : pour tagger le token `t`, on a besoin du contexte **avant** ET **après** `t`. Un LSTM forward ne voit que le passé.

**BiLSTM** : on lance un LSTM forward et un LSTM backward, on concatène leurs hidden states à chaque position :

$$
\mathbf{h}_t = \left[\overrightarrow{\mathbf{h}}_t \; ; \; \overleftarrow{\mathbf{h}}_t\right] \in \mathbb{R}^{2d}
$$

Chaque `h_t` encode "mot + tout son contexte (gauche et droite)". On peut ensuite classer chaque `h_t` en tag.

**Limite** : si on met juste une softmax indépendante sur chaque `h_t`, on perd la **dépendance entre tags adjacents** — le modèle peut prédire `O B-PER O I-PER` (séquence illégale).
<!-- #endregion -->

<!-- #region -->
## 4. CRF — Conditional Random Field
<!-- #endregion -->

<!-- #region -->
La **CRF linear-chain** modélise la probabilité d'une **séquence entière** de tags étant donné une séquence d'inputs.

### Formulation

Pour une séquence de tags `y = (y_1, ..., y_T)` et de scores d'émission `s = (s_1, ..., s_T)` (sortie du BiLSTM, `s_t ∈ ℝ^|tags|`), la CRF définit :

$$
P(y | s) = \frac{\exp\!\big(\sum_{t=1}^{T} s_{t, y_t} + \sum_{t=1}^{T-1} A_{y_t, y_{t+1}}\big)}{Z(s)}
$$

- **`s_{t, y_t}`** : score d'émission — "à quel point la position `t` aime être taggée `y_t`" (vient du BiLSTM).
- **`A_{i, j}`** : score de transition — "à quel point on aime passer de tag `i` à tag `j`". `A` est une matrice apprise `|tags| × |tags|`.
- **`Z(s)`** : la **fonction de partition**, somme sur toutes les séquences possibles — calculable en `O(T·|tags|²)` via l'algo forward.

### Pourquoi c'est mieux qu'une softmax

- La softmax token-par-token est `P(y_t | s_t)` indépendant pour chaque `t`. La CRF est `P(y_1, ..., y_T | s)` joint.
- Concrètement, la CRF **interdit** (apprend à donner score quasi `-∞`) les transitions illégales : `O → I-X` jamais après un `O`, `I-PER` jamais après `B-LOC`.
- Gain F1 typique : **+1 à +2 points** sur CoNLL-2003 par rapport à une simple softmax sur le BiLSTM.

### Inférence

À l'inférence, on cherche `y* = argmax_y P(y | s)`. Calculé en `O(T·|tags|²)` via l'**algo de Viterbi** (programmation dynamique).
<!-- #endregion -->

<!-- #region -->
## 5. BiLSTM-CRF : l'assemblage
<!-- #endregion -->

<!-- #region -->
```
tokens ─→ Embedding (word + char) ─→ BiLSTM ─→ scores d'émission s_t
                                                       │
                                                       ▼
                                                     CRF (matrice A apprise)
                                                       │
                                                       ▼
                                             argmax P(y|s) via Viterbi
```

Entraînement par **maximum de vraisemblance** : on minimise `-log P(y_gold | s)`. Le gradient backprop via la CRF est calculable analytiquement (forward / backward).
<!-- #endregion -->

<!-- #region -->
## 6. Implémentation PyTorch minimale
<!-- #endregion -->

<!-- #region -->
En 2026, on **n'écrit plus la CRF à la main** — on utilise une lib comme **`torchcrf`** (`pip install pytorch-crf`) ou **TorchCRF**. L'archi globale fait moins de 50 lignes.

> Pour reproduire ce notebook, ajouter `pytorch-crf` à l'env. Code ci-dessous fourni comme **pseudo-code éducatif** (non exécuté pour éviter d'alourdir les deps).
<!-- #endregion -->

```python
# Pseudo-code BiLSTM-CRF (référence — à activer avec pytorch-crf installé)
"""
import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_tags: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, bidirectional=True, batch_first=True,
        )
        self.hidden2tag = nn.Linear(2 * hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        emb = self.embedding(input_ids)                # (B, T, E)
        lstm_out, _ = self.bilstm(emb)                 # (B, T, 2H)
        emissions = self.hidden2tag(lstm_out)          # (B, T, num_tags)
        return emissions, mask

    def loss(self, input_ids, tags, mask):
        emissions, mask = self.forward(input_ids, mask)
        # CRF.forward renvoie log-likelihood (à maximiser, donc -log_lik à minimiser)
        return -self.crf(emissions, tags, mask=mask.bool(), reduction='mean')

    def decode(self, input_ids, mask):
        emissions, mask = self.forward(input_ids, mask)
        return self.crf.decode(emissions, mask=mask.bool())  # liste de listes de tag ids


# Training loop minimal
# for input_ids, tags, mask in dataloader:
#     loss = model.loss(input_ids, tags, mask)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
"""
print("Pseudo-code BiLSTM-CRF — voir docstring pour la structure complète.")
```

<!-- #region -->
## 7. Bilan et alternatives 2026
<!-- #endregion -->

<!-- #region -->
### 7.1 Forces (historiques)
<!-- #endregion -->

<!-- #region -->
- Léger : ~10-50 MB sur disque, ~50 MB RAM, tourne CPU 30-100 tokens/ms.
- Pas besoin de pré-entraînement massif — tourne sur quelques milliers d'exemples annotés.
- Architecture **interprétable** (matrice de transition observable).
<!-- #endregion -->

<!-- #region -->
### 7.2 Limites (qui ont causé sa chute)
<!-- #endregion -->

<!-- #region -->
- Embeddings non contextuels (sauf à concaténer ELMo, mais ELMo a aussi été déclassé).
- Capacité limitée — un BiLSTM oublie le contexte au-delà de ~30 tokens en pratique.
- Pas d'auto-attention → pas de capture des dépendances longue distance.
<!-- #endregion -->

<!-- #region -->
### 7.3 Alternatives 2026
<!-- #endregion -->

<!-- #region -->
Par ordre de qualité décroissante :

| Approche | F1 typique CoNLL-2003 | Coût compute | Notes |
|---|---|---|---|
| **DeBERTa-v3 large fine-tuned** | 93.5+ | 700M params | SOTA dur à battre |
| **ModernBERT fine-tuned** | 93+ | 150M | Le meilleur rapport perf/coût en 2026 |
| **DistilBERT fine-tuned** | 91-92 | 66M | Baseline raisonnable |
| **GLiNER fine-tuned** | 90+ | 300M | Si schéma fluide |
| **BiLSTM-CRF (ce notebook)** | 88-89 | ~10 MB | **Edge / low-resource only** |
| **CRF seul (sklearn-crfsuite)** | 80-85 | ~1 MB | Quand vraiment rien d'autre |
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- **Paper fondateur** : Lample et al. (2016) — *Neural Architectures for Named Entity Recognition* — [arXiv:1603.01360](https://arxiv.org/abs/1603.01360)
- **Tutoriel CRF** : Sutton & McCallum — *An Introduction to Conditional Random Fields* — [arXiv:1011.4088](https://arxiv.org/abs/1011.4088)
- **pytorch-crf** : [GitHub kmkurn/pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- **Pour la NER moderne** : voir le notebook `NLP_NER.ipynb` (transformers + GLiNER + LLMs).
<!-- #endregion -->
