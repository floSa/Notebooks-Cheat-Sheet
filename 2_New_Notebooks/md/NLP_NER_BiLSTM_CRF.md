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
# 🕰️ NER avec BiLSTM-CRF (Wiki + implémentation)
<!-- #endregion -->

<!-- #region -->
**Méthode historique** (état de l'art 2016-2019), dépassée par les transformers depuis 2019. **Toujours pertinente** pour :

1. **Pédagogie** — comprendre la CRF reste utile (parsing, segmentation, alignement).
2. **Légèreté** — ~10-50 MB, tourne CPU, déployable edge.
3. **Domaines low-resource** — peut rivaliser avec un transformer mal fine-tuné quand on a < 1000 exemples annotés.

Pour la NER en production en 2026, voir **`NLP_NER`** (transformers + GLiNER).

Ce notebook **implémente vraiment** une BiLSTM-CRF avec `pytorch-crf` sur un mini-dataset NER synthétique (proxy CoNLL-2003).
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
## 5. BiLSTM-CRF — l'assemblage
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
## 6. Implémentation PyTorch avec pytorch-crf
<!-- #endregion -->

<!-- #region -->
On utilise la lib **`pytorch-crf`** (`pip install pytorch-crf`) qui implémente la CRF linear-chain avec forward/backward optimisés et décodage Viterbi.

L'archi globale fait moins de 50 lignes. On va l'entraîner sur un mini-dataset NER synthétique (3 entités : PER, LOC, ORG).
<!-- #endregion -->

<!-- #region -->
### 6.1 Mini dataset NER synthétique
<!-- #endregion -->

```python
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Mini vocab d'entités
NAMES = ["Alice", "Bob", "Marie", "Jean", "Pierre", "Sophie", "Thomas", "Claire"]
LOCS = ["Paris", "Lyon", "Berlin", "Tokyo", "Madrid", "Roma", "Londres", "Geneve"]
ORGS = ["Apple", "Google", "Microsoft", "Sony", "Meta", "OpenAI", "Anthropic", "Tesla"]
OTHER = ["lives", "works", "at", "in", "for", "the", "and", "or", "with", "near"]

# Schémas de phrases template
TEMPLATES = [
    "{PER} lives in {LOC}",
    "{PER} works at {ORG}",
    "{PER} works for {ORG} in {LOC}",
    "{ORG} is based in {LOC}",
    "{PER} and {PER} work at {ORG}",
]


def make_sentence() -> tuple[list[str], list[str]]:
    """Génère une phrase + tags IOB2."""
    tmpl = random.choice(TEMPLATES)
    tokens = []
    tags = []
    for word in tmpl.split():
        if word == "{PER}":
            n = random.choice(NAMES)
            tokens.append(n); tags.append("B-PER")
        elif word == "{LOC}":
            n = random.choice(LOCS)
            tokens.append(n); tags.append("B-LOC")
        elif word == "{ORG}":
            n = random.choice(ORGS)
            tokens.append(n); tags.append("B-ORG")
        else:
            tokens.append(word); tags.append("O")
    return tokens, tags


# Génération du dataset
TRAIN = [make_sentence() for _ in range(500)]
TEST = [make_sentence() for _ in range(100)]
print(f"Train : {len(TRAIN)} phrases | Test : {len(TEST)}")
print(f"Exemple : {TRAIN[0]}")
```

<!-- #region -->
### 6.2 Vocabulaire + mapping label
<!-- #endregion -->

```python
from collections import Counter

# Vocab : tous les tokens du train, + <PAD> et <UNK>
counter = Counter()
for tokens, _ in TRAIN:
    counter.update(tokens)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, _ in counter.most_common():
    word2idx[word] = len(word2idx)
print(f"Vocab size : {len(word2idx)}")

LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
label2idx = {l: i for i, l in enumerate(LABELS)}
idx2label = {i: l for l, i in label2idx.items()}
print(f"Labels : {LABELS}")
```

<!-- #region -->
### 6.3 Dataset PyTorch avec padding et masque
<!-- #endregion -->

```python
class NERDataset(Dataset):
    """Dataset NER : encode tokens et tags en indices entiers."""

    def __init__(self, samples: list[tuple[list[str], list[str]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, tags = self.samples[idx]
        token_ids = torch.tensor([word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens], dtype=torch.long)
        tag_ids = torch.tensor([label2idx[t] for t in tags], dtype=torch.long)
        return token_ids, tag_ids


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad les séquences à la longueur max de la batch + masque pour la CRF."""
    token_seqs, tag_seqs = zip(*batch)
    max_len = max(t.size(0) for t in token_seqs)
    batch_size = len(batch)

    padded_tokens = torch.zeros(batch_size, max_len, dtype=torch.long)  # PAD = 0
    padded_tags = torch.zeros(batch_size, max_len, dtype=torch.long)     # tag 0 (O) en padding
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (toks, tags) in enumerate(zip(token_seqs, tag_seqs)):
        n = toks.size(0)
        padded_tokens[i, :n] = toks
        padded_tags[i, :n] = tags
        mask[i, :n] = True

    return padded_tokens, padded_tags, mask


train_ds = NERDataset(TRAIN)
test_ds = NERDataset(TEST)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_batch)

# Vérification
xb, yb, m = next(iter(train_loader))
print(f"Batch shapes : tokens={xb.shape} tags={yb.shape} mask={m.shape}")
```

<!-- #region -->
### 6.4 Modèle BiLSTM-CRF
<!-- #endregion -->

```python
class BiLSTMCRF(nn.Module):
    """BiLSTM + CRF pour token classification."""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim,
            bidirectional=True, batch_first=True,
        )
        self.hidden2tag = nn.Linear(2 * hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def _emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward jusqu'aux scores d'émission par token."""
        emb = self.dropout(self.embedding(input_ids))   # (B, T, E)
        lstm_out, _ = self.bilstm(emb)                   # (B, T, 2H)
        return self.hidden2tag(lstm_out)                 # (B, T, num_tags)

    def loss(self, input_ids: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood (à minimiser)."""
        emissions = self._emissions(input_ids)
        # CRF.forward renvoie le log-likelihood (positif) → negate pour loss
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def predict(self, input_ids: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decode : meilleure séquence de tags par batch."""
        emissions = self._emissions(input_ids)
        return self.crf.decode(emissions, mask=mask)


model = BiLSTMCRF(vocab_size=len(word2idx), num_tags=len(LABELS), embed_dim=32, hidden_dim=32)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modèle BiLSTM-CRF : {n_params:,} paramètres")
```

<!-- #region -->
### 6.5 Boucle d'entraînement
<!-- #endregion -->

```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

losses_per_epoch = []
for epoch in range(15):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for tokens, tags, mask in train_loader:
        tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = model.loss(tokens, tags, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * tokens.size(0)
        n_samples += tokens.size(0)
    avg = total_loss / n_samples
    losses_per_epoch.append(avg)
    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1:2d}  loss = {avg:.4f}")
```

<!-- #region -->
### 6.6 Évaluation entity-level avec seqeval
<!-- #endregion -->

```python
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

model.eval()
all_true: list[list[str]] = []
all_pred: list[list[str]] = []

with torch.no_grad():
    for tokens, tags, mask in test_loader:
        tokens, mask = tokens.to(device), mask.to(device)
        preds = model.predict(tokens, mask)  # list of list (variable length)

        for i in range(tokens.size(0)):
            true_seq = []
            pred_seq = []
            n_valid = int(mask[i].sum().item())
            for j in range(n_valid):
                true_seq.append(idx2label[int(tags[i, j].item())])
                pred_seq.append(idx2label[int(preds[i][j])])
            all_true.append(true_seq)
            all_pred.append(pred_seq)

f1 = f1_score(all_true, all_pred)
p = precision_score(all_true, all_pred)
r = recall_score(all_true, all_pred)
print(f"Entity-level F1 (seqeval)  : {f1:.4f}")
print(f"Precision                   : {p:.4f}")
print(f"Recall                      : {r:.4f}")
print()
print(classification_report(all_true, all_pred, digits=3))
```

<!-- #region -->
### 6.7 Inspection des transitions apprises par la CRF
<!-- #endregion -->

```python
# La matrice de transition CRF apprise — révèle les "règles" implicites
transitions = model.crf.transitions.detach().cpu().numpy()

import pandas as pd

trans_df = pd.DataFrame(
    transitions,
    index=[f"from_{l}" for l in LABELS],
    columns=[f"to_{l}" for l in LABELS],
)
print("Matrice de transition CRF (logits) :")
print(trans_df.round(2))
print()
print("Note : transitions interdites en IOB2 (O→I-X, B-X→I-Y) ont des scores très négatifs.")
```

<!-- #region -->
### 6.8 Prédiction sur de nouvelles phrases
<!-- #endregion -->

```python
def predict_tags(sentence: str) -> list[tuple[str, str]]:
    """Tokenize + prédit les tags pour une phrase."""
    tokens = sentence.split()
    ids = torch.tensor(
        [[word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]],
        dtype=torch.long,
    ).to(device)
    mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)

    model.eval()
    with torch.no_grad():
        preds = model.predict(ids, mask)[0]

    return list(zip(tokens, [idx2label[i] for i in preds]))


# Test sur des phrases jamais vues
for sentence in [
    "Alice works at OpenAI",
    "Bob lives in Berlin",
    "Sophie and Thomas work at Tesla in Madrid",
]:
    print(predict_tags(sentence))
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
- Architecture **interprétable** (matrice de transition observable, cf section 6.7).
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
