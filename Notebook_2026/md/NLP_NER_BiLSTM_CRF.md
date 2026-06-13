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
# 🕰️ NER avec BiLSTM-CRF — Wiki historique + implémentation
<!-- #endregion -->

<!-- #region -->
**Méthode historique** : état de l'art de la NER entre **2016 et 2019** (Lample et al.), détrônée par les **transformers** à partir de 2019. Toujours pertinente à connaître pour trois raisons :

1. **Pédagogie** — la couche **CRF** (modélisation de séquences sous contraintes) reste utile bien au-delà de la NER : segmentation, alignement, parsing, POS-tagging.
2. **Légèreté** — ~10-50 MB, tourne sur CPU, déployable en edge / embarqué.
3. **Low-resource** — peut rivaliser avec un transformer mal fine-tuné quand on dispose de très peu d'exemples annotés.

> Pour la **NER en production en 2026** (transformers fine-tunés, GLiNER zero-shot, LLMs + function calling), voir le notebook frère **`NLP_NER`**.

Ce notebook **implémente vraiment** une BiLSTM-CRF avec `pytorch-crf`, entraînée sur le vrai dataset **CoNLL-2003**, et compare une baseline BiLSTM+softmax à la version BiLSTM-CRF pour mesurer **l'apport concret de la CRF**.
<!-- #endregion -->

<!-- #region -->
## 1. Setup
<!-- #endregion -->

<!-- #region -->
On importe les libs, on fixe la reproductibilité (seed propagée à NumPy **et** PyTorch) et on définit la palette graphique de la charte. Tout tourne sur **CPU** : un BiLSTM-CRF est suffisamment léger pour ne pas nécessiter de GPU.
<!-- #endregion -->

```python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(4)  # CPU : évite l'over-subscription de threads


def set_seed(seed: int = 42) -> None:
    """Fixe les graines NumPy et PyTorch pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)


SEED = 42
set_seed(SEED)

# Palette CHART (charte projet)
PALETTE = {
    "primary_1": "#00798c", "mauvais": "#d1495b", "moyen": "#edae49",
    "accent": "#66a182", "accent_dark": "#2e4057", "lavender": "#9d83b8",
    "dusty_rose": "#b8848e", "beige": "#c9b78b",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | device = {device}")
```

<!-- #region -->
## 2. Rappel — NER et formats de tags
<!-- #endregion -->

<!-- #region -->
La **NER** (*Named Entity Recognition*) est une tâche de **token classification** : chaque token d'une phrase reçoit un tag indiquant s'il appartient à une entité et de quel type.

Le format **IOB2** (le plus courant) :

- `O` — *outside*, le token n'appartient à aucune entité.
- `B-X` — *begin*, premier token d'une entité de type `X`.
- `I-X` — *inside*, token suivant d'une entité de type `X` déjà commencée.

Exemple :

```
Barack  Obama   visited  Paris   in  1976
B-PER   I-PER   O        B-LOC   O   O
```

Le point crucial pour la suite : **les tags ne sont pas indépendants**. Un `I-PER` ne peut pas suivre un `O` ni un `B-LOC` (illégal en IOB2). Une softmax token-par-token ignore cette contrainte ; c'est exactement ce que la **CRF** va modéliser. Pour les autres formats (BIOES, BILOU) et la NER moderne, voir `NLP_NER`.
<!-- #endregion -->

<!-- #region -->
## 3. Dataset — CoNLL-2003
<!-- #endregion -->

<!-- #region -->
On remplace le dataset historique GMB (perdu / non programmatique) par **CoNLL-2003**, le benchmark NER de référence (articles de presse Reuters, 4 types d'entités : `PER`, `ORG`, `LOC`, `MISC`).

> ⚠️ Le dataset `conll2003` canonique du Hub repose sur un **script de chargement**, désormais refusé par `datasets >= 4`. On passe par le miroir **parquet** `lhoestq/conll2003` (mêmes splits, même contenu).
<!-- #endregion -->

```python
from datasets import load_dataset

raw = load_dataset("lhoestq/conll2003")
print(raw)
print("Exemple brut :", raw["train"][0]["tokens"][:8], "...")
```

<!-- #region -->
Le miroir parquet a perdu les métadonnées `ClassLabel` : `ner_tags` n'est qu'une liste d'entiers. On hardcode donc l'**ordre canonique** des 9 tags CoNLL-2003.
<!-- #endregion -->

```python
NER_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]
id2label = {i: l for i, l in enumerate(NER_LABELS)}
label2id = {l: i for i, l in id2label.items()}
print(f"{len(NER_LABELS)} tags :", NER_LABELS)
```

<!-- #region -->
### 3.1 Conversion IO → BIO (héritage de l'original)
<!-- #endregion -->

<!-- #region -->
CoNLL-2003 est déjà en IOB2. Mais on conserve la logique de conversion **IO → BIO** de l'original (dataset GMB), car c'est un passage obligé dès qu'on annote soi-même un corpus : on dispose souvent d'une simple étiquette de type par token (`PER`, `LOC`, ...) sans la frontière `B`/`I`.

Règle : un token d'entité reçoit `B-` au **début** d'un span (il suit un `O` ou un type différent), et `I-` s'il **continue** le même type.
<!-- #endregion -->

```python
def io_to_bio(io_tags: list[str]) -> list[str]:
    """Convertit une séquence de tags IO (O / PER / LOC ...) en IOB2 (O / B-PER / I-PER ...).

    Args:
        io_tags: tags IO d'une phrase, ex. ['O', 'PER', 'PER', 'O'].
    Returns:
        Les mêmes tags au format IOB2, ex. ['O', 'B-PER', 'I-PER', 'O'].
    """
    bio: list[str] = []
    for idx, tag in enumerate(io_tags):
        if tag == "O":
            bio.append("O")
        elif idx > 0 and io_tags[idx - 1] == tag:
            bio.append(f"I-{tag}")   # continue le même type
        else:
            bio.append(f"B-{tag}")   # début d'un span
    return bio


demo_io = ["O", "PER", "PER", "O", "O", "LOC", "O"]
print("IO  :", demo_io)
print("BIO :", io_to_bio(demo_io))
```

<!-- #region -->
### 3.2 Préparation des splits
<!-- #endregion -->

<!-- #region -->
On transforme chaque split Hugging Face en liste de `(tokens, tags string IOB2)`. Pour garder une exécution **CPU rapide et reproductible**, on sous-échantillonne le train (2500 phrases) et on évalue sur un extrait de validation (700 phrases). Augmenter ces valeurs améliore les scores au prix du temps de calcul.
<!-- #endregion -->

```python
def encode_split(split, max_sentences: int | None = None) -> list[tuple[list[str], list[str]]]:
    """Transforme un split HF en liste de (tokens, tags string IOB2)."""
    out: list[tuple[list[str], list[str]]] = []
    n = len(split) if max_sentences is None else min(max_sentences, len(split))
    for i in range(n):
        ex = split[i]
        tags = [id2label[t] for t in ex["ner_tags"]]
        out.append((ex["tokens"], tags))
    return out


TRAIN = encode_split(raw["train"], max_sentences=2500)
TEST = encode_split(raw["validation"], max_sentences=700)
print(f"Train : {len(TRAIN)} phrases | Test : {len(TEST)} phrases")
print("Exemple :", list(zip(TRAIN[0][0][:6], TRAIN[0][1][:6])))
```

<!-- #region -->
### 3.3 Distribution des tags (graphique préservé de l'original)
<!-- #endregion -->

<!-- #region -->
On recrée le **bar chart des fréquences de tags** de l'original, ici sur les vrais tags CoNLL-2003. On retrouve le **déséquilibre massif** typique de la NER : le tag `O` écrase tout (d'où l'échelle **log**), et les entités sont rares. C'est pourquoi on évalue en **F1 entity-level** (et non en accuracy token, qui serait artificiellement élevée en prédisant `O` partout).
<!-- #endregion -->

```python
from collections import Counter

tag_counts = Counter()
for _, tags in TRAIN:
    tag_counts.update(tags)

labels_plot = NER_LABELS
values_plot = [tag_counts.get(l, 0) for l in labels_plot]

fig, ax = plt.subplots(figsize=(9, 4))
# 'O' = catégorie de fond estompée (beige) ; entités = couleur d'accent (highlight)
colors = [PALETTE["beige"] if l == "O" else PALETTE["primary_1"] for l in labels_plot]
ax.bar(range(len(labels_plot)), values_plot, color=colors)
ax.set_xticks(range(len(labels_plot)))
ax.set_xticklabels(labels_plot, rotation=45, ha="right")
ax.set_yscale("log")  # 'O' écrase tout en échelle linéaire
ax.set_ylabel("Occurrences (échelle log)")
ax.set_title("Distribution des tags IOB2 — CoNLL-2003 (train échantillon)")
fig.tight_layout()
plt.show()
print("Tags les plus fréquents :", tag_counts.most_common(4))
```

<!-- #region -->
## 4. Embeddings de tokens
<!-- #endregion -->

<!-- #region -->
Pour passer un mot à un réseau, il faut un **vecteur**. Trois options historiques :

| Embedding | Idée | Forces | Faiblesses |
|---|---|---|---|
| **Random init** (lookup table) | Embedding appris from scratch | Simple, marche bien avec beaucoup de data | Pas de généralisation aux mots rares |
| **Word2Vec / GloVe / FastText** | Pré-entraîné sur grand corpus | Sémantique pré-acquise | Pas contextuel ("bank" = banque ou rive ?) |
| **Character-level CNN/BiLSTM** | Embedding du mot via ses caractères | Robuste aux fautes / hors-vocabulaire | Plus lourd |

Le pattern gagnant 2016-2018 : **concat(GloVe + char-BiLSTM)** — gère le vocabulaire ouvert ET garde la sémantique. Ici, pour rester autonome, on utilise des embeddings **random init** (appris pendant l'entraînement). Les transformers ont rendu tout cela obsolète en 2019 : les tokenizers sub-word + l'attention bidirectionnelle font mieux que GloVe + char-BiLSTM.
<!-- #endregion -->

<!-- #region -->
## 5. BiLSTM — Bidirectional LSTM
<!-- #endregion -->

<!-- #region -->
Un **LSTM** lit la séquence de gauche à droite et produit un hidden state $\mathbf{h}_t$ par token, résumant "tout ce qu'il a vu jusqu'à $t$". Problème : pour tagger le token $t$, on a besoin du contexte **avant** ET **après** $t$. Un LSTM forward ne voit que le passé.

**BiLSTM** : on lance un LSTM forward et un LSTM backward, et on **concatène** leurs hidden states à chaque position :

$$
\mathbf{h}_t = \big[\,\overrightarrow{\mathbf{h}}_t \; ; \; \overleftarrow{\mathbf{h}}_t\,\big] \in \mathbb{R}^{2d}
$$

Chaque $\mathbf{h}_t$ encode "mot + tout son contexte (gauche et droite)". On projette ensuite chaque $\mathbf{h}_t$ vers les scores de tags (couche linéaire).

**Limite** : avec une softmax indépendante sur chaque $\mathbf{h}_t$, on perd la **dépendance entre tags adjacents** — le modèle peut prédire la séquence illégale `O B-PER O I-PER`. C'est ce trou que la CRF comble (section 8).
<!-- #endregion -->

<!-- #region -->
## 6. Vectorisation PyTorch
<!-- #endregion -->

<!-- #region -->
On construit un **vocabulaire** à partir des tokens du train, avec deux tokens spéciaux : `<PAD>` (indice 0, pour le padding) et `<UNK>` (indice 1, pour les mots hors-vocabulaire au test). On **garde la casse** : la majuscule est un indice fort en NER (un mot capitalisé est souvent une entité).
<!-- #endregion -->

```python
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

counter = Counter()
for tokens, _ in TRAIN:
    counter.update(tokens)

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word, _ in counter.most_common():
    word2idx[word] = len(word2idx)
print(f"Vocab size : {len(word2idx)}")

PAD_TAG_ID = label2id["O"]  # padding des tags = 'O' (neutralisé par le masque)
```

<!-- #region -->
Un `Dataset` encode chaque phrase en indices, et un `collate_fn` **pad** les séquences à la longueur max de la batch tout en produisant un **masque** booléen (`True` = token réel). Ce masque est essentiel : il dit à la CRF (et à la loss) d'**ignorer le padding**.
<!-- #endregion -->

```python
class NERDataset(Dataset):
    """Encode (tokens, tags) en indices entiers."""

    def __init__(self, samples: list[tuple[list[str], list[str]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, tags = self.samples[idx]
        token_ids = torch.tensor(
            [word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens], dtype=torch.long
        )
        tag_ids = torch.tensor([label2id[t] for t in tags], dtype=torch.long)
        return token_ids, tag_ids


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad à la longueur max de la batch + masque booléen (True = token réel)."""
    token_seqs, tag_seqs = zip(*batch)
    max_len = max(t.size(0) for t in token_seqs)
    bs = len(batch)

    padded_tokens = torch.zeros(bs, max_len, dtype=torch.long)             # PAD = 0
    padded_tags = torch.full((bs, max_len), PAD_TAG_ID, dtype=torch.long)
    mask = torch.zeros(bs, max_len, dtype=torch.bool)

    for i, (toks, tags) in enumerate(zip(token_seqs, tag_seqs)):
        n = toks.size(0)
        padded_tokens[i, :n] = toks
        padded_tags[i, :n] = tags
        mask[i, :n] = True
    return padded_tokens, padded_tags, mask


BATCH_SIZE = 32
train_ds, test_ds = NERDataset(TRAIN), NERDataset(TEST)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

xb, yb, m = next(iter(train_loader))
print(f"Batch : tokens={tuple(xb.shape)} tags={tuple(yb.shape)} mask={tuple(m.shape)}")
```

<!-- #region -->
On définit **une fois** les helpers d'évaluation, partagés par la baseline et le modèle CRF. L'évaluation se fait au niveau **entité** (entity-level) avec `seqeval` : une entité n'est correcte que si **tous** ses tokens et sa frontière sont exacts — bien plus strict (et pertinent) que l'accuracy token.
<!-- #endregion -->

```python
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def decode_predictions(
    pred_ids_per_seq: list[list[int]],
    gold_tags: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convertit ids prédits/gold en séquences de strings (positions valides seulement)."""
    true_seqs, pred_seqs = [], []
    for i in range(gold_tags.size(0)):
        n_valid = int(mask[i].sum().item())
        true_seqs.append([id2label[int(gold_tags[i, j])] for j in range(n_valid)])
        pred_seqs.append([id2label[int(pred_ids_per_seq[i][j])] for j in range(n_valid)])
    return true_seqs, pred_seqs
```

<!-- #region -->
## 7. Baseline — BiLSTM + softmax (sans CRF)
<!-- #endregion -->

<!-- #region -->
Avant la CRF, on entraîne la **baseline historique** : un BiLSTM dont chaque hidden state est projeté vers les tags puis classé par une **softmax indépendante par token**. C'est le modèle que la CRF va chercher à dépasser.

Architecture : `Embedding → Dropout → BiLSTM → Linear(2H → tags)`. La loss est une **cross-entropy** restreinte aux positions valides (via le masque).
<!-- #endregion -->

```python
class BiLSTMSoftmax(nn.Module):
    """BiLSTM + classification softmax indépendante par token (baseline pré-CRF)."""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(2 * hidden_dim, num_tags)

    def emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Scores par token : (B, T, num_tags)."""
        emb = self.dropout(self.embedding(input_ids))   # (B, T, E)
        lstm_out, _ = self.bilstm(emb)                   # (B, T, 2H)
        return self.hidden2tag(lstm_out)                 # (B, T, num_tags)
```

<!-- #region -->
Boucle d'entraînement et évaluation. La cross-entropy est calculée **uniquement sur les vrais tokens** (`emis[mask]`), de sorte que le padding n'influence ni le gradient ni le score.
<!-- #endregion -->

```python
import torch.optim as optim


def train_softmax(model: BiLSTMSoftmax, epochs: int = 6) -> list[float]:
    """Entraîne la baseline softmax ; CrossEntropy restreinte aux positions valides."""
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    model.to(device)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for tokens, tags, mask in train_loader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
            opt.zero_grad()
            emis = model.emissions(tokens)             # (B, T, C)
            loss = ce(emis[mask], tags[mask])          # seulement les vrais tokens
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * tokens.size(0)
            n += tokens.size(0)
        history.append(tot / n)
        if (epoch + 1) % 2 == 0:
            print(f"[softmax] epoch {epoch+1:2d}  loss = {history[-1]:.4f}")
    return history


@torch.no_grad()
def evaluate_softmax(model: BiLSTMSoftmax) -> dict:
    """Évaluation entity-level (seqeval) de la baseline softmax."""
    model.eval()
    all_true, all_pred = [], []
    for tokens, tags, mask in test_loader:
        tokens, mask = tokens.to(device), mask.to(device)
        preds = model.emissions(tokens).argmax(-1)  # (B, T)
        pred_lists = [preds[i].cpu().tolist() for i in range(tokens.size(0))]
        t, p = decode_predictions(pred_lists, tags, mask)
        all_true += t
        all_pred += p
    return {
        "f1": f1_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred),
        "recall": recall_score(all_true, all_pred),
    }
```

<!-- #region -->
On instancie, on entraîne et on évalue la baseline.
<!-- #endregion -->

```python
set_seed(SEED)
softmax_model = BiLSTMSoftmax(len(word2idx), len(NER_LABELS))
n_params_sm = sum(p.numel() for p in softmax_model.parameters())
print(f"BiLSTM-softmax : {n_params_sm:,} paramètres")

softmax_hist = train_softmax(softmax_model, epochs=6)
softmax_eval = evaluate_softmax(softmax_model)
print(f"[softmax] F1 entity-level = {softmax_eval['f1']:.4f}")
```

<!-- #region -->
Courbe de la loss d'entraînement (cross-entropy moyenne par epoch).
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(softmax_hist) + 1), softmax_hist, marker="o", color=PALETTE["mauvais"])
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (CE moyenne)")
ax.set_title("BiLSTM + softmax — courbe d'entraînement")
fig.tight_layout()
plt.show()
```

<!-- #region -->
## 8. CRF — Conditional Random Field
<!-- #endregion -->

<!-- #region -->
La **CRF linear-chain** modélise la probabilité d'une **séquence entière** de tags étant donné les scores d'entrée, au lieu de décider tag par tag.

### Formulation

Pour une séquence de tags $y = (y_1, \dots, y_T)$ et des scores d'émission $s = (s_1, \dots, s_T)$ (sortie du BiLSTM, $s_t \in \mathbb{R}^{|\text{tags}|}$), la CRF définit :

$$
P(y \mid s) = \frac{\exp\!\big(\sum_{t=1}^{T} s_{t, y_t} + \sum_{t=1}^{T-1} A_{y_t, y_{t+1}}\big)}{Z(s)}
$$

- **$s_{t, y_t}$** : score d'**émission** — "à quel point la position $t$ aime être taggée $y_t$" (vient du BiLSTM).
- **$A_{i, j}$** : score de **transition** — "à quel point on aime passer du tag $i$ au tag $j$". $A$ est une matrice apprise $|\text{tags}| \times |\text{tags}|$.
- **$Z(s)$** : la **fonction de partition**, somme sur **toutes** les séquences de tags possibles. Calculable en $O(T \cdot |\text{tags}|^2)$ via l'**algorithme forward** (programmation dynamique), au lieu de $|\text{tags}|^T$ naïvement.

### Pourquoi c'est mieux qu'une softmax

- La softmax token-par-token est $P(y_t \mid s_t)$, **indépendant** pour chaque $t$. La CRF est $P(y_1, \dots, y_T \mid s)$, **joint**.
- Via la matrice $A$, la CRF **apprend à interdire** (score $\approx -\infty$) les transitions illégales : `O → I-X` jamais, `I-PER` jamais après `B-LOC`.
- Gain F1 typique : **+1 à +2 points** sur CoNLL-2003 par rapport à une softmax sur le même BiLSTM.

### Entraînement et inférence

- **Entraînement** : maximum de vraisemblance, on minimise $-\log P(y_{\text{gold}} \mid s)$. Le gradient est analytique (forward/backward).
- **Inférence** : on cherche $y^* = \arg\max_y P(y \mid s)$, calculé en $O(T \cdot |\text{tags}|^2)$ via l'**algorithme de Viterbi**.
<!-- #endregion -->

<!-- #region -->
## 9. BiLSTM-CRF — l'assemblage
<!-- #endregion -->

<!-- #region -->
On empile le tout : le BiLSTM produit les scores d'émission, la CRF les transforme en distribution sur les séquences.

```
tokens ─→ Embedding ─→ BiLSTM ─→ Linear ─→ scores d'émission s_t
                                                    │
                                                    ▼
                                         CRF (matrice A apprise)
                                                    │
                                                    ▼
                                    argmax P(y|s) via Viterbi
```

On utilise la lib **`pytorch-crf`** (`pip install pytorch-crf`, import `torchcrf`) qui implémente la CRF linear-chain avec forward/backward optimisés et décodage Viterbi. `crf(emissions, tags, mask)` renvoie le **log-likelihood** (positif) → on le **nie** pour obtenir la loss ; `crf.decode(emissions, mask)` renvoie la meilleure séquence par Viterbi.
<!-- #endregion -->

```python
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    """BiLSTM + CRF linear-chain pour token classification."""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(2 * hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Scores d'émission par token (sortie BiLSTM projetée)."""
        emb = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.bilstm(emb)
        return self.hidden2tag(lstm_out)

    def loss(self, input_ids: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood CRF (à minimiser)."""
        emis = self.emissions(input_ids)
        return -self.crf(emis, tags, mask=mask, reduction="mean")

    def predict(self, input_ids: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Décodage Viterbi : meilleure séquence de tags par phrase."""
        emis = self.emissions(input_ids)
        return self.crf.decode(emis, mask=mask)
```

<!-- #region -->
Boucle d'entraînement et évaluation, en tout point analogues à la baseline — seule la loss change (NLL de la CRF au lieu de la cross-entropy).
<!-- #endregion -->

```python
def train_crf(model: BiLSTMCRF, epochs: int = 6) -> list[float]:
    """Entraîne le BiLSTM-CRF (loss = -log-likelihood CRF)."""
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.to(device)
    history: list[float] = []
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for tokens, tags, mask in train_loader:
            tokens, tags, mask = tokens.to(device), tags.to(device), mask.to(device)
            opt.zero_grad()
            loss = model.loss(tokens, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * tokens.size(0)
            n += tokens.size(0)
        history.append(tot / n)
        if (epoch + 1) % 2 == 0:
            print(f"[crf]     epoch {epoch+1:2d}  loss = {history[-1]:.4f}")
    return history


@torch.no_grad()
def evaluate_crf(model: BiLSTMCRF) -> dict:
    """Évaluation entity-level (seqeval) du BiLSTM-CRF."""
    model.eval()
    all_true, all_pred = [], []
    for tokens, tags, mask in test_loader:
        tokens, mask = tokens.to(device), mask.to(device)
        preds = model.predict(tokens, mask)  # liste de listes (longueur variable)
        t, p = decode_predictions(preds, tags, mask)
        all_true += t
        all_pred += p
    return {
        "f1": f1_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred),
        "recall": recall_score(all_true, all_pred),
        "report": classification_report(all_true, all_pred, digits=3),
    }
```

<!-- #region -->
On instancie, on entraîne, on évalue, et on affiche le rapport détaillé par type d'entité.
<!-- #endregion -->

```python
set_seed(SEED)
crf_model = BiLSTMCRF(len(word2idx), len(NER_LABELS))
n_params_crf = sum(p.numel() for p in crf_model.parameters())
print(f"BiLSTM-CRF : {n_params_crf:,} paramètres")

crf_hist = train_crf(crf_model, epochs=6)
crf_eval = evaluate_crf(crf_model)
print(f"[crf] F1 entity-level = {crf_eval['f1']:.4f}\n")
print(crf_eval["report"])
```

<!-- #region -->
Courbe de la loss CRF (NLL moyenne par epoch — échelle différente de la cross-entropy, ce qui est normal : ce sont deux losses de natures distinctes).
<!-- #endregion -->

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(crf_hist) + 1), crf_hist, marker="o", color=PALETTE["primary_1"])
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (NLL CRF moyenne)")
ax.set_title("BiLSTM-CRF — courbe d'entraînement")
fig.tight_layout()
plt.show()
```

<!-- #region -->
## 10. Comparaison — softmax vs CRF
<!-- #endregion -->

<!-- #region -->
Le payoff pédagogique : à **architecture BiLSTM identique**, la couche CRF apporte un gain de F1 entity-level. L'écart est modeste sur ce mini-setup (embeddings random, 2500 phrases, 6 epochs) mais va **dans le sens attendu** — et s'accentue avec plus de données et d'epochs.
<!-- #endregion -->

```python
comparison = {
    "BiLSTM + softmax": softmax_eval["f1"],
    "BiLSTM-CRF": crf_eval["f1"],
}
print("Comparaison F1 :", {k: round(v, 4) for k, v in comparison.items()})

fig, ax = plt.subplots(figsize=(6, 4))
names = list(comparison.keys())
vals = list(comparison.values())
bars = ax.bar(names, vals, color=[PALETTE["mauvais"], PALETTE["accent"]])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("F1 entity-level (seqeval)")
ax.set_ylim(0, max(vals) * 1.2 + 0.05)
ax.set_title("Apport de la couche CRF")
fig.tight_layout()
plt.show()
```

<!-- #region -->
## 11. Inspection de la matrice de transition apprise
<!-- #endregion -->

<!-- #region -->
La matrice $A$ apprise par la CRF est **directement interprétable** : c'est l'un des charmes de la méthode. On l'affiche en **heatmap** (cmap divergente centrée sur 0). On vérifie ensuite numériquement qu'une transition **illégale** en IOB2 (`O → I-PER`) a bien un score **inférieur** à une transition **légale** (`B-PER → I-PER`) : la CRF a appris la grammaire des tags sans qu'on la lui code en dur.
<!-- #endregion -->

```python
transitions = crf_model.crf.transitions.detach().cpu().numpy()  # (num_tags, num_tags)

bound = np.abs(transitions).max()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(transitions, cmap="RdBu_r", vmin=-bound, vmax=bound)
ax.set_xticks(range(len(NER_LABELS))); ax.set_xticklabels(NER_LABELS, rotation=45, ha="right")
ax.set_yticks(range(len(NER_LABELS))); ax.set_yticklabels(NER_LABELS)
ax.set_xlabel("vers (tag t+1)"); ax.set_ylabel("depuis (tag t)")
ax.set_title("Matrice de transition apprise par la CRF (logits)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

i, j = label2id["O"], label2id["I-PER"]          # illégale en IOB2
li, lj = label2id["B-PER"], label2id["I-PER"]    # légale
print(f"Transition illégale O→I-PER     = {transitions[i, j]:.2f}")
print(f"Transition légale   B-PER→I-PER = {transitions[li, lj]:.2f}")
```

<!-- #region -->
## 12. Prédiction sur de nouvelles phrases
<!-- #endregion -->

<!-- #region -->
On encapsule l'inférence : tokenisation simple (split sur l'espace), encodage, Viterbi. Avec des embeddings random et un mini-train, le modèle attrape bien les entités fréquentes/capitalisées et en rate certaines (rappel modéré) — comportement réaliste pour cette méthode historique sur peu de données.
<!-- #endregion -->

```python
def predict_tags(sentence: str) -> list[tuple[str, str]]:
    """Tokenize (split espace) + prédit les tags IOB2 via le BiLSTM-CRF."""
    tokens = sentence.split()
    ids = torch.tensor(
        [[word2idx.get(t, word2idx[UNK_TOKEN]) for t in tokens]], dtype=torch.long
    ).to(device)
    mask = torch.ones(1, len(tokens), dtype=torch.bool).to(device)
    crf_model.eval()
    with torch.no_grad():
        preds = crf_model.predict(ids, mask)[0]
    return list(zip(tokens, [id2label[i] for i in preds]))


for sentence in [
    "Germany rejected the European Union proposal in Brussels",
    "Barack Obama visited Paris last week",
    "Apple and Microsoft compete in the United States",
]:
    print(predict_tags(sentence))
```

<!-- #region -->
## 13. Bilan et alternatives 2026
<!-- #endregion -->

<!-- #region -->
### 13.1 Forces (historiques)
<!-- #endregion -->

<!-- #region -->
- **Léger** : ~10-50 MB sur disque, faible empreinte RAM, tourne confortablement sur CPU.
- **Frugal en données** : tourne sur quelques milliers d'exemples annotés, sans pré-entraînement massif.
- **Interprétable** : la matrice de transition est observable et lisible (cf. section 11) — rare en deep learning.
<!-- #endregion -->

<!-- #region -->
### 13.2 Limites (qui ont causé sa chute)
<!-- #endregion -->

<!-- #region -->
- **Embeddings non contextuels** (sauf à concaténer ELMo, lui aussi déclassé) : un même mot a toujours le même vecteur d'entrée.
- **Capacité limitée** : un BiLSTM oublie le contexte au-delà de ~30 tokens en pratique.
- **Pas d'auto-attention** → mauvaise capture des dépendances longue distance.
<!-- #endregion -->

<!-- #region -->
### 13.3 Que faire en 2026 ?
<!-- #endregion -->

<!-- #region -->
Par ordre de qualité décroissante (F1 indicatifs sur CoNLL-2003 *full*) :

| Approche | F1 typique | Coût compute | Notes |
|---|---|---|---|
| **DeBERTa-v3 large fine-tuned** | 93.5+ | 700M params | SOTA, dur à battre |
| **ModernBERT fine-tuned** | 93+ | 150M | Meilleur rapport perf/coût en 2026 |
| **DistilBERT fine-tuned** | 91-92 | 66M | Baseline transformer raisonnable |
| **GLiNER fine-tuned** | 90+ | ~300M | Si schéma d'entités fluide / zero-shot |
| **BiLSTM-CRF (ce notebook)** | 88-89 | ~10 MB | **Edge / low-resource uniquement** |
| **CRF seul (sklearn-crfsuite)** | 80-85 | ~1 MB | Features manuelles, quand vraiment rien d'autre |

> Le F1 de **ce** notebook est volontairement plus bas que les 88-89 "full" : on a sous-échantillonné (2500 phrases), pris des embeddings random (pas de GloVe) et 6 epochs, pour rester rapide sur CPU. Pour la NER de production, aller sur **`NLP_NER`**.
<!-- #endregion -->

<!-- #region -->
## 14. Sources
<!-- #endregion -->

<!-- #region -->
- **Paper fondateur** : Lample et al. (2016) — *Neural Architectures for Named Entity Recognition* — [arXiv:1603.01360](https://arxiv.org/abs/1603.01360)
- **Tutoriel CRF** : Sutton & McCallum — *An Introduction to Conditional Random Fields* — [arXiv:1011.4088](https://arxiv.org/abs/1011.4088)
- **pytorch-crf** : [github.com/kmkurn/pytorch-crf](https://github.com/kmkurn/pytorch-crf)
- **CoNLL-2003** : Tjong Kim Sang & De Meulder (2003) — *Introduction to the CoNLL-2003 Shared Task*.
- **Pour la NER moderne** : notebook `NLP_NER` (transformers + GLiNER + LLMs).
<!-- #endregion -->
