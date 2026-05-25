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
# 🏷️ Named Entity Recognition (NER)
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur la **NER** (Named Entity Recognition) — extraction d'entités nommées (personnes, lieux, organisations, dates, montants, ...) depuis du texte brut.

Couverture :

1. Formats de tags (**IOB / IOB2 / BIOES / BILOU**) — la base que tout NLPiste doit connaître.
2. **Approche moderne** : Hugging Face Transformers + token classification (fine-tuning DistilBERT sur CoNLL-2003).
3. **GLiNER (2024-2026)** : **zero-shot NER** universel — extraire n'importe quelles entités sans entraîner un modèle dédié.
4. **NER via LLMs** (Llama, GPT-4) avec function calling pour de la JSON-extraction structurée.
5. **Évaluation rigoureuse** avec `seqeval` (entity-level F1).

> Pour la version historique **BiLSTM-CRF**, voir le notebook dédié `NLP_NER_BiLSTM_CRF` (wiki technique des méthodes pré-transformers).
<!-- #endregion -->

<!-- #region -->
## 1. Le problème NER
<!-- #endregion -->

<!-- #region -->
Étant donné un texte, identifier les **spans** (séquences contiguës de tokens) qui correspondent à des **entités** d'un certain **type**.

Exemple :

```
Apple Inc. was founded in Cupertino by Steve Jobs in April 1976.
[ORG       ]              [LOC     ]    [PER       ]   [DATE     ]
```

Utilisations :
- Extraction d'infos depuis des **documents juridiques, médicaux, financiers**.
- Anonymisation (remplacer noms/dates/lieux par des placeholders).
- Construction de **knowledge graphs**.
- **Search** : faceted search par entité.
- Pré-processing pour **RAG** (filtres / metadata).

C'est une tâche de **token classification** : chaque token reçoit un label (O = outside, ou B-/I-X pour entity).
<!-- #endregion -->

<!-- #region -->
## 2. Formats de tags (IOB, IOB2, BIOES, BILOU)
<!-- #endregion -->

<!-- #region -->
Pour encoder des spans en labels par token, plusieurs conventions :

| Format | Tokens d'une entité multi-mots | Token unique |
|---|---|---|
| **IOB** | `I-X I-X I-X` (ou `B-X` si l'entité suit immédiatement une autre du même type) | `I-X` |
| **IOB2** | `B-X I-X I-X` (B- toujours obligatoire au début) | `B-X` |
| **BIOES** / **BIOLU** | `B-X I-X E-X` (B=Beginning, I=Inside, E=End, S/U=Single) | `S-X` ou `U-X` |
| **BILOU** | `B-X I-X L-X` (L=Last) | `U-X` (Unit) |

**Recommandation 2026** : utiliser **IOB2** (le plus courant, attendu par défaut par `datasets`, `seqeval`, HF Trainer). **BIOES/BILOU** apporte un gain marginal (+0.5-1 % F1) au prix de plus de classes.

Exemple sur la phrase `"Steve Jobs founded Apple"` :

| Token | IOB2 | BIOES |
|---|---|---|
| Steve | B-PER | B-PER |
| Jobs | I-PER | E-PER |
| founded | O | O |
| Apple | B-ORG | S-ORG |
<!-- #endregion -->

```python
def spans_to_iob2(tokens: list[str], spans: list[tuple[int, int, str]]) -> list[str]:
    """Convertit une liste de spans (start_token_idx, end_token_idx_exclusif, label) en tags IOB2.

    Args:
        tokens: liste des tokens du texte.
        spans: liste de tuples (start, end, label) avec end exclusif.
    Returns:
        liste de tags IOB2 alignée sur tokens.
    """
    tags = ["O"] * len(tokens)
    for start, end, label in spans:
        if start >= len(tokens):
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, min(end, len(tokens))):
            tags[i] = f"I-{label}"
    return tags


tokens = "Steve Jobs founded Apple in California".split()
spans = [(0, 2, "PER"), (3, 4, "ORG"), (5, 6, "LOC")]
print(list(zip(tokens, spans_to_iob2(tokens, spans))))
```

<!-- #region -->
## 3. Dataset CoNLL-2003 via Hugging Face datasets
<!-- #endregion -->

<!-- #region -->
**CoNLL-2003** (Reuters news, annoté en PER/LOC/ORG/MISC) est le benchmark de référence depuis 20 ans. Disponible directement via la lib `datasets`.

> En cas de souci d'accès au dataset (gating, network), une alternative est `tner/conll2003` ou `eriktks/conll2003` sur le Hub.
<!-- #endregion -->

```python
from datasets import load_dataset

# Note: peut nécessiter `pip install datasets` et un accès réseau au Hub
# ds = load_dataset("eriktks/conll2003", trust_remote_code=True)
# print(ds)
# print(ds["train"].features["ner_tags"].feature.names)
# → ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

# Pour le smoke test on construit un mini dataset manuel
from datasets import Dataset, Sequence, ClassLabel, Value, Features

label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(ClassLabel(names=label_names)),
})

mini_data = {
    "tokens": [
        ["Steve", "Jobs", "founded", "Apple", "in", "California"],
        ["Marie", "lives", "in", "Paris"],
    ],
    "ner_tags": [
        [1, 2, 0, 3, 0, 5],   # B-PER I-PER O B-ORG O B-LOC
        [1, 0, 0, 5],
    ],
}
mini_ds = Dataset.from_dict(mini_data, features=features)
print(mini_ds)
print("Premier exemple :", mini_ds[0])
```

<!-- #region -->
## 4. Approche moderne : Transformers + Token Classification
<!-- #endregion -->

<!-- #region -->
**Idée** : utiliser un encoder Transformer pré-entraîné (DistilBERT, BERT, ModernBERT) + une tête linéaire au-dessus pour prédire le tag de chaque token.

**Subtilité critique** : le tokenizer du Transformer **découpe en sous-mots**. Un mot peut devenir 1-N sub-tokens. Il faut **aligner les labels** : on attribue le label au premier sub-token, et `-100` (ignored par la loss) aux autres.

C'est ce que fait la fonction `tokenize_and_align_labels` ci-dessous, qui est devenue le standard de fait pour la NER avec HF.
<!-- #endregion -->

```python
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-cased"  # cased pour la NER (case-sensitive)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(examples: dict, label_all_tokens: bool = False) -> dict:
    """Tokenize et aligne les labels NER sur les sub-tokens.

    Args:
        examples: batch avec 'tokens' (list[str]) et 'ner_tags' (list[int]).
        label_all_tokens: si True, propage le label de mot à TOUS ses sub-tokens.
            Si False (recommandé), met -100 aux sub-tokens internes (ignored par la loss).
    Returns:
        Dict tokenized avec 'labels' aligné.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,    # crucial — indique que l'input est déjà pré-tokenisé en mots
        max_length=128,
    )
    new_labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)  # liste de l'indice de mot pour chaque sub-token
        aligned = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:                # tokens spéciaux [CLS], [SEP], [PAD]
                aligned.append(-100)
            elif word_id != prev_word:         # 1er sub-token d'un mot → label du mot
                aligned.append(label_ids[word_id])
            else:                              # sub-tokens suivants
                aligned.append(label_ids[word_id] if label_all_tokens else -100)
            prev_word = word_id
        new_labels.append(aligned)
    tokenized["labels"] = new_labels
    return tokenized


# Test sur le mini dataset
tokenized_ds = mini_ds.map(tokenize_and_align_labels, batched=True, remove_columns=mini_ds.column_names)
print(tokenized_ds)
print("Premier exemple tokenizé :", tokenized_ds[0])
```

<!-- #region -->
### 4.1 Modèle, training, évaluation
<!-- #endregion -->

```python
from transformers import (
    AutoModelForTokenClassification, TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np

id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_names), id2label=id2label, label2id=label2id,
)

# Collator pour padding aligné (input_ids + labels)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_metrics_ner(eval_pred):
    """Calcule l'entity-level F1 via seqeval (gold standard pour la NER)."""
    from seqeval.metrics import classification_report, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Reconvert ids → tags, en ignorant les -100
    true_tags, pred_tags = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        ts, ps = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                ts.append(id2label[int(l)])
                ps.append(id2label[int(p)])
        true_tags.append(ts)
        pred_tags.append(ps)

    return {"f1": f1_score(true_tags, pred_tags)}


training_args = TrainingArguments(
    output_dir="./_artifacts/ner-distilbert",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    logging_steps=100,
)

# ⚠️ trainer.train() à décommenter avec un vrai dataset (CoNLL-2003)
# trainer = Trainer(
#     model=model, args=training_args,
#     train_dataset=tokenized_train, eval_dataset=tokenized_val,
#     tokenizer=tokenizer, data_collator=data_collator,
#     compute_metrics=compute_metrics_ner,
# )
# trainer.train()
```

<!-- #region -->
### 4.2 Inference avec `pipeline("ner")`
<!-- #endregion -->

<!-- #region -->
Pour utiliser un modèle NER entraîné, le wrapper `pipeline` regroupe les sub-tokens en spans complets via `aggregation_strategy`.

| `aggregation_strategy` | Comportement |
|---|---|
| `"none"` | Renvoie chaque sub-token avec son label |
| `"simple"` | Groupe les tokens consécutifs avec même label de base |
| `"first"` | Prend le label du 1er sub-token comme label du mot |
| `"average"` | Moyenne les logits sur les sub-tokens |
| `"max"` | Max des logits par sub-token |
<!-- #endregion -->

```python
from transformers import pipeline

# Modèle pré-entraîné sur CoNLL-2003 (Daniel Bourke / Hugging Face)
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",        # 110M, déjà fine-tuné PER/LOC/ORG/MISC
    aggregation_strategy="simple",
)

text = "Apple was founded by Steve Jobs in Cupertino, California in April 1976."
for ent in ner(text):
    print(f"  {ent['entity_group']:6s}  {ent['word']:25s}  score={ent['score']:.3f}")
```

<!-- #region -->
## 5. GLiNER — Zero-shot NER (2024-2026)
<!-- #endregion -->

<!-- #region -->
**Le game-changer de 2024-2025** : **GLiNER** (Generalist Lightweight NER), un modèle qui prend une **liste arbitraire de types d'entités en input** et les extrait — sans fine-tuning, sans dataset annoté.

### Idée

Encode la phrase ET la liste de labels (`["person", "company", "drug"]`) dans le **même espace latent**. Pour chaque span candidat, calcule sa similarité avec chaque embedding de label. Garde les spans avec un score au-dessus d'un seuil.

### Pourquoi c'est puissant

- **Pas de réannotation** pour un nouveau domaine — décrire les labels en langage naturel suffit.
- **Léger** : 50-300 MB, tourne CPU.
- **Plus rapide qu'un LLM** : extraction parallèle des spans (vs génération séquentielle).
- **Outperforms ChatGPT** sur la plupart des benchmarks NER zero-shot publiés.

### Quand utiliser GLiNER vs un fine-tuning DistilBERT

| Situation | Recommandation |
|---|---|
| Schéma stable + dataset annoté (>1000 ex) | Fine-tuning Transformer (meilleur F1) |
| Schéma exploratoire / labels changent souvent | **GLiNER** (zero-shot) |
| Schéma stable, mais peu de data (<100 ex) | **GLiNER fine-tuné** sur ces 100 ex |
| Besoin d'extraire des relations / structures complexes | LLM avec function calling |
<!-- #endregion -->

```python
# Exemple GLiNER — décommenter, télécharge ~200 MB la première fois
"""
from gliner import GLiNER

gli = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = "Steve Jobs founded Apple in Cupertino in 1976. Tim Cook now leads the company."
labels = ["person", "company", "location", "year"]

entities = gli.predict_entities(text, labels)
for e in entities:
    print(f"  {e['label']:10s}  {e['text']:20s}  score={e['score']:.3f}")
# →   person      Steve Jobs           score=0.96
#     company     Apple                score=0.94
#     location    Cupertino            score=0.92
#     year        1976                 score=0.88
#     person      Tim Cook             score=0.95
"""
```

<!-- #region -->
## 6. NER via LLMs (function calling / JSON mode)
<!-- #endregion -->

<!-- #region -->
Pour des **schémas riches** (objets imbriqués, attributs, relations), on peut demander à un LLM d'extraire le tout en **JSON** :

```python
# Pseudo-code via une lib comme `outlines`, `instructor`, `marvin`, ou `litellm`
prompt = """
Extract entities from the text below as JSON with this schema:
{"persons": [{"name": str, "role": str|null}],
 "companies": [{"name": str, "industry": str|null}]}

Text: "Steve Jobs co-founded Apple, a consumer electronics company, in 1976."
"""
# → LLM renvoie JSON valide grâce au schéma forcé
```

**Quand le faire** :
- Quand le **schéma est complexe** (attributs, relations, hiérarchie).
- Quand on a un **budget tokens** raisonnable (l'inférence LLM coûte plus cher).
- Quand on veut **prototyper vite** sans entraîner.

**Limites** :
- Plus lent qu'un GLiNER ou un fine-tuned BERT.
- Moins reproductible (sauf `do_sample=False`).
- Coût $$ si LLM hébergé.
<!-- #endregion -->

<!-- #region -->
## 7. Évaluation rigoureuse avec `seqeval`
<!-- #endregion -->

<!-- #region -->
**Erreur classique** : utiliser `sklearn.metrics.f1_score` token-par-token. Ça surévalue grossièrement parce que les tokens `O` (majoritaires) sont faciles.

**Bonne pratique** : utiliser **`seqeval`** qui fait l'**entity-level F1** — une entité est considérée correcte uniquement si **toute son span ET son type** sont prédits exactement.

```
True : [B-PER, I-PER, O, B-LOC]   (Steve Jobs ... Cupertino)
Pred : [B-PER, O,     O, B-LOC]   (manque Jobs)
→ Précision PER = 0/1 = 0  (le span n'est PAS exact)
→ Précision LOC = 1/1 = 1
```
<!-- #endregion -->

```python
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

true_tags = [
    ["B-PER", "I-PER", "O", "O", "B-LOC"],
    ["B-ORG", "O", "B-LOC"],
]
pred_tags = [
    ["B-PER", "I-PER", "O", "O", "B-LOC"],
    ["B-ORG", "O", "O"],  # manque le dernier LOC
]

print(f"F1 entity-level   : {f1_score(true_tags, pred_tags):.3f}")
print(f"Precision         : {precision_score(true_tags, pred_tags):.3f}")
print(f"Recall            : {recall_score(true_tags, pred_tags):.3f}")
print(classification_report(true_tags, pred_tags))
```

<!-- #region -->
## 8. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 8.1 Choix d'approche
<!-- #endregion -->

<!-- #region -->
- **Stable + beaucoup de data** → fine-tuning DistilBERT/ModernBERT/DeBERTa-v3.
- **Schéma fluide / nouveau domaine** → GLiNER (zero-shot puis fine-tune si besoin).
- **Multilingue** → `Davlan/xlm-roberta-base-NER` ou GLiNER multilingue.
- **Domain-specific médical/légal** → modèles spécialisés (BioBERT, LegalBERT) + fine-tuning.
- **Schémas complexes** → LLM avec JSON mode.
<!-- #endregion -->

<!-- #region -->
### 8.2 Préparation des données
<!-- #endregion -->

<!-- #region -->
- **Annotation** : outils 2026 — **Argilla** (open-source HF), **Label Studio**, **Prodigy** (commercial mais excellent).
- **Pré-annotation par GLiNER** puis correction humaine = 5-10× plus rapide qu'annoter from scratch.
- Toujours **double-annoter** un sous-ensemble pour mesurer l'**inter-annotator agreement** (Cohen's κ).
- **Augmenter** : substitution d'entités du même type, traduction aller-retour pour multilingue.
<!-- #endregion -->

<!-- #region -->
### 8.3 Inference et déploiement
<!-- #endregion -->

<!-- #region -->
- Si grosse volumétrie : **ONNX** + `optimum` pour 2-5× speedup CPU.
- **Batching** indispensable (1 doc → 50 ms ; 32 docs → 100 ms total).
- **Quantization int8** via `optimum.intel` (Intel Neural Compressor) pour CPU prod.
- Pour des **flux temps réel** : penser à pré-allouer le tokenizer et le modèle au démarrage.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [Hugging Face — Token Classification course](https://huggingface.co/learn/nlp-course/chapter7/2)
- [GLiNER paper — Zaratiana et al. (NAACL 2024)](https://aclanthology.org/2024.naacl-long.300.pdf)
- [GLiNER GitHub](https://github.com/urchade/GLiNER)
- [GLiNER overview — Zilliz blog](https://zilliz.com/blog/gliner-generalist-model-for-named-entity-recognition-using-bidirectional-transformer)
- [seqeval docs](https://github.com/chakki-works/seqeval)
- [Argilla — annotation collaborative](https://docs.argilla.io/)
- [BiLSTM-CRF — notebook NLP_NER_BiLSTM_CRF (wiki historique)](NLP_NER_BiLSTM_CRF.ipynb)
<!-- #endregion -->
