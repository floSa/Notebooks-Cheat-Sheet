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
# 🤖 NLP avec Hugging Face Transformers
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** pour la lib `transformers` de Hugging Face (version 5.x, état 2026).

Le but est triple :

1. Comprendre ce qu'est l'écosystème `transformers` en 2026 (lib + Hub + datasets + accelerate + peft + serving).
2. Savoir utiliser les **briques de base** (`AutoTokenizer`, `AutoModel`, `pipeline`) pour faire de l'inférence en quelques lignes.
3. Savoir mener un **fine-tuning supervisé** complet (classification, ici sur 20 Newsgroups) — et un fine-tuning **paramétrique-efficace** (LoRA / PEFT) qui est devenu la norme pour les LLMs.

> **Pré-requis** : Python 3.10+, PyTorch 2.4+, GPU recommandé pour le fine-tuning (CPU OK pour l'inférence légère avec DistilBERT).
<!-- #endregion -->

<!-- #region -->
## 1. État de la lib transformers (2026)
<!-- #endregion -->

<!-- #region -->
La lib `transformers` est devenue le **standard de définition de modèles** dans tout l'écosystème ML moderne (texte, vision, audio, multimodal). En 2026 :

- **Version 5.x** : refonte autour de PyTorch comme backend principal, simplification de la définition de modèles, alignement avec les frameworks d'inférence modernes (vLLM, SGLang, TGI, llama.cpp, mlx).
- **3M+ téléchargements pip/jour** — l'API `AutoTokenizer` / `AutoModel` / `pipeline` est devenue la lingua franca du NLP/multimodal.
- **Lib pivot** : un modèle défini dans `transformers` est compatible avec Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch Lightning (entraînement) et vLLM, SGLang, TGI (inférence).
- **Évolutions majeures depuis 2024** : ModernBERT (encoder rapide), familles Llama 3/4, Mistral, Qwen, Gemma 3, support natif de la **quantization** (bitsandbytes, GPTQ, AWQ), `chat_template` Jinja standardisée, **Flash Attention 2/3** intégré, `accelerate` pour distribué transparent.

L'écosystème autour de `transformers` :

| Librairie | Rôle |
|---|---|
| `transformers` | Définition de modèles + APIs `pipeline`/`Trainer` |
| `datasets` | Chargement, streaming et préparation de datasets (v4.x) |
| `tokenizers` | Tokenization rapide (BPE/WordPiece/Unigram) en Rust |
| `accelerate` | Lancement distribué CPU/GPU/multi-GPU transparent |
| `peft` | Fine-tuning paramétrique-efficace (LoRA, QLoRA, AdaLoRA, prompt tuning) |
| `evaluate` | Métriques standardisées |
| `safetensors` | Format de sérialisation sûr (remplace `.bin` pickle) |
| `bitsandbytes` | Quantization 4/8-bit |
| `vllm` / `sglang` / `text-generation-inference` | Serveurs d'inférence haute performance |
<!-- #endregion -->

```python
import transformers
import datasets
import accelerate
import torch

print(f"transformers : {transformers.__version__}")
print(f"datasets     : {datasets.__version__}")
print(f"accelerate   : {accelerate.__version__}")
print(f"torch        : {torch.__version__}")
print(f"CUDA dispo   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device     : {torch.cuda.get_device_name(0)}")
```

<!-- #region -->
Tu dois voir au moins `transformers >= 5.0`, `torch >= 2.4`. Si pas de CUDA, l'inférence légère reste possible sur CPU, le fine-tuning sera lent (et le notebook utilisera un modèle distilled comme DistilBERT).
<!-- #endregion -->

<!-- #region -->
## 2. Les 3 briques de base : Tokenizer, Model, Pipeline
<!-- #endregion -->

<!-- #region -->
Toute la lib repose sur 3 objets :

1. **Tokenizer** — transforme du texte en `input_ids` (entiers) que le modèle comprend.
2. **Model** — l'architecture neuronale + ses poids (sortie : `logits`, `hidden_states`, etc.).
3. **Pipeline** — wrapper haut-niveau qui enchaîne `tokenizer → model → post-processing` pour une tâche donnée.

Pour 95 % des cas, on utilise les classes `Auto*` qui détectent automatiquement l'architecture à partir du nom du modèle sur le Hub.
<!-- #endregion -->

<!-- #region -->
### 2.1 Tokenizer
<!-- #endregion -->

<!-- #region -->
Le tokenizer convertit le texte en suites d'**identifiants de tokens** (subword units). Il s'occupe aussi du padding, de la troncature, des tokens spéciaux (`[CLS]`, `[SEP]`, `<s>`, etc.) et du décodage inverse.

Trois algorithmes dominants :

- **WordPiece** (BERT, DistilBERT) — vocabulaire greedy, préfixes `##`.
- **BPE / Byte-level BPE** (GPT, RoBERTa, Llama) — fusion itérative de paires fréquentes.
- **Unigram** (T5, mT5, ALBERT) — modèle probabiliste, drop des tokens les moins utiles.

En pratique on ne choisit pas — le tokenizer est lié au modèle pré-entraîné.
<!-- #endregion -->

```python
from transformers import AutoTokenizer

# DistilBERT : modèle léger (66M params, ~250 MB), parfait pour démos CPU
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

texts = [
    "Hugging Face transformers is the de facto standard for NLP.",
    "Tokenization splits text into subword units.",
]

encoded = tokenizer(
    texts,
    padding=True,        # pad au plus long de la batch
    truncation=True,     # tronque au max_length du modèle (512 pour BERT-like)
    max_length=32,
    return_tensors="pt", # PyTorch tensors (alternative: "np", "tf")
)

print("input_ids shape  :", encoded["input_ids"].shape)
print("attention_mask   :", encoded["attention_mask"][0])
print("Tokens (premier) :", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
print("Decoded (1er)    :", tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False))
```

<!-- #region -->
Points-clés :

- `padding=True` aligne toutes les séquences à la plus longue (ou `max_length` si fourni avec `padding="max_length"`).
- `attention_mask` vaut 1 pour les tokens réels, 0 pour le padding — le modèle l'utilise pour ignorer le padding dans l'attention.
- Les tokens spéciaux `[CLS]` (début) et `[SEP]` (fin) sont ajoutés automatiquement.
- `return_tensors="pt"` renvoie des tenseurs PyTorch directement utilisables par le modèle.
<!-- #endregion -->

<!-- #region -->
### 2.2 Model
<!-- #endregion -->

<!-- #region -->
Les modèles se déclinent en deux familles :

- `AutoModel` — renvoie les **hidden states bruts** (utile pour extraire des embeddings).
- `AutoModelFor<Task>` — ajoute une **tête de tâche** (classification, NER, QA, génération, ...) au modèle de base.

Liste non exhaustive des têtes disponibles :

| Classe | Tâche | Sortie |
|---|---|---|
| `AutoModelForSequenceClassification` | Classification de texte | `logits` `(batch, n_labels)` |
| `AutoModelForTokenClassification` | NER, POS tagging | `logits` `(batch, seq, n_labels)` |
| `AutoModelForQuestionAnswering` | QA extractive (SQuAD) | start/end logits |
| `AutoModelForCausalLM` | Génération autoregressive (GPT, Llama) | `logits` next-token |
| `AutoModelForSeq2SeqLM` | Génération seq2seq (T5, BART) | `logits` decoder |
| `AutoModelForMaskedLM` | Pré-entraînement BERT-like | `logits` masked positions |
<!-- #endregion -->

```python
from transformers import AutoModel, AutoModelForSequenceClassification

# Modèle de base : on récupère les hidden states (embeddings contextualisés)
base_model = AutoModel.from_pretrained("distilbert-base-uncased")

with torch.no_grad():
    outputs = base_model(**encoded)

# hidden_states : (batch, seq_len, hidden_dim) — 768 pour DistilBERT
print("Last hidden state shape :", outputs.last_hidden_state.shape)

# Pour obtenir un embedding par phrase (sentence embedding naïf) :
# moyenne pondérée par l'attention_mask
mask = encoded["attention_mask"].unsqueeze(-1).float()
sentence_emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
print("Sentence embeddings    :", sentence_emb.shape)
```

<!-- #region -->
> **Note** : pour des embeddings de phrase de qualité, utilise plutôt `sentence-transformers` qui ajoute un pooling appris et un fine-tuning contrastif. Voir le notebook `BDD_Vectorielles`.
<!-- #endregion -->

<!-- #region -->
### 2.3 Pipeline
<!-- #endregion -->

<!-- #region -->
`pipeline` est le sucre syntaxique qui enchaîne tokenizer + model + post-processing pour une tâche donnée. En 1 ligne tu fais de la classification, du NER, de la summarization, de la génération, du zero-shot, etc.

C'est le bon outil pour **prototyper** ou faire de l'inférence sans s'embêter. Pour du fine-tuning ou de la prod, on descend d'un cran (voir sections suivantes).
<!-- #endregion -->

```python
from transformers import pipeline

# Pipeline sentiment-analysis (modèle par défaut : distilbert sst-2)
sentiment = pipeline("sentiment-analysis")

results = sentiment([
    "I absolutely love this library!",
    "This is the worst movie I have ever seen.",
])
for r in results:
    print(f"  {r['label']:8s}  score={r['score']:.4f}")
```

<!-- #region -->
On peut spécifier explicitement le modèle, le device, le batch size :
<!-- #endregion -->

```python
# Zero-shot classification : pas besoin d'avoir entraîné sur les classes
zsl = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=0 if torch.cuda.is_available() else -1,
)

result = zsl(
    "The new Python release brings significant performance improvements to the interpreter.",
    candidate_labels=["technology", "sports", "politics", "cooking"],
)
print(result)
```

<!-- #region -->
## 3. Tour des familles de modèles (2026)
<!-- #endregion -->

<!-- #region -->
Trois grandes architectures, plus les modèles multimodaux.
<!-- #endregion -->

<!-- #region -->
### 3.1 Encoders (BERT-likes)
<!-- #endregion -->

<!-- #region -->
**Architecture** : empilement de Transformer blocks **bidirectionnels** (attention sur toute la séquence). Pré-entraînés par **masked language modeling** (MLM).

**Idéal pour** : tâches de **compréhension** — classification, NER, QA extractive, similarité, retrieval.

**Modèles 2026 phares** :

- **BERT / RoBERTa / DistilBERT** — toujours présents, baselines solides.
- **DeBERTa-v3** — décodeur d'attention amélioré + ELECTRA-style pre-training, très fort en NLU.
- **ModernBERT** (2024) — refonte avec rotary embeddings, FlashAttention 2, 8192 tokens de contexte, 2x plus rapide que BERT à qualité comparable. Recommandé en remplacement de BERT.
- **XLM-R / mDeBERTa** — versions multilingues.
<!-- #endregion -->

```python
# ModernBERT — l'encodeur recommandé en 2026 quand on a besoin de contexte long
# (8k tokens vs 512 pour BERT)
from transformers import AutoTokenizer, AutoModel

modern_tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
modern_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

print(f"Vocab size      : {modern_tok.vocab_size}")
print(f"Max position    : {modern_tok.model_max_length}")
print(f"Hidden size     : {modern_model.config.hidden_size}")
print(f"Num layers      : {modern_model.config.num_hidden_layers}")
```

<!-- #region -->
### 3.2 Decoders (LLMs)
<!-- #endregion -->

<!-- #region -->
**Architecture** : empilement de Transformer blocks **causaux** (attention masquée — token `t` ne voit que `0..t-1`). Pré-entraînés par **next-token prediction**.

**Idéal pour** : **génération de texte** — assistants, completion code, raisonnement, fonction d'agents.

**Modèles 2026 phares (open-weights)** :

- **Llama 3.x / 4.x** (Meta) — référence open-weights, tailles 1B → 405B.
- **Mistral / Mixtral** — efficacité MoE.
- **Qwen 3** (Alibaba) — multilingue fort, tailles 0.5B → 72B.
- **Gemma 3** (Google) — multimodal natif.
- **Phi-4** (Microsoft) — petits modèles très performants (3B-14B).

Pour les LLMs, on utilise rarement directement `transformers` en production — préférer **vLLM**, **SGLang** ou **TGI** pour le serving (throughput 10-100x supérieur grâce au paged attention, prefix caching, batching continu).
<!-- #endregion -->

```python
# Génération avec un petit decoder
gen = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-135M-Instruct",  # 135M params, tourne sur CPU
    device=0 if torch.cuda.is_available() else -1,
)

# Format chat moderne (chat template Jinja appliqué automatiquement)
messages = [
    {"role": "user", "content": "Explique en 2 phrases ce qu'est l'attention dans un transformer."}
]
out = gen(messages, max_new_tokens=128, do_sample=False)
print(out[0]["generated_text"][-1]["content"])
```

<!-- #region -->
### 3.3 Encoder-Decoders (seq2seq)
<!-- #endregion -->

<!-- #region -->
**Architecture** : encoder bidirectionnel + decoder causal + cross-attention.

**Idéal pour** : **traduction**, **summarization**, **paraphrase**, toute tâche où la sortie est une **séquence reformulée** de l'entrée.

**Modèles 2026** : T5 / FLAN-T5, BART, mBART, NLLB (traduction 200 langues), Marian.

> Note : les LLMs decoder-only ont absorbé une grosse partie de ces use-cases en 2024-2025 (un prompt "Summarize: ..." sur Llama marche très bien). Les encoder-decoders restent pertinents quand on a besoin de **modèles petits, déterministes, fine-tunés** sur une tâche précise.
<!-- #endregion -->

```python
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

article = """
The Hugging Face Transformers library has become the de facto standard for building, training,
and deploying state-of-the-art natural language processing models. With over 3 million daily
downloads, it powers research labs, startups, and Fortune 500 companies. The library supports
text, vision, audio, and multimodal tasks, and integrates seamlessly with distributed training
frameworks like DeepSpeed, FSDP, and Accelerate.
"""

summary = summarizer(article, max_length=50, min_length=20, do_sample=False)
print(summary[0]["summary_text"])
```

<!-- #region -->
### 3.4 Multimodaux
<!-- #endregion -->

<!-- #region -->
2024-2026 ont vu l'explosion des modèles **vision-language** :

- **CLIP / SigLIP** — embeddings alignés image/texte (retrieval cross-modal).
- **LLaVA, Qwen-VL, Gemma 3** — chat avec images.
- **Whisper** (audio → texte), **MusicGen** (texte → audio).
- **SmolVLM** — petits VLMs efficaces.

Ils s'utilisent comme les decoders, avec en plus un `processor` (combinaison tokenizer + image processor).
<!-- #endregion -->

<!-- #region -->
## 4. Inférence via `pipeline` — tour d'horizon des tâches
<!-- #endregion -->

<!-- #region -->
Tâches disponibles via `pipeline` (toutes prennent un texte/image/audio et renvoient un résultat structuré) :

| Pipeline | Tâche | Modèle par défaut |
|---|---|---|
| `"sentiment-analysis"` | Classif binaire sentiment | distilbert-sst2 |
| `"text-classification"` | Classif multi-classes | dépend |
| `"zero-shot-classification"` | Classif sans entraînement | bart-mnli, deberta-zsl |
| `"ner"` | Named Entity Recognition | bert-large-NER |
| `"question-answering"` | QA extractive | distilbert-squad |
| `"fill-mask"` | Masked LM | distilroberta |
| `"summarization"` | Résumé abstractif | distilbart |
| `"translation_xx_to_yy"` | Traduction | Helsinki-NLP/opus-mt |
| `"text-generation"` | Génération causale | gpt2 (default), spécifier sinon |
| `"text2text-generation"` | Génération seq2seq | t5-base |
| `"feature-extraction"` | Embeddings bruts | dépend |
| `"image-classification"` | Classif image | vit |
| `"image-to-text"` | Captioning | blip |
| `"automatic-speech-recognition"` | ASR | whisper |
<!-- #endregion -->

```python
# Exemple NER : extraction d'entités nommées
ner = pipeline("ner", aggregation_strategy="simple")

text = "Hugging Face was founded in New York City by Clément Delangue and Julien Chaumond in 2016."
entities = ner(text)
for e in entities:
    print(f"  {e['entity_group']:8s}  {e['word']:30s}  score={e['score']:.3f}")
```

<!-- #region -->
## 5. Cas réel — Fine-tuning classification (20 Newsgroups)
<!-- #endregion -->

<!-- #region -->
On va maintenant fine-tuner DistilBERT sur **20 Newsgroups** (classification de posts en 20 catégories). C'est le pipeline complet **prod-ready** que tu adapteras pour tes propres datasets.

Étapes :
1. Chargement et split du dataset.
2. Tokenization en batch.
3. Construction du modèle avec tête de classification.
4. `Trainer` API pour entraîner (compatible CPU, GPU, multi-GPU sans changer de code).
5. Évaluation (accuracy, F1 macro, confusion matrix).
6. Sauvegarde et rechargement.

> **Note temps de calcul** : sur CPU, 1 epoch ≈ 30-60 min. Sur GPU consumer (RTX 4070), 1-2 min. On limite à un sous-ensemble + 1 epoch pour rester rapide en démo.
<!-- #endregion -->

<!-- #region -->
### 5.1 Données : 20 Newsgroups
<!-- #endregion -->

```python
from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset, ClassLabel

# On garde toutes les 20 catégories, mais on enlève les en-têtes/footers/quotes
# pour éviter que le modèle apprenne les artefacts plutôt que le contenu
train_raw = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes"),
    random_state=42,
)
test_raw = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

print(f"Train : {len(train_raw.data):,} documents")
print(f"Test  : {len(test_raw.data):,} documents")
print(f"Classes ({len(train_raw.target_names)}) : {train_raw.target_names[:5]}, ...")
```

```python
# Conversion en datasets HF (interopère avec Trainer)
label_names = train_raw.target_names

train_ds = Dataset.from_dict({"text": train_raw.data, "label": train_raw.target})
test_ds  = Dataset.from_dict({"text": test_raw.data,  "label": test_raw.target})

# On caste la colonne label en ClassLabel pour avoir les noms dans les outputs
train_ds = train_ds.cast_column("label", ClassLabel(names=label_names))
test_ds  = test_ds.cast_column("label",  ClassLabel(names=label_names))

# Pour aller vite en démo : on prend 2 000 train, 500 test (à enlever en prod)
train_ds = train_ds.shuffle(seed=42).select(range(2_000))
test_ds  = test_ds.shuffle(seed=42).select(range(500))

print(train_ds)
print("Exemple :", train_ds[0]["text"][:200], "...")
print("Label   :", label_names[train_ds[0]["label"]])
```

<!-- #region -->
### 5.2 Tokenization
<!-- #endregion -->

```python
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(batch: dict) -> dict:
    """Tokenize une batch de textes — appelée par `.map(batched=True)`."""
    return tokenizer(
        batch["text"],
        padding=False,           # padding dynamique au DataCollator (plus efficace)
        truncation=True,
        max_length=256,          # 256 tokens suffisent pour 20news (textes courts à moyens)
    )


train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
test_tok  = test_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

print(train_tok)
print("Longueurs tokenisées :", [len(x) for x in train_tok[:3]["input_ids"]])
```

<!-- #region -->
### 5.3 Modèle avec tête de classification
<!-- #endregion -->

```python
from transformers import AutoModelForSequenceClassification

num_labels = len(label_names)

# id2label / label2id : indispensable pour avoir les noms dans `pipeline()` plus tard
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# Affichage : DistilBERT a 66M params, dont ~590k pour la tête de classification
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params totaux    : {total_params:,}")
print(f"Params trainable : {trainable_params:,}")
```

<!-- #region -->
### 5.4 Training avec l'API `Trainer`
<!-- #endregion -->

<!-- #region -->
`Trainer` est l'API haut-niveau qui gère :

- Boucle d'entraînement (forward / loss / backward / optimizer step / lr scheduler).
- Logging (TensorBoard, Weights & Biases, etc.).
- Évaluation périodique, early stopping, sauvegarde de checkpoints.
- Distribué (CPU, single-GPU, multi-GPU, multi-node) **sans changer le code** grâce à `accelerate`.
- Mixed precision (FP16/BF16) si GPU compatible.

Pour avoir le contrôle bas-niveau, on écrit sa propre boucle avec `accelerate` directement.
<!-- #endregion -->

```python
import numpy as np
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate

# Padding dynamique : aligne sur le plus long de la batch — économise du compute
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Métriques
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Appelée par Trainer à chaque évaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }


training_args = TrainingArguments(
    output_dir="./_artifacts/distilbert-20news",
    num_train_epochs=1,                 # 1 epoch pour la démo (3-5 en vrai)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,                 # standard pour BERT-likes
    weight_decay=0.01,
    warmup_ratio=0.1,                   # warmup linéaire sur 10% des steps
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),     # mixed precision si GPU
    report_to="none",                   # désactive wandb/tensorboard auto
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ⚠️ Lancement effectif — décommenter pour entraîner
# trainer.train()
# metrics = trainer.evaluate()
# print(metrics)
```

<!-- #region -->
> **Pourquoi `trainer.train()` est commenté** : en mode démo CPU, 1 epoch sur 2 000 docs ≈ 5-10 minutes. À décommenter quand tu lances pour de vrai.
<!-- #endregion -->

<!-- #region -->
### 5.5 Évaluation détaillée
<!-- #endregion -->

```python
from sklearn.metrics import classification_report, confusion_matrix

# Une fois trainer.train() exécuté :
# preds_output = trainer.predict(test_tok)
# y_true = preds_output.label_ids
# y_pred = preds_output.predictions.argmax(axis=-1)
#
# print(classification_report(y_true, y_pred, target_names=label_names, digits=3))
# cm = confusion_matrix(y_true, y_pred)
```

<!-- #region -->
### 5.6 Sauvegarde et rechargement
<!-- #endregion -->

<!-- #region -->
Workflow save/load à exécuter **après** `trainer.train()` :

````python
# Sauvegarde modèle + tokenizer dans le même dossier
# (format safetensors par défaut depuis transformers v5)
trainer.save_model("./_artifacts/distilbert-20news-final")
tokenizer.save_pretrained("./_artifacts/distilbert-20news-final")

# Rechargement plus tard (n'importe où) :
from transformers import pipeline
clf = pipeline("text-classification", model="./_artifacts/distilbert-20news-final")
print(clf("My graphics card is overheating when I play games."))
# → [{'label': 'comp.graphics', 'score': 0.95}]
````
<!-- #endregion -->

<!-- #region -->
## 6. Fine-tuning paramétrique-efficace : LoRA / PEFT
<!-- #endregion -->

<!-- #region -->
Fine-tuner les 66M params de DistilBERT, ça passe. Fine-tuner les **70B params** de Llama, ça ne passe plus (plus de GPU mémoire, plus d'écriture disque sur les checkpoints, plus de temps).

**PEFT** (Parameter-Efficient Fine-Tuning) résout ça en **gelant** le modèle pré-entraîné et en n'entraînant qu'un **petit jeu de paramètres additionnels** (0.1 % à 1 % du total). Les méthodes principales :

- **LoRA** (Low-Rank Adaptation) — ajoute deux matrices low-rank `A ∈ ℝ^{r×k}` et `B ∈ ℝ^{d×r}` qui apprennent la *différence* `ΔW = B A` à appliquer à chaque poids `W` du modèle gelé. Avec `r=8` ou `16`, on entraîne 1000× moins de params.
- **QLoRA** — LoRA + quantization 4-bit du modèle de base (`bitsandbytes`). Permet de fine-tuner un Llama 70B sur un GPU 48GB.
- **AdaLoRA** — LoRA avec rang adaptatif par couche.
- **Prompt tuning / Prefix tuning** — apprend des "soft prompts" injectés en entrée.

### Maths LoRA en 1 paragraphe

Un poids `W ∈ ℝ^{d×k}` du modèle gelé. On apprend `A ∈ ℝ^{r×k}` (init `~N(0, σ²)`) et `B ∈ ℝ^{d×r}` (init zéros, donc `ΔW = 0` au début → pas de perturbation initiale). Forward : `h = W x + (B A) x`. Nombre de params : `d·r + r·k = r(d+k)` au lieu de `d·k`. Pour `d=k=4096, r=16` : 131k au lieu de 16M → **122× moins**.
<!-- #endregion -->

```python
# Exemple LoRA sur DistilBERT pour classification — nécessite `peft`
# Décommenter pour exécuter
"""
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,        # facteur de scaling ΔW = (α/r) · B A
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # noms des couches Q/V dans DistilBERT
)

model_lora = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
)
model_lora = get_peft_model(model_lora, lora_config)
model_lora.print_trainable_parameters()
# → trainable params: 591,124 || all params: 67,547,156 || trainable%: 0.87 %
"""
```

<!-- #region -->
**Quand utiliser quoi en 2026** :

| Situation | Recommandation |
|---|---|
| Modèle ≤ 1B, dataset large, perf max | Full fine-tuning |
| Modèle 1B-13B, budget GPU limité | LoRA (r=8-16) |
| Modèle 13B-70B+ sur GPU consumer | QLoRA (4-bit + LoRA) |
| Plusieurs tâches à servir sur le même base model | LoRA (1 adaptateur par tâche, switch à chaud) |
| Beaucoup de données labellisées, peu de tâches | Full fine-tuning ou Unsloth |
<!-- #endregion -->

<!-- #region -->
## 7. Génération avec LLMs — chat templates et `generate`
<!-- #endregion -->

<!-- #region -->
Les LLMs modernes attendent les messages dans un format de **chat template** (souvent Jinja, défini par le tokenizer). Ne pas concatener à la main — utiliser `apply_chat_template`.
<!-- #endregion -->

```python
# Exemple chat template + generate (sans pipeline)
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(LLM_NAME)
llm = AutoModelForCausalLM.from_pretrained(LLM_NAME)

messages = [
    {"role": "system", "content": "You are a concise Python tutor."},
    {"role": "user", "content": "What is a list comprehension? Answer in one sentence."},
]

# Le tokenizer applique le template Jinja du modèle automatiquement
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== PROMPT FORMATÉ ===")
print(prompt)

inputs = tok(prompt, return_tensors="pt")
outputs = llm.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,                # greedy decoding pour reproductibilité
    pad_token_id=tok.eos_token_id,
)

# On enlève le prompt de la sortie pour ne garder que la réponse
generated = tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n=== RÉPONSE ===")
print(generated)
```

<!-- #region -->
**Paramètres de génération courants** :

- `max_new_tokens` — limite la longueur de la sortie (préférer à `max_length` qui inclut le prompt).
- `do_sample=True` + `temperature` (0.7 par défaut) + `top_p` (0.9) + `top_k` — pour des sorties diverses.
- `do_sample=False` — **greedy decoding**, reproductible (mais répétitif).
- `num_beams=N` — beam search, plus qualitatif sur tâches déterministes (traduction).
- `repetition_penalty=1.1-1.3` — pénalise les répétitions.
- `streamer=TextStreamer(tok)` — affiche la génération token par token (UX chat).
<!-- #endregion -->

<!-- #region -->
## 8. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 8.1 Quantization
<!-- #endregion -->

<!-- #region -->
Pour faire tenir un gros modèle en mémoire :

- **bitsandbytes** : 8-bit (transparent), 4-bit (avec QLoRA) — `load_in_4bit=True`.
- **GPTQ / AWQ** : quantization post-training de haute qualité, formats supportés par vLLM et SGLang.
- **GGUF** : format llama.cpp pour exécution CPU/Mac/edge.
<!-- #endregion -->

<!-- #region -->
Chargement 4-bit avec `bitsandbytes` (nécessite GPU compatible CUDA + `pip install bitsandbytes`) :

````python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
````
<!-- #endregion -->

<!-- #region -->
### 8.2 Serving en prod
<!-- #endregion -->

<!-- #region -->
Ne sers pas un LLM avec `model.generate()` dans une boucle Flask — c'est 10-100x plus lent qu'un serveur dédié. Choix 2026 :

- **vLLM** — paged attention, continuous batching, le standard open.
- **SGLang** — RadixAttention, programmation structurée (function calling natif).
- **TGI** (Hugging Face) — bien intégré avec le Hub, supporte plein de quantizations.
- **llama.cpp** / **mlx** — pour edge / Mac.
<!-- #endregion -->

<!-- #region -->
### 8.3 Versioning et reproductibilité
<!-- #endregion -->

<!-- #region -->
- Toujours pinner `transformers`, `tokenizers`, `torch` dans `pyproject.toml`.
- Push tes modèles fine-tunés sur le Hub avec une **model card** (`README.md` rempli + auto-evaluation results).
- Format **safetensors** uniquement (pas de pickle `.bin`).
- Stocker les datasets d'eval avec `datasets.save_to_disk()`.
<!-- #endregion -->

<!-- #region -->
### 8.4 Évaluation rigoureuse
<!-- #endregion -->

<!-- #region -->
Au-delà de l'accuracy :

- **`evaluate`** pour métriques NLP standards (BLEU, ROUGE, BERTScore, perplexity, exact_match).
- **`lm-eval-harness`** pour benchmarker un LLM sur 200+ tasks.
- **Tests adversaires** : Garak (security), CheckList (robustness).
<!-- #endregion -->

<!-- #region -->
### 8.5 Conseils pratiques
<!-- #endregion -->

<!-- #region -->
- Pré-tokenize ton dataset et `.save_to_disk()` — gain énorme à chaque epoch.
- Pour les LLMs > 7B : utilise **Unsloth** ou **Axolotl** pour fine-tuner, ils accélèrent 2-5x vs `transformers` direct.
- `torch.compile(model)` (PyTorch 2.x) accélère l'inference de 20-50 %.
- Toujours benchmarker une fois en prod : `python -m torch.profiler` ou `nvprof`.
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [Hugging Face Transformers — docs officielles](https://huggingface.co/docs/transformers/)
- [Blog : Transformers v5 announcement](https://huggingface.co/blog/transformers-v5)
- [PEFT library docs](https://huggingface.co/docs/peft/)
- [Datasets library docs](https://huggingface.co/docs/datasets/)
- [ModernBERT — paper & model card](https://huggingface.co/answerdotai/ModernBERT-base)
- [Sachith Dassanayake — Fine-tuning vs adapters vs prompts (2026)](https://www.sachith.co.uk/fine%E2%80%91tuning-vs-adapters-vs-prompts-best-practices-in-2025-practical-guide-mar-27-2026/)
- [vLLM docs](https://docs.vllm.ai/)
- [SGLang docs](https://docs.sglang.ai/)
<!-- #endregion -->
