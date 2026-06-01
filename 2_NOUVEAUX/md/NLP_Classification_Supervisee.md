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
# 📑 Classification supervisée de texte
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** sur la **classification de texte** supervisée — l'une des tâches NLP les plus utilisées en prod (sentiment, intent, spam, routing, modération, ...).

Couvre **3 niveaux de sophistication** et **quand utiliser quoi en 2026** :

1. **Baseline classique** — TF-IDF + sklearn (LogReg / SVM / NB). Ultra-rapide, baseline solide.
2. **Embeddings + classifieur léger** — sentence-transformers + LogReg. Le sweet spot 2026 pour 80 % des cas.
3. **Fine-tuning Transformer** — DistilBERT / ModernBERT. Quand on a >10k exemples et qu'on veut max perf.
4. **Zero-shot / few-shot via LLMs** — quand on n'a pas (ou peu) d'exemples labellisés.

Dataset utilisé : **20 Newsgroups** (mutualisé avec `NLP_Classification_Smote` et `NLP_Transformers`).

> Pour la **classification déséquilibrée** spécifiquement, voir `NLP_Classification_Smote`.
> Pour le **fine-tuning Transformer en détail**, voir `NLP_Transformers`.
<!-- #endregion -->

<!-- #region -->
## 1. Quand utiliser quoi (matrice de décision 2026)
<!-- #endregion -->

<!-- #region -->
| Situation | Approche recommandée |
|---|---|
| **POC en 1 heure** | TF-IDF + LogReg (toujours commencer là) |
| **Dataset < 1000 ex labellisés** | Sentence-Transformers embeddings + LogReg |
| **Dataset 1k-10k, qualité moyenne** | Sentence-Transformers + LogReg ou DistilBERT fine-tuned |
| **Dataset > 10k, qualité haute** | DistilBERT / ModernBERT fine-tuned |
| **Pas de data labellisée** | LLM zero-shot (Llama 3.x, GPT-4) ou ZSL DeBERTa-MNLI |
| **Quelques exemples (5-50)** | LLM few-shot (in-context learning) ou SetFit (sentence-transformers + few-shot training) |
| **Classes très déséquilibrées** | Voir `NLP_Classification_Smote` |
| **Inference temps réel critique** | TF-IDF + LogReg (<1 ms) ou DistilBERT quantizé ONNX |
| **Modèle multilingue** | XLM-R / E5-multilingual fine-tuned, ou GPT-4o zero-shot |
<!-- #endregion -->

<!-- #region -->
## 2. Baseline : TF-IDF + sklearn
<!-- #endregion -->

<!-- #region -->
**Pourquoi commencer là** : 5 lignes de code, training en quelques secondes, F1 souvent >80 % sur tâches simples. Si la baseline atteint déjà ton objectif business, **ne va pas plus loin**.

Stack canonique :

- **TfidfVectorizer** — bigrammes, min_df, sublinear_tf, stop words.
- **LogisticRegression** — robuste, calibrée, interprétable. Alternative : `LinearSVC` (plus rapide), `ComplementNB` (très bon sur texte court).
- **Pipeline** — un seul objet vectorizer + classifier, simplifie save/load et inference.
<!-- #endregion -->

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

# On garde 4 catégories pour aller vite — sur les 20 c'est pareil
CATEGORIES = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

train = fetch_20newsgroups(subset="train", categories=CATEGORIES,
                          remove=("headers", "footers", "quotes"), random_state=42)
test = fetch_20newsgroups(subset="test", categories=CATEGORIES,
                         remove=("headers", "footers", "quotes"), random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),        # uni + bigrammes
        min_df=2,                  # ignore mots qui apparaissent dans <2 docs
        max_df=0.95,               # ignore mots qui apparaissent dans >95 % des docs
        sublinear_tf=True,         # tf devient 1 + log(tf), atténue les outliers
        stop_words="english",
    )),
    ("clf", LogisticRegression(max_iter=1000, C=1.0)),
])

pipe.fit(train.data, train.target)
y_pred = pipe.predict(test.data)
print(f"F1 macro = {f1_score(test.target, y_pred, average='macro'):.3f}")
```

<!-- #region -->
### 2.1 Astuce — analyse des erreurs interprétable
<!-- #endregion -->

<!-- #region -->
Avec un modèle linéaire sur TF-IDF, on peut **lire les coefficients** : pour chaque classe, top-N mots qui poussent vers/contre elle. Indispensable pour debugger.
<!-- #endregion -->

```python
import numpy as np

vectorizer = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]
feature_names = np.array(vectorizer.get_feature_names_out())

for cls_idx, cls_name in enumerate(train.target_names):
    coefs = clf.coef_[cls_idx]
    top_pos = np.argsort(coefs)[-8:][::-1]
    print(f"\n[{cls_name}] top mots positifs :")
    for i in top_pos:
        print(f"    {feature_names[i]:25s}  {coefs[i]:+.3f}")
```

<!-- #region -->
## 3. Sweet spot 2026 : Embeddings + classifieur léger
<!-- #endregion -->

<!-- #region -->
**L'approche qui gagne le rapport perf/coût** pour la plupart des cas avec dataset moyen.

**Idée** :

1. Encode chaque document en un vecteur dense via un modèle pré-entraîné (Sentence-Transformers).
2. Entraîne un classifieur léger (LogReg / SVM) sur ces vecteurs.

**Avantages** :

- **Pas de fine-tuning** du Transformer → 100× moins de compute.
- **Embeddings réutilisables** pour clustering, retrieval, similarité.
- Souvent **+3 à +8 points F1** vs TF-IDF, avec un coût raisonnable.
- Marche bien dès **quelques centaines d'exemples** (TF-IDF demande plus).

**Modèles 2026 conseillés** : `BAAI/bge-small-en-v1.5` (384d), `intfloat/multilingual-e5-large-instruct` (1024d, 100+ langues), `Alibaba-NLP/gte-modernbert-base` (768d, contexte 8k).
<!-- #endregion -->

```python
# Ce bloc télécharge ~90 MB la 1ère fois — décommenter pour exécuter
"""
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_train = encoder.encode(train.data, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
X_test = encoder.encode(test.data, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

clf_emb = LogisticRegression(max_iter=1000, C=1.0)
clf_emb.fit(X_train, train.target)
y_pred = clf_emb.predict(X_test)
print(f"F1 macro = {f1_score(test.target, y_pred, average='macro'):.3f}")
"""
```

<!-- #region -->
### 3.1 SetFit — few-shot avec sentence-transformers
<!-- #endregion -->

<!-- #region -->
Quand on a **5-50 exemples par classe**, **SetFit** (Hugging Face) est l'approche reine en 2026 :

1. Fine-tune contrastif d'un sentence-transformer sur les paires d'exemples (siamese).
2. Classifieur léger (LogReg) au-dessus.

Résultat : **rivalise avec un Transformer fine-tuned sur 1000+ exemples** quand on n'en a que 8-32.

```python
# from setfit import SetFitModel, Trainer, TrainingArguments
# model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5")
# trainer = Trainer(model=model, args=..., train_dataset=mini_train_ds)
# trainer.train()
```
<!-- #endregion -->

<!-- #region -->
## 4. Fine-tuning Transformer
<!-- #endregion -->

<!-- #region -->
Quand on dépasse ~10k exemples labellisés de bonne qualité, fine-tuner directement un encoder (DistilBERT, ModernBERT, DeBERTa-v3) donne le meilleur F1.

Le pipeline complet est couvert en détail dans **`NLP_Transformers`** (sections 5.x : data → tokenization → `AutoModelForSequenceClassification` → `Trainer` API → eval → save).

**Ordre de grandeur en 2026** :

| Modèle | Params | Coût training (20k ex, GPU) | F1 typique vs baseline TF-IDF |
|---|---|---|---|
| DistilBERT-base | 66M | ~10 min | +5 à +10 pts |
| ModernBERT-base | 150M | ~20 min | +7 à +12 pts |
| DeBERTa-v3-large | 700M | ~2 h | +8 à +14 pts |

Au-delà : LoRA sur Llama 3.x 7B, mais le gain marginal vs DeBERTa est faible pour de la classification pure.
<!-- #endregion -->

<!-- #region -->
## 5. Zero-shot et few-shot via LLMs
<!-- #endregion -->

<!-- #region -->
### 5.1 Zero-shot avec un cross-encoder MNLI
<!-- #endregion -->

<!-- #region -->
Avant les LLMs, le standard ZSL était `pipeline("zero-shot-classification")` avec un modèle entraîné sur **MNLI** (entailment). Toujours pertinent en 2026 quand on veut un modèle compact qui tourne CPU.

```python
from transformers import pipeline
zsl = pipeline("zero-shot-classification",
               model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
zsl("My GPU is overheating during gaming sessions.",
    candidate_labels=["hardware", "software", "weather"])
# → {'labels': ['hardware', ...], 'scores': [0.92, ...]}
```
<!-- #endregion -->

<!-- #region -->
### 5.2 Zero-shot / few-shot avec un LLM
<!-- #endregion -->

<!-- #region -->
On envoie un prompt structuré au LLM avec la liste des classes et (optionnellement) quelques exemples. Avec **JSON mode** ou **constrained decoding**, le LLM renvoie directement le label exact.

Quand l'utiliser :

- **Pas d'historique labellisé** → zero-shot.
- **5-30 exemples** → few-shot in-context (ajoute-les au prompt).
- **Schéma de classes change souvent** → un LLM s'adapte sans réentraînement.

Quand NE PAS l'utiliser :

- Inférence à très grande échelle → coût $$ ou latence.
- Latence < 50 ms exigée.
- Tâche très spécifique pour laquelle on a 10k+ exemples (fine-tuning fera mieux et moins cher).
<!-- #endregion -->

<!-- #region -->
## 6. Évaluation : au-delà de l'accuracy
<!-- #endregion -->

<!-- #region -->
L'**accuracy** est un piège dès que les classes sont déséquilibrées (ce qui arrive presque toujours en prod).

**Métriques à monitorer** :

| Métrique | Quand l'utiliser |
|---|---|
| **Accuracy** | Classes équilibrées, coût FP=FN |
| **F1 macro** | Classes déséquilibrées, on veut traiter toutes les classes équitablement |
| **F1 weighted** | Compromis si on accepte que les grosses classes pèsent plus |
| **PR-AUC** | Tâche binaire déséquilibrée (fraud, anomaly, churn) |
| **MCC** | Métrique équilibrée même très déséquilibré, robuste |
| **Top-k accuracy** | Quand on suggère plusieurs labels (autocomplete, recommandation) |
| **Cohen's κ** | Comparaison avec un annotateur humain |
| **Calibration** (Brier, ECE) | Quand le score doit être interprété comme une probabilité (médical, finance) |
<!-- #endregion -->

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score,
)

print(classification_report(test.target, y_pred, target_names=train.target_names, digits=3))
print(f"MCC               : {matthews_corrcoef(test.target, y_pred):.3f}")
print(f"Balanced accuracy : {balanced_accuracy_score(test.target, y_pred):.3f}")
```

<!-- #region -->
## 7. Sources
<!-- #endregion -->

<!-- #region -->
- [scikit-learn — text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [SetFit — efficient few-shot learning](https://github.com/huggingface/setfit)
- [Sentence-Transformers — official docs](https://www.sbert.net/)
- [Zero-shot classification — HuggingFace](https://huggingface.co/tasks/zero-shot-classification)
- [Notebook NLP_Transformers — fine-tuning détaillé](NLP_Transformers.ipynb)
- [Notebook NLP_Classification_Smote — déséquilibre](NLP_Classification_Smote.ipynb)
<!-- #endregion -->
