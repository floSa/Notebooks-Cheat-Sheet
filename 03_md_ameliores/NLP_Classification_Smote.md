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
# ⚖️ Classification de texte déséquilibrée — SMOTE et au-delà
<!-- #endregion -->

<!-- #region -->
Notebook **Tutoriel + Cheat-sheet** centré sur la **classification de texte avec classes déséquilibrées** — situation extrêmement fréquente en prod (spam 1 % vs ham 99 %, fraud, anomalie, churn, modération, ...).

Couvre les **3 leviers** :

1. **Métriques** — accuracy ment, F1 macro / PR-AUC / MCC ne mentent pas.
2. **Niveau données** — resampling : SMOTE (et ses variantes texte), oversampling, undersampling.
3. **Niveau modèle** — class weights, focal loss, threshold tuning.
4. **Niveau augmentation** — back-translation, paraphrase, **LLM-augmentation** (le standard 2026 pour les classes ultra-rares).

Dataset : **20 Newsgroups** ré-équilibré artificiellement pour démonstration (mutualisé avec `NLP_Classification_Supervisee` et `NLP_Transformers`).

> Pour la classif baseline équilibrée, voir `NLP_Classification_Supervisee`.
<!-- #endregion -->

<!-- #region -->
## 1. Le problème du déséquilibre
<!-- #endregion -->

<!-- #region -->
**Situation** : on classifie en N classes, et certaines classes représentent < 10 % (parfois < 0.1 %) des données.

**Pourquoi c'est un piège** :

- Un modèle qui **prédit tout en classe majoritaire** atteint déjà 95 % d'accuracy si une classe fait 95 % → l'accuracy est trompeuse.
- La loss cross-entropy moyennée donne **proportionnellement plus de poids à la classe majoritaire** — le modèle "ignore" naturellement les classes rares.
- Les classes rares ont **peu d'exemples** → le modèle n'a pas vu assez de variations → généralise mal.

**Cas réels typiques** :

| Cas | Ratio typique | Conséquence si ignoré |
|---|---|---|
| Détection de fraude | 0.1 - 1 % | Toute fraude ratée = $ |
| Anomalie médicale | 1 - 5 % | Faux négatif = patient en danger |
| Modération de contenu | 1 - 10 % | Faux négatif = harcèlement non détecté |
| Spam | 5 - 20 % | Faux positif = email légit perdu |
| Multi-classe NLP en long tail | varie | Petites classes invisibles |
<!-- #endregion -->

<!-- #region -->
## 2. Métriques adaptées (et pourquoi accuracy ment)
<!-- #endregion -->

<!-- #region -->
Règle d'or : **dès qu'on est déséquilibré, oublier l'accuracy**.

| Métrique | Formule / idée | Quand utiliser |
|---|---|---|
| **F1 macro** | Moyenne arithmétique des F1 par classe (poids égal) | Multi-classe, toutes les classes comptent autant |
| **F1 weighted** | F1 par classe pondéré par support | Compromis si grosses classes peuvent peser plus |
| **PR-AUC** | Aire sous la courbe Precision-Recall | **Binaire déséquilibré** — bien plus informative que ROC-AUC |
| **MCC** | `(TP·TN - FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Robuste à n'importe quel déséquilibre |
| **Balanced Accuracy** | `(TPR + TNR) / 2` | Vue "équilibrée" simple |
| **Recall à precision fixée** | Recall@P=0.95 | Cas réel : "je veux flagger un max de fraudes avec précision > 95 %" |

**Important** : **ROC-AUC est trompeuse en cas extrême** (par ex. 0.5 % positifs). Privilégier **PR-AUC** ou **F1 à precision fixée**.
<!-- #endregion -->

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, f1_score, matthews_corrcoef,
    balanced_accuracy_score, average_precision_score,
)

# On construit un dataset déséquilibré artificiellement à partir de 20news
# Binaire : "sci.med" (positif, rare) vs reste (négatif, majoritaire)
news = fetch_20newsgroups(
    subset="train",
    categories=["sci.med", "comp.graphics", "rec.sport.hockey", "talk.politics.misc"],
    remove=("headers", "footers", "quotes"), random_state=42,
)
y = (np.array(news.target_names)[news.target] == "sci.med").astype(int)
texts = news.data

# On garde tous les positifs mais sous-échantillonne les négatifs jusqu'à ratio 5 % positifs
rng = np.random.RandomState(42)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
n_pos = len(pos_idx)
n_neg = n_pos * 19  # 1 positif pour 19 négatifs → 5 % positifs
neg_keep = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
keep = np.concatenate([pos_idx, neg_keep])

texts = [texts[i] for i in keep]
y = y[keep]
print(f"Total : {len(y)}  |  positifs : {y.sum()}  ({y.mean():.1%})")
```

```python
# Baseline naïve (LogReg standard)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.3, stratify=y, random_state=42,
)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, stop_words="english")
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, y_train)
y_pred = clf.predict(Xte)
y_score = clf.predict_proba(Xte)[:, 1]

print(f"Accuracy        : {(y_pred == y_test).mean():.3f}  ← TROMPEUR (5 % positifs)")
print(f"F1 macro        : {f1_score(y_test, y_pred, average='macro'):.3f}")
print(f"F1 minoritaire  : {f1_score(y_test, y_pred, pos_label=1):.3f}")
print(f"MCC             : {matthews_corrcoef(y_test, y_pred):.3f}")
print(f"PR-AUC          : {average_precision_score(y_test, y_score):.3f}")
print(f"Balanced acc    : {balanced_accuracy_score(y_test, y_pred):.3f}")
```

<!-- #region -->
## 3. Resampling au niveau données : SMOTE et variantes
<!-- #endregion -->

<!-- #region -->
### 3.1 Vue d'ensemble
<!-- #endregion -->

<!-- #region -->
**3 familles** :

| Stratégie | Idée | Quand |
|---|---|---|
| **Random Oversampling** | Duplique au hasard les exemples minoritaires | Simple, baseline |
| **Random Undersampling** | Drop au hasard des exemples majoritaires | Si on a *énormément* de data |
| **SMOTE** (Synthetic Minority Over-sampling) | Génère de nouveaux exemples synthétiques en interpolant entre voisins | Données numériques tabulaires — limite : textuel via TF-IDF marche, sur embeddings c'est correct |

**Pour le texte spécifiquement** :

- SMOTE marche sur **vecteurs TF-IDF** (sparse → conversion dense) ou **embeddings denses** (sentence-transformers).
- SMOTE **ne marche pas** sur les tokens bruts (ID intervales discrets sans interpolation valide).
- Alternative texte-native : voir section 5 (back-translation, paraphrase, LLM augmentation).

**Note critique** : SMOTE doit être appliqué **uniquement sur train**, jamais sur test (sinon leak). En sklearn moderne, utiliser `imblearn.pipeline` (et pas `sklearn.pipeline`) pour que `fit_resample` ne soit pas appelé sur le test.
<!-- #endregion -->

<!-- #region -->
### 3.2 SMOTE sur TF-IDF + LogReg
<!-- #endregion -->

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Pipeline imblearn : SMOTE est appliqué AVANT le classifieur, mais SEULEMENT au fit
# (pas au predict) — ce que sklearn.Pipeline ne sait pas faire correctement.
pipe_smote = ImbPipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95,
                               sublinear_tf=True, stop_words="english")),
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe_smote.fit(X_train, y_train)
y_pred_s = pipe_smote.predict(X_test)
y_score_s = pipe_smote.predict_proba(X_test)[:, 1]

print(f"F1 minoritaire  SMOTE : {f1_score(y_test, y_pred_s, pos_label=1):.3f}")
print(f"PR-AUC          SMOTE : {average_precision_score(y_test, y_score_s):.3f}")
print(f"MCC             SMOTE : {matthews_corrcoef(y_test, y_pred_s):.3f}")
```

<!-- #region -->
### 3.3 Variantes de SMOTE à connaître
<!-- #endregion -->

<!-- #region -->
| Variante | Apport |
|---|---|
| **SMOTE** classique | Interpolation entre exemples positifs |
| **BorderlineSMOTE** | Concentre la génération près de la frontière de décision |
| **ADASYN** | Génère plus d'exemples près des positifs difficiles à classer |
| **SVMSMOTE** | Utilise un SVM pour identifier où générer |
| **SMOTE-Tomek / SMOTE-ENN** | SMOTE + nettoyage par Tomek links / ENN |
| **RandomOverSampler** | Pas de synthèse, juste duplication — souvent suffisant |
| **RandomUnderSampler** | Drop des négatifs au hasard, utile si N énorme |

Tous disponibles dans `imbalanced-learn` (`imblearn.over_sampling`, `imblearn.under_sampling`, `imblearn.combine`).
<!-- #endregion -->

<!-- #region -->
## 4. Resampling au niveau modèle : class weights, focal loss, threshold
<!-- #endregion -->

<!-- #region -->
### 4.1 Class weights (le levier le plus simple)
<!-- #endregion -->

<!-- #region -->
Plutôt que de modifier les données, on **pondère la loss** : faire compter chaque exemple positif plus que chaque négatif.

- **sklearn** : `class_weight="balanced"` dans `LogisticRegression`, `RandomForestClassifier`, etc.
- **PyTorch / TF** : `weight` ou `class_weight` dans `CrossEntropyLoss` / `model.fit`.
- **HuggingFace Trainer** : override `compute_loss` ou utiliser `WeightedRandomSampler` côté DataLoader.

C'est **équivalent mathématiquement** à dupliquer chaque exemple positif `K` fois, mais sans gonfler la mémoire.
<!-- #endregion -->

```python
clf_w = LogisticRegression(max_iter=1000, class_weight="balanced")
clf_w.fit(Xtr, y_train)
y_pred_w = clf_w.predict(Xte)
y_score_w = clf_w.predict_proba(Xte)[:, 1]

print(f"F1 minoritaire  weights : {f1_score(y_test, y_pred_w, pos_label=1):.3f}")
print(f"PR-AUC          weights : {average_precision_score(y_test, y_score_w):.3f}")
print(f"MCC             weights : {matthews_corrcoef(y_test, y_pred_w):.3f}")
```

<!-- #region -->
### 4.2 Focal Loss
<!-- #endregion -->

<!-- #region -->
**Focal Loss** (Lin et al., 2017) — modifie la cross-entropy pour **down-weight** les exemples faciles (bien classés) et **up-weight** les difficiles :

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- `p_t` : proba prédite de la vraie classe.
- `γ` (gamma, typique 2) : plus c'est grand, plus on focalise sur les hard examples.
- `α_t` : balance positifs/négatifs (typique 0.25 pour la classe rare).

Originalement pour la détection d'objets (où les "facile = background" sont 99 %), s'est généralisé à toute classif déséquilibrée. Implémentation dispo via `torchvision.ops.sigmoid_focal_loss` ou `torch.hub` (kornia).
<!-- #endregion -->

<!-- #region -->
### 4.3 Threshold tuning
<!-- #endregion -->

<!-- #region -->
**Souvent oublié** : le seuil de décision par défaut est `0.5`. Pour un cas déséquilibré, ce n'est presque jamais optimal.

Approche :

1. Calculer `precision`, `recall`, `F1` à tous les seuils.
2. Choisir le seuil qui **maximise la métrique business** (F1, ou recall@precision_min).

Souvent **+5 à +15 points de F1** "gratuits" en bougeant simplement le seuil.
<!-- #endregion -->

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_score)
f1s = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1s[:-1])  # le dernier threshold est NaN
print(f"Seuil par défaut (0.5)  : F1 = {f1_score(y_test, y_score >= 0.5, pos_label=1):.3f}")
print(f"Seuil optimal ({thresholds[best_idx]:.3f}) : F1 = {f1s[best_idx]:.3f}")
```

<!-- #region -->
## 5. Augmentation au niveau texte (2026)
<!-- #endregion -->

<!-- #region -->
SMOTE sur vecteurs numériques marche, mais en 2026 on a **mieux pour le texte** :

| Technique | Idée | Quand |
|---|---|---|
| **Back-translation** | Traduit en/de via une langue tierce (EN → DE → EN) | Conserve la sémantique, change la formulation |
| **Paraphrase model** | T5/BART fine-tuné sur paraphrase | Plus contrôlable que back-translation |
| **EDA** (Easy Data Augmentation) | Synonyme, swap, insertion, deletion aléatoires | Très léger, marche surprenamment bien |
| **LLM augmentation** | Prompt un LLM : "Génère 10 variations de ce texte qui appartiennent à la classe X" | **Standard 2026 pour les classes ultra-rares** |
| **Synthetic data via LLM** | Génère **from scratch** des exemples nouveaux | Quand on a 0-10 exemples seulement |
<!-- #endregion -->

```python
# Pseudo-code LLM augmentation
"""
prompt = '''
Below is a positive example for the class 'medical complaint'.
Generate 5 paraphrases that preserve the meaning and the class.
Return them as a JSON list.

Example: '{positive_text}'
'''
# llm_response = call_llm(prompt)  # via openai / transformers / etc
# augmented_texts = json.loads(llm_response)
# train_texts.extend(augmented_texts)
# train_labels.extend([1] * len(augmented_texts))
"""
```

<!-- #region -->
## 6. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
### 6.1 Ordre des leviers à tester
<!-- #endregion -->

<!-- #region -->
1. **Toujours commencer par** : métriques adaptées + **class_weight="balanced"** + **threshold tuning**.
   → Souvent suffit pour 80 % des cas.
2. Si insuffisant : ajouter **SMOTE** ou ses variantes (en `imblearn.Pipeline`).
3. Si toujours insuffisant : **augmentation texte** (back-translation puis LLM-augmentation).
4. Pour un système prod final : **ensemble** de modèles entraînés sur différents resamplings.
<!-- #endregion -->

<!-- #region -->
### 6.2 Pièges classiques
<!-- #endregion -->

<!-- #region -->
- ❌ **SMOTE appliqué avant le split train/test** → leak massif.
- ❌ **Threshold optimisé sur le test set** → biais d'optimisme.
- ❌ **Évaluation sur dataset déséquilibré avec accuracy** → métrique mensongère.
- ❌ **Resampler le test set pour "équilibrer"** → on évalue sur une distribution qui n'existe pas en prod.
- ✅ Toujours évaluer sur la **distribution réelle de prod** (pas resamplée).
- ✅ Reporter **précision, recall, F1, PR-AUC, MCC** ensemble (vue complète).
- ✅ Pour les LLMs/embeddings : commencer par tester sans SMOTE — souvent les representations sont suffisamment riches pour que le déséquilibre soit moins critique qu'avec TF-IDF.
<!-- #endregion -->

<!-- #region -->
## 7. Sources
<!-- #endregion -->

<!-- #region -->
- [imbalanced-learn — docs officielles](https://imbalanced-learn.org/)
- [SMOTE — Chawla et al. (2002)](https://arxiv.org/abs/1106.1813)
- [Focal Loss — Lin et al. (2017)](https://arxiv.org/abs/1708.02002)
- [EDA — Wei & Zou (2019)](https://arxiv.org/abs/1901.11196)
- [LLM data augmentation — survey 2024](https://arxiv.org/abs/2403.02990)
- [Notebook NLP_Classification_Supervisee — baseline équilibrée](NLP_Classification_Supervisee.ipynb)
<!-- #endregion -->
