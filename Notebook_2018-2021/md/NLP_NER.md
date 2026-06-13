---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="-kMdCzvHmbAQ" -->
# **Reconnaissance d'entités nommée - Named Entity Recognition (2021)**
<!-- #endregion -->

<!-- #region id="S_Lo7_jD--5i" -->
La reconnaissance d'entités nommées (NER) est une technique de traitement automatique du langage visant à identifier et classer des éléments spécifiques dans un texte (noms de personnes, organisations, lieux, dates, etc.).  

Utilisations : structuration de données, extraction d'informations, amélioration de moteurs de recherche, assistance à l'analyse de documents (médicale, juridique, journalistique), ou interaction avec des chatbots et assistants virtuels.
<!-- #endregion -->

<!-- #region id="l6BacK92m4hk" -->
## 0. Chargement des données
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 49995, "status": "ok", "timestamp": 1739365017523, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="-RL9nnWhkD8I" outputId="08c6f1ed-27ca-4772-c5f5-3571d318b0fe"
from google.colab import drive
drive.mount('/content/drive')
```

```python executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1739365017523, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="ooPZhweVpdq_"
path = '/content/drive/MyDrive/Machine_Learning/Data/NLP/'
file_name ='Food_dataset.csv'
```

```python executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1739365017523, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="jIaemWuWMGCh"
import json
import pandas as pd
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/", "height": 129} executionInfo={"elapsed": 2662, "status": "ok", "timestamp": 1739365020178, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="zPusTNFNyqTB" outputId="37864367-ee99-4ead-80f7-7b26ec41160d"
df = pd.read_csv(path+file_name, names=['text','entities'], header=0)
# Appliquer text.replace('\n', ' ') uniquement sur la colonne 'text'
#df['text'] = df['text'].apply(lambda x: x.replace('\n', ', '))

# Fonction pour transformer la clé "type" en "label" dans chaque dictionnaire
def transform_entities(entities_str):
    entities = json.loads(entities_str)
    for entity in entities:
        entity['label'] = entity.pop('type')
    return entities

# Appliquer la transformation à la colonne "entities"
df['entities'] = df['entities'].apply(transform_entities)

print(df.shape)
df.head(2)
```

<!-- #region id="FGkS4ILwyWqz" -->
## 1. Trasformation des données à destination d'un NER
<!-- #endregion -->

<!-- #region id="8WFasLNGyd0_" -->
### A. Tokenization
<!-- #endregion -->

<!-- #region id="Z5y0k1ArAXVU" -->
La tokenisation est une étape de traitement du langage consistant à découper un texte en unités minimales (mots, phrases, symboles), appelées tokens.  

Utilisations : prétraitement de textes pour l'analyse linguistique, préparation de données pour des modèles d'IA (synthaxe, traduction, classification), amélioration de l'efficacité des recherches ou extraction de motifs dans des corpus volumineux.
<!-- #endregion -->

<!-- #region id="uDngcAoRymrh" -->
**Tokenizer par défaut (espaces) :**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1739365020178, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="gpqQ4uy-yVln" outputId="649dab0c-53c2-479d-e170-a7dbd0fcb9ed"
def space_tokenizer(text):
    return text.split()

text_test = "Marie est une docteure!"
space_tokenizer(text_test)
```

<!-- #region id="EQy5ISx8ypzF" -->
**Tokenizer basé sur nltk**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1592, "status": "ok", "timestamp": 1739365021765, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="vHHFKaTVotzw" outputId="61c96ab8-cf46-49a1-ff2b-b80526257eee"
import nltk
nltk.download('punkt_tab')
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1739365021765, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="o5Q9WnSiyVpZ" outputId="6373a630-608c-42e9-936f-7dbdd3a3be2a"
from nltk.tokenize import word_tokenize
def nltk_tokenizer(text):
    return word_tokenize(text)

text_test = "Marie est une docteure!"
nltk_tokenizer(text_test)
```

<!-- #region id="dBtkUsMAyrGV" -->
**Tokenizer basé sur spacy**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21981, "status": "ok", "timestamp": 1739365043743, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="cqqB44HWpQAJ" outputId="7235babe-0988-4ff6-f1c8-d667c2b821db"
import spacy

!python -m spacy download fr_core_news_sm -q
nlp = spacy.load("fr_core_news_sm")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1739365043744, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="uTlPrxP-ygV-" outputId="fb179ea9-0b1b-4faa-ef6b-fc959a5e3ac4"
def spacy_tokenizer(text):
    return [token.text for token in nlp(text)]

text_test = "Marie est une docteure!"
spacy_tokenizer(text_test)
```

<!-- #region id="sA41e1W9ysXE" -->
**Tokenizer basé sur re (expressions régulières) :**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 103} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1739365043745, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="_9vKpjqfygYt" outputId="3ac165dc-d33d-4b4d-af6d-a76a6ab872f7"
import re
def regex_tokenizer(text):
    return re.findall(r'\b\w+\b', text)

spacy_tokenizer
```

<!-- #region id="UwhjN9C_ygf_" -->
### B. Transformations aux formats interpretable par les modèles
<!-- #endregion -->

<!-- #region id="f3J6qsKHAuAr" -->
Les formats IOB, BILOU structurent les étiquettes en NER pour indiquer la position et les limites des entités dans une séquence de tokens.  

Exemple :  

    "B-PER" (début d'une personne), "I-LOC" (intérieur d'un lieu), "O" (hors entité).

But :  
Délimiter les entités : distinguer le début, l'intérieur ou la fin d'une entité.
Standardiser l'apprentissage : faciliter l'entraînement des modèles (CRF, LSTM, transformers) en clarifiant les relations entre tokens.
Améliorer la précision : réduire les erreurs de chevauchement ou de fragmentation d'entités.


Utilisation : traitement de corpus annotés, évaluation de modèles (ex: F1-score), pipelines NLP.
<!-- #endregion -->

<!-- #region id="LGgWOwFbzDnR" -->
#### **IOB**


*   I (Inside): indique qu'on est à l'intérieur d'une entité.
*   O (Outside): indique qu'un token n'appartient à aucune
entité.
*   B (Begin): indique le début d'une entité.


<!-- #endregion -->

```python executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1739365043745, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="4f5MbEjvy7Og"
# Fonction pour générer le format IOB
def generate_iob_format(text, entities):
    tokens = nltk_tokenizer(text)
    tags = ['O'] * len(tokens)

    for entity in entities:
        start, end, label = entity['start'], entity['end'], entity['label']
        entity_text = text[start:end]
        entity_tokens = nltk_tokenizer(entity_text)

        for i, token in enumerate(entity_tokens):
            token_index = next((idx for idx, tok in enumerate(tokens) if tok == token and tags[idx] == 'O'), None)
            if token_index is not None:
                tags[token_index] = f"{'B' if i == 0 else 'I'}-{label}"

    return list(zip(tokens, tags))
```

<!-- #region id="6VwHL4cyzFt9" -->
#### **BILOU**:

*   B (Begin): indique le début d'une entité.
*   I (Inside): indique qu'on est à l'intérieur d'une entité.
*   L (Last): indique la fin d'une entité.
*   O (Outside): indique qu'un token n'appartient à aucune entité.
*   U (Unit) : pour indiquer une entité d'un seul token
<!-- #endregion -->

```python executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1739365043745, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="ks8Q_0nby7TL"
# Fonction pour générer le format BILOU
def generate_bilou_format(text, entities):
    tokens = nltk_tokenizer(text)
    tags = ['O'] * len(tokens)

    # Calcul des offsets des tokens dans le texte
    current_pos = 0
    token_offsets = []
    for token in tokens:
        start = text.find(token, current_pos)
        end = start + len(token)
        token_offsets.append((start, end))
        current_pos = end

    for entity in entities:
        start, end, label = entity['start'], entity['end'], entity['label']
        entity_text = text[start:end]
        entity_tokens = nltk_tokenizer(entity_text)

        entity_indices = [i for i, (t_start, t_end) in enumerate(token_offsets) if t_start >= start and t_end <= end]

        if len(entity_indices) == 1:
            tags[entity_indices[0]] = f"U-{label}"
        elif entity_indices:
            for i, idx in enumerate(entity_indices):
                if i == 0:
                    tags[idx] = f"B-{label}"
                elif i == len(entity_indices) - 1:
                    tags[idx] = f"L-{label}"
                else:
                    tags[idx] = f"I-{label}"

    return list(zip(tokens, tags))
```

<!-- #region id="HZiqAlUtzOM6" -->
#### **IOBES - BIOES**

*   B (Begin): indique le début d'une entité.
*   I (Inside): indique qu'on est à l'intérieur d'une entité.
*   O (Outside): indique qu'un token n'appartient à aucune entité.
*   E (End): indique la fin d'une entité.
*   S (Single): indique un token qui est à la fois le début et la fin d'une entité.
<!-- #endregion -->

```python executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1739365043745, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="cPvf2mE8y7Q1"
# Fonction pour générer le format IOBES
def generate_iobes_format(text, entities):
    tokens = nltk_tokenizer(text)
    tags = ['O'] * len(tokens)

    # Calcul des offsets des tokens dans le texte
    current_pos = 0
    token_offsets = []
    for token in tokens:
        start = text.find(token, current_pos)
        end = start + len(token)
        token_offsets.append((start, end))
        current_pos = end

    for entity in entities:
        start, end, label = entity['start'], entity['end'], entity['label']
        entity_text = text[start:end]
        entity_tokens = nltk_tokenizer(entity_text)

        entity_indices = [i for i, (t_start, t_end) in enumerate(token_offsets) if t_start >= start and t_end <= end]

        if len(entity_indices) == 1:
            tags[entity_indices[0]] = f"S-{label}"
        elif entity_indices:
            for i, idx in enumerate(entity_indices):
                if i == 0:
                    tags[idx] = f"B-{label}"
                elif i == len(entity_indices) - 1:
                    tags[idx] = f"E-{label}"
                else:
                    tags[idx] = f"I-{label}"

    return list(zip(tokens, tags))
```

<!-- #region id="jsUB14OL3EEJ" -->
### Application à notre DataFrame
<!-- #endregion -->

```python executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1739365043745, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="KWzSwxfK5z8i"
# data = [
#     {'text': "Thousands of demonstrators have marched through London to protest the war in Iraq.", 'entities': [{'start': 36, 'end': 41, 'label': "LOC"}, {'start': 55, 'end': 60, 'label': "LOC"}]},
#     {'text': "Marie est une docteure!", 'entities': [{'start': 0, 'end': 5, 'label': "PER"}]}
# ]

# df = pd.DataFrame(data)
# df
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1739365043745, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="gGy9J2wl7apQ" outputId="7673f973-0380-495e-c244-e033f3d24504"
print(df['text'].iloc[0])
print(nltk_tokenizer(df['text'].iloc[0]))
print(df['entities'].iloc[0])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4523, "status": "ok", "timestamp": 1739365048260, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="CtmWFDemiMTP" outputId="01b67245-fb51-4af7-a933-55ff0d5be861"
# Application des fonctions
df['iob_format'] = df.apply(lambda x: generate_iob_format(x['text'], x['entities']), axis=1)
df['bilou_format'] = df.apply(lambda x: generate_bilou_format(x['text'], x['entities']), axis=1)
df['iobes_format'] = df.apply(lambda x: generate_iobes_format(x['text'], x['entities']), axis=1)

print(df['iob_format'].iloc[0])
print(df['bilou_format'].iloc[0])
print(df['iobes_format'].iloc[0])
```

<!-- #region id="7fAe0Qdnat8q" -->
## Créatioin de différent modèles de NER
<!-- #endregion -->

```python id="21btol7vZAtB"
!pip install --upgrade tensorflow -q
!pip install tensorflow-addons -q
```

<!-- #region id="EDf6fiCH50iG" -->
### **Préparations des données**
<!-- #endregion -->

```python id="3deLnhpVMW71"
!pip install seqeval -q
import seqeval
```

```python id="DHCScIm8FQGr"
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import classification_report
```

```python id="AdQTERi7FbUY"
# Préparer les données
sentences = [[word for word, tag in sentence] for sentence in df["iob_format"]]
tags = [[tag for word, tag in sentence] for sentence in df["iob_format"]]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1739265816419, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="jsMego1mEOKq" outputId="bd538f3d-3ec6-4bd1-ff9f-848b2228e10a"
print(sentences[0])
print(tags[0])
```

```python id="fd5O-UOGEON-"
# Créer des dictionnaires pour les mots et les tags
words = list(set(word for sentence in sentences for word in sentence))
tags_set = list(set(tag for sentence in tags for tag in sentence))
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags_set)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Transformer les données
X = [[word2idx[word] for word in sentence] for sentence in sentences]
y = [[tag2idx[tag] for tag in sentence] for sentence in tags]
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1739265819403, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="4N1EuctPYS7P" outputId="f06dca98-2624-4d88-b7e2-7f5b57222d74"
print(X[0],'\n',y[0])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1739265865928, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="8SSBJz6vUmVp" outputId="1cc656e8-bab5-41cb-f271-6cf22ec05ee4"
len(X)
len(X[0])
```

<!-- #region id="suAMtcJ6aQuc" -->
### **Modèle LSTM**
<!-- #endregion -->

<!-- #region id="m-MojlvJZ2Xb" -->
Input de départ (Text et Label)

    {'text': "Marie est une docteure", 'entities': [{'start': 0, 'end': 5, 'label': "PER"}, {'start': 15, 'end': 22, 'label': "PRO"}]}

Transformé par exemple en IOB

    [(Marie, B-PER), (est, O), (une, O), (docteure, B-PRO)]

X : Séquence de mots représentée par leurs indices numériques dans le vocabulaire.

    "Marie est une docteure" devient [1, 2, 3, 4] (index de 4 mots différents dans tout le corpus)

Y : Séquence de tags représentée sous forme one-hot encodée (vecteurs binaires pour chaque tag).

    ["B-PER", "O", "O", "B-PRO"] devient [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]] (pour 3 tags : O, B-PER, B-PRO).
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 305, "status": "ok", "timestamp": 1733932014638, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="thbO4e5Hb1xa" outputId="2651e37e-d11c-4199-81aa-8f7808a675e9"
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Créer des dictionnaires pour les mots et les tags
sentences = [[word for word, tag in sentence] for sentence in df["iob_format"]]
tags = [[tag for word, tag in sentence] for sentence in df["iob_format"]]

words = list(set(word for sentence in sentences for word in sentence))
tags_set = list(set(tag for sentence in tags for tag in sentence))
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags_set)}

# Convertir en indices
X = [[word2idx[word] for word in sentence] for sentence in sentences]
y = [[tag2idx[tag] for tag in sentence] for sentence in tags]

# Padding des séquences
max_len = max(len(s) for s in X)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Conversion des étiquettes en one-hot encoding
y = [to_categorical(seq, num_classes=len(tag2idx)) for seq in y]

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train LSTM Example:", X_train[0])
print("y_train LSTM Example:", y_train[0])  # One-hot encoded
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 68728, "status": "ok", "timestamp": 1733932083363, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="xPIM5ZMay7Yh" outputId="c15f94d6-e02a-4486-dd90-d081b7b87cef"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, TimeDistributed, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam

# Modèle 1 : Bi-LSTM

# Définir le modèle Bi-LSTM
model_lstm = Sequential([
    Embedding(input_dim=len(word2idx) + 1, output_dim=50, input_length=max_len),
    Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    TimeDistributed(Dense(len(tag2idx), activation="softmax"))
])

# Compiler le modèle
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Entraîner le modèle
model_lstm.fit(X_train, np.array(y_train), batch_size=32, epochs=5, validation_split=0.1, verbose=1)


# Prédire sur l'ensemble de test
y_pred_lstm = model_lstm.predict(X_test, verbose=1)
y_pred_lstm = np.argmax(y_pred_lstm, axis=-1)
y_test_lstm = np.argmax(np.array(y_test), axis=-1)

# Convertir les indices en étiquettes
y_pred_lstm_tags = [[idx2tag[idx] for idx in seq] for seq in y_pred_lstm]
y_test_lstm_tags = [[idx2tag[idx] for idx in seq] for seq in y_test_lstm]

# Rapport de classification
print("Bi-LSTM Classification Report:\n", classification_report(y_test_lstm_tags, y_pred_lstm_tags))
```

<!-- #region id="flP0vV9baZT_" -->
### **Modèle CRF**
<!-- #endregion -->

<!-- #region id="yaMKx7f-rEvZ" -->
Input de départ (Text et Label)

    {'text': "Marie est une docteure", 'entities': [{'start': 0, 'end': 5, 'label': "PER"}, {'start': 15, 'end': 22, 'label': "PRO"}]}

Transformé par exemple en IOB

    [(Marie, B-PER), (est, O), (une, O), (docteure, B-PRO)]

X : Liste de dictionnaires représentant chaque mot de la phrase (avec des attributs comme "word").

    Exemple : "Marie est une docteure" devient [{word: "Marie"}, {word: "est"}, {word: "une"}, {word: "docteure"}].

Y : Liste des tags correspondant à chaque mot.

    Exemple : ["O", "O", "O", "B-PER"].
<!-- #endregion -->

```python id="ls6msC8ONUqy"
!pip install sklearn-crfsuite -q
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1733932087355, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="55NZQ_sR0m1s" outputId="73177771-bbf0-4e45-e340-8327538f5f9d"
from sklearn.model_selection import train_test_split

# Extraire les séquences de mots et d'étiquettes
X_crf = [[word for word, tag in sentence] for sentence in df["iob_format"]]
y_crf = [[tag for word, tag in sentence] for sentence in df["iob_format"]]

# Division en ensembles d'entraînement et de test
X_train_crf, X_test_crf, y_train_crf, y_test_crf = train_test_split(X_crf, y_crf, test_size=0.2, random_state=42)

print("X_train CRF Example:", X_train_crf[0])
print("y_train CRF Example:", y_train_crf[0])
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5792, "status": "ok", "timestamp": 1733932093140, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="neXkMQqS1JRc" outputId="3e082f34-6d6b-424a-8a08-2f14344f81cc"
from sklearn_crfsuite import CRF, metrics

# Modèle 1 : CRF
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
crf.fit(X_train_crf, y_train_crf)
y_pred_crf = crf.predict(X_test_crf)
print("CRF Classification Report:\n", classification_report(y_test_crf, y_pred_crf))
```

<!-- #region id="Z9N5NIc-acHs" -->
### **Modèle LSTM + CRF**
<!-- #endregion -->

<!-- #region id="cVScZ81ErZwr" -->
Input de départ (Text et Label)

    {'text': "Marie est une docteure", 'entities': [{'start': 0, 'end': 5, 'label': "PER"}, {'start': 15, 'end': 22, 'label': "PRO"}]}

Transformé par exemple en IOB

    [(Marie, B-PER), (est, O), (une, O), (docteure, B-PRO)]

X : Identique à LSTM (séquences d'indices numériques).

    Exemple : "Marie est une docteure" devient [1, 2, 3, 4].

Y : Séquence de tags entiers (pas de one-hot encoding).

    Exemple : ["O", "O", "O", "B-PER"] devient [0, 0, 0, 1] (où chaque tag est mappé à un entier).

<!-- #endregion -->

```python id="5LWkfZZVDsTE"
!pip install pytorch-crf -q
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 71819, "status": "ok", "timestamp": 1739280284275, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="-a3jzSsiBUGv" outputId="57bf5e80-6b67-4b54-80e2-f7d6d23b2c39"
# Importer les librairies nécessaires
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchcrf import CRF  # pip install pytorch-crf
from sklearn.model_selection import train_test_split  # pour la séparation train/test
from sklearn.metrics import classification_report  # pour afficher le rapport de classification

# Définir le Dataset pour la tâche NER
class NERDataset(Dataset):
    def __init__(self, df, token2idx, tag2idx, column="bilou_format"):
        # Extraire les séquences depuis la colonne indiquée
        self.sentences = []
        self.tags = []
        for seq in df[column]:
            # Chaque séquence est une liste de tuples (token, label)
            tokens, labels = zip(*seq)
            # Convertir les tokens en indices (utiliser <UNK> si absent)
            self.sentences.append([token2idx.get(token, token2idx["<UNK>"]) for token in tokens])
            # Pour les étiquettes, si une entité n'est pas dans tag2idx, utiliser l'indice de "UNK"
            self.tags.append([tag2idx.get(label, tag2idx["UNK"]) for label in labels])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Retourner un couple (séquence d'indices, séquence d'étiquettes)
        return torch.tensor(self.sentences[idx], dtype=torch.long), torch.tensor(self.tags[idx], dtype=torch.long)

# Fonction de padding et création du masque pour le DataLoader
def collate_fn(batch):
    sentences, tags = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    # Padding des séquences de tokens avec l'indice 0 (<PAD>)
    padded_sentences = [torch.cat([s, torch.zeros(max_len - len(s), dtype=torch.long)]) for s in sentences]
    # Padding des étiquettes avec -1 (valeur ignorée par la CRF)
    padded_tags = [torch.cat([t, torch.full((max_len - len(t),), -1, dtype=torch.long)]) for t in tags]
    padded_sentences = torch.stack(padded_sentences)
    padded_tags = torch.stack(padded_tags)
    # Créer un masque : True pour les tokens non padding (<PAD>=0)
    mask = (padded_sentences != 0)
    return padded_sentences, padded_tags, mask

# Construire le vocabulaire à partir des tokens présents dans la colonne "bilou_format" de l'ensemble d'entraînement
def build_vocab(df, column="bilou_format"):
    vocab = set()
    for seq in df[column]:
        tokens, _ = zip(*seq)
        vocab.update(tokens)
    # Réserver l'indice 0 pour <PAD> et 1 pour <UNK>
    vocab = ["<PAD>", "<UNK>"] + sorted(vocab)
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    return token2idx

# Construire les mappings label <-> indice depuis l'ensemble d'entraînement.
# Ajouter l'étiquette "UNK" pour gérer les labels inconnus en test.
def build_tag_mapping(df, column="bilou_format"):
    tags = set()
    for seq in df[column]:
        _, labels = zip(*seq)
        tags.update(labels)
    tags = sorted(list(tags))
    # Ajouter "UNK" si non présent (pour les labels inconnus en test)
    if "UNK" not in tags:
        tags.append("UNK")
    tag2idx = {tag: idx for idx, tag in enumerate(tags)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag

# Définir le modèle NER avec BiLSTM et CRF
class NERCRFModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(NERCRFModel, self).__init__()
        # Couche d'embedding (indice 0 utilisé pour le padding)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Couche BiLSTM bidirectionnelle (hidden_dim divisé par 2 pour chaque direction)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        # Projection de la sortie du LSTM dans l'espace des étiquettes
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # Couche CRF (batch_first=True)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags, mask):
        # Calculer les embeddings
        embeddings = self.embedding(sentences)
        # Passer par le BiLSTM
        lstm_out, _ = self.lstm(embeddings)
        # Calculer les scores d'émission pour chaque token
        emissions = self.hidden2tag(lstm_out)
        # Calculer la perte négative via la CRF
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def predict(self, sentences, mask):
        # Calculer les embeddings
        embeddings = self.embedding(sentences)
        # Passer par le BiLSTM
        lstm_out, _ = self.lstm(embeddings)
        # Calculer les scores d'émission
        emissions = self.hidden2tag(lstm_out)
        # Décoder la meilleure séquence d'étiquettes via la CRF
        predictions = self.crf.decode(emissions, mask=mask)
        return predictions

# Fonction d'entraînement du modèle
def train_model(model, dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for sentences, tags, mask in dataloader:
            sentences, tags, mask = sentences.to(device), tags.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = model(sentences, tags, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

# Fonction d'évaluation du modèle sur un DataLoader donné
def evaluate_model(model, dataloader, device, idx2tag):
    model.eval()
    predictions_all = []
    true_tags_all = []
    with torch.no_grad():
        for sentences, tags, mask in dataloader:
            sentences, tags, mask = sentences.to(device), tags.to(device), mask.to(device)
            predictions = model.predict(sentences, mask)
            # Itérer sur chaque séquence du batch
            for pred, true, m in zip(predictions, tags, mask):
                seq_len = m.sum().item()  # longueur de la séquence sans padding
                pred = pred[:seq_len]
                true = true[:seq_len].tolist()
                predictions_all.extend([idx2tag[p] for p in pred])
                true_tags_all.extend([idx2tag[t] for t in true])
    return true_tags_all, predictions_all

# --------------------------
# Exemple complet d'utilisation
# --------------------------

# Séparer le DataFrame en ensembles d'entraînement et de test (par exemple 80% train - 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Construire le vocabulaire et les mappings d'étiquettes à partir de l'ensemble d'entraînement
token2idx = build_vocab(train_df, column="bilou_format")
tag2idx, idx2tag = build_tag_mapping(train_df, column="bilou_format")

# Créer les Datasets et DataLoaders pour l'entraînement et le test
train_dataset = NERDataset(train_df, token2idx, tag2idx, column="bilou_format")
test_dataset = NERDataset(test_df, token2idx, tag2idx, column="bilou_format")
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Définir le dispositif de calcul (GPU si disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialiser le modèle et l'optimiseur
model = NERCRFModel(vocab_size=len(token2idx), tagset_size=len(tag2idx), embedding_dim=100, hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entraîner le modèle sur l'ensemble d'entraînement
num_epochs = 5
train_model(model, train_dataloader, optimizer, num_epochs, device)

# Évaluer le modèle sur l'ensemble de test
y_true, y_pred = evaluate_model(model, test_dataloader, device, idx2tag)

# Afficher le rapport de classification (classification report)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

```

<!-- #region id="75AXBcQ7O89z" -->
# A foutre ailleur
<!-- #endregion -->

```python id="nvgurUckdPsJ"
import pandas as pd
import json

def merge_entities(entities):
    merged_entities = []

    current_entity = None

    for entity in entities:
        if current_entity is None:
            current_entity = entity.copy()
        elif current_entity['type'] == entity['type'] and current_entity['end'] + 1 >= entity['start'] - 1:
            # Fusionner les entités consécutives du même type
            current_entity['end'] = entity['end']
            current_entity['entity'] = ' '.join([current_entity['entity'], entity['entity']])
        else:
            # Ajouter l'entité fusionnée
            merged_entities.append(current_entity.copy())
            current_entity = entity.copy()

    if current_entity is not None:
        merged_entities.append(current_entity)

    return merged_entities

# Appliquer la fonction sur la colonne 'entities' du DataFrame
df['entities'] = df['entities'].apply(lambda x: merge_entities(json.loads(x)) if pd.notna(x) else [])
```

```python id="qwUeXeCEdPvA"
from IPython.display import display
display(df.loc[[0]].to_string())
```

```python id="egM78DE0do_L"

```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1739281080400, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="e2XcJsUKc6MY" outputId="848891fb-77af-48fe-f1db-9f57bd7ed005"
import re

def split_and_keep_occurrences(text_list, extract):
    """
    Découpe les chaînes dans une liste autour des occurrences de 'extract'
    et insère l'occurrence trouvée elle-même dans la liste.

    Arguments :
    - text_list (list) : Liste de chaînes de texte.
    - extract (str) : La sous-chaîne à rechercher.

    Retourne :
    - list : Liste de chaînes avec les occurrences trouvées insérées.
    """
    result = []
    escaped_extract = re.escape(extract)  # Gérer les caractères spéciaux dans 'extract'

    # Regex pour trouver 'extract' respectant les règles
    pattern = rf"(^|\W)({escaped_extract})(?=$|\W)"

    for line in text_list:
        last_end = 0  # Position de fin de la dernière correspondance
        for match in re.finditer(pattern, line, flags=re.IGNORECASE):
            start, end = match.span(2)  # Début et fin exacts de 'extract'

            # Ajouter la partie avant 'extract' (si elle existe)
            if last_end < start:
                result.append(line[last_end:start])

            # Ajouter l'occurrence trouvée elle-même (groupe 2)
            result.append(match.group(2))
            last_end = end  # Mettre à jour la dernière position traitée

        # Ajouter le reste de la chaîne après la dernière correspondance
        if last_end < len(line):
            result.append(line[last_end:])

    return result

# Exemple de test
text = ["Extract bonjour! Aujourd'hui, extract est ici. Mais_extract, et aussi extract? extracte , l'extract extract."]
extract = "extract"

# Appeler la fonction
result = split_and_keep_occurrences(text, extract)

# Afficher le résultat
print("Texte original :", text)
print("Texte modifié  :", result)

```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1739281067472, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -60} id="PmGFt0hVc99S" outputId="2d6206ab-7b36-4784-9916-69bd2a84c298"
import re

def split_and_keep_occurrences(text, extract):
    """
    Découpe une chaîne autour des occurrences de 'extract' et insère l'occurrence trouvée elle-même dans la liste.

    Arguments :
    - text (str) : Chaîne de texte.
    - extract (str) : La sous-chaîne à rechercher.

    Retourne :
    - list : Liste de chaînes avec les occurrences trouvées insérées.
    """
    result = []
    escaped_extract = re.escape(extract)
    pattern = rf"(^|\W)({escaped_extract})(?=$|\W)"

    last_end = 0
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start, end = match.span(2)
        if last_end < start:
            result.append(text[last_end:start])
        result.append(match.group(2))
        last_end = end
    if last_end < len(text):
        result.append(text[last_end:])

    return result

# Exemple de test avec une seule chaîne
text = "Extract bonjour! Aujourd'hui, extract est ici. Mais_extract, et aussi extract? extracte , l'extract extract."
extract = "extract"

# Appeler la fonction
result = split_and_keep_occurrences(text, extract)

# Afficher le résultat
print("Texte original :", text)
print("Texte modifié  :", result)
```

<!-- #region id="bJLxq7CxbmPm" -->
# Exemple avec modèle pré entrainé
<!-- #endregion -->

<!-- #region id="3wOwDI00Y8TX" -->
Ce code complet, permet de réaliser une tâche de NER en fine-tunant un modèle préentraîné sur des données issues du dataframe. On affichera le F1 score ainsi que le rapport de classification.

- **Préparation des données**
Transformer chaque ligne du dataframe en deux listes (tokens et labels) et utiliser Dataset.from_dict permet d’exploiter les facilités du module datasets.

- **Tokenisation & alignement**
La fonction tokenize_and_align_labels utilise le tokenizer avec is_split_into_words=True. Pour chaque mot, le premier sous-token reçoit le label correspondant et les éventuels sous-tokens suivants sont ignorés (label « -100 ») afin que la loss ne soit pas calculée dessus.

- **Mapping labels**
Construire les dictionnaires label2id et id2label permet de définir la taille de sortie du modèle et d’interpréter les prédictions.

- **Trainer et métriques**
Le Trainer de Hugging Face est configuré pour évaluer à la fin de chaque epoch. La fonction compute_metrics calcule le F1 score avec seqeval. Après l’entraînement, la fonction classification_report de seqeval fournit un résumé détaillé des performances par label.

- **Authentification HF**
Le paramètre use_auth_token=HF_TOKEN permet d’accéder aux modèles privés ou à ceux qui nécessitent une authentification. Remplacer "votre_token_hf" par votre token personnel.


<!-- #endregion -->

```python id="D588McYWfFXL"
# Importer la fonction login depuis huggingface_hub
from huggingface_hub import login

# Insérer directement votre token (il vous sera demandé de le saisir de manière sécurisée)
login("hf_TOKEN_REDACTED_A_REVOQUER")

## OU ##

import os

# Définir la variable d'environnement pour votre token HF
os.environ["HF_HOME"] = "chemin_vers_un_dossier_de_cache"  # optionnel : définir un dossier spécifique
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_TOKEN_REDACTED_A_REVOQUER"
```

```python id="m-nQItITi38-"
!pip install datasets -q
!pip install seqeval -q
!pip install evaluate -q
```

```python id="9fAXRp_saz5e" executionInfo={"status": "ok", "timestamp": 1739368139197, "user_tz": -60, "elapsed": 212, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import warnings
# Ignorer les warnings seqeval sur les tags non standard (U-, L-, etc.)
warnings.filterwarnings("ignore", message=".*seems not to be NE tag.*")
# Ignorer le FutureWarning lié à l'argument tokenizer dans Trainer
warnings.filterwarnings("ignore", message=".*`tokenizer` is deprecated and will be removed in version 5.0.0.*")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 657, "referenced_widgets": ["b7a7362677334d35baa3b95e603f2e3c", "6140a186549c43c88e0fabdb30c80f43", "75a37cd0de5045499c49c50574175cc5", "fd914db91bf64b10b19f52866e9cb52c", "a2640c2acc854e75bd81d79df1c9d8d1", "ce37123f72ae4d859cec77f6e00b6ca4", "8396d0caad374d7da25b9d29268cf1c4", "81b0efd2eab74d0a93bf1b97aa75f68c", "64333c0c1e0f400ca580d5472276ea3e", "1308373abac541aa8ebda9ade71581eb", "b55a0a0f69f343a8b125edd5c3733984", "3f9eee6c9ce1472ca87c3919cbe831f7", "45975267d663447db92e7efb4f0ede08", "706f8db4d0fe43a58d80815a946016de", "49b1e66db26f42369439339b24f0ad27", "6ae617a07bcc4e448165d831062e2b89", "7deeb814d0b14b77be0dd64f33e40deb", "6216988ae89d49af86afea21391c63ff", "a7b5b77ad40a452b98e42648de5b6b4d", "806e87b72d144928928f4231682416c9", "f64a63caeef9479baebe09b4635d7baa", "174899dd960d44ab938d40277319e1d6"]} id="fbYPmwLzDpw3" executionInfo={"status": "ok", "timestamp": 1739369284963, "user_tz": -60, "elapsed": 766171, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="28e00f7c-78d4-4065-c3f8-86de319a826b"
import os
# Désactiver wandb pour éviter son login et ses messages
os.environ["WANDB_DISABLED"] = "true"

import warnings
# Ignorer les warnings seqeval sur les tags non standards (U-, L-, etc.)
warnings.filterwarnings("ignore", message=".*seems not to be NE tag.*")
# Ignorer le FutureWarning lié à l'argument `tokenizer` dans Trainer
warnings.filterwarnings("ignore", message=".*`tokenizer` is deprecated and will be removed in version 5.0.0.*")

import logging
# Masquer les warnings liés aux poids non initialisés de transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from da tasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, TrainingArguments, Trainer)
from seqeval.metrics import f1_score, classification_report

# --- Préparation des données ---

# df est le dataframe existant avec la colonne "bilou_format"
# Exemple d'une ligne : [('5', 'U-QUANTITY'), ('ounces', 'U-UNIT'), ('rum', 'U-FOOD'), ('4', 'U-QUANTITY'), ...]
# Séparer en jeux d'entraînement et d'évaluation
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

def df_to_dataset(dataframe):
    # Transformer le dataframe en Dataset Hugging Face
    tokens_list = []
    ner_tags_list = []
    for _, row in dataframe.iterrows():
        bilou = row["bilou_format"]  # liste de tuples (token, label)
        tokens = [token for token, label in bilou]
        labels = [label for token, label in bilou]
        tokens_list.append(tokens)
        ner_tags_list.append(labels)
    return Dataset.from_dict({"tokens": tokens_list, "ner_tags": ner_tags_list})

train_dataset = df_to_dataset(train_df)
eval_dataset = df_to_dataset(eval_df)

# --- Construction du mapping label2id/id2label ---

# Construire l'union des étiquettes présentes dans train et eval
unique_labels = set()
for tags in train_dataset["ner_tags"]:
    unique_labels.update(tags)
for tags in eval_dataset["ner_tags"]:
    unique_labels.update(tags)
unique_labels = sorted(list(unique_labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# --- Chargement du modèle et du tokenizer ---

# Choisir un modèle léger (ici DistilBERT) et insérer votre token HF
model_checkpoint = "distilbert-base-uncased"
HF_TOKEN = "hf_TOKEN_REDACTED_A_REVOQUER"  # Remplacer par votre token Hugging Face

# Remplacer use_auth_token par token pour éviter le FutureWarning
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, token=HF_TOKEN)

# --- Préparation des inputs pour la token classification ---

def tokenize_and_align_labels(examples):
    # Tokeniser en indiquant que les tokens sont pré-découpés
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Token spécial (CLS, SEP, etc.)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[labels[word_idx]])
            else:
                # Ignorer les sous-tokens additionnels
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

# Appliquer la tokenisation et l'alignement des labels en batch
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# Supprimer les colonnes inutiles et définir le format PyTorch
train_dataset = train_dataset.remove_columns(["tokens", "ner_tags"])
eval_dataset = eval_dataset.remove_columns(["tokens", "ner_tags"])
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# Charger le modèle pour la token classification en utilisant le mapping complet des labels
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    token=HF_TOKEN  # Utiliser token au lieu de use_auth_token
)

# --- Entraînement avec le Trainer ---

# Data collator adapté à la token classification
data_collator = DataCollatorForTokenClassification(tokenizer)

def align_predictions(predictions, label_ids):
    # Transformer les logits en indices de prédictions
    preds = np.argmax(predictions, axis=2)
    out_pred_list = []
    out_label_list = []
    for i, label in enumerate(label_ids):
        pred_labels = []
        true_labels = []
        for j, lab in enumerate(label):
            if lab != -100:
                pred_labels.append(id2label[preds[i][j]])
                true_labels.append(id2label[lab])
        out_pred_list.append(pred_labels)
        out_label_list.append(true_labels)
    return out_pred_list, out_label_list

def compute_metrics(p):
    predictions, labels = p
    preds, refs = align_predictions(predictions, labels)
    f1 = f1_score(refs, preds)
    return {"f1": f1}

# Utiliser eval_strategy à la place de evaluation_strategy et désactiver le reporting (pour éviter wandb)
training_args = TrainingArguments(
    output_dir="./ner_model",
    eval_strategy="epoch",  # Remplace evaluation_strategy (FutureWarning)
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    report_to=[],  # Désactiver wandb et autres reporting
    push_to_hub=False
)

# Initialiser le Trainer (l'argument tokenizer est toujours utilisé, malgré le warning filtré)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Lancer l'entraînement
trainer.train()

# --- Évaluation et affichage du rapport de classification ---

predictions, labels, _ = trainer.predict(eval_dataset)
preds, refs = align_predictions(predictions, labels)
print(classification_report(refs, preds))

```
