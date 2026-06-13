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

```python id="qLQvDnrfK9J9"
import numpy as np
import pandas as pd
import csv
```

```python colab={"base_uri": "https://localhost:8080/"} id="HNXS6rPEK9oU" executionInfo={"status": "ok", "timestamp": 1636402618471, "user_tz": -60, "elapsed": 12006, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1507f72a-3a8c-4bc0-83fa-6a8e65f7a456"
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='all')
```

```python id="pp-DW9onLACC"
df = pd.DataFrame(newsgroups.data, columns=['text'])
df['categories'] = [newsgroups.target_names[index] for index in newsgroups.target]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="X6i7XdFyLBUC" executionInfo={"status": "ok", "timestamp": 1626084662399, "user_tz": -120, "elapsed": 1961, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="c9b9ffb6-93d9-49bb-cefd-0817708a8756"
"""# Convert multiple whitespace characters into a space
df['text'] = df['text'].str.replace('\s+',' ')
# Change newsgroup titles to use underscores rather than periods
df['categories'] = df['categories'].str.replace('.','_')
# Trim leading and tailing whitespace
df['text'] = df['text'].str.strip()
# Truncate all fields to the maximum field length of 128kB
df['text'] = df['text'].str.slice(0,131072)
# Remove any rows with empty fields
df = df.replace('', np.NaN).dropna()
# Drop duplicates
df = df.drop_duplicates(subset='text')
# Limit rows to maximum of 100,000
df = df.sample(min(100000, len(df)))
df.head()"""
```

```python colab={"base_uri": "https://localhost:8080/", "height": 366} id="k-0kw2mdMuBY" executionInfo={"status": "ok", "timestamp": 1636402618964, "user_tz": -60, "elapsed": 496, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="13d02697-1dca-4c36-b8d7-c9fa02dd1734"
orders = list(df['categories'].value_counts().index)
import seaborn as sns
import matplotlib.pyplot as plt
order = list(df['categories'].value_counts().index)
plt.figure(figsize = (25, 5))

graph  = sns.countplot(x ="categories", data = df ,order=orders)
plt.xticks(rotation=-20)
plt.show()
```

<!-- #region id="TdsiwJZ0M8Sp" -->
#Vectorisation en TF-IDF
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CF08MuTKM9Yg" executionInfo={"status": "ok", "timestamp": 1636402621497, "user_tz": -60, "elapsed": 1687, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="8149f39a-513a-4b11-d130-e4efaeeae115"
 import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
```

```python id="CHXY-qyeM_F9"
stop_words = set(stopwords.words('english'))

# Suppriment les affixes morphologiques des mots, ne laissant que le radical du mot.
stemmer = PorterStemmer()

def remove_SW_Stem(text):
    text=[stemmer.stem(words) for words in text.split(" ") if words not in stop_words]
    return " ".join(text)

special_chars = re.compile('[^0-9a-z#+_]')
add_space = re.compile('[/(){}\[\]\\@;]')

def clean_text(text):
    text=text.lower()
    text = add_space.sub(" ",text)
    text = special_chars.sub(" ",text)
    text = remove_SW_Stem(text)
    return text
    
df['text'] = df['text'].apply(lambda text:clean_text(text))
```

```python id="vrm1dF0ELpRX"
#Vectorisation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
```

```python colab={"base_uri": "https://localhost:8080/"} id="9Fw85nHnLGKC" executionInfo={"status": "ok", "timestamp": 1636402713340, "user_tz": -60, "elapsed": 6752, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6e698e9a-8955-419b-c986-920045661fb3"
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english').fit(df["text"].values)
print("NB features: %d" %(len(tfidf.vocabulary_)))
features = tfidf.transform(df["text"].values)
#X_test = transformer.transform(test_df["description_lower"].values)
labels = df.categories
```

```python id="WU3FxysvOF1I"
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['categories'] ,test_size=0.25 , random_state = 0)
```

```python id="-WiMcaC2LJi5"
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
```

```python id="9T81fftpLN7K"
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
```

<!-- #region id="IbueJkSMPHYm" -->
## Entrainement des modeles 
<!-- #endregion -->

```python id="ufoKjzFyLsqy"
models = [
    SGDClassifier(max_iter=500, tol=1e-3),
    #RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 3
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='f1_macro', cv=CV)
  for fold_idx, f1_macro in enumerate(accuracies):
    entries.append((model_name, fold_idx, f1_macro))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1_macro'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 606} id="0nh9aU5WLs8r" executionInfo={"status": "ok", "timestamp": 1636402962767, "user_tz": -60, "elapsed": 724, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6b1ef148-2db6-4fad-8205-0df57ac0ddd8"
import seaborn as sns
plt.figure(figsize=(20, 10))
sns.boxplot(x='model_name', y='f1_macro', data=cv_df)
sns.stripplot(x='model_name', y='f1_macro', data=cv_df, 
              size=12, jitter=True, edgecolor="gray", linewidth=2   )
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="f6IGvLj_RMiS" executionInfo={"status": "ok", "timestamp": 1636402975659, "user_tz": -60, "elapsed": 322, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="79c56b3d-0957-46f5-e922-f1ec6bc6b589"
cv_df.groupby('model_name').f1_macro.mean()
```

```python id="IG0RBsVBQ6hT" executionInfo={"status": "ok", "timestamp": 1636403756243, "user_tz": -60, "elapsed": 312, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6c183a26-a510-4319-e01b-7a9f02a7b027" colab={"base_uri": "https://localhost:8080/", "height": 206}
df.head()
```

```python id="QRwXrCjcRwtP"
category_id_df = df[['categories_name', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'categories_name']].values)
#Dict
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'categories_name']].values)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ul4Jcf5dRX8a" executionInfo={"status": "ok", "timestamp": 1636403635044, "user_tz": -60, "elapsed": 3997, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="20214e97-6692-4bf5-e964-b250e0a6c13b"
model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=df.categories.unique(), yticklabels=df.categories.unique(), vmin=0, vmax=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="nmQsap2SRadi" executionInfo={"status": "ok", "timestamp": 1636403661626, "user_tz": -60, "elapsed": 318, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="61afc8ee-396a-4b27-c67d-6e0e0a974c91"
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['categories'].unique()))
```

```python id="kyQ6Y1wtQkit"

```
