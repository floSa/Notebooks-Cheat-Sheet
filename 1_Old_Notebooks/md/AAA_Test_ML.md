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

```python id="rwrHPSKtjEQD" executionInfo={"status": "ok", "timestamp": 1657049558046, "user_tz": -120, "elapsed": 1721, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="51033a3c-faea-4ddb-9f59-cb4b2d82ee95" colab={"base_uri": "https://localhost:8080/"}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns


titanic = sns.load_dataset('titanic')
y = titanic['survived']
X = titanic.drop('survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Selection des colonnes numériques / catégorielles
numerical_features = ['pclass', 'age', 'fare']
categorical_features = ['sex', 'deck', 'alone']

# Création de pipelines pour traiter chaque type de colonnes
numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

# Assemblage des pipeline dans un ColumnTransformer
preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                   (categorical_pipeline, categorical_features))

# Création d'une pipeline globale qui assemble le preprocessor avec un estimateur
model = make_pipeline(preprocessor, SGDClassifier())

# Etrainement et évaluation
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

```python id="gr4FJK96jFFd"

```
