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

```python id="w8R3q-HYOKGJ" executionInfo={"status": "ok", "timestamp": 1704461607101, "user_tz": -60, "elapsed": 2188, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score
```

```python colab={"base_uri": "https://localhost:8080/"} id="exe77y_iNVqj" executionInfo={"status": "ok", "timestamp": 1704461608879, "user_tz": -60, "elapsed": 1781, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="783295e0-7670-4110-9904-4ba81cf6908d"
# Charger le jeu de données Boston Housing
boston_data = fetch_openml(name="boston", version=2)

# Créer un DataFrame à partir des données
X, y = pd.DataFrame(boston_data.data, columns=boston_data.feature_names) , boston_data.target
```

```python colab={"base_uri": "https://localhost:8080/", "height": 473} id="kn6yWZ-eFsXa" executionInfo={"status": "ok", "timestamp": 1704461839941, "user_tz": -60, "elapsed": 32954, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4befa1d8-f4df-4b0b-a21c-3a055a6e13ad"
# Listes pour stocker les résultats
percentages_missing = list(range(0, 31, 2))
accuracy_approach1_list = []
accuracy_approach2_list = []

# Définition du nombre de partitions (4 dans ce cas)
num_partitions = 4
kf = KFold(n_splits=num_partitions, shuffle=True, random_state=42)

# Boucle sur les différents pourcentages de données manquantes
for percent_missing in percentages_missing:
    # Initialisation des listes pour stocker les résultats de chaque partition
    accuracy_approach1_cv = []
    accuracy_approach2_cv = []

    # Boucle sur les différentes partitions
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Création de quelques valeurs manquantes artificielles
        X_train_missing = X_train.copy()
        X_test_missing = X_test.copy()

        for col in X.columns:
            missing_mask_train = np.random.rand(X_train.shape[0]) < (percent_missing / 100)
            missing_mask_test = np.random.rand(X_test.shape[0]) < (percent_missing / 100)

            X_train_missing.loc[missing_mask_train, col] = np.nan
            X_test_missing.loc[missing_mask_test, col] = np.nan

        # Approche 1 : Modèle qui gère les données manquantes
        model_approach1 = RandomForestClassifier(random_state=42)
        imputer_approach1 = SimpleImputer(strategy='mean')
        X_train_approach1 = imputer_approach1.fit_transform(X_train_missing)
        X_test_approach1 = imputer_approach1.transform(X_test_missing)
        model_approach1.fit(X_train_approach1, y_train)
        accuracy_approach1 = accuracy_score(y_test, model_approach1.predict(X_test_approach1))
        accuracy_approach1_cv.append(accuracy_approach1)

        # Approche 2 : Imputation avec k-NN + Modèle qui gère les données manquantes
        model_approach2 = RandomForestClassifier(random_state=42)
        imputer_approach2 = KNNImputer(n_neighbors=3)
        X_train_approach2 = imputer_approach2.fit_transform(X_train_missing)
        X_test_approach2 = imputer_approach2.transform(X_test_missing)
        model_approach2.fit(X_train_approach2, y_train)
        accuracy_approach2 = accuracy_score(y_test, model_approach2.predict(X_test_approach2))
        accuracy_approach2_cv.append(accuracy_approach2)

    # Calcul de la moyenne des précisions pour chaque approche
    mean_accuracy_approach1 = np.mean(accuracy_approach1_cv)
    mean_accuracy_approach2 = np.mean(accuracy_approach2_cv)

    # Ajout des résultats à la liste
    accuracy_approach1_list.append(mean_accuracy_approach1)
    accuracy_approach2_list.append(mean_accuracy_approach2)

# Tracé du graphique
plt.plot(percentages_missing, accuracy_approach1_list, label='Modèle seul')
plt.plot(percentages_missing, accuracy_approach2_list, label='Knn imput + Modèle')
plt.xlabel('Pourcentage de données manquantes')
plt.ylabel('Précision moyenne (Validation croisée)')
plt.title('Comparaison des approches Modèle seul et Knn imput + Modèle avec validation croisée')
plt.legend()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 473} id="Hwr3scxyG2-r" executionInfo={"status": "ok", "timestamp": 1704461793883, "user_tz": -60, "elapsed": 24203, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="117a10f1-62b6-4c78-bd96-946595133a49"
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Convertir les classes en valeurs numériques
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Convertir les variables catégorielles en numériques
categorical_cols = X.select_dtypes(include=['category']).columns
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Listes pour stocker les résultats
percentages_missing = list(range(0, 31, 2))
accuracy_approach1_list = []
accuracy_approach2_list = []

# Définition du nombre de partitions (4 dans ce cas)
num_partitions = 4
kf = KFold(n_splits=num_partitions, shuffle=True, random_state=42)

# Boucle sur les différents pourcentages de données manquantes
for percent_missing in percentages_missing:
    # Initialisation des listes pour stocker les résultats de chaque partition
    accuracy_approach1_cv = []
    accuracy_approach2_cv = []

    # Boucle sur les différentes partitions
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_numeric[train_index], y_numeric[test_index]

        # Création de quelques valeurs manquantes artificielles
        X_train_missing = X_train.copy()
        X_test_missing = X_test.copy()

        for col in X.columns:
            missing_mask_train = np.random.rand(X_train.shape[0]) < (percent_missing / 100)
            missing_mask_test = np.random.rand(X_test.shape[0]) < (percent_missing / 100)

            X_train_missing.loc[missing_mask_train, col] = np.nan
            X_test_missing.loc[missing_mask_test, col] = np.nan

        # Approche 1 : Modèle qui gère les données manquantes
        model_approach1 = XGBClassifier(random_state=42, enable_categorical=True)
        model_approach1.fit(X_train_missing, y_train)
        accuracy_approach1 = accuracy_score(y_test, model_approach1.predict(X_test_missing))
        accuracy_approach1_cv.append(accuracy_approach1)

        # Approche 2 : Imputation avec k-NN + Modèle qui gère les données manquantes
        model_approach2 = XGBClassifier(random_state=42, enable_categorical=True)
        imputer_approach2 = KNNImputer(n_neighbors=5)
        X_train_approach2 = imputer_approach2.fit_transform(X_train_missing)
        X_test_approach2 = imputer_approach2.transform(X_test_missing)
        model_approach2.fit(X_train_approach2, y_train)
        accuracy_approach2 = accuracy_score(y_test, model_approach2.predict(X_test_approach2))
        accuracy_approach2_cv.append(accuracy_approach2)

    # Calcul de la moyenne des précisions pour chaque approche
    mean_accuracy_approach1 = np.mean(accuracy_approach1_cv)
    mean_accuracy_approach2 = np.mean(accuracy_approach2_cv)

    # Ajout des résultats à la liste
    accuracy_approach1_list.append(mean_accuracy_approach1)
    accuracy_approach2_list.append(mean_accuracy_approach2)

# Tracé du graphique
plt.plot(percentages_missing, accuracy_approach1_list, label='Modèle seul (XGBClassifier)')
plt.plot(percentages_missing, accuracy_approach2_list, label='Knn imput + Modèle (XGBClassifier)')
plt.xlabel('Pourcentage de données manquantes')
plt.ylabel('Précision moyenne (Validation croisée)')
plt.title('Comparaison des approches Modèle seul et Knn imput + Modèle avec validation croisée (XGBClassifier)')
plt.legend()
plt.show()
```

```python id="norGYhIO1AmN"

```
