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

<!-- #region id="DBfwLahOPheK" -->
**connexion au server MLFlow**
<!-- #endregion -->

```python id="6lQu7l4SwtKX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1715873110169, "user_tz": -120, "elapsed": 29709, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ad39b82a-3158-4d82-accf-fc0a7da216a8"
pip install mlflow
```

```python id="RrtQ4JT-PWOQ" executionInfo={"status": "ok", "timestamp": 1715873111501, "user_tz": -120, "elapsed": 1335, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import mlflow
tracking_uri = "http://"+ "admin" +":"+"password"+"@192.168.10.90:5000"
mlflow.set_tracking_uri(tracking_uri)
```

<!-- #region id="w4PEYxocPpEd" -->
**Import des Library**
<!-- #endregion -->

```python id="YKnRC8yhPfYD" executionInfo={"status": "ok", "timestamp": 1715873112330, "user_tz": -120, "elapsed": 832, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# MLFlow
import mlflow
import mlflow.sklearn

# Base
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Modèles
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

#Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

```python id="E9fqVNoxyOHW" executionInfo={"status": "ok", "timestamp": 1715873112332, "user_tz": -120, "elapsed": 6, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
```

<!-- #region id="nT7z-Xd9QXcZ" -->
**Création du Projet**
<!-- #endregion -->

```python id="HEDOHy5wQoDR"
# Crée une expérience si elle n'existe pas
mlflow.set_experiment("Iris")
```

<!-- #region id="hikyVQ89QMsC" -->
**Chargement des Données**
<!-- #endregion -->

```python id="OS2Rsu5sQQUy"
# Charger le jeu de données Iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df= pd.DataFrame(pd.concat([X,y]))
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train, X_test  = X.train_test_split( test_size = 0.3 , random_state=42  )
```

<!-- #region id="pXR5ZyTGQvUZ" -->
**Lier le Dataset**
<!-- #endregion -->

```python id="QKfYRis8Qx9x"
lien = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/iris.csv"
dataset: PandasDataset = mlflow.data.from_pandas(df, source=lien)
```

```python id="qVVE440mwmv_"
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer , mean_squared_error
from sklearn.metrics import confusion_matrix
```

```python colab={"base_uri": "https://localhost:8080/"} id="DHQ7OJJiwm2g" executionInfo={"status": "ok", "timestamp": 1698675689734, "user_tz": -60, "elapsed": 388, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="27cebbd6-ea29-4a4e-b80e-19b9b3d38d0d"
cv_results = cross_validate(LinearRegression(), X_train, y_train, cv=4,  scoring='neg_mean_squared_error', return_train_score=True, return_estimator=True)
sorted(cv_results.keys())
```

```python colab={"base_uri": "https://localhost:8080/"} id="QbUo3pQk0Sh3" executionInfo={"status": "ok", "timestamp": 1698675692424, "user_tz": -60, "elapsed": 314, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f7b278e5-90b8-4422-eeb8-a66976003519"
-cv_results['train_score']
```

```python colab={"base_uri": "https://localhost:8080/"} id="v3g-b7fR7pgl" executionInfo={"status": "ok", "timestamp": 1698675696313, "user_tz": -60, "elapsed": 272, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f4e60501-5b41-4c4b-9579-ae0cf479e4d6"
-cv_results['test_score']
```

```python colab={"base_uri": "https://localhost:8080/"} id="TgOTkHBcwm9K" executionInfo={"status": "ok", "timestamp": 1698675211309, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4776a031-b7b8-458a-881f-20c22d537f24"
neg_mse_scorer = get_scorer(mean_squared_error)
np.array([neg_mse_scorer(y_test,  cv_results['estimator'][i].predict(X_test) ) for i in  range(len(cv_results['estimator']))])
```

<!-- #region id="k6TnKkKhR-z_" -->

<!-- #endregion -->

<!-- #region id="9XdXdo23R-6P" -->
**Définition des modèles de Benchs**
<!-- #endregion -->

```python id="EhFjHNu5STEp"
# Liste des modèles à entraîner avec différentes valeurs de n_neighbors
n_neighbors_values = [2, 5, 10]

    # Dictionnaire de modèles
models = {
    'LinearRegression': LinearRegression(),
    'KNeighborsRegressor' : KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor()
}
parameters = {
    'KNeighborsRegressor': {'n_neighbors': [2,5,10]},
    'DecisionTreeRegressor': {'max_leaf_nodes': [5,10,15] }
}
```

```python id="1Ob69k9SugZS"
for model_type, model in models.items():

    for param in parameters[model_type]
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", model_type)
            mlflow.log_input(dataset, context="iris_data")
            mlflow.set_tag("model_name", model.__class__.__name__)
            if

            mlflow.log_params(model.get_params())

            # Création, entraînement et évaluation du modèle sur l'ensemble de test
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            # Calcul des métriques
            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            # Enregistrement des métriques pour le modèle sur l'ensemble de test
            metrics = {
                'MSE_test' : mse_test,
                'MAE_test' : mae_test,
                'R2_test' : r2_test
            }
            mlflow.log_metrics(metrics)

            # Évaluation du modèle sur l'ensemble d'entraînement
            y_pred_train = model.predict(X_train)

            # Calcul des métriques pour l'ensemble d'entraînement
            mse_train = mean_squared_error(y_train, y_pred_train)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)

            # Enregistrement des métriques pour le modèle sur l'ensemble d'entraînement
            metrics = {
                'MSE_train' : mse_train,
                'MAE_train' : mae_train,
                'R2_train' : r2_train
            }
            mlflow.log_metrics(metrics)
            # Enregistrement du modèle
            mlflow.sklearn.log_model(model,  'model_' + model_type)

            # Affichage de l'URI du modèle enregistré
            model_uri = mlflow.get_artifact_uri('model_' + model_type )

            #print(f"{model_type} Model saved in: {model_uri}")
            print("model_type1", model_type ,"Done")
        mlflow.end_run()

```

```python id="EFFgU1MxR0Fv"
# Liste des modèles à entraîner avec différentes valeurs de n_neighbors
n_neighbors_values = [2, 5, 10]

    # Dictionnaire de modèles
models_base = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
}
models_knn = {"KNeighborsRegressor_"+str(element): KNeighborsRegressor(n_neighbors=element) for element in [2,5,10]}

models = {**models_base, **models_knn}
```

```python id="EEFDZVIhRNjB"
for model_type, model in models.items():
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", model_type)
        mlflow.log_input(dataset, context="iris_data")
        # Enregistrement de paramètres
        mlflow.set_tag("model_name", model.__class__.__name__)        # training)
        mlflow.log_params(model.get_params())

        # Création, entraînement et évaluation du modèle sur l'ensemble de test
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        # Calcul des métriques
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        # Enregistrement des métriques pour le modèle sur l'ensemble de test
        metrics = {
            'MSE_test' : mse_test,
            'MAE_test' : mae_test,
            'R2_test' : r2_test
        }
        mlflow.log_metrics(metrics)

        # Évaluation du modèle sur l'ensemble d'entraînement
        y_pred_train = model.predict(X_train)

        # Calcul des métriques pour l'ensemble d'entraînement
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        # Enregistrement des métriques pour le modèle sur l'ensemble d'entraînement
        metrics = {
            'MSE_train' : mse_train,
            'MAE_train' : mae_train,
            'R2_train' : r2_train
        }
        mlflow.log_metrics(metrics)
        # Enregistrement du modèle
        mlflow.sklearn.log_model(model,  'model_' + model_type)

        # Affichage de l'URI du modèle enregistré
        model_uri = mlflow.get_artifact_uri('model_' + model_type )

        #print(f"{model_type} Model saved in: {model_uri}")
        print("model_type1", model_type ,"Done")
    mlflow.end_run()

```
