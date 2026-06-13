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

<!-- #region id="RRwQKBnndHlk" -->
#Optuna
<!-- #endregion -->

<!-- #region id="-WD3_bJydTKF" -->
**Optuna est un framework logiciel d'optimisation automatique d'hyperparamètres, particulièrement conçu pour l'apprentissage automatique**

L'optimisation des hyperparamètres parallélisés est un sujet qui apparaît assez fréquemment dans les problèmes et les discussions d'Optuna
<!-- #endregion -->

```python id="gcwiJq4Oiz_l"
pip install optuna
```

```python id="URMN-SsNc4BS" executionInfo={"status": "ok", "timestamp": 1681798409173, "user_tz": -120, "elapsed": 1666, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import optuna
```

<!-- #region id="4kTJ8E6ufrtA" -->
## Principales caractéristiques 
<!-- #endregion -->

<!-- #region id="CmuuPOgxixP1" -->
###1. Architecture légère, polyvalente et indépendante de la plate-forme

Avant de chercher à Optimiser des modèles on prendra l'exemple de d'une simple fonction
<!-- #endregion -->

```python id="DTre9uRCgLLa" executionInfo={"status": "ok", "timestamp": 1681798409173, "user_tz": -120, "elapsed": 9, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2
```

<!-- #region id="7_f1oSqngaX5" -->
- Cette fonction renvoie la valeur de **(x-2)^2**
Notre objectif est de trouver la valeur de x qui minimise la sortie de la fonction objectif. C'est "l'optimisation". Au cours de l'optimisation, Optuna appelle et évalue à plusieurs reprises la fonction objectif avec différentes valeurs de x.

- L'objet **Trial**  correspond à une seule exécution de la fonction objectif et est instancié en interne à chaque invocation de la fonction.

- Les API de suggestion (par exemple, **suggest_float()**) sont appelées dans la fonction d'objectif pour obtenir les paramètres d'un essai. suggest_float() sélectionne uniformément les paramètres dans la plage fournie. Dans notre exemple, de -10 à 10.

- Pour démarrer l'optimisation, nous créons un objet d'étude et passons la fonction objectif à la méthode **optimize()** :
<!-- #endregion -->

```python id="HS0U-MWogQgr"
study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

<!-- #region id="VaNWr1B-hnEd" -->
**On peut obtenir le meilleur paramètre: **
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 318, "status": "ok", "timestamp": 1681798446606, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="FpKKCPrXhb0w" outputId="3b8df953-6440-4c96-b439-0a1a85bcef0b"
best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1681798411008, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="an2M42nIhmUI" outputId="a3eecfba-96ad-427e-ecef-d6867e6cd36b"
study.best_params
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 644, "status": "ok", "timestamp": 1681798411647, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="Qy8sYyADiLHa" outputId="d95c6dce-9ef7-482b-978e-618eb8806392"
study.best_value
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 226, "status": "ok", "timestamp": 1681798489131, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="-uco3nk9iQBi" outputId="ba2548f6-ecea-4a33-8a1f-7e565b2c285e"
study.best_trial
```

<!-- #region id="mOUdbcnLiWO7" -->
* nombre d'essais :
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1681798491048, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="bNlsLYkqiSMR" outputId="64e49b1a-94d9-4fd4-f898-381c5c587dcd"
len(study.trials)
```

<!-- #region id="Thpj5AR9i-Pa" -->
###2. Espace de recherche pythonique
<!-- #endregion -->

<!-- #region id="BBnctssYjD3B" -->
Pour l'échantillonnage d'hyperparamètres, Optuna fournit les fonctionnalités suivantes :

* *optuna.trial.Trial.suggest_categorical()* pour les paramètres catégoriels

* *optuna.trial.Trial.suggest_int()* pour les paramètres entiers

* *optuna.trial.Trial.suggest_float()* pour les paramètres à virgule flottante
<!-- #endregion -->

<!-- #region id="tr1cD1ttkC-2" -->
Voici les possibilité de parcours des paramètres proposé par Optuna 
(il propose des arguments optionnels de **step** (discrétiser) et **log** (passage au log)
<!-- #endregion -->

```python id="oQ2Td9dFiaKx"
def objective(trial):
    # Paramètre catégorique
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Paramètre entier
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Paramètre entier (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Paramètre entier (discrétisé)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Paramètre à virgule flottante
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Paramètre à virgule flottante (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Paramètre à virgule flottante (discrétisé)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
```

<!-- #region id="Pc-ld5PZm-2H" -->
* **Exemple simple sur des modèles en Branches**
<!-- #endregion -->

```python id="vlw_aDJGktw7"
import sklearn.ensemble
import sklearn.svm

def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)
```

<!-- #region id="XcHOFP7nnBcc" -->
* **Exemple simple sur une Boucle**
<!-- #endregion -->

```python id="dADHNqadlYQC"
import torch
import torch.nn as nn

def create_model(trial, in_size):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)
```

<!-- #region id="XbAG4zmvqVro" -->
##3. Algorithmes d'optimisation efficaces
<!-- #endregion -->

<!-- #region id="smSNLTFJqahF" -->
**Algorithmes d'échantillonnage**

Les échantillonneurs réduisent en permanence l'espace de recherche en utilisant les enregistrements des valeurs de paramètre suggérées et des valeurs objectives évaluées, ce qui conduit à un espace de recherche optimal qui donne des paramètres conduisant à de meilleures valeurs objectives. Une explication plus détaillée de la façon dont les échantillonneurs suggèrent des paramètres se trouve dans BaseSampler.

Optuna fournit les algorithmes d'échantillonnage suivants :

* Grid Search implémenté dans *GridSampler*
* Recherche aléatoire implémentée dans *RandomSampler*
* Algorithme arborescent Parzen Estimator implémenté dans *TPESampler*
* Algorithme basé sur CMA-ES implémenté dans *CmaEsSampler*
* Algorithme permettant d'activer des paramètres fixes partiels implémentés dans *PartialFixedSampler*
* Algorithme génétique de tri non dominé II mis en œuvre dans *NSGAIISampler*
* Un algorithme d'échantillonnage Quasi Monte Carlo implémenté dans *QMCSampler*
* L'échantillonneur par défaut est *TPESampler*
<!-- #endregion -->

<!-- #region id="mKnZVU_LrKwF" -->
choisir un échantilloneur
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 225, "status": "ok", "timestamp": 1681742703814, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="UM4IZd1knRt5" outputId="890f66ee-bf24-4fa3-941d-b002ef0fd04a"
study = optuna.create_study()
print(f"Sampler is {study.sampler.__class__.__name__}")
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 220, "status": "ok", "timestamp": 1681742755080, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="tytj0QlArSoa" outputId="b5b3036a-44f3-4653-dd40-3346022d6150"
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")

study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")
```

<!-- #region id="tdbDCaRpsGXy" -->
Algorithmes d'élagage
**Prunersarrêter** automatiquement les essais peu prometteurs aux premiers stades de la formation (c'est-à-dire l'arrêt précoce automatisé).

Optuna fournit les algorithmes d'élagage suivants :

Algorithme d'élagage médian implémenté dans **MedianPruner**

Algorithme sans élagage implémenté dans **NopPruner**

Algorithme pour faire fonctionner le sécateur avec tolérance implémenté dans **PatientPruner**

Algorithme pour élaguer le centile spécifié des essais mis en œuvre dans **PercentilePruner**

Algorithme Asynchronous Successive Halving implémenté dans **SuccessiveHalvingPruner**

Algorithme hyperbande implémenté dans **HyperbandPruner**

Algorithme d'élagage au seuil implémenté dans **ThresholdPruner**


**SuccessiveHalvingPruneret** et **HyperbandPrunercomme** sont à privilégier.
<!-- #endregion -->

<!-- #region id="3fzRVsyCt-BM" -->
## Exemple pour Scikit-Learn
<!-- #endregion -->

<!-- #region id="i6yRQt9YuHrf" -->
Vous pouvez optimiser les hyperparamètres de Scikit-Learn, tels que le paramètre C du SVC et la profondeur maximale du RandomForestClassifier, en trois étapes :

1. Envelopper l'apprentissage du modèle avec une fonction **objective** et renvoyer la précision.
2. Suggérer des hyperparamètres à l'aide d'un objet **trial**
3. Créer un objet **study** et exécuter l'optimisation
<!-- #endregion -->

```python id="OMVSsMuVuF_S"
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

iris = sklearn.datasets.load_iris()
X, y = iris.data, iris.target

# 1. Définir une fonction objective à maximiser.

def objective(trial):
    # 2. Proposer des valeurs pour les hyperparamètres à l'aide d'un objet d'essai.
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])

    if classifier_name == "SVC":
        # Classifier PARAMS
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        # RandomForest PARAMS
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )
    # Evaluation du score
    score = sklearn.model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy

# 3. Créez un objet d'étude et optimisez la fonction objectif.
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 228, "status": "ok", "timestamp": 1681799312354, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="3l0_ESMxupdX" outputId="2ece93b0-c639-47ca-9071-ed1a1afa3bc4"
print("best_value :" , study.best_value)
print("best_trial :" , study.best_trial.params)
```

<!-- #region id="AGL45rsmxPlJ" -->
## Exemple pour XGBoost
<!-- #endregion -->

```python id="E5ywpSwuu63W"
import numpy as np

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

(data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x, label=valid_y)

def objective(trial):
    # A. PARAMS en commun
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # utiliser exact pour les petits ensembles de données.
        "tree_method": "exact",
        # définit booster, gblinear pour les fonctions linéaires.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # Poids de régularisation L2.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # Poids de régularisation L1.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # taux d'échantillonnage pour les données d'apprentissage.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # échantillonnage en fonction de chaque arbre.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }
    # B. PARAMS selon l'amplificateur
    if param["booster"] in ["gbtree", "dart"]:
        # profondeur maximale de l'arbre, signifie la complexité de l'arbre.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # poids minimum de l'enfant, plus le terme est grand, plus l'arbre est conservateur.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # définit le degré de sélectivité de l'algorithme.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Evaluation du score
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)
```

```python colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 249, "status": "ok", "timestamp": 1681800403074, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}, "user_tz": -120} id="g-7CB21CxgwC" outputId="bf9fc3b4-86e9-4d4c-894d-7dabba7d3762"
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

<!-- #region id="XT8H4Tzm4goi" -->
## Exemple pour Catboost
<!-- #endregion -->

```python id="eSHgIYeJ5FL3"
pip install catboost
```

```python id="PEwZj2o2x0tV"
import numpy as np
import optuna

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data, target = load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.3)

def objective(trial):
    # A. PARAMS en commun
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }
    # B. PARAMS selon l'amorçage 
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)
```

```python id="UFLsU3e4491B" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1681806265849, "user_tz": -120, "elapsed": 888, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c3be97d0-519d-4b49-c757-a108b9659f06"
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

<!-- #region id="nqMku9Oqd1qf" -->
##Exemple LightGBM
<!-- #endregion -->

```python id="jWUfXeMgv4HP" executionInfo={"status": "ok", "timestamp": 1681811135733, "user_tz": -120, "elapsed": 269, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
```

```python colab={"base_uri": "https://localhost:8080/"} id="ynOyJE9MNDBr" executionInfo={"status": "ok", "timestamp": 1681811686261, "user_tz": -120, "elapsed": 8912, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c100e5e6-6f3a-48f2-e8dc-bfbd2590c3be"
data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
dtrain = lgb.Dataset(train_x, label=train_y)

def objective(trial):

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    dtrain = lgb.Dataset(train_x, label=train_y)
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

```python colab={"base_uri": "https://localhost:8080/"} id="R-QTtuDceK2-" executionInfo={"status": "ok", "timestamp": 1681811713686, "user_tz": -120, "elapsed": 230, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d720dafe-cef1-457a-af34-e66ff2a06f17"
print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
```

<!-- #region id="WFX-AYfn2L_v" -->
## Exemple Keras

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GbMJXZf3wPSp" executionInfo={"status": "ok", "timestamp": 1681830177065, "user_tz": -120, "elapsed": 629312, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7ba3652c-7965-420e-f43a-ec17659f880b"
import urllib
import warnings

import optuna

from keras.backend import clear_session
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


# TODO(crcrpar): Remove the below three lines once everything is ok.
# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype("float32") / 255
    x_valid = x_valid.reshape(-1, img_x, img_y, 1)[:N_VALID_EXAMPLES].astype("float32") / 255
    y_train = y_train[:N_TRAIN_EXAMPLES]
    y_valid = y_valid[:N_VALID_EXAMPLES]
    input_shape = (img_x, img_y, 1)

    model = Sequential()
    model.add(
        Conv2D(
            filters=trial.suggest_categorical("filters", [32, 64]),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5]),
            strides=trial.suggest_categorical("strides", [1, 2]),
            activation=trial.suggest_categorical("activation", ["relu", "linear"]),
            input_shape=input_shape,
        )
    )
    model.add(Flatten())
    model.add(Dense(CLASSES, activation="softmax"))

    # We compile our model with a sampled learning rate.
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        shuffle=True,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        verbose=False,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score[1]


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF:https://github.com/keras-team/keras/releases/tag/2.4.0"
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```

```python colab={"base_uri": "https://localhost:8080/"} id="0eSZ-r6b2S8q" executionInfo={"status": "ok", "timestamp": 1681830241937, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="07e19363-ed6b-45ab-b63d-0274b6393a59"
trial.params
```

```python id="Isk0hLKh5OSI"

```
