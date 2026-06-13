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

<!-- #region id="aRCCU90Bmi5M" -->
# Introduction modèles ARIMA
<!-- #endregion -->

<!-- #region id="20hMjkyPmog-" -->
## Import des bibliothèques et données
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="voM86mWHiNjG" executionInfo={"status": "ok", "timestamp": 1636746745053, "user_tz": -60, "elapsed": 3397, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="2bb18591-1800-405a-aa3a-04b66177d730"
pip install statsmodels --upgrade
```

```python id="D3uYiL3egFr3"
import pandas as pd
import numpy as np

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
```

```python id="76i85XRwiGQd"
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
```

```python colab={"base_uri": "https://localhost:8080/"} id="VRXX9IzlgQSU" executionInfo={"status": "ok", "timestamp": 1636746751119, "user_tz": -60, "elapsed": 1850, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3b06d565-a491-49c5-8bcf-fd27dea44c38"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/Time_Series/"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 225} id="-b-kkAfogQ2b" executionInfo={"status": "ok", "timestamp": 1636746751120, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="e2c1c4a5-4e00-44ea-8bdd-197bb22737d9"
df = pd.read_csv(path+'daily-min-temperatures.csv', header=0, parse_dates=[0])
df.set_index('Date',inplace=True)
df.index.name = None
print(df.shape)
df.head()
```

<!-- #region id="LTWeFI3Zg5oG" -->
1. Chaque pas de temps dans l'ensemble de données de test est itéré.
2. Dans chaque itération, un nouveau modèle ARIMA est entraîné sur toutes les données historiques disponibles.
3. Le modèle est utilisé pour faire une prédiction pour le jour suivant.
4. La prédiction est stockée et l'observation "réelle" est extraite de l'ensemble de test et ajoutée à l'historique pour être utilisée dans l'opération suivante.ajoutée à l'historique pour être utilisée lors de l'itération suivante.
5. Les performances du modèle sont résumées à la fin en calculant la racine de l'erreur quadratique moyenne (RMSE) de la prédiction.
L'erreur quadratique moyenne (RMSE) de toutes les prédictions effectuées par rapport aux valeurs attendues dans l'ensemble de données de test.



<!-- #endregion -->

```python id="-G3R1yrVgadg"
def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat
```

<!-- #region id="2PMrb-IAm7Sv" -->
## Simple
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ufnk7YnTiEYq" executionInfo={"status": "ok", "timestamp": 1636745645482, "user_tz": -60, "elapsed": 2798, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="f4f397af-5219-44e0-f33b-fe875addcf58"
series = df
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,0))
	model_fit = model.fit()
	ar_coef = model_fit.arparams
	yhat = predict(ar_coef, history)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

<!-- #region id="h_3yx_g1ikTg" -->
* Notez que l'implémentation ARIMA modélisera automatiquement une tendance dans la série temporelle. Cela ajoute une constante à l'équation de régression dont nous n'avons pas besoin pour la démonstration. Nous désactivons cette commodité en définissant l'argument 'trend' de la fonction fit() à la valeur 'nc' pour 'no constant'.  
* La fonction fit() produit également un grand nombre de messages verbeux que nous pouvons désactiver en définissant l'argument 'disp' sur False.  
* L'exécution de l'exemple imprime la prédiction et la valeur attendue à chaque itération pendant 7 jours. La valeur RMSE finale est imprimée et montre une erreur moyenne d'environ 1,9 degré Celsius pour ce modèle simple.
<!-- #endregion -->

<!-- #region id="W3c28N9BjMOz" -->
## Modèle Moyenne Mobile (MA)
<!-- #endregion -->

<!-- #region id="qJjdktnvjjIs" -->
Un modèle MA avec un retard de k peut être spécifié dans le modèle ARIMA comme suit :

    - model = ARIMA history, order=(0,0,k))

les coefficients MA du modèle ajusté et que nous les utilisions avec le décalage des valeurs d'erreurs résiduelles et que nous appelions la fonction personnalisée predict() définie ci-dessus.
Les erreurs résiduelles pendant la formation sont stockées dans le modèle ARIMA sous le paramètre 'resid de l'objet ARIMAResults.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="zhdzp5USiJEg" executionInfo={"status": "ok", "timestamp": 1636745649157, "user_tz": -60, "elapsed": 3677, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="efdbf3d1-560f-42ef-fbf3-74389cc5b14f"
series = df
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,0,1))
	model_fit = model.fit()
	ma_coef = model_fit.maparams
	resid = model_fit.resid
	yhat = predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

<!-- #region id="ViLN-VN1kaR3" -->
* 'exécution de l'exemple imprime les prédictions et les valeurs attendues à chaque itération pendant 7 jours et se termine par un résumé de la RMSE de toutes les prédictions.
* La compétence du modèle n'est pas grande et vous pouvez en profiter pour explorer les modèles MA avec d'autres ordres et les utiliser pour faire des prédictions manuelles.

<!-- #endregion -->

<!-- #region id="3s4GS66VkgTI" -->
## Modèle d'autorégression à moyenne mobile (ARMA)
<!-- #endregion -->

<!-- #region id="8oL7png1lBgg" -->
Cette approches peut être directement combinées pour effectuer des prédictions manuelles pour un modèle ARMA plus complet.
Nous allons ajuster un modèle ARMA(1,1) qui peut être configuré dans un modèle ARIMA comme ARIMA(1,0,1) sans différenciation.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yrmR7bsTj9sQ" executionInfo={"status": "ok", "timestamp": 1636745657708, "user_tz": -60, "elapsed": 8554, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1d5d6c89-30d8-4e90-b215-e022332ffb77"
series = df
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,0,1))
	model_fit = model.fit()
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid
	yhat = predict(ar_coef, history) + predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

<!-- #region id="hrYtqO_PlbKo" -->
* La prédiction (yhat) est la somme du produit scalaire des coefficients AR et des observations de décalage et des coefficients MA et des erreurs résiduelles de décalage.

    yhat = predict(ar_coef, history) + predict(ma_coef, resid)  

<!-- #endregion -->

<!-- #region id="hZ4K-v2vl1ry" -->
## Modèle Autoregression Integrated Moving Average (ARIMA)
<!-- #endregion -->

```python id="7-_I8Eq1lO5f"
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return np.array(diff)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BCxEPgWCl_zg" executionInfo={"status": "ok", "timestamp": 1636746770989, "user_tz": -60, "elapsed": 3500, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="d98b6360-d3d7-40a8-9cbe-10f640b4fc3d"
series = df
X = series.values
size = len(X) - 7
train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit()
	ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
	resid = model_fit.resid
	diff = difference(history)
	yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
```

<!-- #region id="JUfQ76HznZmV" -->
## Grille de recherche des meilleurs hyperparamètres
<!-- #endregion -->

```python id="ruhfaBsYntpi"
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in tqdm(p_values, position=2, leave=True):
		for d in tqdm(d_values, position=1, leave=True):
			for q in tqdm(q_values, position=0, leave=True):
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
```

```python id="nCmMaAL7oSAW"
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	#train_size = int(len(X) * 0.95)
    train_size = len(X) - 7
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse
```

```python colab={"base_uri": "https://localhost:8080/"} id="iCg8eJn_oI53" executionInfo={"status": "ok", "timestamp": 1636748632108, "user_tz": -60, "elapsed": 1851785, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6cf7051e-bb2c-4240-f963-c409ac31544e"
# evaluate parameters
series = df
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

<!-- #region id="40snazhNgohr" -->
Best ARIMA(0, 1, 2) MSE=1.148
<!-- #endregion -->

<!-- #region id="wj9Ux2iq1Drx" -->
La méthode de recherche par grille utilisée dans ce tutoriel est simple et peut facilement être étendue.

Cette section énumère quelques idées d'extension de l'approche que vous pourriez souhaiter explorer.

* **Grille d'ensemencement**(Seed Grid). Les outils de diagnostic classiques que sont les graphiques ACF et PACF peuvent toujours être utilisés, les résultats servant à ensemencer la grille des paramètres ARIMA à rechercher.
* **Mesures alternatives**. La recherche cherche à optimiser l'erreur quadratique moyenne hors échantillon. Cette mesure peut être remplacée par une autre statistique hors échantillon, une statistique dans l'échantillon, telle que AIC ou BIC, ou une combinaison des deux. Vous pouvez choisir la métrique qui est la plus significative pour votre projet.
* **Diagnostics résiduels**. Des statistiques peuvent être calculées automatiquement sur les erreurs résiduelles de prévision pour fournir une indication supplémentaire de la qualité de l'ajustement. Les exemples incluent des tests statistiques pour déterminer si la distribution des résidus est gaussienne et s'il existe une autocorrélation dans les résidus.
* **Modèle de mise à jour**. Le modèle ARIMA est créé à partir de zéro pour chaque prévision à une étape. En inspectant attentivement l'API, il peut être possible de mettre à jour les données internes du modèle avec de nouvelles observations plutôt que de le recréer à partir de zéro.
* **Conditions préalables**. Le modèle ARIMA peut faire des hypothèses sur l'ensemble des données de séries chronologiques, telles que la normalité et la stationnalité. Ces hypothèses pourraient être vérifiées et un avertissement pourrait être émis pour un ensemble de données donné avant qu'un modèle donné ne soit entraîné.
<!-- #endregion -->

```python id="JL-tLELaGvyO"
import statsmodels
```

```python id="cnImTQ0pJg_A"
from statsmodels.tsa.arima.model import ARIMA
```

<!-- #region id="WATIpPnMKESD" -->
méthode Box-Jenkins
Acf et Pacf
fonction d'autocorélation c'est la corrélation (Pearson) entre les valeurs d'une series. Par exemple entre X-t et Xt-2 Cette valeur coprend à la fois les effets discret et indiscrets

Paarrtial Autocorélation function  C'est la corélation de Pearson entre les valeurs d'une série isolé de l'impact des autres valeurs de la série . Pour obtenir cette valeur il suffit de créer un modèle linéaire et d'estmer les coefficients
Xt = a1Xt-1 + a2Xt-2 + a3Xt-3 + Vt
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 354} id="FxgYTrbGLlAw" executionInfo={"status": "error", "timestamp": 1679503106545, "user_tz": -60, "elapsed": 337, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2f43beee-f1cc-4a89-c988-cdc053b20938"
# Méthode de Box Jenkins

from statsmodels import Arima
```
