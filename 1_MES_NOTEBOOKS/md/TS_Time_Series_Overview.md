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
    language: python
    name: python3
---

<!-- #region id="y_4zsLsM_y97" -->
##Import des Bibliothèque Chargement de données
<!-- #endregion -->

```python id="M3VZ_4oa_Vby"
pip install statsmodels --upgrade
```

```python id="SkEGTSQ4_y9_"
import pandas as pd
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/"} id="HuO29GWQBOTh" executionInfo={"status": "ok", "timestamp": 1636725299300, "user_tz": -60, "elapsed": 24118, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="555b4a10-e96b-4ba1-e8cf-86d31213d00f"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/Time_Series/"
```

<!-- #region id="R7E3JshdFZHy" -->
##Dataframe **csv**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 263} id="Mc8-Gu9H_y9_" executionInfo={"status": "ok", "timestamp": 1636725299301, "user_tz": -60, "elapsed": 17, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3110cc74-81cf-4eae-e0bc-07f50a5b3efc"
# Download csv file from resources and put it in working directory
dataframe = pd.read_csv(path+'daily-total-female-births-CA.csv', header=0 , parse_dates=[0])
print(dataframe.shape)
print(dataframe['date'].dtype)
dataframe.head()
```

<!-- #region id="FU6ay9gz_y-F" -->
#### Data Type
<!-- #endregion -->

<!-- #region id="c3rxYXkq_y-G" -->
* Autre foncton
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv(.........,parse_dates=[0], date_parser=dateparse)

[source](https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior)
<!-- #endregion -->

<!-- #region id="OKudor1j_y-G" -->
### Loading Data as a Series
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="LKQPYm8q_y-G" executionInfo={"status": "ok", "timestamp": 1636725299302, "user_tz": -60, "elapsed": 15, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="913967c8-1fea-4cb1-e051-33b7d13f44ca"
series = pd.read_csv(path+'daily-total-female-births-CA.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(series.shape)
series.head()
```

<!-- #region id="P8cudB3O_y-H" -->
# Exploration des données
<!-- #endregion -->

<!-- #region id="6027ELs2_y-I" -->
### Sous selection par date
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="I0gUuzY2_y-I" executionInfo={"status": "ok", "timestamp": 1636725299302, "user_tz": -60, "elapsed": 15, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="e50653a8-484f-40ab-fef6-cc5f40e99d94"
dataframe[(dataframe['date'] > '1959-01-01') & (dataframe['date'] <= '1959-01-21')].head()
print('nb de données pour "1959-01" : '+str(len(series['1959-01']))+'\n',  series['1959-01'][:6] )
```

<!-- #region id="XzENM_MQ_y-I" -->
### Statistique Descriptive
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 320} id="BheIib7F_y-J" executionInfo={"status": "ok", "timestamp": 1636725299303, "user_tz": -60, "elapsed": 13, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="bf9e08d3-a3b3-4a04-b63b-a56912cac57f"
series.describe()
dataframe.describe()
```

<!-- #region id="A1Ae9A22_y-J" -->
# Feature Engineering
<!-- #endregion -->

<!-- #region id="9YHv1wq6_y-J" -->
* Caractéristiques de la "date et de l'heure"
* Fonctions de "décalage"
* Fonctions de "fenêtre"
* Fonctionnalité "d'expansion"
<!-- #endregion -->

<!-- #region id="Rhyhpv4c_y-K" -->

### Date time features
<!-- #endregion -->

```python id="-HRfEjYv_y-K"
features = dataframe.copy()
```

```python id="HDnLVk06_y-K"
features['year'] = features['date'].dt.year
features['month'] = features['date'].dt.month
features['day'] = features['date'].dt.day
```

```python colab={"base_uri": "https://localhost:8080/", "height": 226} id="jZ-MxCoQ_y-M" executionInfo={"status": "ok", "timestamp": 1636725300035, "user_tz": -60, "elapsed": 36, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="33881698-0290-4e63-bde7-e838a1a17a1e"
features.head()
```

<!-- #region id="4saMzAop_y-M" -->
[source](https://pandas.pydata.org/pandas-docs/stable//reference/series.html#datetimelike-properties)
<!-- #endregion -->

<!-- #region id="4XqaodkK_y-M" -->
### Lag features
<!-- #endregion -->

<!-- #region id="M3QmiWc__y-M" -->
Fonction de lag : donner
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="9qzW_YFN_y-N" executionInfo={"status": "ok", "timestamp": 1636725300035, "user_tz": -60, "elapsed": 35, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="e39aad52-43df-4e00-b753-70341a39b48e"
features['lag1'] =  features['births'].shift(1)
features['lag2'] =  features['births'].shift(365)
features.head(7)
```

<!-- #region id="8hKJ7-MV_y-N" -->
[source](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.shift.html#pandas.Series.shift)
<!-- #endregion -->

<!-- #region id="LRJp0JqA_y-N" -->
### Fenêtre glissante
<!-- #endregion -->

```python id="MLGxviiA_y-N" colab={"base_uri": "https://localhost:8080/", "height": 226} executionInfo={"status": "ok", "timestamp": 1636725300036, "user_tz": -60, "elapsed": 35, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="666da9d7-9d20-46c4-8ffb-b25ae88a0e18"
features['Roll_mean'] = features['births'].rolling(window = 2).mean()
features['Roll_max'] = features['births'].rolling(window = 3).max()
features.head()
```

<!-- #region id="CGrXvZdo_y-O" -->
[Source](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.rolling.html#pandas.Series.rolling)
<!-- #endregion -->

<!-- #region id="YIfKb2IN_y-O" -->
### Compléter features

* **min_periods** : Nombre minimum d'observations dans la fenêtre requis pour avoir une valeur (sinon le résultat est NA).
<!-- #endregion -->

```python id="NUmGCkE4_y-P" colab={"base_uri": "https://localhost:8080/", "height": 226} executionInfo={"status": "ok", "timestamp": 1636725300036, "user_tz": -60, "elapsed": 33, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1548be92-5d55-4261-9ebc-f6e3d6307979"
features['Expand_max'] = features['births'].expanding( min_periods=1, axis=0).max()
features.head()
```

<!-- #region id="ToUhtzMG_y-Q" -->
[Source](https://pandas.pydata.org/pandas-docs/stable//reference/api/pandas.Series.expanding.html#pandas.Series.expanding)
<!-- #endregion -->

<!-- #region id="yxKrTJMu_y-Q" -->
#Data Visualization
<!-- #endregion -->

```python id="fiIIzcvj_y-Q"
from matplotlib import pyplot as plt
%matplotlib inline
```

```python id="DrYrFKN2_y-Q"
Dataviz_df = dataframe.copy()
```

```python id="QPTK9Cm4_y-R" colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"status": "ok", "timestamp": 1636725300038, "user_tz": -60, "elapsed": 33, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="7409404f-d546-46af-ebcf-f2bdb469143c"
fig = plt.figure(1, figsize=(22, 5))
plt.plot(Dataviz_df['births'])
plt.show()
```

```python id="x9mFJu0B_y-R" colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"status": "ok", "timestamp": 1636725300039, "user_tz": -60, "elapsed": 33, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="687339be-6cb4-4e42-952b-2794bec4f373"
fig = plt.figure(1, figsize=(22, 5))
Dataviz_df.index = Dataviz_df['date']
Dataviz_df.index.name = None
plt.plot(Dataviz_df['births'])
plt.show()
```

<!-- #region id="BT2JSSaq_y-R" -->
### Zoomer sur une fenêtre
<!-- #endregion -->

```python id="93ZMPiIX_y-R" colab={"base_uri": "https://localhost:8080/", "height": 318} executionInfo={"status": "ok", "timestamp": 1636725300039, "user_tz": -60, "elapsed": 31, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="4006753f-7195-44e2-d38f-ad93bf3c5827"
Dataviz_df2 = Dataviz_df[(Dataviz_df['date'] > '1959-01-01') & (Dataviz_df['date'] <= '1959-01-30')].copy()
Dataviz_df2['births'].plot()
```

<!-- #region id="IkDOC7qu_y-R" -->
### Ligne de Tendance
<!-- #endregion -->

```python id="Wz6Yfqbz_y-S"
import seaborn as sns
```

```python id="E6Hbej-J_y-S" colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"status": "ok", "timestamp": 1636725300040, "user_tz": -60, "elapsed": 26, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="71985923-3299-492b-d94c-10fe79d03913"
fig = plt.figure(1, figsize=(22, 5))
ordinal = list(pd.to_datetime(Dataviz_df['date']).apply(lambda date: date.toordinal()))
ax = sns.regplot(x= ordinal, y=Dataviz_df['births'].values)
ax.set_xticklabels(list(Dataviz_df.index.astype(str)))
plt.show()
```

```python id="TT3zHKgI_y-S" colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"status": "ok", "timestamp": 1636725300040, "user_tz": -60, "elapsed": 25, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1e56fca7-1d5d-43a3-c29a-ba821f1a606a"
miles_df = pd.read_csv(path+'us-airlines-monthly-aircraft-miles-flown.csv', header=0 , parse_dates=[0])
miles_df.head()
```

```python id="ZDRAydW4_y-T" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1636725301732, "user_tz": -60, "elapsed": 1715, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="76861857-059b-40b1-b604-a9eb4dd8beeb"
fig = plt.figure(1, figsize=(22, 5))
plt.plot(miles_df['MilesMM'])
plt.show()
```

```python id="0U58OPAY_y-T" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1636725301733, "user_tz": -60, "elapsed": 14, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="875710d2-047a-4a02-e265-c40c1dc9c752"
fig = plt.figure(1, figsize=(22, 5))
ordinal = list(pd.to_datetime(miles_df['Month']).apply(lambda date: date.toordinal()))
ax = sns.regplot(x= ordinal, y=miles_df['MilesMM'].values)
ax.set_xticklabels(list(miles_df['Month'].astype(str)))
plt.show()
```

<!-- #region id="0R8UuTMn_y-T" -->
### Suppression de la saisonnalité
<!-- #endregion -->

```python id="23EeUdI1_y-T" colab={"base_uri": "https://localhost:8080/", "height": 226} executionInfo={"status": "ok", "timestamp": 1636725301733, "user_tz": -60, "elapsed": 13, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="8e4eb47b-7c7d-4985-aebc-22fd5b65b12d"
miles_df['year'] = miles_df['Month'].dt.year
miles_df.head()
```

```python id="U7T7nsaa_y-U" colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"status": "ok", "timestamp": 1636725301734, "user_tz": -60, "elapsed": 13, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="7bdd27a6-804d-4a88-d892-7cedc1936318"
fig = plt.figure(1, figsize=(22, 5))
plt.plot(miles_df.groupby(['year'])['MilesMM'].mean())
plt.show()
```

<!-- #region id="VaiuKTWV_y-U" -->
### Lag plots
<!-- #endregion -->

```python id="fThs54kz_y-U" colab={"base_uri": "https://localhost:8080/", "height": 73} executionInfo={"status": "ok", "timestamp": 1636725302421, "user_tz": -60, "elapsed": 699, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3d00f6a1-953f-4c3d-9547-80d9e1a8779e"
miles_df['lag1'] =  miles_df['MilesMM'].shift(1)
sns.scatterplot(x=miles_df['lag1'], y=miles_df['MilesMM'])
```

```python id="CDliRHwt_y-V" colab={"base_uri": "https://localhost:8080/", "height": 73} executionInfo={"status": "ok", "timestamp": 1636725302421, "user_tz": -60, "elapsed": 32, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="ad0bef45-28d4-4dc2-cdb3-690398beab0a"
from pandas.plotting import lag_plot
lag_plot(miles_df['MilesMM'])
```

<!-- #region id="WPEPYYSB_y-W" -->
### Graphiques d'autocorrélation
<!-- #endregion -->

```python id="S24jFfa1_y-W"
from pandas.plotting import autocorrelation_plot
```

```python id="cF471EyH_y-W" colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"status": "ok", "timestamp": 1636725302422, "user_tz": -60, "elapsed": 32, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="bea99470-0820-454d-b3cb-321055333caf"
fig = plt.figure(1, figsize=(22, 5))
autocorrelation_plot(miles_df['MilesMM'])
plt.show()
```

<!-- #region id="J8XAwR0S_y-X" -->
## Downsampling and Upsampling
<!-- #endregion -->

```python id="lNrxkDQ8_y-X" colab={"base_uri": "https://localhost:8080/", "height": 225} executionInfo={"status": "ok", "timestamp": 1636725302422, "user_tz": -60, "elapsed": 31, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="347f6c2d-970d-4cd3-e246-059baf919182"
miles_df = pd.read_csv(path+'us-airlines-monthly-aircraft-miles-flown.csv', header=0 , parse_dates=[0])
print(miles_df.shape)
miles_df.head()
```

<!-- #region id="Ol_HAUHS_y-Y" -->
| Alias  | Description           |
|--------|-----------------------|
| B      | Business day          |
| D      | Calendar day          |
| W      | Weekly                |
| M      | Month end             |
| Q      | Quarter end           |
| A      | Year end              |
| BA     | Business year end     |
| AS     | Year start            |
| H      | Hourly frequency      |
| T, min | Minutely frequency    |
| S      | Secondly frequency    |
| L, ms  | Millisecond frequency |
| U, us  | Microsecond frequency |
| N, ns  | Nanosecond frequency  |
<!-- #endregion -->

<!-- #region id="tOlT0y38_y-X" -->
### Downsampling
<!-- #endregion -->

```python id="c0Uf84fH_y-X" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636725302422, "user_tz": -60, "elapsed": 28, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="284d9d5b-5c5d-4aa3-d511-6ead007de31a"
quarterly_miles_df = miles_df.resample('Q', on='Month').mean()
print(quarterly_miles_df.shape)
```

```python id="wcsq6DAB_y-Y" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636725302423, "user_tz": -60, "elapsed": 26, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3b196f4c-6bee-4aad-ac73-1b809c79e592"
yearly_total_miles_df = miles_df.resample('A', on='Month').sum()
print(yearly_total_miles_df.shape)
```

<!-- #region id="QqBFLwhY_y-Z" -->
### Upsampling avec interpolation
<!-- #endregion -->

```python id="xpMkUheX_y-Z" colab={"base_uri": "https://localhost:8080/", "height": 256} executionInfo={"status": "ok", "timestamp": 1636725302423, "user_tz": -60, "elapsed": 23, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3605ebf4-7f8d-4cda-9123-0f5b8cdc8701"
interpolated_miles_df = miles_df.resample('D', on='Month').mean()
interpolated_miles_df = interpolated_miles_df.interpolate(method='linear')
print(interpolated_miles_df.shape)
interpolated_miles_df.head()
```

```python id="xWprTw1F_y-b" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1636725302423, "user_tz": -60, "elapsed": 20, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="9c72767f-774f-44c2-82da-e2717c8749c5"
fig = plt.figure(1, figsize=(22, 5))
plt.plot(interpolated_miles_df)
plt.show()
```

```python id="cbKvCNFw_y-b"
upsampled_miles_df = interpolated_miles_df = miles_df.resample('D', on='Month').mean()
poly_interpolated_miles_df = upsampled_miles_df.interpolate(method='spline', order=2)
```

```python id="a1ebZ2i6_y-b" colab={"base_uri": "https://localhost:8080/", "height": 317} executionInfo={"status": "ok", "timestamp": 1636725302424, "user_tz": -60, "elapsed": 20, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="8cc3402c-e012-4e4c-913f-36d9f6371005"
fig = plt.figure(1, figsize=(22, 5))
plt.plot(poly_interpolated_miles_df)
plt.show()
```

<!-- #region id="7WHvDy3C_y-b" -->
| Method  | Description                                               |
|---------|-----------------------------------------------------------|
| bfill   | Backward fill                                             |
| count   | Count of values                                           |
| ffill   | Forward fill                                              |
| first   | First valid data value                                    |
| last    | Last valid data value                                     |
| max     | Maximum data value                                        |
| mean    | Mean of values in time range                              |
| median  | Median of values in time range                            |
| min     | Minimum data value                                        |
| nunique | Number of unique values                                   |
| ohlc    | Opening value, highest value, lowest value, closing value |
| pad     | Same as forward fill                                      |
| std     | Standard deviation of values                              |
| sum     | Sum of values                                             |
| var     | Variance of values                                        |
<!-- #endregion -->

<!-- #region id="4vHsAjOe_y-c" -->
## Modèles de type ARIMA
<!-- #endregion -->

<!-- #region id="r48GULu8z_oA" -->
* Une série temporelle est stationnaire si ses propriétés statistiques ne dépendent pas de la valeur absolue de la variable temporelle **t**.
* Autrement dit, ces propriétés ne sont pas affectées par une translation quelconque de la série dans le temps.
* Une série avec une tendance (moyenne varie selon t) n'est pas stationnaire.

*  Pour éliminer la saisonnalité, nous pouvons faire la différence entre valeurs consécutives de la même saison.  
Ex.: y'(t) = y(t) – y(t-12) pour des données mensuelles par exemple.
<!-- #endregion -->

<!-- #region id="zePv3-J6_y-d" -->
**Additive Model**

y(t) = Level + Trend + Seasonality + Noise

**Multiplicative Model**

y(t) = Level * Trend * Seasonality * Noise
<!-- #endregion -->

```python id="NGrXJbS2_y-d"
from statsmodels.tsa.seasonal import seasonal_decompose
```

```python id="QuKlz4qq_y-d" colab={"base_uri": "https://localhost:8080/", "height": 237} executionInfo={"status": "ok", "timestamp": 1636725302424, "user_tz": -60, "elapsed": 19, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="5c733cf4-258c-43db-b41b-21c26a1aa338"
miles_decomp_df = pd.read_csv(path+'us-airlines-monthly-aircraft-miles-flown.csv', header=0 , parse_dates=[0])
miles_decomp_df.index = miles_decomp_df['Month']
miles_decomp_df.head()
```

<!-- #region id="0P15RlZ32zRh" -->
### Décomposition saisonnaire
<!-- #endregion -->

```python id="9SSiISPv_y-d" colab={"base_uri": "https://localhost:8080/", "height": 722} executionInfo={"status": "ok", "timestamp": 1636725304087, "user_tz": -60, "elapsed": 1681, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="49305985-f7a7-4f79-f951-bad4d50d21c9"
result = seasonal_decompose(miles_decomp_df['MilesMM'], model='additive')
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(22,12))
result.trend.plot(ax=ax1)
result.resid.plot(ax=ax2)
result.seasonal.plot(ax=ax3)
ax1.set_title('Tendance', loc = 'left')
ax2.set_title('Résidus', loc='left')
ax3.set_title('Saisonnalité', loc='left')
plt.show()
```

```python id="ojxSWFsy_y-e" colab={"base_uri": "https://localhost:8080/", "height": 723} executionInfo={"status": "ok", "timestamp": 1636725304911, "user_tz": -60, "elapsed": 828, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="10f0ec4e-99b8-4ccc-bb71-1734b7c140ea"
result = seasonal_decompose(miles_decomp_df['MilesMM'], model='multiplicative')
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(22,12))
result.trend.plot(ax=ax1)
result.resid.plot(ax=ax2)
result.seasonal.plot(ax=ax3)
ax1.set_title('Tendance', loc = 'left')
ax2.set_title('Résidus', loc='left')
ax3.set_title('Saisonnalité', loc='left')
plt.show()
```

<!-- #region id="4OoS-N0C_y-f" -->
### Différenciation
<!-- #endregion -->

<!-- #region id="d9uEvBZo0QvI" -->
* La différenciation est une méthode générale pour éliminer uune tendance d'une série temporelle.

* On peut corriger cela en prenant la différence entre deux valeurs consécutives d'une marche aléatoire est stationnaire
    - ordre 1 :  y'(t) = y(t) – Y(t-1)  
    - ordre 2 :  y''(t) = ( y(t) – Y(t-1) ) – (Y(t-1) – Y(t-2) )
<!-- #endregion -->

```python id="5QsoUtEr_y-f"
miles_df = pd.read_csv(path+'us-airlines-monthly-aircraft-miles-flown.csv', header=0 , parse_dates=[0])
```

<!-- #region id="m0cRzGOK3Zdw" -->
#### Ordre 1
<!-- #endregion -->

```python id="16woEXWo_y-h" colab={"base_uri": "https://localhost:8080/", "height": 54} executionInfo={"status": "ok", "timestamp": 1636725304912, "user_tz": -60, "elapsed": 11, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="046e3762-cc67-4268-f1ce-fd87f0a3c90a"
miles_df['MilesMM_diff_1'] =  miles_df['MilesMM'].diff(periods=1)

miles_df.index = miles_df['Month']
result = seasonal_decompose(miles_df.iloc[1:,miles_df.columns.get_loc("MilesMM_diff_1")], model='additive')
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(22,12))
result.trend.plot(ax=ax1)
result.resid.plot(ax=ax2)
result.seasonal.plot(ax=ax3)
ax1.set_title('Tendance', loc = 'left')
ax2.set_title('Résidus', loc='left')
ax3.set_title('Saisonnalité', loc='left')
plt.show()
```

<!-- #region id="mD3GCKO53c2y" -->
#### Ordre 12
<!-- #endregion -->

```python id="hY2KhTST_y-h" colab={"base_uri": "https://localhost:8080/", "height": 722} executionInfo={"status": "ok", "timestamp": 1636725304913, "user_tz": -60, "elapsed": 10, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="30d72dd5-2756-4f87-b5d9-464940c4799a"
miles_df['MilesMM_diff_12'] = miles_df['MilesMM'].diff(periods=12)

fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(22,12))
miles_df['MilesMM'].plot(ax=ax1  )
miles_df['MilesMM_diff_1'].plot(ax=ax2  )
miles_df['MilesMM_diff_12'].plot(ax=ax3  )
ax1.set_title('MilesMM', loc = 'left')
ax2.set_title('diff 1', loc='left')
ax3.set_title('diff 12', loc='left')
plt.show()
```

```python id="u9kQAB4V_y-h" colab={"base_uri": "https://localhost:8080/", "height": 722} executionInfo={"status": "ok", "timestamp": 1636725307523, "user_tz": -60, "elapsed": 2619, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1719941f-9dd3-4620-8615-ffb917f7b30b"
result = seasonal_decompose(miles_df.iloc[13:,miles_df.columns.get_loc("MilesMM_diff_12")], model='additive')
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(22,12))
result.trend.plot(ax=ax1)
result.resid.plot(ax=ax2)
result.seasonal.plot(ax=ax3)
ax1.set_title('Tendance', loc = 'left')
ax2.set_title('Résidus', loc='left')
ax3.set_title('Saisonnalité', loc='left')
plt.show()
```

<!-- #region id="euwpstQD_y-l" -->
![image.png](data:image/png;base64,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)
<!-- #endregion -->

<!-- #region id="A4QoNeoq_y-n" -->
### Autoregression Model (AR)
<!-- #endregion -->

<!-- #region id="z-Q03-TY2EZs" -->
**Modèle AR(p)**  
![image.png](data:image/png;base64,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)  

• **y(t)** est une fonction des p valeurs précédentes, plus un bruit blanc.
• Les coefficients **teta** doivent être **< 1** pour une série stationnaire.
• L'autocorrélation est présente au-delà d'un délai **p**, mais l'effet s'estompe avec le temps.
• Par exemple: pour AR(1), y(t) dépend de y(t-1), mais yt-1 dépend de y(t-2), donc y(t) dépend
indirectement de (t-2).

<!-- #endregion -->

```python id="QPjmZe_J_y-o" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636725308083, "user_tz": -60, "elapsed": 563, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="58cc3db0-adda-40e0-f2d7-2b5058d23675"
# load dataset
series = pd.read_csv(path+'daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
print('train: ',train.shape,'\ntest: ',test.shape)
```

```python id="QnofUQdx_y-o" colab={"base_uri": "https://localhost:8080/", "height": 471} executionInfo={"status": "ok", "timestamp": 1636725337736, "user_tz": -60, "elapsed": 405, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="c357f74f-1dae-477d-b826-c1d172a6cfed"
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print('Coefficients:\n %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
rmse = mean_squared_error(test, predictions)**(1/2)
print('\nTest RMSE: %.3f' % rmse)
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
```

<!-- #region id="Jd3EBHS4ajMi" -->
yhat = b0 + b1*X1 + b2*X2 ... bn*Xn
<!-- #endregion -->

```python id="q_Cw8vgP_y-p" colab={"base_uri": "https://localhost:8080/", "height": 284} executionInfo={"status": "ok", "timestamp": 1636725370416, "user_tz": -60, "elapsed": 30, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6c6205da-5511-445b-883f-999b67ada165"
# train autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
rmse = mean_squared_error(test, predictions)**(1/2)
print('Test RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
```

<!-- #region id="tYc2EJvc_y-q" -->
#### Autre façon
<!-- #endregion -->

```python id="vTMz-uqK_y-q"
df = pd.read_csv(path+'daily-min-temperatures.csv', header=0 , parse_dates=[0])
```

```python id="Ei2MaM4D_y-q"
train, test = df.Temp[1:df.shape[0]-7], df.Temp[df.shape[0]-7:]
```

```python id="6K-lW1-q_y-r" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636725370417, "user_tz": -60, "elapsed": 26, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="cd190352-692e-4d34-ab4b-89a3cc5b436f"
data = train
predict =[]
for t in test:
    model = AutoReg(data, lags=29)
    model_fit = model.fit()
    y = model_fit.predict(start=len(data), end=len(train)+len(test)-1)
    predict.append(y.values[0])
    data = np.append(data, t)
    data = pd.Series(data)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test.values, predict)
print('Test RMSE: %.3f' % mse**(1/2))
```

```python id="PvR4hs36_y-r" colab={"base_uri": "https://localhost:8080/", "height": 284} executionInfo={"status": "ok", "timestamp": 1636725429368, "user_tz": -60, "elapsed": 16, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3eac0f56-3849-4e23-dd3f-96a4b77b59cf"
plt.plot(test.values)
plt.plot(predict, color='red')
```

<!-- #region id="x2WCk5Dj_y-r" -->
### Moyenne mobile (MA)
<!-- #endregion -->

<!-- #region id="pEP4TlYV05Mn" -->
**Modèle MA(q)**  

* Il s'agit d'un modèle naïf qui suppose que les composantes de tendance et de saisonnalité de la série chronologique ont déjà été supprimées ou ajustées.  

![image.png](data:image/png;base64,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)  
* y est la moyenne pondérée des **q+1** dernières valeurs d'un bruit blanc.
* L'effet d'un "choc" et disparaît au temps **t+q+1**. L'autocorrélation temporelle est limitée à un
délai **q**.
* Dans le modèle d’autorégressif, la valeur d’**y** dépend de valeur précédente et non du bruit.
<!-- #endregion -->

<!-- #region id="Cx-7K9vX1vic" -->
#### Retardé
<!-- #endregion -->

```python id="06z5nufw_y-r"
from matplotlib.pyplot import figure

df = pd.read_csv(path+'daily-min-temperatures.csv', header=0 , parse_dates=[0])
# Tail-rolling average transform
rolling = df.Temp.rolling(window=4)
rolling_mean = rolling.mean()

# plot original and transformed dataset
#fig, ax = plt.subplots()
#fig.set_size_inches(22, 6)
#ax.plot(df.Temp)
#ax.plot(rolling_mean, color='red')

#ax.set_xticklabels(list(df.Date.astype(str)))
#plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} id="JyHPCx73snkx" executionInfo={"status": "ok", "timestamp": 1636725429369, "user_tz": -60, "elapsed": 10, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6e74dca1-d589-4eec-979f-920b8b2d6e8e"
# plot original and transformed dataset
fig, ax = plt.subplots()
fig.set_size_inches(22, 6)
ax.plot(df.Temp[0:60])
ax.plot(rolling_mean[0:60], color='red')

ax.set_xticklabels(list(df.Date[0:60].astype(str)))
plt.show()
```

<!-- #region id="Jt70IMoe1Aht" -->
#### Centré
<!-- #endregion -->

```python id="8ld_B09JnBiN"
def roling_center(df,col,w,):
    if not w%2 and w>=2:
        df['rolling_mean_'+str(w)] = ( df[col].rolling(int(w-1)).sum().shift(-int((w/2)-1)) +  (1/2)*(df[col].shift(int(w/2)) + df[col].shift(-int(w/2))) ) /w
    else:
        df['rolling_mean_'+str(w)] = df[col].rolling(w).mean().shift(-int(w/2))
    return df
```

```python id="vDc8apZUrSV3"
df = roling_center(df,'Temp',2)
df = roling_center(df,'Temp',3)
df = roling_center(df,'Temp',4)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} id="O17WJ8c8j1Ep" executionInfo={"status": "ok", "timestamp": 1636725430101, "user_tz": -60, "elapsed": 739, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="98f84287-998d-4119-99f2-5430a41e71d1"
# plot original and transformed dataset
fig, ax = plt.subplots()
fig.set_size_inches(22, 6)
ax.plot(df.Temp[0:60])
ax.plot(df.rolling_mean_2[0:60], color='green', label='rolling mean_2')
ax.plot(df.rolling_mean_3[0:60], color='red', label='rolling mean_3')
ax.plot(df.rolling_mean_4[0:60], color='orange', label='rolling mean_4')
plt.legend()
ax.set_xticklabels(list(df.Date[0:60].astype(str)))
plt.show()
```

<!-- #region id="wbo522XVx9Ap" -->
### AutoRegressive Integrated Moving Average (ARIMA)
<!-- #endregion -->

<!-- #region id="sylH5z3Irbh3" -->
ARIMA est un modèle statistique pour analyser ou prédire les données de séries temporelles ou time series. Le principe est également connu sous l'appellation moyenne mobile autorégressive intégrée.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 225} id="6NOcrpDVduQr" executionInfo={"status": "ok", "timestamp": 1636726857009, "user_tz": -60, "elapsed": 1708, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="7c5a58b3-73ce-4670-f093-79720732a3bf"
df = pd.read_csv(path+'shampoo.csv', header=0, parse_dates=[0])
print(df.shape)
df.head()
```

```python id="CsohSJkH_y-u" colab={"base_uri": "https://localhost:8080/", "height": 320} executionInfo={"status": "ok", "timestamp": 1636726910503, "user_tz": -60, "elapsed": 2404, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="133eaff3-6eba-424e-c650-f52ea6c462e0"
fig, ax = plt.subplots(figsize=(22,5))
ax.set_xticklabels(list(df['Month'].astype(str)))
plt.plot(df['Sales'])
plt.show()
```

<!-- #region id="6EOdwLwi_y-v" -->
### Autocorrelation Plot
<!-- #endregion -->

```python id="8ncgLtrHjlR4"
from pandas.plotting import autocorrelation_plot
```

```python id="B8-bYOJe_y-v" colab={"base_uri": "https://localhost:8080/", "height": 440} executionInfo={"status": "ok", "timestamp": 1636726893475, "user_tz": -60, "elapsed": 2375, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="28f17f25-da91-4ca8-e07e-9108c2cd2ac8"
fig, ax = plt.subplots(figsize=(22,6))
autocorrelation_plot(df['Sales'])
plt.show()
```

<!-- #region id="DTIGL6J3huI-" -->
* ici on prendra **p = 5** la première valeur à entrer dans l'intervalle de confiance
<!-- #endregion -->

<!-- #region id="nw5JLb13_y-v" -->
### Partial Autocorrelation Graph
<!-- #endregion -->

<!-- #region id="WUqXwmeAcECf" -->
Corrélation entre y(t) et y(t-k) après avoir tenu compte de l'effet des délais inférieur à k
<!-- #endregion -->

```python id="c9rwzduL_y-v"
from statsmodels.graphics.tsaplots import plot_pacf
```

```python id="u_lcw51a_y-v" colab={"base_uri": "https://localhost:8080/", "height": 390} executionInfo={"status": "ok", "timestamp": 1636727024924, "user_tz": -60, "elapsed": 297, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="8a1c0560-fe81-4a3a-a467-a266c31cf609"
ax = plot_pacf(df['Sales'], lags=15)
ax.set_size_inches(22, 5)
plt.show()
```

<!-- #region id="SJ1cUDX6_y-v" -->
l'intervalle de cofiance est traverssé entre 2 et 3 on prendra q = 2
<!-- #endregion -->

<!-- #region id="_8-53w9_-ske" -->
### Modèle non saisonier
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 225} id="BCxew4ZphW1g" executionInfo={"status": "ok", "timestamp": 1636737323596, "user_tz": -60, "elapsed": 301, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="a2ea177e-59b7-4e8f-890c-3874c153fb43"
df = pd.read_csv(path+'daily-min-temperatures.csv', header=0, parse_dates=[0])
df.set_index('Date',inplace=True)
df.index.name = None
print(df.shape)
df.head()
```

<!-- #region id="6m5y_dK7ddLU" -->
• ARIMA(p,d,q): Combinaison d'un modèle autorégressif d'ordre p et d'une moyenne mobile d'ordre
q sur la variable y différenciée d fois.
<!-- #endregion -->

```python id="CqTAY4w3dgzY"
split_point = len(series) - 7
train, validation = df[0:split_point], series[split_point:]
```

<!-- #region id="cPXw7W4ufRF8" -->
nous allons rendre les données stationnaires et développer un modèle ARIMA simple
<!-- #endregion -->

```python id="_k9uNH_RfQSA"
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
```

```python id="NkDojqrff-D7"
X = train['Temp'].values
days_in_year = 365
differenced = difference(X, days_in_year)
```

```python id="rzacMlTm5kTp"
from statsmodels.tsa.arima.model import ARIMA
```

```python colab={"base_uri": "https://localhost:8080/"} id="7EAjKht-gQHO" executionInfo={"status": "ok", "timestamp": 1636737403045, "user_tz": -60, "elapsed": 8992, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="f1fb2a92-51f4-4686-b7f3-fad59105467e"
# fit le model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit()
#  summary
print(model_fit.summary())
```

```python id="WLYWo6iJ_y-w"
residuals = model_fit.resid
```

```python colab={"base_uri": "https://localhost:8080/", "height": 319} id="M7pV9-NahC_l" executionInfo={"status": "ok", "timestamp": 1636737404491, "user_tz": -60, "elapsed": 338, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="04135a59-c917-4979-cb63-20fbde12e1a9"
fig, ax = plt.subplots(figsize=(22,5))
ax.set_xticklabels(list(df.index.astype(str)))
plt.plot(residuals.flatten())
plt.show()
```

```python id="kc4hDLbm_y-w" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636737404491, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="c23b4478-6e0f-4517-c924-e1e402c2213e"
pd.Series(residuals.flatten()).describe()
```

<!-- #region id="-labt0sR_y-x" -->
### Rappels
* **ARIMA** - <br />
    model = ARIMA(df['Sales'], order=(q,d,p)) <br />
* **Autoregression** - <br />
    model = ARIMA(df['Sales'], order=(p,d,0)) <br />
* **Moving Avaerage Model** - <br />
    model = ARIMA(df['Sales'], order=(0,d,q)) <br />
<!-- #endregion -->

<!-- #region id="coT2gqJ5mPBn" -->
#### Prédictions hors échantillon avec le modèle
<!-- #endregion -->

```python id="GfMUvBDE_y-x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636737404492, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="844d2e64-4b23-42df-dceb-3484f1ba9eac"
output = model_fit.forecast()
output
forecast = inverse_difference(X, output, days_in_year)
forecast
```

<!-- #region id="jX9WaVwMnOGH" -->
* La fonction predict peut être utilisée:
    * pour prévoir des pas de temps arbitraires dans l'échantillon et hors de l'échantillon, y compris le prochain pas de temps de prévision hors de l'échantillon.
    * elle requiert la spécification d'un début et d'une fin, qui peuvent être les indices des pas de temps relatifs au début des données d'apprentissage utilisées pour ajuster le modèle.
<!-- #endregion -->

```python id="Tr9ZAIBsNyis"
from sklearn.metrics import mean_squared_error
```

```python id="5xYIaCGD_y-x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636737405362, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="c0199bd4-9b58-4622-e147-718a63befcce"
from pandas import datetime
start_index = df.index.get_loc(pd.to_datetime('1990-12-25'))
end_index = df.index.get_loc(pd.to_datetime('1990-12-26'))

forecast = model_fit.predict(start=start_index, end=end_index)
forecast = inverse_difference(X, forecast, days_in_year)
forecast
```

```python colab={"base_uri": "https://localhost:8080/"} id="isklOBXhF9a3" executionInfo={"status": "ok", "timestamp": 1636737860676, "user_tz": -60, "elapsed": 662, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="ed8c0792-15c8-4767-ad4e-605be06dba2d"
start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
pred = inverse_difference(X, forecast, days_in_year)
list(np.round(pred,2))
```

```python colab={"base_uri": "https://localhost:8080/"} id="HVJvI84NLeWq" executionInfo={"status": "ok", "timestamp": 1636737878140, "user_tz": -60, "elapsed": 304, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="08421802-784a-430f-def0-f608dc199597"
mean_squared_error(list(validation.values), list(np.round(pred,2)))
```

<!-- #region id="GjtI8La-FpBX" -->
On peut effectuer une prédiction, une par une pour obtenir une meilleur prédiction, Notez que pour inverser la valeur prévue pour t(n+1), nous avons besoin de la valeur prévue inversée pour t(n). Ici, nous les ajoutons à la fin d'une liste appelée historique pour les utiliser lors de l'appel de inverse_difference().
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="RXL2csMUFnTP" executionInfo={"status": "ok", "timestamp": 1636737778784, "user_tz": -60, "elapsed": 299, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3ba07ae2-d3ec-4fdb-cc10-4ebc4932e480"
forecast = model_fit.forecast(steps=7)
history = [x for x in X]
pred= []
day = 1
for yhat in forecast:
    inverted = inverse_difference(history, yhat, days_in_year)
    #print('Day %d: %f' % (day, inverted))
    pred.append(inverted)
    history.append(inverted)
    day += 1
np.round(pred,2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="BCUnvh3rJyVO" executionInfo={"status": "ok", "timestamp": 1636737715962, "user_tz": -60, "elapsed": 273, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="07b0e726-c6c1-4a6f-f4a5-6a4071fdcf99"
mean_squared_error(list(validation.values), pred)
```

<!-- #region id="963rOv-5_y-y" -->
def parser(x):
return datetime.strptime('190'+x, '%Y-%m')
<!-- #endregion -->

<!-- #region id="e9LwoEGF_y-z" -->
## SARIMA or Seasonal ARIMA


SARIMA(p,d,q)(P,D,Q)m

* p: Trend autoregression order.
* d: Trend difference order.
* q: Trend moving average order.

**Seasonal Elements** -   
There are four seasonal elements that are not part of ARIMA that must be configured; they are:  

* P: Seasonal autoregressive order.
* D: Seasonal difference order.
* Q: Seasonal moving average order.
* m: The number of time steps for a single seasonal period.
<!-- #endregion -->

```python id="PY4Lg1O2_y-z"
from statsmodels.tsa.statespace.sarimax import SARIMAX
```

```python id="lo7OAV7a_y-0"
df = pd.read_csv('us-airlines-monthly-aircraft-miles-flown.csv', header=0 , parse_dates=[0])
```

```python id="4hH8YnXG_y-0"
df.head()
```

```python id="U24kIMmn_y-0"
df.tail()
```

```python id="nQVBDcKc_y-0"
df.index = df['Month']
```

```python id="DYvOhnJ2_y-1"
result_a = seasonal_decompose(df['MilesMM'], model='multiplicative')
result_a.plot()
```

```python id="-qaU0YX5_y-1"
model = SARIMAX(df['MilesMM'], order=(5,1,3), seasonal_order=(1,1,1,12))
```

```python id="mXchoJJk_y-1"
model_fit = model.fit()
```

```python id="vWLNnJD7_y-1"
residuals = model_fit.resid
```

```python id="ymq85Xcf_y-1"
residuals.plot()
```

```python id="jvEkQuo4_y-1"
output = model_fit.forecast()
```

```python id="ybdZHZAN_y-1"
output
```

```python id="ZE4kNFhP_y-1"
model_fit.forecast(12)
```

```python id="iUm-buW-_y-1"
yhat = model_fit.predict()
```

```python id="e-PGLNtf_y-2"
yhat.head()
```

```python id="7Z9T39VQ_y-2"
pyplot.plot(df['MilesMM'])
pyplot.plot(yhat, color='red')
```
