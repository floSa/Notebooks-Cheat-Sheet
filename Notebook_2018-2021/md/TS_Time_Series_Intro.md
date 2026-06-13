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

<!-- #region id="DqVKpfEomYGU" -->
#Introduction aux séries temporelles
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="wUgBEppzmWPW" executionInfo={"status": "ok", "timestamp": 1636878741398, "user_tz": -60, "elapsed": 9664, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="dd33ba86-d4f7-4cb9-f7f3-e2f4cce4f831"
pip install statsmodels --upgrade
```

```python id="W814M4Wyjwy0"
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
```

```python id="dvpXhHeLr6qa"
from statsmodels.tsa.tsatools import detrend
```

```python colab={"base_uri": "https://localhost:8080/"} id="u6ATLv4hmhdH" executionInfo={"status": "ok", "timestamp": 1636903338585, "user_tz": -60, "elapsed": 77062, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="4ef02ba5-b381-4f23-feed-b33f3c96c418"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/Time_Series/"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 225} id="qm6a79bbD5qj" executionInfo={"status": "ok", "timestamp": 1636903692130, "user_tz": -60, "elapsed": 435, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="bf740e08-d43c-4b3c-caaa-b2d024995339"
df = pd.read_csv(path+"chocolat.csv", sep=",", header=0 , parse_dates=[0])
print(df.shape)
df.head()
```

<!-- #region id="gvoz9Tm5qXkX" -->
### Tendance

La fonction detrend retourne la tendance. On l’obtient en réalisant une régression linéaire de Y sur le temps t.
<!-- #endregion -->

```python id="CazWqUA7D-VP"
col = "prix"
```

```python colab={"base_uri": "https://localhost:8080/"} id="kKwOSVC_qaqv" executionInfo={"status": "ok", "timestamp": 1636903754678, "user_tz": -60, "elapsed": 1405, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="0476528c-2dc1-49b5-a3db-9f630fc147e0"
from statsmodels.api import OLS

y = df[col]
X = np.ones((len(y), 2))
X[:,1] = np.arange(0,len(y))
reg = OLS(y,X)
results = reg.fit()
results.params
```

```python colab={"base_uri": "https://localhost:8080/", "height": 318} id="vbG4kckar7Ad" executionInfo={"status": "ok", "timestamp": 1636903765916, "user_tz": -60, "elapsed": 709, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="d7113dff-89d4-40f2-9a94-40db70e2929e"
from statsmodels.graphics.regressionplots import abline_plot

fig = abline_plot(model_results=results ,  color="red" )
fig.set_size_inches(22, 5)
ax = fig.axes[0]
ax.plot(X[:,1], y, 'b')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 371} id="pzUze28TjJ79" executionInfo={"status": "ok", "timestamp": 1636903540404, "user_tz": -60, "elapsed": 771, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1cc052a8-d963-4605-f0ad-4709fde3595a"
from statsmodels.tsa.tsatools import detrend

df["notrend"] = detrend(df[col])
df["trend"] = df[col] - df["notrend"]
df.plot(x="date", y=[col, "notrend","trend"], figsize=(22,5))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 302} id="js9ANypColTs" executionInfo={"status": "ok", "timestamp": 1636878762831, "user_tz": -60, "elapsed": 11, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="694760d0-f595-4cac-9e75-e2cd0a027279"
from statsmodels.tsa.stattools import acf

cor = acf(df.px_notrend)
print(cor.shape)
plt.plot(cor)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 302} id="_2o2mUSS4-_o" executionInfo={"status": "ok", "timestamp": 1636878763164, "user_tz": -60, "elapsed": 340, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="36babd5d-49f6-424f-872a-aee5eb7bcd35"
from statsmodels.tsa.stattools import pacf
pcor = pacf(df.px_notrend)
print(pcor.shape)
plt.plot(pcor)
```

<!-- #region id="yjH6qUwDEvTd" -->
* On essaye de calculer une tendance en minimisant : ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAAAcCAYAAACDKkZcAAAEDElEQVR4nO2a35GrIBTG6SDpIHZgOtAOtAO2A9MBdsB2wHZAOiAd8HyfKIESvvtwB8c/YBQ1N5vhN+PMLpp4hO8cPjAEicQBkP8dQOIzScJKHEISVuIQkrAS0UgpIaWEEAJFUUBK2Z1LwkpEobXG7Xbr/ldK4XQ6wVoLoCcsrTXqukbbtmjbFl9fX+Cce7+0qqru2qZpDn6E34kxBowxCCHAOQeltOv0T0BrjdPpNGgjhEAp9e/v8QcopYMLfEgpkec5jDF7xvoxWGsnCSeEAGNscq1Sarav3wVrLYQQwfPBiuXQWoMQEqxErtM+Kfv69H1CLL6+k1KiKIpJe57nwZnhGa8UZNM03vgdeZ4PhOf1WJfLBVmWBW/wqaICMNt5SwhVoKZpJoKz1oIQAq111L22xrqGPM+9FRf492zjhPQKi3MOQsjk4k8XFbB9sHydb63F5XKZWAcp5cSnrOGVwgrZo6ZpusTQWnd/e4VljAEhBJTSro1zHp1ZR2GtHRjkPaaxrYPlpjUXkzPu/b6jlKIsS2RZhvP5jLIsB339ilgZYyjLEnVdT8bVGNN5qrIscb1eQQhBWZYoy7JLEErpYMFX13XYYzmqqgIhpLtB7KAZY/B4PFYdS9BaoyiKQRWglA7inDObIbYMljGmu78z60qp4EJni78C4mOllKIoCjDGQCkdmG5gWnV9/sr1//hwBIUlhAAhBLfbLWqAHFJKMMZWHc+mW601zufzJNOUUp031FpHJcMWYQkhvLG7qtVnq78C4mLlnE/GUynVCdwYMzk/569CzG6QjqfDd6EoClRVNWlXSnVV9llHCCG6Et4/sizzti8RaeieSilcr9dB2xp/pZRaFevPz8/qGF37+LxLgLUr0KCw3CC92x6L83++KupiXuK33OptfOR57m1fUllCg8Y5n1SXpmm8yeFDa70q1pgx45x796qklCBk/Qua4CcYY1FfOEZK6c2quWNu49WJxzfQ7tyWKhs7Fc5NvZTSSUxb/RWw76pQKeVd9T/bv/LzJyyssRmLxRgTzKqYbHPimTu3pcrGPjPn3HtfV2H7yTL2V9baKJHtLSxfBXUm37HMb88IixCy2rC9isvlMqlY7o2AE5a1NuqVU+xgMcYm4rDW4nq9es1yPzmWLFh87C0sX8U9nU5dwqyZZifCqqqq27fIsgx1XW8K+Ai01qCU4n6/436/DzxV0zRo2/blg9XfT3s8Hvj+/h5sHo7J8xz3+x2MsbfYeXcey9funm3NKvtX/2wmJJwtbwdiBmvsr5ZWyq1vMfYU1txCIibOXy2sI9iyono1e67Yl65Ql5KEtQPv6kWX0t8g3YskrB3Y8mbiHWCM7f7buiSsxCEkYSUOIQkrcQhJWIlD+AuiDcw33B0mEgAAAABJRU5ErkJggg==)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 338} id="mUbqMkBiv3Xu" executionInfo={"status": "ok", "timestamp": 1636903973184, "user_tz": -60, "elapsed": 1033, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="3686a669-c5cc-4b38-ee89-71a6c9ea6777"
notrend2 = detrend(df[col], order=2)
df["notrend2"] = notrend2
df["trend2"] = df[col] - df["notrend2"]
df.plot(y=[col, "notrend2", "trend2"], figsize=(22,5))
```

<!-- #region id="b9w1TpeBFYY7" -->
* Log
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 320} id="0et2lQubFWVI" executionInfo={"status": "ok", "timestamp": 1636904094370, "user_tz": -60, "elapsed": 750, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="ba59d9cf-7755-4ec0-e560-9a82f94f141a"
df["logSess"] = df[col].apply(lambda x: np.log(x+1))
lognotrend = detrend(df['logSess'])
df["lognotrend"] = lognotrend
df["logtrend"] = df["logSess"] - df["lognotrend"]
df.plot(y=["logSess", "lognotrend", "logtrend"], figsize=(22,5))
```

<!-- #region id="NtTqzMxgFroO" -->
### Saisonlité
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 339} id="Ae200YvMFwOt" executionInfo={"status": "ok", "timestamp": 1636904460812, "user_tz": -60, "elapsed": 1011, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="f8551322-3d04-4f27-d8ff-d0ce1763deb3"
from statsmodels.tsa.seasonal import seasonal_decompose

res = seasonal_decompose(df[col].values.ravel(), freq=10, two_sided=False)
df["season"] = res.seasonal
df["trendsea"] = res.trend
df.plot(y=[col, "season", "trendsea"], figsize=(22,5))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 339} id="5-Jf-RIEFwRQ" executionInfo={"status": "ok", "timestamp": 1636904546942, "user_tz": -60, "elapsed": 756, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="bcc21542-8556-4304-ec3e-20141ab0519d"
res = seasonal_decompose(df[col].values.ravel() + 1, freq=7, two_sided=False, model='multiplicative')
df["season"] = res.seasonal
df["trendsea"] = res.trend
df.plot(y=[col, "season", "trendsea"], figsize=(22,5))
```

<!-- #region id="OG4bhZ96H5vH" -->
### Enlever la saisonnalité
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1pq6L1ZpJOvg" executionInfo={"status": "ok", "timestamp": 1636905103661, "user_tz": -60, "elapsed": 4462, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="b8f94ecc-7363-4eb9-da4c-682607f81ca8"
pip install seasonal
```

```python colab={"base_uri": "https://localhost:8080/", "height": 526} id="EQbFlJMcH7ij" executionInfo={"status": "ok", "timestamp": 1636905129742, "user_tz": -60, "elapsed": 887, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="6fe338d8-cc9d-4087-dd21-f8fd1a13ea7e"
from seasonal import fit_seasons

cv_seasons, trend = fit_seasons(df[col])
print(cv_seasons)
# data["cs_seasons"] = cv_seasons
df["trendcs"] = trend
df.plot(y=[col, "trendcs", "trendsea"], figsize=(22,5));
```

<!-- #region id="1oYHUfHkJtXI" -->
### Autocorrélograme
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 627} id="R8SOI8tYJya7" executionInfo={"status": "ok", "timestamp": 1636905244833, "user_tz": -60, "elapsed": 866, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="5fe1fa36-0557-462a-f54c-62acd3773e9d"
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plt.figure(figsize=(22,10))
ax1 = fig.add_subplot(211)
fig = plot_acf(df[col], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df[col], lags=40, ax=ax2);
```

<!-- #region id="QcMw4v2BKGOJ" -->
Le premier graph indique 10   
Le second indique 3
<!-- #endregion -->

<!-- #region id="Ekw4_Q9Ewk8j" -->
## Machine learning
<!-- #endregion -->

```python id="KxhIUT8TKx-X"
df = pd.read_csv(path+"chocolat.csv", sep=",", header=0 , parse_dates=[0])
col = "prix"
print(df.shape)
```

<!-- #region id="Wo7XuWsmFwxT" -->
### Ajout du Lag
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="7qm_z-aRqyEr" executionInfo={"status": "ok", "timestamp": 1636878763776, "user_tz": -60, "elapsed": 11, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="53fd03af-485a-4e24-8083-4384071f4c55"
n_slice = 1
df['diff_'+str(n_slice)] = df[col].diff(periods=n_slice)
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 352} id="CgufL99BvkzR" executionInfo={"status": "ok", "timestamp": 1636878764388, "user_tz": -60, "elapsed": 622, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="88c20614-e523-4450-fa47-efe534e615df"
df.plot(x="date", y="diff_1", figsize=(22,5))
```

<!-- #region id="PAhk4hauwqrx" -->
* avec statmodel


<!-- #endregion -->

```python id="49cpg_3Rwhvq"
"""from statsmodels.tsa.tsatools import lagmat
lag = 8
col = "diff_1"
X = lagmat(df[col] , lag)
lagged = df.copy()
for c in range(1,lag+1):
    lagged["lag%d" % c] = X[:, c-1]
pd.concat([lagged.head(), lagged.tail()])"""
```

```python id="tqPyF9xJwVHX"
df = pd.read_csv(path+"chocolat.csv", sep=",", header=0 , parse_dates=[0])
df['diff'] = df[col].diff(periods=1)
lag = 8
col = "diff"
for l in range(1,lag+1):
    df["lag%d" % l] =df[col].shift(periods=l)
```

```python id="9iOAIDjcwmuA"
split_point = 7
df_train = df.iloc[:, 2:]
train, validation = df_train[0:-split_point], df_train[-split_point:]
train.fillna(3,inplace=True)
```

<!-- #region id="z7qU0JJiKmEQ" -->
### Utilsation de modèles de type régression
<!-- #endregion -->

```python id="3VNkYua0xgIt"
def make_prediction(model,train,nam_col , nb_pred):

    num_col = train.columns.get_loc(nam_col)
    X_train , y_train  = train[train.columns.difference([nam_col])].values , train[nam_col].values

    for p in range(nb_pred):
        X_test  =  np.array( [np.append( np.array( [y_train[-1]] ) , X_train[-1][0:-1], axis=0) ] )
        model.fit(X_train, y_train)
        tp1_pred = model.predict(X_test) +
        new_row = np.append( tp1_pred ,  X_test[0][1:] , axis=0)
        X_train , y_train = np.append( X_train , np.array([new_row]) , axis=0) , np.append( y_train , tp1_pred , axis=0)

    return y_train[-nb_pred:]
```

```python id="o7NNFz0AxgLi" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636880729414, "user_tz": -60, "elapsed": 260, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="48b3d7c0-69f0-4ce5-bdb5-a3c7cfbefcdd"
np.array(validation['diff'])
```

```python id="toSPKkn-xgOs" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1636880990759, "user_tz": -60, "elapsed": 303, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="936f0c37-2b14-48d7-9434-ee71112e445e"
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=54)

pred_knn = make_prediction(model,train,'diff' , 7)
pred_knn
```

```python colab={"base_uri": "https://localhost:8080/"} id="dhxOyVtYv7XZ" executionInfo={"status": "ok", "timestamp": 1636880756053, "user_tz": -60, "elapsed": 297, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="a2df9ed8-50a0-4b1d-a4b2-f13984766029"
from sklearn.linear_model import LinearRegression
model = LinearRegression()

pred_regL = make_prediction(model,train,'diff' , 7)
pred_regL
```

```python colab={"base_uri": "https://localhost:8080/"} id="QNg7CFYQq-lX" executionInfo={"status": "ok", "timestamp": 1636882181070, "user_tz": -60, "elapsed": 1819, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="2ee13d4d-13b4-4614-a2f5-ff53f6485600"
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

pred_RF = make_prediction(model,train,'diff' , 7)
pred_RF
```

<!-- #region id="VTrX3Gsm9x52" -->
# Transformation
<!-- #endregion -->

<!-- #region id="fPqJrO--Z5Ya" -->
### Etudes des valeurs manquante sur les Dates
<!-- #endregion -->

```python id="zX-1C5eQ9z6i"
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Générer des dates entre 2020 et 2023 (200 données)
date_range = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

# Créer un dataframe avec les dates générées
df = pd.DataFrame({'Date': date_range})

# Ajouter 4 autres colonnes avec des données manquantes
num_rows = len(df)
for i in range(39):
    col_name = f'Column{i+1}'
    missing_data_indices = np.random.choice(range(num_rows), size=int(num_rows*0.2), replace=False)
    df[col_name] = np.nan
    df.loc[missing_data_indices, col_name] = np.random.rand(len(missing_data_indices))

# Filtrer les dates du 1er janvier de chaque année parmi les dates uniques dans le dataframe
dates_january_first = df['Date'][df['Date'].dt.month == 1 & (df['Date'].dt.day == 1)].unique()

# Créer une liste vide de la même longueur que le dataframe
yticks_positions = [None] * len(df)

# Placer les dates du 1er janvier de chaque année aux bonnes positions dans la liste
for date in dates_january_first:
    index = df.index[df['Date'] == date][0]
    year = pd.to_datetime(date).year  # Convertir en datetime et extraire l'année
    yticks_positions[index] = year

# Définir le nombre de parties pour diviser le dataframe
num_parts = 4

# Calculer le nombre de colonnes par partie (arrondi supérieur pour que toutes les parties soient remplies)
columns_per_part = int(np.ceil(len(df.columns) / num_parts))

# Diviser les colonnes du dataframe en groupes égaux (ou moins pour le dernier groupe)
groups = [df.columns[i:i + columns_per_part] for i in range(0, len(df.columns), columns_per_part)]

# Affichez les matrices de données manquantes pour chaque groupe
for i, group in enumerate(groups):
    sub_df = df[group]
    plt.figure(figsize=(15, 6))  # Ajustez la taille de la figure selon vos préférences
    sns.heatmap(sub_df.isnull(), cmap="Blues", cbar=False)
    # Utilisez uniquement l'année comme étiquettes des ordonnées
    plt.yticks(ticks=range(len(yticks_positions)), labels=yticks_positions)
    plt.title(f"Partie {i+1}")
    plt.show()
```

<!-- #region id="5x5s-zRtZ_Ii" -->
## compléter les dates manquantes et interpoler
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="aJqDcxHjg-hM" executionInfo={"status": "ok", "timestamp": 1690379519934, "user_tz": -120, "elapsed": 209, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="761ccbaf-1a0c-4dac-b765-2c771f81793b"
# Exemple de DataFrame avec des dates manquantes et des valeurs quantitatives
data = {
    'date': ['2016-01-01', '2016-01-03', '2016-01-07', '2016-01-11'],
    'value': [10, 20, 40, 0]
    }

df = pd.DataFrame(data)

# Convertir la colonne 'date' en format de date
df['date'] = pd.to_datetime(df['date'])

# Afficher le DataFrame initial
df
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="pUbsAbT4hE-0" executionInfo={"status": "ok", "timestamp": 1690379521094, "user_tz": -120, "elapsed": 38, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="84e7c6d7-e9ec-4e2f-e112-e893f2988ee4"
# Étape 1 : Compléter les dates manquantes
df = df.set_index('date')
df_resampled = df.resample('D').asfreq().fillna(np.NaN)
df_resampled.reset_index( inplace=True)
df_resampled
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="Ef8HB9sOhQD3" executionInfo={"status": "ok", "timestamp": 1690296545147, "user_tz": -120, "elapsed": 225, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a00c0220-0a8e-4bfd-8487-5d1b5f0a047f"
# Étape 2 : Compléter les valeurs quantitatives linéairement
df_resampled['value_interpo'] = df_resampled['value'].interpolate(method='linear')

# Réinitialiser l'index
df_resampled = df_resampled.reset_index()
df_resampled
```

<!-- #region id="sMgG38UhesGL" -->
### Génération de Date avec frequence
<!-- #endregion -->

<!-- #region id="dQvdybXxewRr" -->
**ici on aura une date toutes semaines et 20 observations à partir de la date 2023-07-31**
<!-- #endregion -->

```python id="ep1HMzdKhmyo"
import pandas as pd
import numpy as np

# Création d'un DataFrame avec 20 observations (dates) qui sont toutes un lundi
data = {
    'Dates': pd.date_range(start='2023-07-31', periods=20, freq='W-MON'),
    'Value': np.random.randint(1, 100, size=20)  # Génère des entiers aléatoires entre 1 et 100
}

df = pd.DataFrame(data)
print(df)
```

```python id="bq2ojL1Rjxc9"
import pandas as pd
import numpy as np

# Création d'un DataFrame initial avec 5 observations (dates) qui sont toutes un lundi
data = {
    'Dates': pd.date_range(start='2023-07-31', periods=5, freq='W-MON'),
    'Value': np.random.randint(0, 9, size=5)  # Génère des entiers aléatoires entre 0 et 9
}

df = pd.DataFrame(data)
df['Dates'] = pd.to_datetime(df['Dates'])

# Ajout de la date "2023-06-05" avec la valeur 5
df = df.append({"Dates": pd.to_datetime("2023-06-05 00:00:00"), "Value": 5}, ignore_index=True)

# Génération des nouvelles dates chaque semaine entre "2023-06-05" et "2023-07-31" (le lundi)
new_dates = pd.date_range(start='2023-06-05', end='2023-07-30', freq='W-MON')

# Concaténation des nouvelles lignes au DataFrame original
for date in new_dates:
    df = df.append({"Dates": date, "Value": np.nan}, ignore_index=True)

# Tri du DataFrame par la colonne "Dates"
df.sort_values(by='Dates', inplace=True)

print(df)
```

<!-- #region id="DpL69DoxnMK0" -->
**Pour la granularité à la semaine**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="yYUzmsovmdHR" executionInfo={"status": "ok", "timestamp": 1690371999998, "user_tz": -120, "elapsed": 704, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4239fe9c-773b-4b6a-83e9-08eff5ce6eeb"
import pandas as pd
import numpy as np

Periode =10
# Générer les dates chaque semaine du lundi au dimanche
debut_semaine = pd.date_range(start='2019-09-16', periods=Periode, freq='W-MON')
fin_semaine = debut_semaine + pd.offsets.Week(weekday=6)

# Créer la nouvelle colonne au format "AAAA-MM-JJ/AAAA-MM-JJ" représentant la fenêtre de la semaine
df = pd.DataFrame({
    'Dates': debut_semaine.strftime('%Y-%m-%d') + '/' + fin_semaine.strftime('%Y-%m-%d'),
    'Value': np.random.randint(0, 9, size=Periode)  # Génère des entiers aléatoires entre 0 et 9
})

df.head(3)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="O1B50MLNm0nF" executionInfo={"status": "ok", "timestamp": 1690372426476, "user_tz": -120, "elapsed": 233, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9f54c797-f6a8-4d72-b935-6c971200dc90"
df['Dates'] = df['Dates'].apply(lambda x: x.split('/')[0])
df['Dates'] = pd.to_datetime(df['Dates'])
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="_0Vhpq15oQ2j" executionInfo={"status": "ok", "timestamp": 1690372000539, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2ad0f706-d9b3-4dc3-b53c-a627da9954e1"
# Ajout de la date "2023-06-05" avec la valeur 5
#df = df.append({"Dates": pd.to_datetime("2019-08-12"), "Value": 5}, ignore_index=True)
df = pd.concat([df, pd.DataFrame([{"Dates": pd.to_datetime("2019-08-12"), "Value": 5}])], ignore_index=True)
df.sort_values(by='Dates', inplace=True)
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 238} id="ED46CKysn4M6" executionInfo={"status": "ok", "timestamp": 1690372000539, "user_tz": -120, "elapsed": 12, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d53337e3-f190-446b-ce25-90d83e7acf77"
# Génération des nouvelles dates chaque semaine entre "2023-06-05" et "2023-07-31" (le lundi)
new_dates = pd.date_range(start='2019-08-13', end='2019-09-15', freq='W-MON')

# Concaténation des nouvelles lignes au DataFrame original
for date in new_dates:
    #df = df.append({"Dates": date, "Value": np.nan}, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{"Dates": date, "Value": np.nan}])], ignore_index=True)
df.sort_values(by='Dates', inplace=True)
# Transformation de chaque date en format "AAAA-MM-JJ/AAAA-MM-JJ" représentant la fenêtre de la semaine
df['Dates'] = df['Dates'].dt.strftime('%Y-%m-%d') + '/' + df['Dates'].dt.to_period('W').dt.end_time.dt.strftime('%Y-%m-%d')
df.head(6)
```

<!-- #region id="QN5V6VPn7xjS" -->
#Preprocessing
<!-- #endregion -->

```python id="Pi5m3ynNo1SR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1692686958355, "user_tz": -120, "elapsed": 316, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f05fb0d6-8b25-4c03-86d8-4cd842b8a6aa"
import pandas as pd

# Création du premier dataframe
df1 = pd.DataFrame({'Date': ['2022-01-01',
                             '2022-01-08',
                             '2022-01-15',
                             '2022-01-22'],
                    'Valeur1': [10, 20, 30, 40]})

df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d')
df1 = df1.sort_values(by='Date', ascending=True)

# Création du deuxième dataframe
df2 = pd.DataFrame({'Date': ['2022-01-01',
                             '2022-01-08',
                             '2022-01-16',
                             '2022-01-23'],
                    'Valeur2': [100, 200, 300, 400]})

df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d')
df2 = df2.sort_values(by='Date', ascending=True)

print(df1.shape)
print(df1)
print()
print(df2.shape)
print(df2)
```

<!-- #region id="mcDr-jWH74vL" -->
### Simple Merge
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="KynAHOZ48Nqt" executionInfo={"status": "ok", "timestamp": 1692686960624, "user_tz": -120, "elapsed": 8, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8e471943-564e-461a-ee28-62c3c4480d2e"
# Fusion des dataframes en utilisant la colonne de dates comme clé
merged_df = pd.merge(df1, df2, on='Date',how="outer")

# Affichage du dataframe fusionné
print(merged_df.shape)
print(merged_df)
```

<!-- #region id="RllBR-OU8c_V" -->
### Merge_asof
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5ORF3bPb8MMH" executionInfo={"status": "ok", "timestamp": 1692687360651, "user_tz": -120, "elapsed": 266, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3d2e6282-5056-4582-ede4-74cd827d7385"
merged_df = pd.merge_asof(df1, df2, on="Date", direction="nearest")
# Affichage du dataframe fusionné
print(merged_df.shape)
print(merged_df)
print()

merged_df = pd.merge_asof(df2, df1, on="Date", direction="nearest")
# Affichage du dataframe fusionné
print(merged_df.shape)
print(merged_df)
```

<!-- #region id="njU1yUI0fHAZ" -->
### Drop Duplicate avec Moyenne des Elements
<!-- #endregion -->

```python id="vOzmyZkh8gPB"
# Exemple de DataFrame
donnes = {'Date': ['2023-08-01 12:30:00', '2023-08-01 12:30:00', '2023-08-01 13:00:00', '2023-08-01 13:00:00'],
        'Valeur1': [10, 20, 15, 25],
        'Valeur2': [5, 10, 7, 12]}
df = pd.DataFrame(donnes)
df['Date'] = pd.to_datetime(df['Date'])

print(df, '\n')
# Supprimer les lignes en doublon et calculer la moyenne des valeurs dupliquées
resultat = df.groupby('Date', as_index=False).mean()

# Afficher le résultat
print(resultat)
```

<!-- #region id="ufLZVd3foFsp" -->

<!-- #endregion -->

```python id="9mn5Lft-oF3Y"
import pandas as pd

donnees = {
    'time': ['2023-07-19 16:32:08',
             '2023-07-19 16:32:28',
             '2023-07-19 16:32:50',
             '2023-07-19 17:33:10',
             '2023-07-19 19:33:28',
             '2023-07-19 20:33:00',
             '2023-07-19 20:33:50'],
    'value': [1, 2, 3, 6, 1, 2, 1]
}

df = pd.DataFrame(donnees)
df['time'] = pd.to_datetime(df['time'])

# Calculate the difference in hours between consecutive rows
df['time_diff'] = (df['time'].shift(-1) - df['time']).dt.total_seconds() / 3600

# Create a new dataframe for the rows with truncated time where the time difference is greater than 1 hour
new_df = pd.DataFrame(columns=df.columns)
to_add = df[df['time_diff'] > 1].copy()
to_add['time'] = to_add['time'] + pd.offsets.Hour(1)
to_add['time'] = to_add['time'].dt.floor('h')
to_add['value'] = 0
new_df = pd.concat([new_df, to_add]).sort_values('time').reset_index(drop=True)

# Drop the temporary 'time_diff' column from the original dataframe
df.drop(columns=['time_diff'], inplace=True)
new_df.drop(columns=['time_diff'], inplace=True)
print(df)
print(new_df)
```

<!-- #region id="kR_kt8dywzqX" -->

<!-- #endregion -->

```python id="xM--it_yoF6g"
# Créer un DataFrame avec vos données
donnees = {
    'time': ['2023-07-19 16:32:08', '2023-07-19 16:32:28', '2023-07-19 16:32:50',
             '2023-07-19 17:21:08', '2023-07-19 17:21:25', '2023-07-19 17:25:25','2023-07-19 17:44:03',
             '2023-07-19 17:45:10', '2023-07-19 17:46:20' , '2023-07-19 17:47:20'],
    'value': [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 1.0 , 0.0]
}
df = pd.DataFrame(donnees)
df['time'] = pd.to_datetime(df['time'])
print("\nOrigin\n",df)


# Calculer la différence de temps en secondes
df['diff_time_h'] = df['time'].diff().dt.total_seconds() /3600
# On décale d'un cran
df['diff_time_h'] = df['diff_time_h'].shift(-1)
print("\nAp calc2diff\n",df)
# Filtrer les lignes où la vitesse est supérieure à 0 km/h
df = df[df['value'] > 0]
# Ensuite, groupez les valeurs par heure et calculez la somme des valeurs pour chaque jour et date heure
# df['sum_by_hour'] = df.groupby(df['time'].dt.hour)['diff_time_h'].transform('sum')
df['sum_by_hour'] = df.groupby([df['time'].dt.date, df['time'].dt.hour])['diff_time_h'].transform('sum')

# On recupère la proba
df['proba'] = df['diff_time_h'] / df['sum_by_hour']

df['value'] = df['proba'] * df['value']

print("\nCalculé\n",df)

# Regrouper par heure et calculer la somme des distances et des temps
df = df.resample('H', on='time').agg({'diff_time_h': 'sum', 'value': 'sum'})

df.drop('diff_time_h', axis=1, inplace=True)
df.reset_index(inplace=True)
print("\nFinal\n",df)
```

```python id="4hXgtGRsoF-V"
# Création du DataFrame
donnees = {
    'time': ['2023-07-19 16:32:08', '2023-07-19 16:32:28', '2023-07-19 16:32:50', '2023-07-19 16:33:10', '2023-07-19 16:33:28', '2023-07-19 16:33:50'],
    'value': [np.nan, 1.0, np.nan, np.nan,  np.nan, 3.0]
}

df = pd.DataFrame(donnees)
df['time'] = pd.to_datetime(df['time'])

print(df.to_string(index=False))

# Appliquer la logique pour conserver uniquement la première ligne NaN dans chaque groupe
#df['group'] = (df['value'].notnull() != df['value'].notnull().shift()).cumsum()
#df = df.groupby('group').apply(lambda x: x if not x['value'].isnull().iloc[0] else x.iloc[[0]]).reset_index(drop=True)

df['group'] = (df['value'].notnull() != df['value'].notnull().shift()).cumsum()
for group_id, group_df in df.groupby('group'):
    if group_df['value'].isnull().any():
        first_nan_index = group_df.index[group_df['value'].isnull()].min()
        df.at[first_nan_index, 'value'] = 0

df.drop(columns='group', inplace=True)

print(df.to_string(index=False))
```

```python id="sXmhjfpozH8_"
# Créer un DataFrame avec vos données
donnees = {
    'time': ['2023-07-19 16:32:08', '2023-07-19 16:32:28', '2023-07-19 16:32:50',
             '2023-07-19 17:21:08', '2023-07-19 17:21:25', '2023-07-19 17:44:03',
             '2023-07-19 17:45:10', '2023-07-19 17:46:20' , '2023-07-19 17:47:20'],
    'vitesse': [1.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0 , 0.0]
}

df = pd.DataFrame(donnees)
df['time'] = pd.to_datetime(df['time'])

print(df)

# Calculer la différence de temps en secondes
df['diff_time_h'] = df['time'].diff().dt.total_seconds() /3600
# On décale d'un cran
df['diff_time_h'] = df['diff_time_h'].shift(-1)

# Filtrer les lignes où la vitesse est supérieure à 0 km/h
df = df[df['vitesse'] > 0]

# Ensuite, groupez les valeurs par heure et calculez la somme des valeurs pour chaque heure
df['sum_by_hour'] = df.groupby(df['time'].dt.hour)['diff_time_h'].transform('sum')

# On recupère la proba
df['proba'] = df['diff_time_h'] / df['sum_by_hour']

df['km_heure'] = df['proba'] * df['vitesse']

print(df)

# Calculer la distance parcourue dans chaque intervalle
#df['km'] = df['vitesse'] * df['diff_time_h'] * 3600

# Regrouper par heure et calculer la somme des distances et des temps
hourly_stats = df.resample('H', on='time').agg({'diff_time_h': 'sum', 'km_heure': 'sum'})

print(hourly_stats)
```
