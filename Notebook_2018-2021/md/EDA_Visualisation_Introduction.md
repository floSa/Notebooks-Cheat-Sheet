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

```python id="750Lc75kgKMK"
# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="IzMgA6d3i4Ct" executionInfo={"status": "ok", "timestamp": 1685649712379, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2ff8077e-ffef-4592-fbaa-9b473d0cfd2c"
# load the dataset

df = sns.load_dataset('titanic')
df.sample(2)
```

```python colab={"base_uri": "https://localhost:8080/"} id="s3gFPWk7i4Fr" executionInfo={"status": "ok", "timestamp": 1685649712380, "user_tz": -120, "elapsed": 15, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ae439038-74b1-4f19-85b9-04035384f025"
# print the DataFrame's shape
print(df.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="r0prYcFPi4IZ" executionInfo={"status": "ok", "timestamp": 1685649712380, "user_tz": -120, "elapsed": 12, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="dc1dcb63-d2a8-4717-b345-22e7d20321c0"
# print the DataFrame's data types
print(df.dtypes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="HWP04sTkjMrx" executionInfo={"status": "ok", "timestamp": 1685649730731, "user_tz": -120, "elapsed": 2, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e76d8a98-f618-487f-9fdb-4f5302673b9e"
# check for missing values
print(df.isnull().sum())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 544} id="9SWahDFtjbGt" executionInfo={"status": "ok", "timestamp": 1685649857807, "user_tz": -120, "elapsed": 6690, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f4253b04-5fac-4ca8-c071-d6aea0d10340"
sns.relplot(x="age", y="fare", col="pclass",
            hue="sex", style="sex",
            kind="line", data=df)
```

```python id="9g3RvjFzkNEf"
sns.relplot(data = df, x = 'age', y = 'fare', row = 'sex',
 col = 'Embarked', hue = 'Survived', size = 'Pclass',
 style = 'Parch', palette = 'flare')
```

<!-- #region id="k2X_V-FmkytD" -->
## Display Visualisation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="BjJLTie5jMug" executionInfo={"status": "ok", "timestamp": 1681304392351, "user_tz": -120, "elapsed": 713, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2c8232a6-e3dd-4664-d11b-3ff1c8417a79"
# visualize the distribution of a numeric column
plt.hist(df['age'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 466} id="j_B16iQCjMxP" executionInfo={"status": "ok", "timestamp": 1681304398539, "user_tz": -120, "elapsed": 981, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a1dd6c04-df34-4f37-c731-551775c4687e"
# visualize the distribution of a categorical column
df['sex'].value_counts().plot(kind='bar')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="RYJ8d8LHjM0D" executionInfo={"status": "ok", "timestamp": 1681304401813, "user_tz": -120, "elapsed": 235, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="93367596-17cd-4a94-cdb4-da8a92025068"
# calculate basic statistics for a numeric column
print(df['fare'].describe())
```

```python colab={"base_uri": "https://localhost:8080/"} id="lcz4b37BjM2y" executionInfo={"status": "ok", "timestamp": 1681304407538, "user_tz": -120, "elapsed": 402, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="023d1f20-1835-4f69-9d84-9bc6b794837e"
# calculate the correlation between two numeric columns
print(df['fare'].corr(df['survived']))
```

```python colab={"base_uri": "https://localhost:8080/"} id="agLpIJoQjWhY" executionInfo={"status": "ok", "timestamp": 1681304417992, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b98bc67d-8661-49f8-eae6-875be3cad224"
# group the data by a categorical column and calculate statistics
grouped_df = df.groupby('pclass')['survived'].mean()
print(grouped_df)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="MXTdbYv8jWj_" executionInfo={"status": "ok", "timestamp": 1681304425587, "user_tz": -120, "elapsed": 768, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="32db2631-0057-4ffe-b840-1d6fffb5e7cc"
# create a scatter plot to visualize the relationship between two numeric columns
plt.scatter(df['age'], df['fare'])
plt.xlabel('age')
plt.ylabel('fare')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 430} id="-klE51QRjWnE" executionInfo={"status": "ok", "timestamp": 1681304430978, "user_tz": -120, "elapsed": 522, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="eb4f1713-b592-4368-9cd8-6725ff3cd251"
# create a box plot to visualize the distribution of a numeric column
plt.boxplot(df['fare'])
plt.ylabel('fare')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 483} id="-ebh4bP7jasP" executionInfo={"status": "ok", "timestamp": 1681304442442, "user_tz": -120, "elapsed": 970, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9195062e-0a61-4ed4-bfee-8043ea9ac2a1"
# create a bar plot to visualize the mean of a numeric column for each category of a categorical column
df.groupby('sex')['age'].mean().plot(kind='bar')
plt.ylabel('Average age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="nMkS4Ua1javO" executionInfo={"status": "ok", "timestamp": 1681304455453, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7e5677a5-340d-4eb2-d19a-376b839310f5"
# create a pivot table to summarize the data
pivot_table = df.pivot_table(index='sex', columns='pclass', values='fare', aggfunc='mean')
print(pivot_table)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 435} id="ygRYKKdqjayO" executionInfo={"status": "ok", "timestamp": 1681304469686, "user_tz": -120, "elapsed": 765, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9a488f46-f500-456c-eabe-a0dec493e28a"
# create a heatmap to visualize the pivot table
plt.pcolor(pivot_table, cmap='Reds')
plt.colorbar()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 758} id="nFyMPEiIja1N" executionInfo={"status": "ok", "timestamp": 1681304477659, "user_tz": -120, "elapsed": 3604, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="536ace78-3fa5-4985-bd0f-d6ec06138d12"
# create a pairplot to visualize the relationships between multiple numeric columns
import seaborn as sns
sns.pairplot(df, vars=['age', 'fare', 'sibsp'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 427} id="UGOFS5AUja5B" executionInfo={"status": "ok", "timestamp": 1681304485055, "user_tz": -120, "elapsed": 999, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="bc0e1d8f-e1dd-47c8-c6bf-18644a669c2e"
# create a bar plot to visualize the count of a categorical column
df['embarked'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="6gfEcyBpjo0y" executionInfo={"status": "ok", "timestamp": 1681304491654, "user_tz": -120, "elapsed": 10, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="48775f9d-54cd-4a95-b17f-e9e6f47fb534"
# create a countplot to visualize the count of a categorical column by the categories of another categorical column
sns.countplot(x='sex', hue='pclass', data=df)
plt.show()
```

```python id="HVIyzwmPjqoB"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="2lvrwrupjqq4" executionInfo={"status": "ok", "timestamp": 1681304502808, "user_tz": -120, "elapsed": 734, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e2e503ae-1954-4af8-9933-6f384ca94747"
# create a point plot to visualize the mean of a numeric column by the categories of a categorical column
sns.pointplot(x='sex', y='age', data=df)
plt.ylabel('Average age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="Vl4G0UmkjqtY" executionInfo={"status": "ok", "timestamp": 1681304508045, "user_tz": -120, "elapsed": 631, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e5bf8f14-001c-4721-aab5-5ce68883e9e1"
# create a violin plot to visualize the distribution of a numeric column by the categories of a categorical column
sns.violinplot(x='sex', y='age', data=df)
plt.ylabel('age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="kfS_n0AXjqwM" executionInfo={"status": "ok", "timestamp": 1681304514043, "user_tz": -120, "elapsed": 922, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="dffb5de9-28a9-4118-be11-bd1606ec3ef2"
# create a box plot to visualize the distribution of a numeric column by the categories of a categorical column
sns.boxplot(x='sex', y='age', data=df)
plt.ylabel('age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="RcqkInJKjv-r" executionInfo={"status": "ok", "timestamp": 1681304524681, "user_tz": -120, "elapsed": 2236, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="dacd4dc5-28c1-462f-c37c-5bfa673350d1"
# create a swarm plot to visualize the distribution of a numeric column by the categories of a categorical column
sns.swarmplot(x='sex', y='age', data=df)
plt.ylabel('age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 307} id="tbg5xYggjwBX" executionInfo={"status": "ok", "timestamp": 1681304532454, "user_tz": -120, "elapsed": 490, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="470d195c-d9e5-41b6-aa7e-2835f2690d27"
# create a faceting grid to visualize the distribution of multiple numeric columns by the categories of a categorical column
g = sns.FacetGrid(df, col='sex')
g.map(plt.hist, 'age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 737} id="4ozuEY-QjwDr" executionInfo={"status": "ok", "timestamp": 1681304537894, "user_tz": -120, "elapsed": 1192, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="288ffd49-4bfa-4a18-c388-b602823f5385"
# create a heatmap to visualize the correlation between multiple numeric columns
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='RdYlGn', annot=True)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="a0Ur_Ln5jwG0" executionInfo={"status": "ok", "timestamp": 1681304559717, "user_tz": -120, "elapsed": 822, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="83520828-5871-40c2-f9fe-f08422c736ae"
# create a lag plot to check for autocorrelation in a numeric column
from pandas.plotting import lag_plot
lag_plot(df['fare'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 455} id="sxfQAf0ij7fq" executionInfo={"status": "ok", "timestamp": 1681304570989, "user_tz": -120, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="66f63738-dc3c-41df-fdfd-25a8b87cd2d6"
# create an autocorrelation plot to visualize the autocorrelation in a numeric column
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['fare'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 549} id="jeDIosW9j7i3" executionInfo={"status": "ok", "timestamp": 1681304585880, "user_tz": -120, "elapsed": 1303, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="64137458-08ac-433d-9d6e-df77398a1f6b"
# create a scatter plot matrix to visualize the relationships between multiple numeric columns
from pandas.plotting import scatter_matrix
scatter_matrix(df[['age', 'fare', 'sibsp']], alpha=0.2, figsize=(6, 6))
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="e36nVl05j72X" executionInfo={"status": "ok", "timestamp": 1681304603962, "user_tz": -120, "elapsed": 2272, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d821d4b2-8c14-47ac-d4ce-de4c183347f7"
# create a regression plot to visualize the relationship between two numeric columns
sns.regplot(x='age', y='fare', data=df)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="4MGGygTykGD7" executionInfo={"status": "ok", "timestamp": 1681304613963, "user_tz": -120, "elapsed": 1000, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3671240d-530b-468f-d20c-fce070322602"
# create a barplot to visualize the mean of a numeric column by the categories of a categorical column
sns.barplot(x='sex', y='age', data=df)
plt.ylabel('Average age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 536} id="YMjITReTkGHA" executionInfo={"status": "ok", "timestamp": 1681304620496, "user_tz": -120, "elapsed": 1441, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4a464b94-6da1-427e-ee32-8869fdf86488"
# create a pointplot to visualize the mean and confidence interval of a numeric column by the categories of a categorical column
sns.pointplot(x='sex', y='age', data=df, ci=95)
plt.ylabel('Average age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 507} id="6vbJ-qDykGJo" executionInfo={"status": "ok", "timestamp": 1681304644829, "user_tz": -120, "elapsed": 1883, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b0cfe558-8b02-4e70-8ac6-a5144abd9291"
# create a lmplot to visualize the relationship between two numeric columns and the categories of a categorical column
sns.lmplot(x='age', y='fare', hue='sex', data=df)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 506} id="ylagS1eKkGMv" executionInfo={"status": "ok", "timestamp": 1681304652586, "user_tz": -120, "elapsed": 1307, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="fe12c602-59e9-41b4-ef02-29c324e05c12"
# create a factorplot to visualize the distribution of a numeric column by the categories of a categorical column
sns.catplot(x='sex', y='age', data=df)
plt.ylabel('Average age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="sCu52WlekGOe" executionInfo={"status": "ok", "timestamp": 1681304668323, "user_tz": -120, "elapsed": 750, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7d2ef99f-8542-4174-964d-6a3fd841b025"
# create a boxenplot to visualize the distribution of a numeric column by the categories of a categorical column
sns.boxenplot(x='sex', y='age', data=df)
plt.ylabel('age')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="zq8qbKkTkWjz" executionInfo={"status": "ok", "timestamp": 1681304679802, "user_tz": -120, "elapsed": 1609, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8dba2921-1c7f-454c-9dd9-31b89673c659"
# create a distplot to visualize the distribution of a numeric column
sns.histplot(data=df, x="fare", kde=True)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="S6xIxf4KkYlp" executionInfo={"status": "ok", "timestamp": 1681304688014, "user_tz": -120, "elapsed": 278, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a6632033-8ee2-45f0-c988-e3faf57f820b"
# create a kdeplot to visualize the kernel density estimate of a numeric column
sns.kdeplot(df['fare'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 455} id="eD1d93oJkaWx" executionInfo={"status": "ok", "timestamp": 1681304695099, "user_tz": -120, "elapsed": 737, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c4e03c14-8773-453f-d284-6e502a8373fe"
# create a rugplot to visualize the distribution of a numeric column
sns.rugplot(df['fare'])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 607} id="fAumwPELgG4K" executionInfo={"status": "ok", "timestamp": 1681304702059, "user_tz": -120, "elapsed": 2804, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="febb2164-82e6-4fb5-992b-cb2452edeb49"
# create a jointplot to visualize the relationship between two numeric columns and their distributions
sns.jointplot(x='age', y='fare', data=df)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="MgqDMx8ENCas" executionInfo={"status": "ok", "timestamp": 1684151549689, "user_tz": -120, "elapsed": 670, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ab563dad-280c-47be-a13e-d747e7967419"
import pandas as pd
import matplotlib.pyplot as plt

# Créer le DataFrame
data = [[1.0, 2.0, "Normal", False],
        [1.0, 3.0, "Normal", True],
        [1.0, 2.0, "Normal", False],
        [2.0, 1.0, "Fault", True],
        [2.0, 2.0, "Maintenance", False],
        [3.0, 1.0, "Normal", True],
        [3.0, 3.0, "Normal", False],
        [3.0, 2.0, "Fault", False]]

df = pd.DataFrame(data, columns=["sensor_00", "sensor_01", "machine_status", "outliers"])

# Définir les couleurs à utiliser pour chaque catégorie de machine_status
colors = {"Normal": "green", "Fault": "red" , "Maintenance":"orange"}

# Définir les types de marqueur à utiliser pour chaque valeur de la colonne "outliers"
markers = {False: "o", True: "*"}

# Créer une colonne "color" dans le DataFrame en fonction de la valeur de "machine_status"
df["color"] = df["machine_status"].apply(lambda x: colors[x])

# Créer une colonne "marker" dans le DataFrame en fonction de la valeur de "outliers"
df["marker"] = df["outliers"].apply(lambda x: markers[x])

for mark in markers:
    d = df[df["outliers"]==mark]
    plt.scatter(d["sensor_00"], d["sensor_01"],
                c = d["color"],
                marker = markers[mark])

# Afficher le graphique

# Créer le scatterplot en utilisant les colonnes "color" et "marker" pour définir les couleurs et les types de marqueurs
#plt.scatter(df["sensor_00"], df["sensor_01"], c=df["color"], marker=df["marker"])

# Ajouter un titre et des étiquettes d'axe
plt.title("Relation entre sensor_00 et sensor_01")
plt.xlabel("sensor_00")
plt.ylabel("sensor_01")

# Ajouter une légende avec les couleurs correspondant à chaque catégorie de machine_status
plt.legend(handles=[plt.plot([],[],color=val, marker="s", ls="", label=key)[0] for key,val in colors.items()])
plt.legend(handles=[plt.plot([],[],color="black", marker=val, ls="", label=key)[0] for key,val in markers.items()])
# Ajouter une légende supplémentaire pour la catégorie "broken"
#plt.plot([],[],color="blue", marker="x", ls="", label="broken")
plt.legend()

# Afficher le graphique
plt.show()
```

<!-- #region id="VY00HBv5krQ4" -->
##Preprocessing de Base
<!-- #endregion -->

```python id="f6A9kRUolCMh"
# create a copy of the original DataFrame
df_preprocessed = df.copy()
```

```python id="belwJ2BHlCPD"
# handle missing values in the DataFrame
df_preprocessed['Age'].fillna(df_preprocessed['Age'].median(), inplace=True)
df_preprocessed.dropna(inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZPGGQsb6kraR" executionInfo={"status": "ok", "timestamp": 1681305358730, "user_tz": -120, "elapsed": 270, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e199e597-2652-45c4-b668-3c173bd3b313"
# create a copy of the original DataFrame
df_preprocessed = df.copy()

# handle missing values in the DataFrame
df_preprocessed['age'].fillna(df_preprocessed['age'].median(), inplace=True)
df_preprocessed.dropna(inplace=True)

# encode categorical variables using one-hot encoding
df_preprocessed = pd.get_dummies(df_preprocessed, columns=['sex', 'pclass'], prefix=['sex', 'pclass'])

# standardize the values of a numeric column
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_preprocessed['age_Std_Scale'] = scaler.fit_transform(df_preprocessed[['age']])

# normalize the values of a numeric column
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
df_preprocessed['age_normalized'] = normalizer.fit_transform(df_preprocessed[['age']])

# bin the values of a numeric column
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal')
df_preprocessed['age_binned'] = discretizer.fit_transform(df_preprocessed[['age']])

# apply a min-max scaling to a numeric column
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_preprocessed['age_MinMax_Scale'] = scaler.fit_transform(df_preprocessed[['age']])

# apply a robust scaling to a numeric column
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_preprocessed['age_Robust_Scale'] = scaler.fit_transform(df_preprocessed[['age']])

# apply a power transformation to a numeric column
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
df_preprocessed['age_trnsfo_yeo-johnson'] = transformer.fit_transform(df_preprocessed[['age']])

# apply a quantile transformation to a numeric column
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(output_distribution='normal')
df_preprocessed['age_trnsfo_normal'] = transformer.fit_transform(df_preprocessed[['age']])

# apply a box-cox transformation to a numeric column
from scipy.stats import boxcox

df_preprocessed['age_trnsfo_boxcox'], lambda_ = boxcox(df_preprocessed['age'])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 287} id="3xhZbRj_lhb_" executionInfo={"status": "ok", "timestamp": 1681305403582, "user_tz": -120, "elapsed": 280, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ea44c17b-3c31-448d-b271-2ab460829a90"
df_preprocessed.iloc[:,-13:].sample(5)
```

<!-- #region id="XByBie-yc0fp" -->
# Time Series
<!-- #endregion -->

```python id="L5mVEjLvlkHF"
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

```python colab={"base_uri": "https://localhost:8080/", "height": 76} id="EACMNR-pdAb-" executionInfo={"status": "ok", "timestamp": 1687415961130, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4e3b6a8a-4a80-4b91-a2db-493ef98f4f0f"
sns.color_palette(["#00798c", "#d1495b", '#edae49', '#66a182'])
```

```python id="Fv_nxTFSdBuW"
pd.options.display.max_columns = 999
```

```python id="AdwxAKcneSbF"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/"
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="5jbMjj3udJvN" executionInfo={"status": "ok", "timestamp": 1687416085210, "user_tz": -120, "elapsed": 382, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="39809f23-2e46-4874-e467-14fdb88a7cae"
data = pd.read_csv(path+'House_Prices_Advanced_Regression_Techniques.csv', usecols=["YearBuilt",'HouseStyle'])
data.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="GKxhPj3beg7V" executionInfo={"status": "ok", "timestamp": 1687416088119, "user_tz": -120, "elapsed": 9, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="34464bdf-04ea-4f1e-e670-a916e047fe3f"
data.describe(include='O')
```

<!-- #region id="Vyeat8PLjFRX" -->
## Bar plot
<!-- #endregion -->

<!-- #region id="nm-0Od-2jScx" -->
### Single
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 349} id="tWCCnx1ae_cd" executionInfo={"status": "ok", "timestamp": 1687416093494, "user_tz": -120, "elapsed": 718, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="14acf6df-1962-4fec-c5f3-37b5ab063398"
# Extraction de la colonne 'YearBuilt' du DataFrame 'data' et comptage des occurrences de chaque valeur
built = data['YearBuilt'].value_counts().sort_index()
# Création des objets figure et axes pour le graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
# Attribution des couleurs aux barres en fonction de leur hauteur (valeurs) dans la variable 'built'
color = ['#4a4a4a' if val != max(built) else '#e3120b' for val in built]
# Création d'un graphique à barres en utilisant l'indice de 'built' comme axe des x et les valeurs comme hauteurs des barres
ax.bar(built.index, built, color=color)
# Suppression des bordures supérieure et droite du graphique
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
# Ajout de lignes de grille sur le graphique
ax.grid()
# Affichage du graphique
plt.show()
```

<!-- #region id="8yxQtTGPgk-r" -->
### Multiple Subplots
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 692} id="qiIjZdFggAWa" executionInfo={"status": "ok", "timestamp": 1687416098742, "user_tz": -120, "elapsed": 2088, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="69ebd3ac-7827-4f06-cbce-918a26a3c7f8"
# Modification de la colonne 'HouseStyle' du DataFrame 'data'
# Si la valeur est dans ['SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'], elle est remplacée par 'ETC', sinon elle reste inchangée
data['HouseStyle'] = data['HouseStyle'].apply(lambda x : 'ETC' if x in ['SLvl', 'SFoyer', '1.5Unf', '2.5Unf', '2.5Fin'] else x)

# Création des objets figure et axes pour les sous-graphiques
fig, ax = plt.subplots(4, 1, figsize=(20, 12), sharex=True)

# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Comptage des maisons construites pour chaque année pour le style de maison actuel
    hs_built = data[data['HouseStyle']==hs]['YearBuilt'].value_counts()
    # Création d'un graphique à barres pour le style de maison actuel
    ax[i].bar(hs_built.index, hs_built, color=color[i], label=hs)
    ax[i].set_ylim(0, 50)  # Définition de la limite de l'axe des y pour mieux visualiser les barres
    ax[i].legend(loc='upper left')  # Ajout d'une légende dans le coin supérieur gauche du sous-graphique

    # Suppression des bordures supérieure et droite du sous-graphique
    for s in ['top', 'right']:
        ax[i].spines[s].set_visible(False)

# Affichage des sous-graphiques
plt.show()
```

<!-- #region id="jeBzjlD1hz7S" -->
### Multiple Overlaped
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="Bpsk6xFOhVna" executionInfo={"status": "ok", "timestamp": 1687416102672, "user_tz": -120, "elapsed": 874, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d9cba811-f311-423e-e37b-b82378bfcee0"
# Création des objets figure et axes pour le graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))

# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Comptage des maisons construites pour chaque année pour le style de maison actuel
    hs_built = data[data['HouseStyle']==hs]['YearBuilt'].value_counts()

    # Création d'un graphique à barres pour le style de maison actuel
    ax.bar(hs_built.index, hs_built, color=color[i], label=hs, alpha=0.4, edgecolor=color[i])

# Suppression des bordures supérieure et droite du graphique
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Définition de la limite de l'axe des y pour mieux visualiser les barres
ax.set_ylim(0, 50)

# Ajout d'une légende dans le coin supérieur gauche du graphique
ax.legend(loc='upper left')

# Affichage du graphique
plt.show()
```

<!-- #region id="1u6Im5jSiO8T" -->
###  Multiple Stacked Amout
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 349} id="l2vxwxA_iANx" executionInfo={"status": "ok", "timestamp": 1687416108848, "user_tz": -120, "elapsed": 1278, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="32ee9907-6e14-41e4-daaa-fcc4ecf26a44"
# Regroupement des données par 'HouseStyle' et comptage des maisons construites pour chaque année
data_sub = data.groupby('HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC','1.5Fin','2Story', '1Story']].cumsum(axis=0).T

# Création des objets figure et axes pour le graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))

# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données de maisons construites pour le style de maison actuel
    hs_built = data_sub[hs]

    # Création d'un graphique à barres pour le style de maison actuel
    ax.bar(hs_built.index, hs_built, color=color[i], label=hs)

# Suppression des bordures supérieure et droite du graphique
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Ajout d'une légende dans le coin supérieur gauche du graphique
ax.legend(loc='upper left')

# Ajout de lignes de grille sur le graphique
ax.grid()

# Affichage du graphique
plt.show()
```

<!-- #region id="L4aZS0XYir6S" -->
###  Multiple Stacked Ratio
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 345} id="2m-uUGXrijfB" executionInfo={"status": "ok", "timestamp": 1687416112416, "user_tz": -120, "elapsed": 2031, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="08332f8f-dcd3-4e0f-d242-cff1ccb3be4a"
# Regroupement des données par 'HouseStyle' et comptage des maisons construites pour chaque année
data_sub = data.groupby('HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC','1.5Fin','2Story', '1Story']].T
# Calcul de la somme des maisons construites par année
data_sum = data_sub.sum(axis=1)
# Normalisation des données en divisant par la somme totale des maisons construites par année
data_sub = (data_sub.T / data_sum).cumsum().T
# Création des objets figure et axes pour le graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données de maisons construites pour le style de maison actuel
    hs_built = data_sub[hs]
    # Création d'un graphique à barres pour le style de maison actuel
    ax.bar(hs_built.index, hs_built, color=color[i], label=hs)

# Suppression des bordures supérieure et droite du graphique
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Ajout d'une légende dans le coin supérieur gauche du graphique
ax.legend(loc='upper left')
# Ajout de lignes de grille sur le graphique
ax.grid()
# Affichage du graphique
plt.show()
```

<!-- #region id="ChMsGYr5jXIg" -->
## Line
<!-- #endregion -->

<!-- #region id="uAavrTy0jj5a" -->
### Single
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 349} id="KkwUzpGpi74G" executionInfo={"status": "ok", "timestamp": 1687416115629, "user_tz": -120, "elapsed": 796, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8f0bea7b-696b-4d8f-fd5d-7ff38452089c"
# Comptage des maisons construites pour chaque année et tri par index
built = data['YearBuilt'].value_counts().sort_index()
# Création des objets figure et axes pour le graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
# Tracé de la courbe en utilisant les index et les valeurs du comptage
ax.plot(built.index, built, color='#4a4a4a')

# Suppression des bordures supérieure et droite du graphique
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Ajout de lignes de grille sur le graphique
ax.grid()
# Affichage du graphique
plt.show()
```

<!-- #region id="cmm1gfc4jmAR" -->
### Single Area
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 349} id="HK0lrUTBjgKW" executionInfo={"status": "ok", "timestamp": 1687416118599, "user_tz": -120, "elapsed": 389, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f3b501d8-1f56-40d1-9e5c-6de23ff2d930"
# Comptage des maisons construites pour chaque année et tri par index
built = data['YearBuilt'].value_counts().sort_index()
# Création des objets figure et axes pour le graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
# Tracé de la courbe en utilisant les index et les valeurs du comptage
ax.plot(built.index, built, color='#4a4a4a')
# Remplissage de l'espace entre la courbe et l'axe des x
ax.fill_between(built.index, 0, built, color='#4a4a4a')

# Suppression des bordures supérieure et droite du graphique
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Ajout de lignes de grille sur le graphique
ax.grid()
# Affichage du graphique
plt.show()
```

<!-- #region id="oMlq3y7UkB_z" -->
### Multiple Subplots
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 692} id="rHHJfhqkj5jw" executionInfo={"status": "ok", "timestamp": 1687416121697, "user_tz": -120, "elapsed": 1072, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0219b629-a11a-4e3b-c150-a9a098ccb20e"
# Création des objets figure et axes pour le graphique avec 4 sous-graphiques arrangés en 4 lignes
fig, ax = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données de maisons construites pour le style de maison actuel
    hs_built = data[data['HouseStyle']==hs]['YearBuilt'].value_counts().sort_index()
    # Tracé de la courbe en utilisant les index et les valeurs du comptage
    ax[i].plot(hs_built.index, hs_built, color=color[i], label=hs)
    # Définition de la limite supérieure de l'axe des y pour chaque sous-graphique
    ax[i].set_ylim(0, 50)
    # Ajout d'une légende dans le coin supérieur gauche de chaque sous-graphique
    ax[i].legend(loc='upper left')

    # Suppression des bordures supérieure et droite de chaque sous-graphique
    for s in ['top', 'right']:
        ax[i].spines[s].set_visible(False)

# Affichage du graphique
plt.show()
```

<!-- #region id="ub9hu2ZikYoI" -->
###  Multiple Horizon Chart
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 692} id="3LLHJNPVkUK_" executionInfo={"status": "ok", "timestamp": 1687416127036, "user_tz": -120, "elapsed": 2771, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d304d5c5-b19d-4522-fe9e-96a3dbf37256"
# Création des objets figure et axes pour le graphique avec 4 sous-graphiques arrangés en 4 lignes
fig, ax = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données de maisons construites pour le style de maison actuel
    hs_built = data[data['HouseStyle']==hs]['YearBuilt'].value_counts().sort_index()
    # Tracé de la courbe en utilisant les index et les valeurs du comptage
    ax[i].plot(hs_built.index, hs_built, color=color[i], label=hs)
    # Remplissage de l'espace entre la courbe et l'axe des x
    ax[i].fill_between(hs_built.index, 0, hs_built, color=color[i])
    # Définition de la limite supérieure de l'axe des y pour chaque sous-graphique
    ax[i].set_ylim(0, 30)
    # Ajout d'une légende dans le coin supérieur gauche de chaque sous-graphique
    ax[i].legend(loc='upper left')

# Ajustement de l'espacement vertical entre les sous-graphiques
plt.subplots_adjust(hspace=0)
# Affichage du graphique
plt.show()
```

<!-- #region id="pqwl8h6Nk5gX" -->
### Multiple Overlaped
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="_QJ-4mKQkwZn" executionInfo={"status": "ok", "timestamp": 1687416130141, "user_tz": -120, "elapsed": 963, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a05cf3ca-9971-45e4-c249-0abf2347c5d3"
# Création des objets figure et axes pour le graphique avec un seul sous-graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 6))
# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données de maisons construites pour le style de maison actuel
    hs_built = data[data['HouseStyle']==hs]['YearBuilt'].value_counts().sort_index()
    # Tracé de la courbe en utilisant les index et les valeurs du comptage
    ax.plot(hs_built.index, hs_built, color=color[i], label=hs)

# Définition de la limite supérieure de l'axe des y pour le sous-graphique principal
ax.set_ylim(0, 50)
# Ajout de la légende dans le coin supérieur gauche du sous-graphique principal
ax.legend(loc='upper left')

# Suppression des bordures supérieure et droite du sous-graphique principal
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Activation de la grille pour le sous-graphique principal
ax.grid()
# Affichage du graphique
plt.show()

```

<!-- #region id="KjQJZotclUy3" -->
### Multiple Overlaped Area
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 410} id="m4vx7BiulEq-" executionInfo={"status": "ok", "timestamp": 1687416133146, "user_tz": -120, "elapsed": 572, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="91879efe-95ba-478f-d528-f6d6b1b18ca3"
# Création des objets figure et axes pour le graphique avec un seul sous-graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 6))
# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données de maisons construites pour le style de maison actuel
    hs_built = data[data['HouseStyle']==hs]['YearBuilt'].value_counts().sort_index()
    # Tracé de la courbe en utilisant les index et les valeurs du comptage
    ax.plot(hs_built.index, hs_built, color=color[i], label=hs)
    # Remplissage de la zone entre la courbe et l'axe des x avec une couleur et une transparence définies
    ax.fill_between(hs_built.index, 0, hs_built, color=color[i], alpha=0.4)

# Définition de la limite supérieure de l'axe des y pour le sous-graphique principal
ax.set_ylim(0, 50)
# Ajout de la légende dans le coin supérieur gauche du sous-graphique principal
ax.legend(loc='upper left')

# Suppression des bordures supérieure et droite du sous-graphique principal
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Activation de la grille pour le sous-graphique principal
ax.grid()
# Affichage du graphique
plt.show()
```

<!-- #region id="-i4nuUQfljcP" -->
### Multiple Stacked Amout
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 349} id="OBYPctAplazz" executionInfo={"status": "ok", "timestamp": 1687416136578, "user_tz": -120, "elapsed": 863, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f2b93909-8c3d-4083-9cd6-d97dc0ff2c9b"
# Création du tableau de données avec le regroupement par 'HouseStyle' et 'YearBuilt'
data_sub = data.groupby('HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC','1.5Fin','2Story', '1Story']].cumsum(axis=0).T
# Création des objets figure et axes pour le graphique avec un seul sous-graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
# Définition des couleurs pour chaque style de maison
color = ["#00798c", "#d1495b", '#edae49', '#66a182']

# Boucle sur chaque index (style de maison) dans les valeurs comptées de 'HouseStyle'
for i, hs in enumerate(data['HouseStyle'].value_counts().index):
    # Sélection des données cumulatives de maisons construites pour le style de maison actuel
    hs_built = data_sub[hs]
    # Remplissage de la zone entre la courbe et l'axe des x avec une couleur définie
    ax.fill_between(hs_built.index, 0, hs_built, color=color[i], label=hs)

# Suppression des bordures supérieure et droite du sous-graphique principal
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# Ajout de la légende dans le coin supérieur gauche du sous-graphique principal
ax.legend(loc='upper left')
# Activation de la grille pour le sous-graphique principal
ax.grid()
# Affichage du graphique
plt.show()

```

<!-- #region id="TfKsZ8yfl44m" -->
 ### Multiple Stacked Stream graph
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 390} id="V0bIpfm7lwg8" executionInfo={"status": "ok", "timestamp": 1687416139654, "user_tz": -120, "elapsed": 624, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c24d645d-6172-4f8b-d152-07002463c921"
# Création du tableau de données avec le regroupement par 'HouseStyle' et 'YearBuilt'
data_sub = data.groupby('HouseStyle')['YearBuilt'].value_counts().unstack().fillna(0).loc[['ETC','1.5Fin','2Story', '1Story']].cumsum(axis=0).T
# Insertion d'une colonne 'base' contenant des zéros au début du tableau
data_sub.insert(0, "base", np.zeros(len(data_sub)))
# Décalage des données cumulatives par rapport aux valeurs comptées de 'YearBuilt'
data_sub = data_sub.add(-data['YearBuilt'].value_counts() / 2, axis=0)
# Création des objets figure et axes pour le graphique avec un seul sous-graphique
fig, ax = plt.subplots(1, 1, figsize=(18, 5))
# Définition des couleurs pour chaque style de maison (inversées)
color = ["#00798c", "#d1495b", '#edae49', '#66a182'][::-1]
# Liste des styles de maison
hs_list = data_sub.columns

# Boucle sur chaque index (style de maison) dans la liste des styles de maison
for i, hs in enumerate(hs_list):
    # Si l'index est 0 (correspondant à la colonne 'base'), on passe à l'itération suivante
    if i == 0:
        continue
    # Remplissage de la zone entre les courbes précédente et actuelle avec une couleur définie
    ax.fill_between(hs_built.index, data_sub.iloc[:, i-1], data_sub.iloc[:, i], color=color[i-1])

# Suppression des bordures supérieure, droite, inférieure et gauche du sous-graphique principal
for s in ['top', 'right', 'bottom', 'left']:
    ax.spines[s].set_visible(False)

# Suppression des ticks de l'axe des y
ax.set_yticks([])
# Ajout de la légende dans le coin supérieur gauche du sous-graphique principal
ax.legend(loc='upper left')
# Activation de la grille pour l'axe des x du sous-graphique principal
ax.grid(axis='x')
# Affichage du graphique
plt.show()
```

<!-- #region id="m0s1NjDoi_w1" -->
# Autres Plot (sur colonne de dataframe)
<!-- #endregion -->

```python id="8I0R7a5gmD0V"
import pandas as pd

# Créer un dictionnaire de données
data = {
    'machine': ['A', 'A',  'B', 'B', 'C', 'C'],
    'x': [[-10,1], [-12,2], [-13,3], [-14,4], [-15,5], [-16,6]],
    'y': [[1,10], [2,12], [3,13], [4,14], [5,15], [6,16]],
    'sex': ['f', 'h',  'f', 'h', 'f', 'h']
}

# Créer le dataframe à partir du dictionnaire
df = pd.DataFrame(data)

# Afficher le dataframe
print(df)
```

```python id="IRrsJGRyjb72"
# Créer les subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Obtenir les modalités uniques de la colonne "machine"
machines = df['machine'].unique()

# Parcourir les modalités et tracer les line plots correspondants pour chaque sexe
for i, machine in enumerate(machines):
    # Sélectionner le subplot correspondant
    ax = axes.flatten()[i]

    # Filtrer les données pour la machine spécifique
    machine_data = df[df['machine'] == machine]

    # Obtenir les modalités uniques de la colonne 'sex' pour la machine spécifique
    sexes = machine_data['sex'].unique()

    # Parcourir les modalités de sexe et tracer les line plots correspondants
    for sex in sexes:
        # Filtrer les données pour le sexe spécifique
        sex_data = machine_data[machine_data['sex'] == sex]
        #print(sex_data)
        print("type", type(sex_data['x']))
        print(sex_data['x'].to_list()[0], sex_data['y'].to_list()[0], sex)
        # Tracer le line plot de la colonne 'x' par rapport à la colonne 'y' pour le sexe spécifique
        ax.plot(list(sex_data['x'])[0], list(sex_data['y'])[0], label=sex)

    # Ajouter des labels, une légende et un titre pour chaque subplot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Machine {machine}")
    ax.legend()

# Ajuster l'espacement entre les subplots
plt.tight_layout()

# Afficher le graphique
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 419} id="UhFwM4rAjNz-" executionInfo={"status": "ok", "timestamp": 1689175274846, "user_tz": -120, "elapsed": 689, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="579be0d8-0606-43c1-d51d-5501e74413e3"
# Exemple de données
data = {
    'Date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
    'Etat': ['A', 'C', 'B', 'C', 'A']
}
# Changer la taille de la figure
plt.figure(figsize=(20, 5))

# Créer le DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Trier le DataFrame selon la colonne 'Date'
df = df.sort_values('Date')

# Spécifier l'ordre prédéfini des modalités sur l'axe y
ordre_modalites = ['B', 'A', 'C']

# Mapper les modalités à des valeurs numériques correspondant à l'ordre souhaité
mapping_modalites = {modalite: i for i, modalite in enumerate(ordre_modalites)}

# Créer le line plot en utilisant les valeurs de l'axe y
plt.plot(df['Date'], df['Etat'].map(mapping_modalites))

# Spécifier l'ordre prédéfini des modalités sur l'axe y
modalites_positions = np.arange(len(ordre_modalites))
plt.yticks(modalites_positions, ordre_modalites)


# Afficher le graphique
plt.show()

```

```python id="uu7v5y3QnNgN"
# Créer une nouvelle figure avec des sous-graphiques
fig, axes = plt.subplots(nrows=len(df_activite.columns)-1, ncols=1, figsize=(20, 30))
couleurs = ['red', 'blue', 'orange', 'green']
ordre_modalites = ["State: 0(Set: 25)" , "State: 1(Set: 25)"]
# Tracer chaque colonne dans un sous-graphique différent
j = -1
for i, colonne in enumerate(df_activite.columns[1:]):
    if i % 4 == 0 :
        j += 1
    ax = axes[i]  # Récupérer le sous-graphique correspondant
    test = df_activite[df_activite[colonne].isin(["State: 1(Set: 25)","State: 0(Set: 25)"])]

    # Mapper les modalités à des valeurs numériques correspondant à l'ordre souhaité
    mapping_modalites = {modalite: i for i, modalite in enumerate(ordre_modalites)}


    ax.plot(test['Date'], test[colonne].map(mapping_modalites) , color=couleurs[j])  # Tracer la colonne spécifique
    ax.set_xlabel('Date')
    ax.set_ylabel(pattern_Dict[colonne])
    ax.set_title(f'Line Plot de la colonne {pattern_Dict[colonne]} en fonction de la colonne Date')


    # Spécifier l'ordre prédéfini des modalités sur l'axe y
    modalites_positions = np.arange(len(ordre_modalites))
    ax.set_yticks(modalites_positions)
    ax.set_yticklabels(ordre_modalites)

# Ajuster les espacements entre les sous-graphiques
plt.tight_layout()

# Afficher les sous-graphiques
plt.show()
```

<!-- #region id="INb_mEaqHZ2F" -->
# Visualisations Expertes
<!-- #endregion -->

<!-- #region id="sNfvTcKRH7l5" -->
**Ici on s'appuiera sur un dataframe de variables Quategorielles**
<!-- #endregion -->

```python id="qdMTgg7gHy39"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
%matplotlib inline
```

<!-- #region id="4Kuc5JfTHobB" -->
**Génération du dataframe**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="vEys_UwtHglx" executionInfo={"status": "ok", "timestamp": 1704701580871, "user_tz": -60, "elapsed": 10, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b7e85c1f-767f-43ea-9fcb-8451fd508ba8"
n=1000
dict_loca = {1:'New York', 2:'Boston', 3:'Other'}
list_loca = [dict_loca.get(random.randint(1,3)) for i in range(0,n)]

dict_prod = {1:'food', 2:'beverage'}
list_prod = [dict_prod.get(random.randint(1,2,)) for i in range(0,n)]

dict_pay = {1:'cash', 2:'credit card'}
list_pay = [dict_pay.get(random.randint(1,2,)) for i in range(0,n)]

dict_gender = {1:'male', 2:'female'}
list_gender = [dict_gender.get(random.randint(1,2)) for i in range(0,n)]

dict_age = {1:'<25', 2:'25-50', 3:'>=50'}
list_age = [dict_age.get(random.randint(1,3)) for i in range(0,n)]

df = pd.DataFrame(zip(list_loca, list_prod, list_pay, list_gender, list_age),
                  columns=['location', 'product', 'payment', 'gender', 'age'])
df.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="ayYxqhmiHvMZ" executionInfo={"status": "ok", "timestamp": 1704701600072, "user_tz": -60, "elapsed": 473, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0184ff92-9ad0-4b18-fe70-30407c475ebc"
count_gb = df.groupby(['location', 'product', 'payment',
                       'gender', 'age'])['age'].count()
df_m = df.groupby(['location', 'product', 'payment',
                   'gender', 'age']).count()
df_m['freq'] = list(count_gb)
df_m.reset_index(inplace=True)
df_m.head()
```

<!-- #region id="V-zHKQ9FKNZq" -->
#### Diagramme circulaire à plusieurs niveaux en étoile
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="J9wh72UIIhyz" executionInfo={"status": "ok", "timestamp": 1704701996282, "user_tz": -60, "elapsed": 8400, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e7acdc32-86b1-4de6-9172-f25a6dba416d"
import plotly.express as px
fig = px.sunburst(df_m, path=['location', 'product',
                              'payment', 'gender', 'age'],
                  values='freq',
                  color='freq',
                  color_continuous_scale='rdbu_r',
                  width=960, height=600
                 )
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
fig.show()
```

<!-- #region id="xK-GSdByKnvK" -->
#### Diagramme rectengulaire
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 617} id="oGZf11CnKAlp" executionInfo={"status": "ok", "timestamp": 1704702125330, "user_tz": -60, "elapsed": 1023, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c8ad6fd1-61b9-4a4f-f2c4-4ab97091d288"
import plotly.express as px
fig = px.treemap(df_m, path=[px.Constant("all"), 'location', 'product',
                             'payment', 'gender', 'age'],
                 values='freq',
                 color='freq',
                 color_continuous_scale='viridis',
                 width=960, height=600
                )
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
fig.show()
```

<!-- #region id="rE0-4vTuLY4J" -->
#### Multiple Heatmap
<!-- #endregion -->

```python id="JJ6lKTKyL53j"
import itertools
pair_loca_prod_gend = list(itertools.product(dict_loca.values(),
                                             dict_prod.values(),
                                             dict_gender.values()))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 846} id="zrJnMyahKh5J" executionInfo={"status": "ok", "timestamp": 1704702499926, "user_tz": -60, "elapsed": 5052, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2789d096-775d-44c5-b1d1-604555b0cee5"
#get cartesian product of column and row index for assigning subplot
columns = list(range(4))
rows = list(range(3))
pair_cr = list(itertools.product(rows, columns))

fig, axes = plt.subplots(3, 4, figsize=(16.5, 12))
for i,cr in zip(pair_loca_prod_gend, pair_cr):
    df_s = df_m[(df_m['location']==i[0]) & (df_m['product']==i[1]) & (df_m['gender']==i[2])]
    df_ct = pd.crosstab(df_s.payment, df_s.age,
                        values=df_s.freq, aggfunc='mean')
    sns.heatmap(df_ct, annot=True, cmap='coolwarm', cbar=False,
                vmin=10, vmax=20, ax=axes[cr[0],cr[1]] )
    axes[cr[0],cr[1]].set(ylabel=None)
    axes[cr[0],cr[1]].set(xlabel=None)
    axes[cr[0],cr[1]].set_title(i[0]+' - '+i[1]+' - '+i[2])
plt.show()
```

<!-- #region id="d630DHP7MIBL" -->
#### Multiple bar chart
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 880} id="MOxkKD0cMLcv" executionInfo={"status": "ok", "timestamp": 1704702578070, "user_tz": -60, "elapsed": 4708, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a71228c0-8b1f-43a6-d4bb-4ebf26f90d56"
sns.set_style('darkgrid')
fig, axes = plt.subplots(3, 4, figsize=(16.5, 12))
for i,cr in zip(pair_loca_prod_gend, pair_cr):
    df_s = df_m[(df_m['location']==i[0]) & (df_m['product']==i[1]) & (df_m['gender']==i[2])]
    sns.barplot(data=df_s, x='age', y='freq', hue='payment',
                ax=axes[cr[0],cr[1]], palette=['red','orange'])
    axes[cr[0],cr[1]].set(ylabel=None)
    axes[cr[0],cr[1]].set(xlabel=None)
    axes[cr[0],cr[1]].get_legend().remove()
    axes[cr[0],cr[1]].set_title(i[0]+' - '+i[1]+' - '+i[2])

#legend
handles, labels = axes[cr[0],cr[1]].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1,
           bbox_to_anchor=(0.9, 0.95), frameon=False)
plt.show()
```

<!-- #region id="_al_3EDDMfCo" -->
#### Diagramme en barres empilées
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="uHGqdOhAMPip" executionInfo={"status": "ok", "timestamp": 1704702671035, "user_tz": -60, "elapsed": 6647, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8b2dab77-58a2-4520-9e37-795842e39e8d"
fig, axes = plt.subplots(3, 4, figsize=(16.5, 14))
for i,cr in zip(pair_loca_prod_gend, pair_cr):
    df_s = df_m[(df_m['location']==i[0]) & (df_m['product']==i[1]) & (df_m['gender']==i[2])]
    df_ct = pd.crosstab(df_s.payment, df_s.age, values=df_s.freq, aggfunc='mean')
    df_ct.plot(kind='bar', stacked=True, color=['red', 'orange', 'blue'],
               ax=axes[cr[0],cr[1]])
    axes[cr[0],cr[1]].tick_params(axis='x', labelrotation = 0)
    axes[cr[0],cr[1]].set(ylabel=None)
    axes[cr[0],cr[1]].set(xlabel=None)
    axes[cr[0],cr[1]].get_legend().remove()
    axes[cr[0],cr[1]].set_title(i[0]+' - '+i[1]+' - '+i[2])

#legend
handles, labels = axes[cr[0],cr[1]].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1,
           bbox_to_anchor=(0.9, 0.95), frameon=False)

plt.show()
```

<!-- #region id="TGn_wTwIM2tu" -->
#### Manipuler plusieurs dimensions à l'aide d'un diagramme de coordonnées parallèles
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="Imfj-vkBMluN" executionInfo={"status": "ok", "timestamp": 1704702897891, "user_tz": -60, "elapsed": 288, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="26178f5b-cab8-4654-c2bd-26653e59e80a"
# Créer les plages de fréquence
f_range = pd.cut(x=df_m['freq'], bins=[0, 5, 10, 15, 20, 25])
df_m['freq_range'] = [str(i) for i in f_range]

# Mapper les valeurs aux emplacements
df_m['location_map'] = df_m['location'].map({'New York':1, 'Boston':2, 'Other':3})

# Assigner les couleurs aux catégories
color_mapping = {'New York': 'red', 'Boston': 'orange', 'Other': 'blue'}
df_m['location_color'] = df_m['location'].map(color_mapping)

# Créer la visualisation parallel categories
fig = px.parallel_categories(df_m, color='location_color',
                             dimensions=['location', 'product', 'payment',
                                         'gender', 'age', 'freq_range'])
fig.update(layout_coloraxis_showscale=False)
fig.show()
```

<!-- #region id="MUgMts6FOBTu" -->
#### Diagramme en mosaïque des relations entre les composantes  
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="tYHNRzOvNErU" executionInfo={"status": "ok", "timestamp": 1704706198350, "user_tz": -60, "elapsed": 1220, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="bd2e2a17-d1e5-44f1-da72-d60d6a071700"
from statsmodels.graphics.mosaicplot import mosaic
plt.rcParams["figure.figsize"] = [16, 29]
plt.rcParams.update({'font.size': 16})
ax = mosaic(df, ['location', 'age', 'payment', 'gender'])
plt.show()
```

```python id="iRXbxFFoOQdd" colab={"base_uri": "https://localhost:8080/", "height": 752} executionInfo={"status": "ok", "timestamp": 1707747623024, "user_tz": -60, "elapsed": 5253, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="34c24ec1-8043-492c-bcba-56a161646a36"
import holoviews as hv
from holoviews import opts
import pandas as pd
import numpy as np
hv.extension('bokeh')

# Sample matrix representing the export volumes between 5 countries
export_data = np.array([[0, 50, 30, 20, 10],
                        [10, 0, 40, 30, 20],
                        [20, 10, 0, 35, 25],
                        [30, 20, 10, 0, 40],
                        [25, 15, 30, 20, 0]])

labels = ['USA', 'China', 'Germany', 'Japan', 'India']

# Creating a pandas DataFrame
df = pd.DataFrame(export_data, index=labels, columns=labels)
df = df.stack().reset_index()

df.columns = ['source', 'target', 'value']

# Creating a Chord object
chord = hv.Chord(df)

# Styling the Chord diagram
chord.opts(
    opts.Chord(
        cmap='Category20', edge_cmap='Category20',
        labels='source', label_text_font_size='10pt',
        edge_color='source', node_color='index',
        width=700, height=700
    )
).select(value=(5, None))

# Display the plot
chord
```

```python id="LULi8ER-sIiT"
import plotly.express as px
import numpy as np

df = px.data.gapminder().query("year == 2007")

fig = px.sunburst(df, path=['continent', 'country'],
                  values='pop',
                  color='lifeExp',
                  hover_data=['iso_alpha'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))
fig.show()
```
