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

<!-- #region id="_yYIm95r0NdJ" -->
* **Import**
<!-- #endregion -->

```python id="L1MHtDP70Omu"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import random

import matplotlib.pyplot as plt
import seaborn as sns
```

```python id="oJmCkTqf0l10"
import warnings
warnings.filterwarnings("ignore")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="ixqcf2CLv41p" executionInfo={"status": "ok", "timestamp": 1694176122339, "user_tz": -120, "elapsed": 1042, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2acfd1d0-3705-4fca-b9c4-278fcf7cdd04"
import seaborn as sns
df = sns.load_dataset("tips")
df.head()
```

<!-- #region id="74JVkjvLwpZk" -->
# Satistiques Descriptives
<!-- #endregion -->

<!-- #region id="YZ1Uj8bd3IKd" -->
# Analyses Univariées
<!-- #endregion -->

<!-- #region id="wlY7LCKi2gC0" -->
### Résumé
<!-- #endregion -->

<!-- #region id="8nuskm6J2JKw" -->
* **Aperçu des varaibles et leur type**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 381} id="j554BLxfydt3" executionInfo={"status": "ok", "timestamp": 1694178462867, "user_tz": -120, "elapsed": 2086, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4fc66ce3-f8d6-495a-e690-c53ba5498c5f"
# Calculez le pourcentage de chaque type de variable
type_counts = df.dtypes.value_counts(normalize=True).reset_index()
type_counts.columns = ['type', 'freq']

# Convertissez les types catégoriels en 'object'
type_counts['type'] = type_counts['type'].astype(str)

# Regroupez par "type" et faites la somme des valeurs de "freq"
type_counts_grouped = type_counts.groupby('type')['freq'].sum().reset_index()

# Créez une liste de couleurs pour les barres
couleurs = ['royalblue', 'darkorange', 'forestgreen', 'crimson', 'darkviolet', 'gold']

# Créez une figure avec deux sous-graphiques (un camembert et un barplot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

# Sous-graphique 1 : Camembert
ax1.pie(type_counts_grouped['freq'], labels=type_counts_grouped['type'], autopct='%1.1f%%', startangle=140, colors=couleurs)
ax1.axis('equal')  # Pour s'assurer que le camembert est un cercle.
ax1.set_title("Camembert - Pourcentage de chaque type de variable")

# Sous-graphique 2 : Barplot
for i, (t, f) in enumerate(zip(type_counts_grouped['type'], type_counts_grouped['freq'])):
    ax2.bar(i, f, color=couleurs[i], label=t)
    ax2.text(i, f, f'{f:.1%}', ha='center', va='bottom', fontsize=12)

ax2.set_xlabel("Type de variable")
ax2.set_ylabel("Fréquence")
ax2.set_title("Barplot - Fréquence de chaque type de variable")
ax2.set_xticks(range(len(type_counts_grouped['type'])))
ax2.set_xticklabels(type_counts_grouped['type'])
ax2.legend()

plt.tight_layout()
plt.show()
```

<!-- #region id="j_nhRlrQ2QEP" -->
* **Varibles qualitatives Sumary**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 175} id="xKZlJDvYwuhm" executionInfo={"status": "ok", "timestamp": 1694176124157, "user_tz": -120, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="41eedbf2-ab9a-4b30-88b4-05feca9ae164"
df_quali = df.select_dtypes(['object','category','bool'])
df_quali.describe()
```

<!-- #region id="HiyWUuS82ZdH" -->
* **Varibles quantitatives Sumary**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="xvRPGl8gwo7T" executionInfo={"status": "ok", "timestamp": 1694176127592, "user_tz": -120, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b84db79c-df16-456d-bcec-e7105acd3b0c"
df_quanti = df.select_dtypes(['float','int64'])
df_quanti.describe()
```

<!-- #region id="mAH-bDHW1hW6" -->
###Distribution
<!-- #endregion -->

<!-- #region id="RY-RIzw52l_o" -->
* **Varibles quantitatives**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 434} id="yLQeG9dl0Hm5" executionInfo={"status": "ok", "timestamp": 1694176222515, "user_tz": -120, "elapsed": 2294, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d4a3bfc2-80fa-4743-e28b-35baed12f740"
# Distribution des Variables quantitatives.

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(30, 8))
for col ,ax in zip( df_quanti , axes.flat):
    #sns.distplot(df_quanti[col],ax=ax)
    sns.histplot(df_quanti[col],ax=ax, kde=True,stat='density', linewidth=0)

plt.show()
```

<!-- #region id="URVhGs4V2zoQ" -->
* **Varibles qualitatives**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 301} id="nsV4hZl41Nsx" executionInfo={"status": "ok", "timestamp": 1681993286754, "user_tz": -120, "elapsed": 632, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="122f3b3d-b04e-40ec-c8c9-e1947ebe2516"
# Distribution des Variables qualitatives.

fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(25, 15))
for col ,ax in zip( df_quali, axes.flat):
  if len(df_quali[col].value_counts())<30:
     df_quali[col].value_counts().plot.pie(ax=ax)

plt.show()
```

<!-- #region id="JFnXgI4zUutE" -->
### Distrib sans Outliers (quantile)
<!-- #endregion -->

```python id="iEtH4upRQsh0"
# Distribution des Variables quantitatives without outlier:
def quali_reprez(df, quantil = 0):

  fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(25, 5))

  df_WO = pd.DataFrame()

  for col ,ax in zip( df_quanti, axes.flat):

      df_WO["temp"] = df_quanti[col].copy()
      # Sans les outliers enlevant les 1%  des valeurs extremes
      df_WO["temp"]  = df_WO["temp"][df_WO["temp"].between(df_WO["temp"].quantile(quantil), df_WO["temp"].quantile(1-quantil))]
      #sns.distplot(df_WO["temp"] ,ax=ax , axlabel=col )
      sns.histplot(df_WO["temp"], ax=ax, kde=True,stat='density', linewidth=0)

  plt.show()
```

```python id="0bwZTEweTpmd" colab={"base_uri": "https://localhost:8080/", "height": 345} executionInfo={"status": "ok", "timestamp": 1694178519537, "user_tz": -120, "elapsed": 3677, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0ea42e47-746c-418d-f8e5-1d2d804a441b"
quali_reprez(df, 0.015)
```

```python id="mlVppOGCRy3I"
def quali_reprez(df, frek = 0.05 ):

    df = df.select_dtypes(['object','category','bool'])
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(25, 10)) # choisir le cadrillage des camabert

    for col ,ax in zip( df , axes.flat):
    #for col in df:

        ddf = pd.DataFrame(columns=[col])
        ddf[col] = df[col].copy()
        moda = ddf[col].unique()

        if ddf.dtypes[col] != 'object':
            ddf =  ddf.astype({col: 'object'})
            #type_col = ddf.dtypes[col]
            print(ddf.dtypes[col])

        lst_moda = list(ddf[col].unique())
        ddf['freq'] = ddf.groupby([col])[col].transform('count')/ddf.shape[0]
        ddf[col] = ddf[col].where(ddf['freq'] > frek  , "other")

        nb = len(moda) - len(ddf[col].unique()) +1

        #nb = df[df[col] == "other"].shape[0]
        autre = "other_(" + str(nb) +")"
        #ddf[col] = ddf[col].where(ddf[col] != "other" , autre )
        ddf[col].replace("other", autre,inplace=True )
        couleurs = list(np.random.rand(len(lst_moda)))

        ddf[col].value_counts().plot(kind='pie',autopct='%1.0f%%' , ax=ax) # a verifier pour la coherence avec les differents label
        label_removed = list( set( moda ).symmetric_difference( set( ddf[col].unique() ) ) )
        if autre in label_removed:
            label_removed.remove(autre)
            if  label_removed :
                print(col, "- moda |x|<",frek,":")
                print( label_removed[:5], "\n")
    plt.figure()
```

```python id="FI-8_wYvSe3n" colab={"base_uri": "https://localhost:8080/", "height": 388} executionInfo={"status": "ok", "timestamp": 1681993288120, "user_tz": -120, "elapsed": 679, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="42578534-13fa-469a-beb3-f9c2e694c8f9"
quali_reprez(df_quali , 0.05)
```

<!-- #region id="-5aY5ZRq4wNQ" -->
### Heatmap
<!-- #endregion -->

<!-- #region id="OvRRV8D145Hu" -->
Carte de fréquentation, est une représentation graphique de données statistiques qui fait correspondre à l'intensité d'une grandeur variable une gamme de tons
<!-- #endregion -->

```python id="WIkBf5UpSmXf" colab={"base_uri": "https://localhost:8080/", "height": 521} executionInfo={"status": "ok", "timestamp": 1681993288612, "user_tz": -120, "elapsed": 501, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0be48f10-16c4-4a87-b70f-d459a349b2ca"
corr = df.corr()
plt.figure(figsize=(6, 6))
ax = sns.heatmap(
    corr,
    vmin=0, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    fmt='.1g'
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```

<!-- #region id="Y_LSvFO36Igx" -->
### Count Values
<!-- #endregion -->

```python id="ru2IUpUBYFsc" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1681993288613, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="64fe7e6f-0224-4e8b-b9c5-c8ab102345e3"
# Valeurs de chaque modalite d'une colonne + NaN value
print(df["size"].value_counts(dropna=False), # somme
      "\n\n",
      df["size"].value_counts(dropna=False , normalize= True)  )# En porcentage
```

<!-- #region id="sqXfSLKM_wlt" -->
### Quantitatives en grand nombre
<!-- #endregion -->

```python id="Bkp_9InXk109"
# Fonction pour générer des colonnes avec une distribution normale
def generate_dataframe(num_columns=10, num_rows=100):
    """
    Fonction pour générer un DataFrame avec des colonnes de données aléatoires.
    """
    data = {}
    for i in range(num_columns):
        # Générer une moyenne aléatoire (entre 0 et 1) multipliée par 10^Z, où Z est un entier aléatoire entre -3 et 3
        Z = random.choice([-3,-2,-1, 1,2,3])
        mean = random.random() * (10 ** Z)
        sigma = 4 * abs(Z)  # Sigma proportionnel à Z
        # Générer des données selon une distribution normale
        data[f'col_{i+1}'] = np.random.normal(loc=mean, scale=sigma, size=num_rows)

    return pd.DataFrame(data)

# Générer un DataFrame avec 20 colonnes et 100 lignes
df = generate_dataframe(num_columns=20, num_rows=100)
```

```python id="Mbz3XrEc9B86"
def introduce_missing_values(df, min_percent, max_percent):
    """
    Introduit des valeurs manquantes aléatoirement dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        min_percent (float): Pourcentage minimum de valeurs manquantes par colonne.
        max_percent (float): Pourcentage maximum de valeurs manquantes par colonne.

    Returns:
        pd.DataFrame: Le DataFrame avec les valeurs manquantes introduites.
    """

    for col in df.columns:
        # Générer un pourcentage aléatoire entre min_percent et max_percent
        percent_missing = np.random.uniform(min_percent, max_percent)

        # Calculer le nombre de valeurs à remplacer par NaN
        num_missing = int(len(df) * percent_missing)

        # Générer des indices aléatoires pour remplacer
        missing_indices = np.random.choice(df.index, num_missing, replace=False)

        # Remplacer les valeurs par NaN aux indices choisis
        df.loc[missing_indices, col] = np.nan

    return df

    df = introduce_missing_values(df, 20, 50)
```

```python id="hRqiqnvb_wzh"
def process_and_plot(df, factor_threshold=7):
    """
    Fonction qui sélectionne les colonnes numériques d'un DataFrame, calcule des statistiques
    sur ces colonnes (médiane, quantiles), regroupe les colonnes selon des critères basés
    sur ces statistiques, et affiche les regroupements sous forme de subplots.

    Entrées:
    - df: DataFrame pandas contenant les données
    - factor_threshold: seuil de facteur pour limiter la taille des groupes de colonnes (par défaut: 7)

    Sorties:
    - grouped_columns: liste des groupes de colonnes regroupées selon les critères définis
    - Subplots des groupes de colonnes tracés en lignes avec Seaborn
    """

    # Sélectionner uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=np.number)

    # Calculer les statistiques pertinentes (médianes, quantiles 1 et 9)
    stats = {}
    for col in df_numeric.columns:
        median = df_numeric[col].median()
        q1 = df_numeric[col].quantile(0.1)
        q9 = df_numeric[col].quantile(0.9)
        stats[col] = {'median': median, 'q_diff': abs(q9 - q1)}

    # Fonction pour regrouper les colonnes selon les critères définis
    def group_columns_by_criteria(stats, factor_threshold):
        # Trier les colonnes par leur médiane
        sorted_columns = sorted(stats.items(), key=lambda x: x[1]['median'])

        # Initialisation des groupes
        grouped_columns = []

        # Tant qu'il reste des colonnes non assignées
        while sorted_columns:
            # Prendre la première colonne restante pour démarrer un groupe
            current_group = [sorted_columns.pop(0)]
            current_medians = [current_group[0][1]['median']]
            current_qdiffs = [current_group[0][1]['q_diff']]

            # Parcourir les colonnes restantes et tenter de les ajouter au groupe
            i = 0
            while i < len(sorted_columns) and len(current_group) < factor_threshold:
                col, stat = sorted_columns[i]
                med_c = stat['median']
                diffQ_c = stat['q_diff']
                med_g = np.mean(current_medians)  # Moyenne des médianes du groupe
                diffQ_g = np.mean(current_qdiffs)  # Moyenne des différences de quantiles du groupe

                # Vérifier les critères des médianes et quantiles
                median_ratio = abs(min(med_c, med_g)) / abs(max(med_c, med_g))
                qdiff_ratio = abs(min(diffQ_c, diffQ_g)) / abs(max(diffQ_c, diffQ_g))

                if median_ratio < 0.1 or qdiff_ratio < 0.1:
                    # Si la variable ne respecte pas les critères, passer à la suivante
                    i += 1
                else:
                    # Ajouter la variable au groupe
                    current_group.append(sorted_columns.pop(i))
                    current_medians.append(med_c)
                    current_qdiffs.append(diffQ_c)

            # Ajouter le groupe à la liste finale
            grouped_columns.append([col[0] for col in current_group])

        return grouped_columns

    # Regrouper les colonnes selon les critères définis
    grouped_columns = group_columns_by_criteria(stats, factor_threshold)

    # Affichage des groupes et génération de subplots
    num_groups = len(grouped_columns)
    fig, axes = plt.subplots(num_groups, 1, figsize=(20, 5 * num_groups), sharex=True)

    # Si un seul groupe, on s'assure que 'axes' soit une liste
    if num_groups == 1:
        axes = [axes]

    # Tracer les subplots pour chaque groupe de variables
    for i, group in enumerate(grouped_columns):
        sns.lineplot(data=df_numeric[group], ax=axes[i], errorbar=None,  dashes=False, markers=False) # linestyle='-',
        axes[i].set_title(f"Groupe {i+1} - Variables: {', '.join(group)}", fontsize=14)
        axes[i].set_ylabel("Valeurs")

    # Ajouter des étiquettes globales
    plt.xlabel("Index")
    plt.tight_layout()
    plt.show()

    return None

```

```python colab={"base_uri": "https://localhost:8080/", "height": 845} id="WLSMFbTg6ZF8" executionInfo={"status": "ok", "timestamp": 1727784384007, "user_tz": -120, "elapsed": 2861, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="fdc7b747-fa82-44b0-dbda-ff8b0796adc5"
process_and_plot(df, factor_threshold=7)
```

<!-- #region id="5xCIVITecHXK" -->
# Analyses Multivariées
<!-- #endregion -->

<!-- #region id="TKAI2xQiuhMW" -->
## Analyse Bivarier
<!-- #endregion -->

<!-- #region id="zuwZGrXDKgeO" -->
### Qualitative X Qualitative
<!-- #endregion -->

```python id="qngomuhFbmby" executionInfo={"status": "ok", "timestamp": 1681993289143, "user_tz": -120, "elapsed": 541, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f3831ab4-5e1d-4cf3-f686-a136c6ea4ab3" colab={"base_uri": "https://localhost:8080/", "height": 657}
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data1 = pd.DataFrame(np.random.rand(17,3), columns=['A','B','C']).assign(Location=1)
data2 = pd.DataFrame(np.random.rand(17,3)+0.2, columns=['A','B','C']).assign(Location=2)
data3 = pd.DataFrame(np.random.rand(17,3)+0.4, columns=['A','B','C']).assign(Location=3)

cdf = pd.concat([data1, data2, data3])
print('cdf taille: ',cdf.shape , '\n\n' , cdf.head(2))
print('\n')
mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Letter'])
print('mdf taille: ',mdf.shape , '\n\n' ,mdf.head(2))


ax = sns.boxplot(x="Location", y="value", hue="Letter", data=mdf)
plt.show()
```

<!-- #region id="nROM4rtT3Xmo" -->
* **Table de contingence**
<!-- #endregion -->

```python id="6xSvLf6rLlRq" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1681993289143, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="03ba7ffe-7fe0-49c8-d018-071f0d6012e6"
print( pd.crosstab(df["sex"],df["time"] , margins=True , dropna=True ),"\n\n",
      pd.crosstab(df["sex"],df["time"] , dropna=False  ).apply(lambda r: r/r.sum(), axis=1),"\n\n",
      pd.crosstab(df["sex"],df["time"], margins=True , dropna=False , normalize='all')*100 )
```

<!-- #region id="6GaamaGWCQSh" -->
* **Count plot**
<!-- #endregion -->

```python id="CE6vstWzcmBz"
def draw(graph):
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
#print("p.get_x()",p.get_x())
#print("p.get_y()",p.get_y())
#print("p.get_width()",p.get_width())
#print("p.get_height()", p.get_height())
#print("\n\n")"""
```

```python colab={"base_uri": "https://localhost:8080/", "height": 465} id="gs4xV5a3rEw_" executionInfo={"status": "ok", "timestamp": 1681994757909, "user_tz": -120, "elapsed": 2327, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="98a66ad9-5c6c-4ec2-a698-300d7739341a"
plt.figure(figsize=(5, 5))
graph = sns.countplot(x="sex", hue="time", data=df)

for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x() + p.get_width() / 2., height -6, height, ha="center")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 465} id="RcgrgYtzEuaj" executionInfo={"status": "ok", "timestamp": 1681995436545, "user_tz": -120, "elapsed": 1328, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d62bbc94-b5f2-4056-e02a-ed0d74840ef8"
plt.figure(figsize = (5, 5))
g = sns.countplot(x="sex", hue="time", data=df)

for p in g.patches:
    txt = str(p.get_height().round(2))
    txt_x = p.get_x() + 0.2
    txt_y = p.get_height()+ 1
    g.text(txt_x,txt_y,txt,ha="center")
```

<!-- #region id="kmPVePoLCU5U" -->
* **Frequence plot**
<!-- #endregion -->

```python id="nMzoJ1yHZcHx" colab={"base_uri": "https://localhost:8080/", "height": 470} executionInfo={"status": "ok", "timestamp": 1681995572311, "user_tz": -120, "elapsed": 452, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b1e4f562-f6fd-430c-8180-da907cf97f07"
occupation_counts = (df.groupby(['sex'])['time']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('sex'))
plt.figure(figsize = (5, 5))
g = sns.barplot(x="sex", y="percentage", hue="time", data=occupation_counts)
#_ = plt.setp(g.get_xticklabels(), rotation=90)  # Rotate labels
g.set_ylim(0,100)

for p in g.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.2
    txt_y = p.get_height()+2
    g.text(txt_x,txt_y,txt,ha="center")
```

<!-- #region id="KK9g7YCOJVCc" -->
#### Autre ajensements possibles
<!-- #endregion -->

```python id="bQiDXCLi4jiF" colab={"base_uri": "https://localhost:8080/", "height": 466} executionInfo={"status": "ok", "timestamp": 1681995760661, "user_tz": -120, "elapsed": 535, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="22704185-e9ba-4cb2-88b6-7ee08232e7d6"
x, y, hue = "sex", "proportion", "time"
#hue_order = ["Male", "Female"]

(df[x]
 .groupby(df[hue])
 .value_counts(normalize=True)
 .rename(y)
 .reset_index()
 .pipe((sns.barplot, "data"), x=x, y=y, hue=hue) #[.text(v , i ) for i, v in enumerate(y)  ]
 )
#show_values_on_bars(sns_t, "h", 0.3)
```

```python id="VoQSHM5SlDPt" colab={"base_uri": "https://localhost:8080/", "height": 512} executionInfo={"status": "ok", "timestamp": 1681995957448, "user_tz": -120, "elapsed": 1337, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a28893f7-aea3-4324-9d5b-a483989e650a"
x,y = 'sex', 'time'

df1 = df.groupby(x)[y].value_counts(normalize=True)
df1 = df1.mul(100)
df1 = df1.rename('percent').reset_index()

g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1)
g.ax.set_ylim(0,100)
#sns.set_style("white")

for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x() + 0.2
    txt_y = p.get_height()+2
    g.ax.text(txt_x,txt_y,txt,ha="center")

#draw(g.ax)
```

```python id="qmIK4ZtbZ9xp" colab={"base_uri": "https://localhost:8080/", "height": 503} executionInfo={"status": "ok", "timestamp": 1681996036534, "user_tz": -120, "elapsed": 494, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="041643b5-c1bd-43c7-d202-a7448370347f"
plt.figure(figsize = (5, 5))
ct = pd.crosstab(df["sex"],df["time"] , dropna=False  ).apply(lambda r: r/r.sum(), axis=1)
ax = ct.plot(kind='bar')

for p in ax.patches:
    ax.annotate(str(round(p.get_height()*100,2))+"%", (p.get_x()+ 0.02, p.get_height()+ 0.005))
```

```python id="TveCQzS90HiH" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1681996000718, "user_tz": -120, "elapsed": 502, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3a0280b5-67c9-42bf-ae06-463f5ee0ee80"
print((df["time"].values == 'Lunch').sum(),
(df["sex"].values == 'Male').sum())
```

<!-- #region id="E2nZdsHsdTkI" -->
###  Quantitative X Quantitative
<!-- #endregion -->

```python id="J7uTdksCnr_J"
df = sns.load_dataset('tips')
```

```python id="mjGVDHf72Hs3" colab={"base_uri": "https://localhost:8080/", "height": 523} executionInfo={"status": "ok", "timestamp": 1681996079107, "user_tz": -120, "elapsed": 815, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3d9bcfbf-b1ee-4536-970a-cf7fdc787592"
g = sns.lmplot(x="total_bill", y="tip", data=df,ci=None)
g.set_axis_labels("Total Bill", "Tip ")

#f, ax = plt.subplots(figsize=(5, 6))
#sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 506} id="n8bPpKn7nlyU" executionInfo={"status": "ok", "timestamp": 1681996079645, "user_tz": -120, "elapsed": 556, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c8fd6642-aa5e-4f2a-f475-a32a931538cf"
# Approche consistant à ajuster une régression non paramétrique
g = sns.lmplot(x="total_bill", y="tip", data=df,
           lowess=True)

```

<!-- #region id="fr2TXSfhTlf9" -->
### Qualitative(s) X Quantitative(s)
<!-- #endregion -->

<!-- #region id="WdmM3oRXSV_2" -->
* **2 Quanti X 1 Quali**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 523} id="V8yXn7EPRXsk" executionInfo={"status": "ok", "timestamp": 1681996086470, "user_tz": -120, "elapsed": 1048, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9380717b-ea04-4f79-f501-3c62b096efe8"
g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=df)
g.set_axis_labels("Total Bill", "Tip ")
```

```python id="_m2Fdp9teVh4" colab={"base_uri": "https://localhost:8080/", "height": 348} executionInfo={"status": "ok", "timestamp": 1681996091155, "user_tz": -120, "elapsed": 4273, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e3901550-75d7-4907-af98-157a8325e3f3"
g = sns.lmplot(x="total_bill", y="tip", hue="day", col="day", data=df);
g.set_axis_labels("Total Bill", "Tip ")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 449} id="weQ_6yjCiWLp" executionInfo={"status": "ok", "timestamp": 1681996091884, "user_tz": -120, "elapsed": 735, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="1e608dae-9a76-41fc-c61d-903daa5a4b64"
# Draw a categorical scatterplot to show each observation
ax = sns.swarmplot(data=df, x="time", y="total_bill", hue="smoker")
#ax.set(ylabel="")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="2WHzJpzMPqy_" executionInfo={"status": "ok", "timestamp": 1681996091885, "user_tz": -120, "elapsed": 19, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c02ca05e-fbcc-41dc-c554-a52aab579d5a"
ax = sns.boxplot(x="total_bill", y="day", data=df, whis=np.inf)
#ax = sns.swarmplot(x="total_bill", y="day", data=df, color=".2")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 450} id="8MtBwEMwP1_s" executionInfo={"status": "ok", "timestamp": 1681996092370, "user_tz": -120, "elapsed": 492, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="1f27ab77-b490-4177-9a77-c7fc995bebb7"
ax = sns.boxplot(x="total_bill", y="day", data=df, whis=np.inf)
ax = sns.swarmplot(x="total_bill", y="day", data=df, color=".2")
```

<!-- #region id="7hit_F3dQ1av" -->
* **3 Quanti X 1 Quali**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 624} id="TS0IVoYaRFBA" executionInfo={"status": "ok", "timestamp": 1681996211434, "user_tz": -120, "elapsed": 2744, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7ecf6519-a480-46dc-f64b-a7cfd922bdcd"
iris = sns.load_dataset("iris")
# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="sepal_length", y="petal_width", hue="species", size="sepal_width",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=iris)
```

<!-- #region id="yB22MLWlSkxE" -->
* **N Quanti X 1 Quali**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 513} id="tZCrk22A5myh" executionInfo={"status": "ok", "timestamp": 1681996238672, "user_tz": -120, "elapsed": 2632, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="108f9111-6028-4c4a-ed0c-340366c760d1"
sns.pairplot(df, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="day", height=5, aspect=.8, kind="reg");
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1s0R6MlsSAz0" executionInfo={"status": "ok", "timestamp": 1681996424046, "user_tz": -120, "elapsed": 19869, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="5209970e-05da-405c-c96f-73b70cca5acd"
sns.pairplot(iris ,hue="species" , diag_kind="hist")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="LlPLdHSZOcsm" executionInfo={"status": "ok", "timestamp": 1681996403193, "user_tz": -120, "elapsed": 11184, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b96f5183-7ec9-492e-8b92-8870f56ec9aa"
sns.pairplot(iris , hue="species" ,  markers=["o", "s", "D"])
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 718} id="EADOWPg_OczV" executionInfo={"status": "ok", "timestamp": 1681997302114, "user_tz": -120, "elapsed": 1891, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="769dc244-98b2-47b7-c66c-065c0913dbae"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib.ticker as ticker

# Some random data
dfWIM = pd.DataFrame({'AXLES': np.random.normal(8, 2, 5000).astype(int)})
ncount = len(dfWIM)

plt.figure(figsize=(12,8))

ax = sns.countplot(x="AXLES", data=dfWIM, order=[3,4,5,6,7,8,9,10,11,12] ,palette = "rocket_r")
plt.title('Distribution of Truck Configurations')
plt.xlabel('Number of Axles')

# Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),
            ha='center', va='bottom') # set the alignment of the text

# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 718} id="eUn_UHO27Llt" executionInfo={"status": "ok", "timestamp": 1682058893760, "user_tz": -120, "elapsed": 2277, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f7cf6d30-07fa-4330-d536-614276238037"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# Some random data
dfWIM = pd.DataFrame({'AXLES': np.random.normal(8, 2, 5000).astype(int)})
ncount = len(dfWIM)

plt.figure(figsize=(12,8))

ax = sns.countplot(x="AXLES", data=dfWIM, order=[3,4,5,6,7,8,9,10,11,12] ,palette = "rocket_r")
plt.title('Distribution of Truck Configurations')
plt.xlabel('Number of Axles')

# Make twin axis
ax2=ax.twinx()

# Switch so count axis is on right, frequency on left
ax.yaxis.tick_right()
ax2.yaxis.tick_left()

# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]')

# Use a LinearLocator to ensure the correct number of ticks
ax.yaxis.set_major_locator(ticker.LinearLocator(11))
# Set count axis y limit
ax.set_ylim(0, dfWIM['AXLES'].value_counts().max() * 1.1)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
# Fix the frequency range to 0-100 and set y axis label format
ax2.set_ylim(0,((dfWIM['AXLES'].value_counts().max() * 1.1) / dfWIM['AXLES'].value_counts().sum() ) *100)

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y),
            ha='center', va='bottom') # set the alignment of the text

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)
```

<!-- #region id="mksNjDzBNl7v" -->
**Affichage du Count**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 528} id="iF_J96JGx7WR" executionInfo={"status": "ok", "timestamp": 1695894883139, "user_tz": -120, "elapsed": 1195, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="661adbac-d07f-4e8e-b4d4-964c9ecfab24"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

# création du DataFrame
data = {'titre': ['A', 'B', 'C', 'D', 'E'],
        'count': [10, 15, 8, 12, 20],
        'fréquence': [0.1, 0.3, 0.2, 0.15, 0.25]}
df = pd.DataFrame(data)

# tri des valeurs par ordre décroissant de 'count'
df = df.sort_values(by='count', ascending=False)

# création de la figure
fig, ax = plt.subplots()

# création du plot pour le count
ax.bar(df['titre'], df['count'], color='blue')

# ajout de la légende et des titres
ax.set_xlabel('Titre')
ax.set_ylabel('Count')
ax.set_title('Barplot avec la fréquence en intensité de couleur')

# normalisation de la fréquence pour l'utiliser comme intensité de couleur
norm = plt.Normalize(0, 0.4)
colors = plt.cm.Blues(norm(df['fréquence'].values))

# création du plot pour la fréquence en intensité de couleur
rects = ax.bar(df['titre'], df['count'], color=colors)

# ajout de la couleur et de la valeur de la fréquence pour chaque bar
for i, rect in enumerate(rects):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height - 1, f'{height:.0f}', ha='center', va='bottom')
    rect.set_facecolor(colors[i])

# création de la légende pour la couleur
sm = ScalarMappable(cmap=plt.cm.Blues, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('Fréquence')

# affichage du plot
plt.show()
```

<!-- #region id="e5kHeV1SNu4O" -->
**Affichage de la Frequence**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="bdLYqTcRR6hG" executionInfo={"status": "ok", "timestamp": 1682062756589, "user_tz": -120, "elapsed": 743, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="6d54d2fd-40bf-4143-dfcb-68e23ce3c7be"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

# création du DataFrame
data = {'Lettre': ['A', 'B', 'C', 'D', 'E'],
        'Count': [10, 15, 8, 12, 20],
        'Fréquence': [0.1, 0.3, 0.2, 0.15, 0.25]}
df = pd.DataFrame(data)

# tri des valeurs par ordre décroissant de 'count'
df = df.sort_values(by='Count', ascending=False)

# création de la figure
fig, ax = plt.subplots()

# création du plot pour le count
ax.bar(df['Lettre'], df['Fréquence'], color='blue')

# ajout de la légende et des titres
ax.set_xlabel('Lettre')
ax.set_ylabel('Count')
ax.set_title('Barplot avec la fréquence en intensité de couleur')

# normalisation de la fréquence pour l'utiliser comme intensité de couleur
norm = plt.Normalize(0, 0.4)
colors = plt.cm.Blues(norm(df['Fréquence'].values))

# création du plot pour la fréquence en intensité de couleur
rects = ax.bar(df['Lettre'], df['Count'], color=colors)

# ajout de la couleur et de la valeur de la fréquence pour chaque bar
for i, rect in enumerate(rects):
    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2., height - 1, f'{df["Fréquence"].iloc[i]:.2f}', ha='center', va='bottom') #, color='black'
    rect.set_facecolor(colors[i])

# création de la légende pour la couleur
sm = ScalarMappable(cmap=plt.cm.Blues, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.ax.set_ylabel('Fréquence')

# affichage du plot
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="SJx8kJYoFkft" executionInfo={"status": "ok", "timestamp": 1682337224648, "user_tz": -120, "elapsed": 224, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="cf085f86-b0bf-439b-9ed1-d39e3b3f4f63"
df.columns
```

```python colab={"base_uri": "https://localhost:8080/", "height": 564} id="RQTiZCjQE3zC" executionInfo={"status": "ok", "timestamp": 1682340088878, "user_tz": -120, "elapsed": 1447, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="008bbe84-8e05-43c7-d23f-57a1c83280a7"
# Create as many colors as there are unique df['day']
categories = np.unique(df['day'])
colors = [plt.cm.Set1(i / float(len(categories) - 1)) for i in range(len(categories))]

# Draw Plot for Each day
plt.figure(figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')

for i, category in enumerate(categories):
    plt.scatter('total_bill','tip',data=df.loc[df.day == category, :],s=20,c=colors[i],
    label=str(category))

# Decorations
#plt.gca().set(xlim=(0.0, 0.1),ylim=(0, 90000),)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('total_bill', fontdict={'fontsize': 10})
plt.ylabel('tip', fontdict={'fontsize': 10})
plt.title("Scatterplot of df total_bill vs tip", fontsize=12)
plt.legend(fontsize=10)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Wg5j7eUqKwkc" executionInfo={"status": "ok", "timestamp": 1682341030024, "user_tz": -120, "elapsed": 7, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d293dcb0-b32a-41e7-e0dd-467c29c1d601"
df["size"].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 462} id="U_fXNT8LYT5v" executionInfo={"status": "ok", "timestamp": 1693314111979, "user_tz": -120, "elapsed": 3242, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="da20adaf-2359-4d4c-ba2c-d0c97c8a3b12"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

sns.set_style("whitegrid")

# Votre DataFrame df doit être préalablement défini
# categories = np.unique(df['day'])  # Assurez-vous que df['day'] existe dans votre DataFrame
categories = np.unique(df['sex'])  # Utilisation de 'sex' à la place de 'day' pour l'exemple
colors = ["green", "blue", "red", "orange"]

fig = plt.figure(figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

for i, category in enumerate(categories):
    sub_df = df[df['sex'] == category]
    plt.scatter(sub_df['total_bill'], sub_df['tip'],
                s=30,  # Taille des points
                c=colors[i],
                label=str(category),
                edgecolors='black'
                )

def encircle(x, y, ax=None, **kw):
    if not ax:
        ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)

df_encircle_data1 = df.loc[df.sex == 'Female', :]
encircle(df_encircle_data1.total_bill, df_encircle_data1.tip, ec="b", fc="none", alpha=0.3)
encircle(df_encircle_data1.total_bill, df_encircle_data1.tip, ec="b", fc="none", linestyle='--')

df_encircle_data6 = df.loc[df.sex == 'Male', :]
encircle(df_encircle_data6.total_bill, df_encircle_data6.tip, ec="r", fc="none", alpha=0.3)
encircle(df_encircle_data6.total_bill, df_encircle_data6.tip, ec="r", fc="none", linestyle='--')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('total_bill', fontdict={'fontsize': 14})
plt.ylabel('tip', fontdict={'fontsize': 14})
plt.title("Bubble Plot with Encircling", fontsize=14)
plt.legend(fontsize=10)
plt.show()

```

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="LVV05EeAWyU7" executionInfo={"status": "ok", "timestamp": 1682341334785, "user_tz": -120, "elapsed": 235, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2941342a-e923-419c-b25b-ccde414dded1"
df.sample(3)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 644} id="4w3_uOujKMhm" executionInfo={"status": "ok", "timestamp": 1682341422994, "user_tz": -120, "elapsed": 1859, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a4617ecc-73b6-44f5-82a1-91474bb2e2c3"
plt.figure(dpi=500)

#df_select = df.loc[df.cyl.isin([4, 8]), :]

# Plot

gridobj = sns.lmplot(
    x="total_bill",
    y="tip",
    hue="time",
    data=df,
    height=7,
    aspect=1.6,  #robust=True,
    palette='Set1',
    scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))

# Decorations
sns.set(style="whitegrid", font_scale=1.5)
#gridobj.set(xlim=(0.5, 7.5), ylim=(10, 50))
gridobj.fig.set_size_inches(10, 6)
plt.title("Scatterplot with line of best fit grouped by number of cylinders")
plt.show()
```

```python id="sepKfgtIXAAl"
df_counts = df.groupby(['hwy', 'cty']).size().reset_index(name='counts')
# Draw Stripplot
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
sns.stripplot(df_counts.cty,
              df_counts.hwy,
              sizes=df_counts.counts * 30,
              ax=ax,
              palette='Set1')

# Decorations
sns.set(style="whitegrid", font_scale=1.1)
plt.title('Counts Plot - Size of circle is bigger as more points overlap')
plt.show()
```

<!-- #region id="lbyRKLVB3AnF" -->
# Pareto
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="FFYeHbSS3C6B" executionInfo={"status": "ok", "timestamp": 1732198087787, "user_tz": -60, "elapsed": 555, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="16b8fe10-fc95-4246-c6e4-7fb6399ca5df"
import pandas as pd
import plotly.graph_objects as go

def plot_pareto(df, x_col):
    """
    Affiche un diagramme de Pareto avec deux axes : fréquence et pourcentage cumulé.

    :param df: DataFrame pandas contenant les données
    :param x_col: Nom de la colonne qualitative
    :return: Figure Plotly
    """
    # Comptage des occurrences pour chaque catégorie
    counts = df[x_col].value_counts().reset_index()
    counts.columns = [x_col, 'Frequency']

    # Calcul des pourcentages cumulatifs
    counts['Cumulative'] = counts['Frequency'].cumsum()
    counts['Cumulative Percentage'] = counts['Cumulative'] / counts['Frequency'].sum() * 100

    # Création de la figure
    fig = go.Figure()

    # Ajout des barres pour les fréquences
    fig.add_trace(go.Bar(
        x=counts[x_col],
        y=counts['Frequency'],
        name='Frequency',
        marker_color='blue',
        yaxis='y1'
    ))

    # Ajout de la ligne pour les pourcentages cumulatifs
    fig.add_trace(go.Scatter(
        x=counts[x_col],
        y=counts['Cumulative Percentage'],
        name='Cumulative Percentage',
        mode='lines+markers',
        line=dict(color='red'),
        yaxis='y2'
    ))

    # Mise en forme du diagramme
    fig.update_layout(
        title='Pareto Chart',
        xaxis=dict(title=x_col),
        yaxis=dict(
            title='Frequency',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Cumulative Percentage',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    return fig

# Test de la fonction
data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'C', 'B', 'A']}
df = pd.DataFrame(data)

fig = plot_pareto(df, 'Category')
fig.show()

```

```python id="Klbx5QPX3DZ-"

```
