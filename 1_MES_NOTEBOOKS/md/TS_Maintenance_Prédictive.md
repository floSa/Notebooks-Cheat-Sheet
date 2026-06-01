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

<!-- #region id="bZtngOt6WS6w" -->
#LSTM pour la maintenance prédictive sur les données des capteurs de pompe
<!-- #endregion -->

<!-- #region id="1ILbCBrDfXlJ" -->
Contexte

Pompe à eau d'une petite zone. 7 pannes du système ressencé l'an dernière.

Les données proviennent de tous les capteurs disponibles et sont toutes des valeurs brutes. Le nombre total de capteurs est de 52 unités.
Remerciements


<!-- #endregion -->

<!-- #region id="82L2JiDgRRn4" -->
## Chargements des données
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tWn3RX3pXI7U" executionInfo={"status": "ok", "timestamp": 1683875224963, "user_tz": -120, "elapsed": 25952, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8b6c782b-ce4d-49db-df5b-bc1bb6086226"
# google.colab
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/Maint_Pred/"
```

```python id="BLHi1cdql1Y6"
import warnings
warnings.filterwarnings('ignore')
```

```python id="sTHVngwDXfqs"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
```

```python id="Z-aBUzJ_7TlC"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 329} id="YznAmtm_XJaU" executionInfo={"status": "ok", "timestamp": 1683787724180, "user_tz": -120, "elapsed": 3670, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9cf41eaa-1e4e-4ba0-ef79-ec72ed450475"
data = pd.read_csv(path + 'pump_sensor.csv')
data.head(3)
```

<!-- #region id="yuE6eo0UfWaE" -->

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Sb8m3VIeXeia" executionInfo={"status": "ok", "timestamp": 1683787724180, "user_tz": -120, "elapsed": 21, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="717be491-5100-474c-c5fe-8b1e3d07c3d1"
print(data.shape)
data.machine_status.value_counts()
```

<!-- #region id="PHaUeKGURI-k" -->
##Representation des données par sample
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dxDK41HLpKsP" executionInfo={"status": "ok", "timestamp": 1683787724181, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b93ba258-2bea-4b62-b9af-c5d1852c273b"
Sample = data[["timestamp","machine_status"]]
print("test",Sample.shape)
BRO = Sample[Sample["machine_status"] == 'BROKEN']
print("Bro",BRO.shape)
OTHER = Sample[~(Sample["machine_status"] == "BROKEN")]
print("OTHER",OTHER.shape)
OTHER = OTHER.sample(n=2993, replace=False , random_state=0)
Sample = pd.concat([OTHER,BRO] , axis=0)
Sample.sort_values(by='timestamp', ascending=True , inplace=True)
print("Sample",Sample.shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3MEnsKWWUzGH" executionInfo={"status": "ok", "timestamp": 1683787724181, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ef37d636-5e91-4809-f495-036d987ed9a1"
Sample.machine_status.value_counts()
```

```python id="nlVsI-wiWfT8"
Sample["machine_status"].replace({'NORMAL': 2 ,'BROKEN' : 0, 'RECOVERING':1 },inplace=True)
data["machine_status"].replace({'NORMAL': 2 ,'BROKEN' : 0, 'RECOVERING':1 },inplace=True)
```

<!-- #region id="8N5tGToi423t" -->
## Visualisation des Capteurs
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="nQD3u-gouNH9" executionInfo={"status": "ok", "timestamp": 1683787731250, "user_tz": -120, "elapsed": 7079, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="bf80a082-8b96-4a9b-c71a-8fac46341c52"
import seaborn as sns
from matplotlib.pyplot import figure

figure(figsize=(20, 6), dpi=80)

#le = preprocessing.LabelEncoder()
#Sample["machine_status"] = le.fit_transform(Sample["machine_status"])
sns.lineplot(data=Sample, x="timestamp", y="machine_status")
labels = ['Broken','Recovering', 'Normal']
plt.yticks([0,1,2], labels, rotation='vertical')
N = 20
plt.xticks(range(0, len(Sample["timestamp"]), len(Sample["timestamp"])//N), Sample["timestamp"][::len(Sample["timestamp"])//N], rotation=45)

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "16gyfz6CqzX22IIvfj53F7nCvmP3hERdx"} id="JrWTm2QX6-YL" executionInfo={"status": "ok", "timestamp": 1683788132804, "user_tz": -120, "elapsed": 401574, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3f88127c-ed54-4295-86b7-c68c1b941299"
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.pyplot import figure

Sensors = ["sensor_0"+str(i) for i in range(0,10) ] + ["sensor_"+str(i) for i in range(10,52) ]

for sensor in Sensors:

    Sample = pd.merge(Sample, data[[sensor]] , left_index=True, right_index=True)

    # Création du graphique
    fig, ax1 = plt.subplots( figsize=(20, 5))
    ax2 = ax1.twinx()
    fig.set_size_inches(20, 4)

    # Tracer la courbe pour y sur l'axe des ordonnées ax1
    ax1.plot(Sample["timestamp"], Sample["machine_status"], label='machine_status')
    ax1.set_ylabel('machine_status')

    labels = ['Broken','Recovering', 'Normal']
    ax1.set_yticks([0, 1, 2 ] )
    ax1.set_yticklabels(labels, rotation='vertical')
    ax1.set_xticklabels(labels, rotation=25)

    # Tracer la courbe pour z sur l'axe des ordonnées ax2
    ax2.plot(Sample["timestamp"], Sample[sensor], color='red', label=sensor)
    ax2.set_ylabel(sensor)

    # Configuration des étiquettes de l'axe des abscisses
    N= 20
    plt.xticks(range(0, len(Sample["timestamp"]), len(Sample["timestamp"])//N), Sample["timestamp"][::len(Sample["timestamp"])//N], rotation=45)

    # Affichage des légendes et du graphique
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
```

<!-- #region id="hOV8zBatRxEJ" -->
## Reconstruction des données des capteurs par interpolation
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 589} id="l88kXNiPvi6R" executionInfo={"status": "ok", "timestamp": 1683788136285, "user_tz": -120, "elapsed": 3490, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="bf1c9c23-7d84-4613-a3f4-0d96277f4fdb"
import missingno as msno
msno.matrix(Sample , labels=True , label_rotation=-90 )
```

<!-- #region id="bMO6SjmJ4raF" -->
## Gestion des données manquantes par Interpolation et Fill
<!-- #endregion -->

<!-- #region id="DX9EN405ZTPN" -->
* **Fill 0** : sensors 15 50 51
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="D567P_uwa_gB" executionInfo={"status": "ok", "timestamp": 1683788136287, "user_tz": -120, "elapsed": 66, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e75b310e-c7da-4843-f058-54225e6cedcd"
Sample['sensor_15'].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 219} id="OWRvC0PZuKhk" executionInfo={"status": "ok", "timestamp": 1683788137504, "user_tz": -120, "elapsed": 1277, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="6fc5f610-0b31-4838-92b3-1088e808ba5e"
Sample[[ 'sensor_50', 'sensor_51' ]].fillna(0)
Sample['sensor_15'] = 0
Sample.interpolate(limit_direction='both' , inplace=True)
msno.matrix(Sample , figsize=(20,3))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 215} id="03uTEfO5fD87" executionInfo={"status": "ok", "timestamp": 1683788141325, "user_tz": -120, "elapsed": 3842, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0282ebb9-9aba-472b-ab9b-dafea8d31216"
data[[ 'sensor_50', 'sensor_51' ]].fillna(0)
data['sensor_15'] = 0
data.interpolate(limit_direction='both' , inplace=True)
msno.matrix(data , figsize=(20,3))
```

<!-- #region id="NgPptdnY4s2D" -->
## Detection d'Outliers
<!-- #endregion -->

<!-- #region id="r9K8xs-e4-rN" -->
### Local Outlier Factor (LOF) Algorithm (Unsupervised)
<!-- #endregion -->

```python id="NCHiF3kg_lgF"
from sklearn.neighbors import LocalOutlierFactor
```

```python id="pkafxlQc5u6e"
test_out = Sample[["sensor_00","sensor_01", "machine_status"]]

lof = LocalOutlierFactor(n_neighbors=7, algorithm='auto',
                         metric='minkowski', contamination=0.04,
                         novelty=False, n_jobs=-1)

# Returns 1 of inliers, -1 for outliers
pred = lof.fit_predict(test_out[["sensor_00","sensor_01"]])
test_out["outliers"] = pred
test_out["outliers"].replace({-1: True , 1 : False  },inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="Raf1ygeVkgUZ" executionInfo={"status": "ok", "timestamp": 1683788143115, "user_tz": -120, "elapsed": 1426, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="85e3e859-4f70-4ff0-cdb2-ae39221a0a49"
# Définir les couleurs à utiliser pour chaque catégorie de machine_status
colors = {2: "green",  1: "orange",  0: "red"}
statU = {2: "Normal",  1: "Recovering",  0: "Broken"}

# Définir les types de marqueur à utiliser pour chaque valeur de la colonne "outliers"
markers = {False: "o", True: "*"}
Outliers = {False: "not outliers", True: "outliers"}

# Créer une colonne "color" dans le DataFrame en fonction de la valeur de "machine_status"
test_out["color"] = test_out["machine_status"].apply(lambda x: colors[x])

# Créer une colonne "marker" dans le DataFrame en fonction de la valeur de "outliers"
test_out["marker"] = test_out["outliers"].apply(lambda x: markers[x])

for mark in markers:
    d = test_out[test_out["outliers"]==mark]
    plt.scatter(d["sensor_00"], d["sensor_01"],
                c = d["color"],
                marker = markers[mark])

# Afficher le graphique

# Ajouter un titre et des étiquettes d'axe
plt.title("Relation entre sensor_00 et sensor_01")
plt.xlabel("sensor_00")
plt.ylabel("sensor_01")

# Ajouter une légende avec les couleurs correspondant pour chaque cas

plt.legend(handles=[plt.plot([],[],color=val, marker="o", ls="", label= statU[key] )[0] for key,val in colors.items()])
plt.legend(handles=[plt.plot([],[],color="black", marker=val, ls="", label= Outliers[key] )[0] for key,val in markers.items()])

# Ajouter une légende supplémentaire pour la catégorie "broken"
plt.legend()

# Afficher le graphique
plt.show()
```

<!-- #region id="a_DoI_2YOxgf" -->
## Supression des Capteurs Correlés et définir les capteurs pertinants
<!-- #endregion -->

<!-- #region id="zMzbqCRnh17X" -->
**Visualisation**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "11h2i5NCB3_FbBIoTBSHEd1fE6saDg2Gm"} id="jxtixtRQOXBV" executionInfo={"status": "ok", "timestamp": 1683788222423, "user_tz": -120, "elapsed": 79352, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ffb6fa16-3140-4262-a938-1a6c9158f268"
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(20, 5), dpi=80)

roll = 30

for sensor in Sensors:
    figure(figsize=(20, 5), dpi=80)
    temp = Sample[["timestamp", sensor , "machine_status"]]
    temp[str(sensor)+'_'+str(roll)] = temp[sensor].rolling(roll).mean()


    # Créer un dictionnaire de couleurs pour chaque modalité de la colonne "machine_status"
    colors = {0: "red", 1: "orange", 2: "green"}

    # Grouper les données par "machine_status" et parcourir chaque groupe pour tracer la courbe correspondante
    groups = temp.groupby("machine_status")
    statU = {2: "Normal",  1: "Recovering",  0: "Broken"}
    for name, group in groups:
        plt.plot(group["timestamp"], group[str(sensor)+'_'+str(roll)], label=statU[name], color=colors[name])

    # Ajouter une légende pour les couleurs
    plt.legend(title="Machine status")
    # Configuration des étiquettes de l'axe des abscisses
    N= 20
    plt.xticks(range(0, len(temp["timestamp"]), len(temp["timestamp"])//N), temp["timestamp"][::len(temp["timestamp"])//N], rotation=45)
    plt.title(str(sensor)+' avec rolling de '+str(roll))

    # Afficher le graphique
    plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="_QO2xpDgOXEL" executionInfo={"status": "ok", "timestamp": 1683788226540, "user_tz": -120, "elapsed": 4122, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="5c9c45a1-4cdc-43b7-d0e0-d4ddd6ae3955"
temp = pd.DataFrame()

roll = 30

for sensor in Sensors:
    temp[sensor] = data[sensor].rolling(roll).mean()
temp = temp.iloc[roll:]
corr = temp[Sensors].corr()
corr.style.background_gradient(cmap='coolwarm')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mdFWMsK74CEd" executionInfo={"status": "ok", "timestamp": 1683788237019, "user_tz": -120, "elapsed": 10490, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="28d290dd-114c-4829-b791-0ae7f87c38cc"
f = plt.figure(figsize=(19, 15))
plt.matshow(temp[Sensors].corr(), fignum=f.number)
plt.xticks(range(temp.select_dtypes(['number']).shape[1]), temp.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(temp.select_dtypes(['number']).shape[1]), temp.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
```

<!-- #region id="G5Ogvz2TyO_x" -->
## PCA
<!-- #endregion -->

```python id="Z7S0oW0G3ztT"
from itertools import chain
#classe pour standardisation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.decomposition
from sklearn.impute import SimpleImputer
```

```python colab={"base_uri": "https://localhost:8080/"} id="sX3eKxb5J4-S" executionInfo={"status": "ok", "timestamp": 1683788237020, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e0678adf-ba5e-4372-a6a4-3c672111229d"
roll = 30
temp2 = temp.iloc[roll:]
temp2.shape
```

<!-- #region id="d6QghSTLxJDg" -->
L'idée de faire une PCA est de regrouper des variables corréler entre elles afin de facilité l'apprentissage.

On posera plusieurs hypothèsess:
- sur les valeurs propres (toutes celles > à 1)
- sur le pourcentage d'explicabilité de la variance (> 95% de la VT)
- sur le gap entre deux dimension
<!-- #endregion -->

```python id="7sCFGGrf2EHd"
# Fit Transforme

def DataFrame_to_PCA(df,components):
    """
    Selectionne les varaibles qualitatives d'un dataFrame et applique une décomposition en N composante principales passé en paramètre.
    Retourne le model PCA , les données transformé et la liste des nom des colonnes.
    """
    df = df.select_dtypes(exclude=['object','bool','category','datetime64','timedelta'])

    Columns = list(df.columns)
    # On replace nos valeurs manquante par une valuer arbitraire.

    imputer = SimpleImputer( missing_values=np.nan , strategy='mean' )
    X = imputer.fit_transform(df.values)

    # On centre et reduit nos variable quantitatives
    X =  StandardScaler().fit_transform(X)

    if components == 0:
        components = df.shape[1]-1

    pca = PCA(n_components=components)

    pca.fit(X)
    return pca, X, Columns
```

```python id="bf-m5Ic02ENJ"
compo = 12
pca, X, varNames = DataFrame_to_PCA(data.drop(columns=["timestamp" , "machine_status" , "sensor_15"] ), compo)
```

```python colab={"base_uri": "https://localhost:8080/"} id="EtPssy5tNjOp" executionInfo={"status": "ok", "timestamp": 1683788240201, "user_tz": -120, "elapsed": 25, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ed0f974a-d2e6-4b0d-8e66-dfc267964e73"
print("X",X.shape)
print("compo",compo)
```

```python id="dTGFlAbQ2EPI"
def Variance_Exp(pca):
    """
    Retourne tableau du % de variance expliqué pour les N Axes selectionné
    """
    n = len(pca.explained_variance_ratio_)
    eig = pd.DataFrame(
        {
            "Dim" : ["Dim " + str(x + 1) for x in range(n)],
            "Val prop" : np.round(pca.explained_variance_ , decimals=2),
            "% var expli" : np.round(pca.explained_variance_ratio_ * 100 , decimals=2),
            "Sum % var" : np.round(np.cumsum(pca.explained_variance_ratio_ * 100) , decimals=2 )
        },
        columns = ["Dim", "Val prop", "% var expli", "Sum % var"]
    )
    return eig
```

```python colab={"base_uri": "https://localhost:8080/"} id="oIhxBL9TC3a4" executionInfo={"status": "ok", "timestamp": 1683788240202, "user_tz": -120, "elapsed": 24, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="638a5c61-7ba5-4bea-9b90-ba7642dc7609"
eigs = Variance_Exp(pca)
print(eigs)
```

<!-- #region id="0XT-25Fo1dhv" -->
- Sur les valeurs propres (toutes celles > à 1) ici on prendra          
 - 7 Dims:      0.95         1.87      89.79
- Sur le pourcentage d'explicabilité de la variance (> 95 de la VT)    
 - 12 Dims:       0.37         0.72      94.98
- Sur le gap entre deux dimension
 - 5 Dims:       2.09         4.09      85.79
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="l08Sa8o1Na9q" executionInfo={"status": "ok", "timestamp": 1683788240202, "user_tz": -120, "elapsed": 19, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b2917e53-440f-4611-e826-b636fddf23b6"
print("X",X.shape)
print("compo",compo)
print("varNames",len(varNames))
```

```python id="j6mO5LmsC3oN"
def Contrib_Var(pca ,X , varNames , compo ):
    """
    fonction permettant de visualiszer les variables les plus contributrices, selon les axes de chacunes de composantes de la PCA
    """

    # construction de notre matrice identite
    iden = np.identity(X.shape[1])

    #transformation sur les variables
    contrib_var = pca.transform(iden)

    # Contruction du dtaframe à partir de notre matrice transformé
    All_contribs = pd.DataFrame(np.round(contrib_var,3), columns =  ["Dim "+str(i) for i in  range(1, compo+1) ]  , index = varNames)

    Cols = list(chain.from_iterable(("Var D"+str(i), "Dim "+str(i) ) for i in range(1, compo+1)))
    Contribs = pd.DataFrame(columns=Cols)

    for i , col in zip( range(1,compo+1) , All_contribs.columns):
        temp = All_contribs[[col]]
        temp = temp.sort_values(by=col, ascending=False)
        Contribs["Var D"+str(i)] = list(temp.index)
        Contribs["Dim "+str(i)] = list(temp[col])

    return All_contribs ,Contribs
```

```python colab={"base_uri": "https://localhost:8080/"} id="SLO1YFSPDiD4" executionInfo={"status": "ok", "timestamp": 1683788240521, "user_tz": -120, "elapsed": 333, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ad9b7fa6-d735-4eec-9daf-71abc06a5d82"
df_contrib ,Contrib = Contrib_Var(pca ,X , varNames , compo)
print(Contrib.head(5))
print(Contrib.tail(5))
```

```python id="t9Wzj2cLEERy"
 def Contrib_indiv(pca ,X , compo):

    Xt =  pd.DataFrame(pca.transform(X).round(decimals=6) , columns=["Dim " + str(x) for x in range(1,compo+1) ])

    return Xt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 129} id="brPdFmOqEmys" executionInfo={"status": "ok", "timestamp": 1683788241357, "user_tz": -120, "elapsed": 845, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="25dcbb02-86a7-4eb2-85ac-05c91f968ccb"
Xt = Contrib_indiv(pca ,X , compo )
print(Xt.shape)
Xt.head(2)
```

```python id="g3hRrtn39RwQ"
df_info = data[["timestamp","machine_status"]].reset_index(drop=True)

df_VP = Xt[[ "Dim " + str(col) for col in range(1,8)]]
df_VP = pd.concat([df_VP, df_info ], ignore_index=True , axis=1 )
df_per = Xt[[ "Dim " + str(col) for col in range(1,13)]]
df_per = pd.concat([df_per, df_info ], ignore_index=True , axis=1 )
df_gap = Xt[[ "Dim " + str(col) for col in range(1,6)]]
df_gap = pd.concat([df_gap, df_info ] , ignore_index=False , axis=1  )
```

<!-- #region id="SWpcZBo136dN" -->
**df_gap**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 129} id="waMVvdorDkH9" executionInfo={"status": "ok", "timestamp": 1683788241358, "user_tz": -120, "elapsed": 30, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="fcd50879-24bf-4801-88fd-96dfd0095dde"
print(df_gap.shape)
df_gap.head(2)
```

```python id="PcVtzv8Ks8dJ"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(   df_gap[[ "Dim " + str(col) for col in range(1,6)]] , df_gap["machine_status"], test_size=0.30, shuffle = False, stratify = None)
```

<!-- #region id="v5SZo7w8PNzS" -->
## LSTM
<!-- #endregion -->

```python id="xqW3xwwgNYrv"
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
```

```python colab={"base_uri": "https://localhost:8080/"} id="hbYqhQ0QZI8o" executionInfo={"status": "ok", "timestamp": 1683788252086, "user_tz": -120, "elapsed": 17, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="204e94f9-a447-4b2a-e363-bef4afe612b7"
print(y_train.value_counts())
print(y_train[0:3])
y_train = to_categorical(y_train)
print(y_train[0:3])
```

```python colab={"base_uri": "https://localhost:8080/"} id="l3V0iKBPZT7o" executionInfo={"status": "ok", "timestamp": 1683788252087, "user_tz": -120, "elapsed": 17, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4fd9c4d8-130d-413f-a3c2-ff0627a54bfa"
y_test.value_counts()
print(y_test[0:3])
y_test = to_categorical(y_test)
print(y_test[0:3])
```

```python id="rHrmjnZ_NYx_"
seq_size = length = 30
batch_size = 64
#data_gen = TimeseriesGenerator(df_gap[["Dim "+str(col) for col in range(1,6)]], df_gap["machine_status"],
#                               length=length, batch_size=batch_size)

data_gen_train = TimeseriesGenerator(X_train.values, y_train,
                               length=length, batch_size=batch_size)

data_gen_test = TimeseriesGenerator(X_test.values,y_test,
                               length=length, batch_size=batch_size)
```

```python id="c4pPAU6vNY1i"
X, y = data_gen_train[0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="n-4tpgUVNY4d" executionInfo={"status": "ok", "timestamp": 1683788252088, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d28d739d-9c76-4138-b056-10cc61ce0fad"
print("X" ,  X.shape)
print( "y" , y.shape)
```

```python id="-QqOsNHMPMdn"
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
```

```python colab={"base_uri": "https://localhost:8080/"} id="9q0xFlYvRGiT" executionInfo={"status": "ok", "timestamp": 1683788253684, "user_tz": -120, "elapsed": 1604, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="bcbfe383-d0a6-4c96-dd18-bf98d5f45ac0"
num_features = X_train.shape[1]

# Créer le modèle LSTM
model = Sequential()
model.add(LSTM(30, activation='relu' , input_shape=(length, num_features) , return_sequences = True  ))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu' ,  return_sequences=True ))
model.add(Dropout(0.2))
model.add(LSTM(20, activation='relu' ,  return_sequences=False  ))
model.add(Dropout(0.2))
model.add(Dense(3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="mfzzIdJfRlpB" executionInfo={"status": "ok", "timestamp": 1683788848370, "user_tz": -120, "elapsed": 594691, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="130850c8-e78f-4f81-ffed-83746e1908c0"
# Entraîner le modèle
history = model.fit_generator(generator = data_gen_train , verbose=1 , validation_data = data_gen_test , epochs=3)
```

<!-- #region id="qxtx28CUhaMh" -->
# Test
<!-- #endregion -->

```python id="P58gU2gNArLR"

```
