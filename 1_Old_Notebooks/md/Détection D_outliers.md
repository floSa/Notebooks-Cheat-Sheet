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

```python colab={"base_uri": "https://localhost:8080/"} id="LjM_eSGcjQRV" executionInfo={"status": "ok", "timestamp": 1684156859936, "user_tz": -120, "elapsed": 19740, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b003d13f-63cb-4c75-c52c-164b9e19330c"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Etudes/"
```

```python id="xtveVodajUhf" executionInfo={"status": "ok", "timestamp": 1684156859939, "user_tz": -120, "elapsed": 11, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

```python id="GwwKd66ckjYg"
from sklearn.datasets import make_blobs
```

<!-- #region id="6e_AWh5P81RN" -->
## Détection D'outliers
<!-- #endregion -->

<!-- #region id="s1AbEVD7jwRU" -->
### Elliptic envelope
<!-- #endregion -->

<!-- #region id="XZaObLAdjwVO" -->
La principale hypothèse de cette méthode est que les données doivent être normalement distribuées (ensemble de données distribuées gaussiennes). L'intuition derrière l'enveloppe elliptique est très simple. Nous dessinons une ellipse autour des points de données en fonction de certains critères et nous classons tout point de données à l'intérieur de l'ellipse comme une observation aberrante (en vert) et toute observation à l'extérieur de l'ellipse comme une observation aberrante (en rouge).
<!-- #endregion -->

<!-- #region id="aLKywV_Uj2Ko" -->
![image.png](data:image/png;base64,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lSrewZvmNareunQLTWVNOPzTw9j08KbQI3zF5Z3q56vhqfTAN+Azbd5ENHIwAJKg7WgbtvzzlrBKufLBStS9U4dWVyt6LvTA260/XHLnuU4072jG+WXnceznx7Dvh/uw5Z+3YMWYFZoVfsmfl2D393fj3H+dQ8fp9L81lYiSjwGQRD1NPTj57kndo3S58v5KCcr/vhwb/mEDVvyRdgWvfBV+vhCbJm/C4fmH0VTWhP6W2IalICJSYgBYxNvtxYXVF1D171VYl7UOxX9WHLWiFwUR67LWYdcTu1C7oBZNZU3oPNdp96IQ0QjBALDZ7Vu30ftxL9qOtuH6wevoPNuJvuY+Pk2LiJKOAUBE5FAMACIih2IAEBE5FAOAiMihGABERA7FACAicigGABGRQzEAiIgcigFARORQDAAiIodiABAROdTIDwCPB6isBBYsCP7X47G7REREKWHkBkBXFzBvHiAI4a9584LvExE52MgMgK4uYMKEYGU/enRo5S/9e8IEhgAROdrIDICcHO0jf/UrJ8fukhIR2WbkBYDHo33kr35J77NNgIgcauQFQEVFbEf/0quiwu4SExHZYuQFQG6usQDIzbW7xEREthh5AcAzACKimIy8AJDaAEaNilzxS++zDYCIHGrkBQDAu4CIiGIwMgOgqwvIytI+E5D+nZXFfgBE5GgjMwCAYOWudyaQk8PKn4gcb+QGgMTjCTb05uYG/8tr/kREAJwQAEREpIkBQETkUAwAIiKHYgAQETkUA4CIyKEYAEREDsUAICJyKAYAEZFDMQCIiByKAUBE5FAMACIih2IAEBE5FAOAiMihGABERA7FACAicigGABGRQzEAiIgcigFARORQDAAiIodiABARORQDgIjIoRgAREQOxQAgIsu53W54vV67i+F4DAAispzb7UZRURFqa2sZBDZiABCRLUpLSyGKIoPARgwAIrKF2+2GKIryi0FgPQYAEdlGOgtgENjD1ADYtGkTamtr+eKLL75iem3fvh2iKGLp0qVYunQpg8BipgTAtWvXsHnz5rAk54svvviK9aUOAOlVWlqK3t5eM6oqUjElAACgpaXF9qMJvvjiK71e0hmAFADKEOAZQPKZFgBEREaxDcBeDAAisgXvArIfA4CIbMF+APZjABCR5dgTODUwAIjIchwLKDUwAIiIHIoBQETkUAwAIiKHYgAQ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)
<!-- #endregion -->

```python id="QQGX1o6cjxUG"
# La fonction crée un ensemble de données distribuées gaussiennes.
# deux caractéristiques pour X et pas de y.
X, _ = make_blobs(n_samples=300, n_features=2, centers=1, shuffle=True, random_state=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 284} id="Sd2xG6i0kHfp" executionInfo={"status": "ok", "timestamp": 1617383558594, "user_tz": -120, "elapsed": 1112, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="d6e16bf8-b05d-4a73-bb20-b70441218a1f"
sns.set_style("darkgrid")
sns.scatterplot(x=X[:,0], y=X[:,1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 326} id="O6OVzaHdkKXN" executionInfo={"status": "ok", "timestamp": 1617383591878, "user_tz": -120, "elapsed": 1899, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="e23c8087-7dd3-4504-f5a9-14a47a5d80dd"
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,3))
sns.distplot(X[:,0], ax=ax[0], color="darkblue")
sns.distplot(X[:,1], ax=ax[1], color='green')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 284} id="Ly9dwj9Tkwgo" executionInfo={"status": "ok", "timestamp": 1617383678016, "user_tz": -120, "elapsed": 1383, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="f6bdb096-9e02-4ba6-c011-d617d5aaf35d"
from sklearn.covariance import EllipticEnvelope

elpenv = EllipticEnvelope(contamination=0.025,  random_state=1)

# Renvoie 1 pour les valeurs aberrantes, -1 pour les valeurs aberrantes.
pred = elpenv.fit_predict(X)

# Extraire les valeurs aberrantes
outlier_index = np.where(pred==-1)
outlier_values = X[outlier_index]

# Affichage
sns.scatterplot(x=X[:,0], y=X[:,1])
sns.scatterplot(x=outlier_values[:,0], 
                y=outlier_values[:,1], color='r')
```

<!-- #region id="hsLeCU0xlUbR" -->
La méthode **fit_predict()** de l'objet enveloppe elliptique (elpenv) renvoie 1s pour les valeurs aberrantes et -1s pour les valeurs aberrantes. Nous pouvons extraire les valeurs aberrantes à l'aide de la fonction **np.where()**. Les points de données colorés en rouge sont des valeurs aberrantes. Les observations aux valeurs d'indice suivantes sont des valeurs aberrantes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="847MGDTGlFf0" executionInfo={"status": "ok", "timestamp": 1617383817793, "user_tz": -120, "elapsed": 868, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="389abc7d-c121-46d7-f353-3bbfbebb4841"
print(outlier_index[0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="etS996dpljEt" executionInfo={"status": "ok", "timestamp": 1617383904783, "user_tz": -120, "elapsed": 521, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="ba6fd21c-ba37-4c0b-99c4-0815625738fd"
# Extract inliers
inlier_index = np.where(pred==1)
inlier_values = X[inlier_index]

df = pd.DataFrame(inlier_values, columns=['x1', 'x2'])
df.head()
```

<!-- #region id="qMjntRafmC-3" -->
###IQR-based detection
<!-- #endregion -->

<!-- #region id="z-bjJcdzmFEx" -->
La détection basée sur l'IQR est une approche statistique. Cette technique est appliquée pour des caractéristiques individuelles, et non pour des observations entières comme dans l'enveloppe elliptique. L'intuition derrière la détection basée sur l'IQR est également très simple. Tout d'abord, nous calculons le premier quartile (Q1) et le troisième quartile (Q3) des données. Ensuite, nous obtenons la différence entre ces quartiles. Cette différence s'appelle l'IQR (InterQuartile Range).  
*   IQR = Q3-Q1
*   Lower bound = Q1–1.5(IQR)
*   Upper bound = Q3+1.5(IQR)


<!-- #endregion -->

<!-- #region id="WNHp7xfCmg_D" -->
![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAkUAAADxCAYAAADFusOLAAAgAElEQVR4nO3df4hcV93H8Sttd9u10DXSQKh/FCyFNpS1RAKN7bMLhSyB9KH0124pkkJj1n8KQkj/k1mGBoz2SUOhwraiYMZfrcWiNGAi7CKDGihEFGVDFf9YKSgTRAKr5kn5PH/kOdOzZ8+dmTt7Z84997xfcNDuzO6cOefM93zmzr2TTAAAAFAWugMAAABVQCgCAAAQoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQgAAEASoQglaDabyrJM7Xa75/3W19c1MzOjLMu6bWZmRuvr6937tFqt7s86nY7m5+e33UeS2u32lr+TZZlardaW+5jfd+/nu7/5e/2eA0bPrJP5+Xl1Op3uzzc3N7W0tLRlDt37SDfWUN6c++6P8aJeoMoIRdgRu9j0KxC+wmQXMFOUms3mlv92i5wpqv02vX5Fzu6z2XCXlpa0ubk5iqHCgHyhyLdB5m1YvUJRlmXMcUDUC1QdoQhDczeffkXO3N99h+bebv6Or8jlFVW7oLlFstfRBHNf3+MjDDcU2UeI3E3IzJm9RvLWWa8jCRg96gViQCjCUOzN6ODBg32Lg++jD7t4mdt97wTtn5l3fb5C6W6mvYqcKZb2Jmt+Zhc+jJ87j3kfpxnumsjbTO01yEY2XtQLxIJQhKG0Wi01m82BN5q8Q9O9CpJb5Pq903cLZa8iZ4qlXdBMkeSQeFjuZuV7l25zN6e8UGT+LkeKxo96gVgQirAjgxa5Xh+JtFotb4EpWuSkrSdx9jtHwP07vYoixqdoKHLXTr9zinhnHw71AlVHKMKO7OQjCXuz8x2eHvad3yBFzvfujnNOqmHYI0WDhCI+NguLeoGqIxRhRwYtcr1OVBy0yEnbzxEw92m329s+HvG9kzO/7ytkFLlqGOScolar1V0r7kcb7sdn9pVrHCUKi3qBqiMUYUeKHg43BcT9vUEOh0v+q0ncIwP9Tpw0hc79OYfDq2GQq8987+p7nWhtB6O8q5kwetQLVB2hCDuSV+R87/R83xfibnK9Tpzs9Xd8m2Ne0fJdjitx4mRVDPM9Re5RJF/48V2+j/GiXqDqCEXYkSJFzr3M1i4mg15ia/i+2M09kXaQS2ztfnOJbTUU+UZr30mwg1ySz0YWBvUCVUcoQmWU8WVo9jkDIR4fYdjnGCEN1AuMAqEIleF+bf848bX9QFyoFxgFQhEqxf4HHseJf+ARiA/1AmUjFAEAAIhQBAAAIIlQBAAAIIlQhEh86lOfCt0FVBRrA66vfe1robuASBGKEIUsY6nCj7UB1yc+8YnQXUCkqCaIAhsf8rA24GJNYFisHERheno6dBdQUawNuAhFGBYrB1HIskznzp0L3Q1UEGsDrizL9MEHH4TuBiJEKEIUsizTwsJC6G6gglgbcGVZptdeey10NxAhQhEq7/Tp0zp06JBeffVVPf300/rwww9DdwkVwdqAz8MPP6y77rpLf/jDH0J3BZEhFCG4zc1NXblyRX/961/15z//WWtra1pZWdHx48d1//336/HHH9f169clSW+99Zb27Nmjffv2qdFo6MKFC7p48aI++OADXblyJfAzQdlYGxjU1atX9dZbb+nAgQNaXV2VJN1///06duyYfvazn4XtHKJBKIrQ+++/r+XlZT3zzDPav3+/7rnnHu3atUtzc3PKsiyqNjc3p9tuu027du3SXXfdpYceekizs7M6duyYXnnlldx3enUbg127dumee+7R/v37tbCwoOXlZb3//vvJrAPWxujXi3H58mWtra3pnXfe0ZtvvqlTp05peXk5inbmzBm99NJLevHFF3X06FF9+ctf1tzcnPbs2aPbb79dTz/9tH79619veb4rKys6fPiwsizToUOHdOjQIT333HN68cUXdebMmeDPadB26tQpvfnmm3rnnXe0tramy5cvDzTfKIZQFBHeCdfXlStX9MEHH+jixYu6cOGCGo2G9u3bpz179uitt97acl/WAYqsF+nGP2B6/Phx3XvvvXryySc1OzurJ554QkePHtVLL72kRqMRRXv11Vd16tQpvfbaa3rzzTf1wx/+UKurqwN/bPqb3/xG586dU6vV0muvvaZXX301+HMatL300ks6evSonnjiCc3OzurJJ5/Uvffeq+PHj/MP05aIUBSJM2fOcM5Egj788EM9/fTTOnPmjCTWAXpz14skvf7663rhhRf0yiuvcHShZi5fvqxXXnlFL7zwgl5//fXQ3akFQlEE3nvvPa6uSZz5iIR1gEEsLCzovffe09tvv60vfelLobuDMTh27Jjefvvt0N2IHqEo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)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DTeYJyKXl9Mj" executionInfo={"status": "ok", "timestamp": 1617384069745, "user_tz": -120, "elapsed": 1218, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="2b3718c5-92c1-42a3-a392-b58e9573f62a"
X, _ = make_blobs(n_samples=300, n_features=2,
                  centers=1, shuffle=True, random_state=1)

# Cette fonction renvoie les indices aberrants d'une caractéristique.
# L'entrée est une caractéristique
def outlier_indices(x):
    Q1, Q3 = np.percentile(x, [25, 75])
    IQR = Q3-Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    outlier_index = np.where((x < lower_bound) | (x > upper_bound))
    return outlier_index

print("Outlier Indices of x1: ", outlier_indices(X[:,0])[0])
print("\nOutlier Values of x1: \n", X[outlier_indices(X[:,0])])
print("\n")
print("Outlier Indices of x2: ", outlier_indices(X[:,1])[0])
print("\nOutlier Values of x2: \n", X[outlier_indices(X[:,1])])
```

```python colab={"base_uri": "https://localhost:8080/"} id="1vktjBphmlWJ" executionInfo={"status": "ok", "timestamp": 1617384150986, "user_tz": -120, "elapsed": 835, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="25f752e3-447f-4215-86eb-ac4749b638d8"
index_union = np.union1d(outlier_indices(X[:,0])[0], 
                         outlier_indices(X[:,1])[0])
index_union
```

```python colab={"base_uri": "https://localhost:8080/", "height": 366} id="m5mMg01pm5Ra" executionInfo={"status": "ok", "timestamp": 1617384185052, "user_tz": -120, "elapsed": 1613, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="095f0319-3874-4e96-b169-514e580c1b3d"
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

sns.scatterplot(x=X[:,0], y=X[:,1], ax=ax[0])
sns.scatterplot(x=outlier_values[:,0], ax=ax[0],
                y=outlier_values[:,1], color='r')

sns.scatterplot(x=X[:,0], y=X[:,1], ax=ax[1])
sns.scatterplot(x=X[index_union][:,0], y=X[index_union][:,1], ax=ax[1],
                color='r')

ax[0].set_title("Elliptic Envelope Detection", fontsize=15, pad=15)
ax[1].set_title("IQR-based Detection", fontsize=15, pad=15)
```

<!-- #region id="7fuhRW7Fnz9X" -->
**Suppression des outliers du DF**

l'index_union pour supprimer les valeurs aberrantes de l'ensemble de données.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 380} id="nC221elonBZb" executionInfo={"status": "ok", "timestamp": 1617384370449, "user_tz": -120, "elapsed": 1544, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="a6d00411-ae54-4006-cb4a-69dd7ac10fd4"
df = pd.DataFrame(inlier_values, columns=['x1', 'x2'])
cleaned_df = df.drop(labels=index_union, axis=0, inplace=False)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

sns.scatterplot(x=X[:,0], y=X[:,1], ax=ax[0])
sns.scatterplot(x=X[index_union][:,0], y=X[index_union][:,1], ax=ax[0],
                color='r')

sns.scatterplot(x=cleaned_df['x1'], y=cleaned_df['x2'], ax=ax[1])

ax[0].set_title("Data with outliers", fontsize=15, pad=15)
ax[1].set_title("Data without outliers", fontsize=15, pad=15)
```

<!-- #region id="cTXn1RLxoDR9" -->
**Visualisation par boxplot**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 441} id="II-OCp7Inurh" executionInfo={"status": "ok", "timestamp": 1617384477036, "user_tz": -120, "elapsed": 1961, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="3b755d1e-2542-48cd-f8b0-43101d4a96a8"
sns.set_style("white")
plt.tight_layout(pad=10)

fig, ax = plt.subplots(nrows=2, ncols=2,
                       figsize=(20,6), sharex='col')
sns.boxplot(x=X[:,0], ax=ax[0,0], color='yellow')
sns.boxplot(x=X[:,1], ax=ax[0,1], color='red')
sns.boxplot(x=cleaned_df['x1'], ax=ax[1,0], color='yellow')
sns.boxplot(x=cleaned_df['x2'], ax=ax[1,1], color='red')

ax[0,0].set_title("Boxplot of x1 with outliers", fontsize=15)
ax[0,1].set_title("Boxplot of x2 with outliers", fontsize=15)
ax[1,0].set_title("Boxplot of x1 without outliers", fontsize=15)
ax[1,1].set_title("Boxplot of x2 without outliers", fontsize=15)
```

<!-- #region id="jg5DiZUQ5qmo" -->
### One-Class SVM Algorithm
<!-- #endregion -->

<!-- #region id="W86-3UoZ59Wz" -->
Détection des nouveautés lorsque les données d'apprentissage ne sont pas trop polluées par des valeurs aberrantes. Cet algorithme peut être appliqué à des ensembles de données hautement dimensionnels et il n'y a pas d'hypothèse sous-jacente dans la distribution des données.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 311} id="jriqnrf84rp_" executionInfo={"status": "ok", "timestamp": 1617389189274, "user_tz": -120, "elapsed": 2265, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="b529e44e-d518-457c-ef89-f6ef2179972d"
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM
sns.set_style("darkgrid")

# Make a simulated dataset with 2 features
# Normally distributed dataset
"""X, _ = make_blobs(n_samples=300, n_features=2,
                  centers=1, shuffle=True, random_state=1)"""

one_class_svm = OneClassSVM(kernel='rbf', degree=3, gamma='scale')

# Define a new observation
new_data = np.array([[-4, 8.5]])

one_class_svm.fit(X)

# Make prediction
pred = one_class_svm.predict(new_data)

# Extract outliers
outlier_index = np.where(pred==-1)
outlier_values = new_data[outlier_index]

# Plot the data
sns.scatterplot(x=X[:,0], y=X[:,1])
sns.scatterplot(x=outlier_values[:,0], 
                y=outlier_values[:,1], color='orange')
plt.title("One-Class SVM Novelty Detection", fontsize=15, pad=15)
#plt.savefig("One-Class SVM Detection.png", dpi=80)
```

<!-- #region id="WMwAAHnX6MSt" -->
### Local Outlier Factor (LOF) Algorithm
<!-- #endregion -->

<!-- #region id="Y1qHv-QT6S5Y" -->
Local Outlier Factor (LOF) est un algorithme d'apprentissage automatique non supervisé qui a été créé à l'origine pour la détection de valeurs aberrantes, mais qui peut désormais être utilisé pour la détection de nouveautés. Il fonctionne bien sur les ensembles de données à haute dimension.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="GlCeWq7S6Dlj" executionInfo={"status": "ok", "timestamp": 1617389362776, "user_tz": -120, "elapsed": 1384, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="46f8cf5b-d2d8-4fe4-fb3e-781ba79ae2b3"
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
sns.set_style("darkgrid")

# Make a simulated dataset with 2 features
# Normally distributed dataset
"""X, _ = make_blobs(n_samples=300, n_features=2,
                  centers=1, shuffle=True, random_state=1)"""

lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                         metric='minkowski', contamination=0.04,
                         novelty=False, n_jobs=-1)

# Returns 1 of inliers, -1 for outliers
pred = lof.fit_predict(X)

# Extract outliers
outlier_index = np.where(pred==-1)
outlier_values = X[outlier_index]

# Plot the data
sns.scatterplot(x=X[:,0], y=X[:,1])
sns.scatterplot(x=outlier_values[:,0], 
                y=outlier_values[:,1], color='r')
plt.title("LOF Outlier Detection", fontsize=15, pad=15)
plt.savefig("LOF Outlier Detection.png", dpi=80)
```

<!-- #region id="x0ufbQBU7RM1" -->
**Détection de nouveauté**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="IRuTG5pE7Q3I" executionInfo={"status": "ok", "timestamp": 1617389577565, "user_tz": -120, "elapsed": 1305, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="97a6138d-a5ae-4d08-8e46-800a792071b6"
lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto',
                         metric='minkowski', contamination=0.04,
                         novelty=True, n_jobs=-1)

# Define a new observation
new_data = np.array([[-4, 8.5]])

lof.fit(X)

# Make prediction
pred = lof.predict(new_data)

# Extract outliers
outlier_index = np.where(pred==-1)
outlier_values = new_data[outlier_index]

# Plot the data
sns.scatterplot(x=X[:,0], y=X[:,1])
sns.scatterplot(x=outlier_values[:,0], 
                y=outlier_values[:,1], color='orange')
plt.title("LOF Novelty Detection", fontsize=15, pad=15)
plt.savefig("LOF Novelty Detection.png", dpi=80)
```

<!-- #region id="tj1EtmtSoL7U" -->
### Isolation Forest Algorithm
<!-- #endregion -->

<!-- #region id="JJnPcHjPoN0a" -->
  L'Isolation forest isole les observations en sélectionnant aléatoirement une caractéristique, puis en choisissant aléatoirement une valeur de division entre les valeurs maximale et minimale de la caractéristique sélectionnée.
<!-- #endregion -->

<!-- #region id="FDTkXT9844Hg" -->
1/ application de  l'analyse en composantes principales pour réduire les données multidimensionnelles en données bidimensionnelles qui peuvent être représentées dans un graphique en 2D, tout en conservant un pourcentage élevé % de variation dans les données d'origine
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 325} id="xk8Qg3fNoIfY" executionInfo={"status": "ok", "timestamp": 1617388825746, "user_tz": -120, "elapsed": 1583, "user": {"displayName": "Florian H.", "photoUrl": "", "userId": "05323848775778397032"}} outputId="b8b288d6-02f6-4717-806e-68ec2172edc5"
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sns.set_style("darkgrid")

# Import the iris dataset
data = load_iris()
X = data.data

iforest = IsolationForest(n_estimators=100, max_samples='auto', 
                          contamination=0.05, max_features=1.0, 
                          bootstrap=False, n_jobs=-1, random_state=1)

# Returns 1 of inliers, -1 for outliers
pred = iforest.fit_predict(X)

# Extract outliers
outlier_index = np.where(pred==-1)
outlier_values = X[outlier_index]

# Feature scaling 
sc=StandardScaler()
X_scaled = sc.fit_transform(X)
outlier_values_scaled = sc.transform(outlier_values)

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
outlier_values_pca = pca.transform(outlier_values_scaled)

# Plot the data
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1])
sns.scatterplot(x=outlier_values_pca[:,0], 
                y=outlier_values_pca[:,1], color='r')
plt.title("Isolation Forest Outlier Detection (Iris Data)", 
           fontsize=15, pad=15)
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.savefig("Isolation Forest Detection.png", dpi=80)
```

<!-- #region id="oHr-F9r35Z4c" -->
l'hyperparamètre de contamination - une valeur que nous ne connaissons pas. Il représente la proportion de valeurs aberrantes dans l'ensemble de données.  
Les valeurs de cet hyperparamètre sont comprises entre 0 et 0,5. Si nous pensons qu'il y aura beaucoup de valeurs aberrantes dans nos données, nous pouvons fixer la contamination à une valeur plus grande. Le fait de ne pas connaître la proportion exacte de valeurs aberrantes dans l'ensemble de données est la principale limite de l'utilisation de cette méthode.
<!-- #endregion -->

<!-- #region id="l0ycRmMmEtgz" -->
## Autres
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 143} id="jlBprVFDkK-j" executionInfo={"status": "ok", "timestamp": 1684156908830, "user_tz": -120, "elapsed": 9, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="44c6d176-116e-490b-916d-edae4247c119"
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/"
df = pd.read_csv(path+"glass.csv")
df.sample(3)
```

<!-- #region id="0KA5nhoabYfj" -->
Le graphique résiduel est un outil très utile non seulement pour détecter les mauvais algorithmes d'apprentissage automatique, mais également pour identifier les valeurs aberrantes . 

Dans un graphique résiduel, la variable indépendante est représentée sur l'axe horizontal x et la valeur résiduelle sur l'axe vertical y.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 490} id="0E1nsOpRatbV" executionInfo={"status": "ok", "timestamp": 1684157116850, "user_tz": -120, "elapsed": 1058, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e95a867c-0ea3-4978-f6f7-4bbc3b4fcf5e"
# residual plot with seaborn library
sns.residplot(x='RI', y='K', data=df, scatter_kws=dict(s=50))

# ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# labels and title
plt.xlabel('RI', fontsize=14)
plt.ylabel('residuals', fontsize=14)
plt.title('Residual plot', fontsize=20);
```

<!-- #region id="GqSmtblYdFHD" -->
###Distance de Cook
<!-- #endregion -->

<!-- #region id="qalazAFvdKiC" -->
Distance de cuisiniera été introduit par le statisticien américain Dennis Cook en 1977. 

Il mesure l'effet de la suppression d'une observation donnée dans l'analyse de régression.  
En d'autres termes, c'est un moyen de détecter les points de données qui modifient considérablement le coefficients de régression(données très influentes).  
On peut considérer l'influence comme le produit d'effet de levier(une observation avec des valeurs extrêmes sur une variable indépendante) et  d'aberration (une observation avec de grandes valeurs résiduelles).  
Mathématiquement la distance de Cook s'exprime comme suit :

![image.png](data:image/png;base64,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)  

où y_hat j est la valeur de la jème réponse prédite lorsque toutes les observations sont incluses, y_hat j(i) est la valeur de la jème réponse prédite lorsque l'observation i n'est pas incluse pour le calcul du modèle de régression, p est le nombre de coefficients, et s² est l'erreur quadratique moyenne du modèle de régression.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 490} id="a0xxNnqpekuM" executionInfo={"status": "ok", "timestamp": 1684157354964, "user_tz": -120, "elapsed": 2393, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="abf13897-832c-4f0e-e257-fc7fc3d2dc5f"
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence
import seaborn as sns
import matplotlib.pyplot as plt


# fit the regression model using statsmodels library 
f = 'RI ~ K'
model = ols(formula=f, data=df).fit()

# calculate the cooks_distance - the OLSInfluence object contains multiple influence measurements
cook_distance = OLSInfluence(model).cooks_distance
(distance, p_value) = cook_distance

# scatter plot - x axis (independent variable height), y-axis (dependent variable weight), size and color of the marks according to its cook's distance
sns.scatterplot(x='RI', y='K', data=df, hue=distance, size=distance, sizes=(50, 200), edgecolor='black', linewidth=1)

# ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# labels and title
plt.xlabel('RI', fontsize=14)
plt.ylabel('K', fontsize=14)
plt.title('Cook\'s distance', fontsize=20);
```

```python colab={"base_uri": "https://localhost:8080/", "height": 490} id="N8J6vn2Zfpq2" executionInfo={"status": "ok", "timestamp": 1684157494768, "user_tz": -120, "elapsed": 477, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="404e0d7e-0838-49eb-8870-555ab0d4ed88"
threshold = 4/100

# stem plot - the x-axis represents the index of the observation and the y-axis its Cook's distance
plt.stem(distance, basefmt=" ")

# horizontal line showing the threshold value
plt.hlines(threshold, -2, 214, 'r')

# the observations with Cook's distances higher than the threshold value are labeled in the plot
influencial_data = distance[distance > threshold]

for index, value in influencial_data.items():
    plt.text(index, value, str(index), fontsize=14)

# ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# labels and title
plt.xlabel('index', fontsize=14)
plt.ylabel('cook\'s distance', fontsize=14)
plt.title('Cook\'s distance', fontsize=20);
```

<!-- #region id="HRZXKpLinuA_" -->
**Autre**
<!-- #endregion -->

```python id="r7q7o7wDEvV0"
df.drop('new', axis=1, inplace=True)
df.drop(5, axis=0, inplace=True)
df.insert(0, 'new', np.random.random(5))
pd.melt(df, id_vars='names').head()
df.pivot_table(index='name', columns='ctg', aggfunc='mean')
pd.get_dummies(df)
dfa.merge(df, on='id')
```
