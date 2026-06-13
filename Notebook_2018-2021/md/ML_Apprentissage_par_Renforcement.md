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

<!-- #region id="B8Sb2OL_Y0EC" -->
#Prédiction du taux de clics publicitaires (CTR) à l'aide de l'apprentissage par renforcement 
<!-- #endregion -->

<!-- #region id="_jQ68DM7Y6iv" -->
**Contexte**

Sur les pages de sites Internet affichant des publicités, les entreprises qui souhaitent faire de la publicité ciblé pour leurs produits. L'objectif est que pour l'entreprise disposant d'une gamme de versions de publicité, est de savoir quelle vesion  obtenient le taux de conversion (nombre de clics sur l'annonce) le plus élevé.

L' ensemble de données comprend la réponse de 10 000 visiteurs à 10 publicités affichées sur une plateforme Web. Ces 10 publicités sont en fait les 10 versions publicitaires du même produit. Les réponses sont représentées en termes de récompenses attribuées à ces 10 annonces par les visiteurs. Si le visiteur a cliqué sur une annonce, la récompense est 1 et si le visiteur a ignoré l'annonce, la récompense est 0. 

la tâche est d'identifier laquelle parmi les 10 annonces a le CTR le plus élevé afin que l'annonce avec le taux de conversion le plus élevé doit être placée sur la plateforme Web.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="D0uOYIJBYzdt" executionInfo={"status": "ok", "timestamp": 1635953440339, "user_tz": -60, "elapsed": 47376, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="78ca9e74-3119-4838-cdd9-0d71394d2210"
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/"
```

```python id="wJDgJFAVbHcb" executionInfo={"status": "ok", "timestamp": 1635953490291, "user_tz": -60, "elapsed": 382, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}}
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random
import math
```

```python colab={"base_uri": "https://localhost:8080/", "height": 224} id="O_EAV-bFa6T_" executionInfo={"status": "ok", "timestamp": 1635953531088, "user_tz": -60, "elapsed": 335, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="4434332f-fee1-4846-f88f-70ce7a9be421"
df = pd.read_csv(path+'Ads_CTR_Optimisation.csv')
print('taille:',df.shape)
df.head()
```

<!-- #region id="jqU6663sbljx" -->
## Utilisation de la méthode de sélection aléatoire

Pour voir la différence, nous devons d'abord voir comment fonctionne la sélection aléatoire des versions d'annonces. Parmi l'ensemble de 10 versions d'annonces, une annonce est sélectionnée au hasard et affichée au visiteur. 
<!-- #endregion -->

<!-- #region id="-tWmEyddcoGZ" -->
algorithme sélectionne une annonce au hasard et l'affiche au visiteur du site. Si le visiteur clique sur l'annonce, la récompense 1 est ajoutée et si le visiteur l'ignore, la récompense 0 est ajoutée. Voir dans la capture d'écran ci-dessous cette sélection aléatoire d'annonces.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vWl1kHEKbXn9" executionInfo={"status": "ok", "timestamp": 1635954405813, "user_tz": -60, "elapsed": 300, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="d6610885-8c9d-4ef3-9068-388f9d7ad0cc"
N = 10000
d = 10
total_prize = 0
selected = []
for n in range(0,N):
    ad = random.randrange(d) # on récupère le numéro d'une publicité au hazard 
    selected.append(ad)
    prize = df.values[n,ad] # si la nième ligne de la publicité numéro (ad) vaut 1 alors le prix est 1 sinon 0
    total_prize = total_prize + prize
print("Total Prize:",total_prize)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 307} id="XdXzBr4ib6vv" executionInfo={"status": "ok", "timestamp": 1635953693477, "user_tz": -60, "elapsed": 646, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="e62d179e-fb66-4b58-d2a2-9c3f02c951a2"
# Visualisation des résultats 
plt.hist(selected) 
plt.title('Histogramme des sélections d\'annonces') 
plt.xlabel('Annonces') 
plt.ylabel('Nombre de fois où chaque annonce a été sélectionnée' ) 
plt.show()
```

<!-- #region id="CGMjpipjbIBH" -->
## Upper Confidence Bound (UCB)
<!-- #endregion -->

<!-- #region id="0bGFiuCXfBbs" -->
La **Limite de confiance supérieure** (UCB) utilise la borne supérieure d'un intervalle de confiance autour de cette moyenne associée à chaque document  

![image.png](data:image/png;base64,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)  

* **xi** est la proportion de clics observée pour la publicité **i**
* **ti** est le nombre de fois où la publicité **i** a été recommandé 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="fe1JYDIyb7S0" executionInfo={"status": "ok", "timestamp": 1635960972470, "user_tz": -60, "elapsed": 587, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="f404d178-86de-4f4b-fa3e-53bbf118ffae"
N = 10000
d = 10
prizes = [0] * d
click = [0] * d
total_prize = 0
selected = [0]
for n in range(1,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if(click[i] > 0):
            avg_prize = prizes[i] / click[i]
            delta = math.sqrt(2*math.log(n)/click[i])
            ucb = avg_prize + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
            
    selected.append(ad)   
    click[ad] = click[ad] + 1
    prize = df.values[n,ad]
    prizes[ad] = prizes[ad] + prize
    total_prize = total_prize + prize
    
print("Total Prize:" , total_prize)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 310} id="hmGyUbADfWtn" executionInfo={"status": "ok", "timestamp": 1635960973178, "user_tz": -60, "elapsed": 382, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="5e0d51eb-8f3d-4710-c6e3-cdeeffa34892"
# Visualisation des résultats
plt.hist(selected)
plt.title('Histogramme des sélections d\'annonces')
plt.xlabel('Annonces')
plt.ylabel('Nombre de fois que chaque annonce a été sélectionnée')
plt.show()
```

<!-- #region id="Y_CCqRdWysNn" -->
## Thompson Sampling
<!-- #endregion -->

<!-- #region id="t9jP6HBuzPGU" -->
L'échantillonnage de Thompson est un algorithme pour les problèmes de décision en ligne où les actions sont prises séquentiellement d'une manière qui doit trouver un équilibre entre l'exploitation de ce qui est connu pour maximiser la performance immédiate et l'investissement pour accumuler de nouvelles informations qui peuvent améliorer la performance future.  

Maintenant, pour chaque inernaute, on parcours chaque publicité et sélectionne la publicité avec la distribution bêta aléatoire la plus élevée.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="VNfEO6uTgLcW" executionInfo={"status": "ok", "timestamp": 1635960828000, "user_tz": -60, "elapsed": 352, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="fc3153a2-c05e-4290-97eb-8b2317688cc2"
N = 10000
d = 10
prizes = [0] * d
total_prize = 0
selected = []
rewards = [0] * d
penalties = [0] * d

for n in range(1,N):
    ad = 0
    max_th = 0
    # pour chaque observation, on parcourir chaque pub et sélectionne la pub avec la distribution bêta aléatoire la plus élevée.
    for i in range(0,d):
        random_beta = random.betavariate ( rewards[i] + 1 , penalties[i] + 1)
        if random_beta > max_th:
            max_th = random_beta
            ad = i
    # Mise à jour la liste 'pub selected' avec le internaute/pub sélectionné par Thompson Sampling.
    selected.append(ad) 
    # apres la sélection de la pub, on vérifie l'ensemble de données pour cette pub et le nombre d'observations, 
    # Si il s'agit d'une récompense, nous mettrons à jour la liste des « récompenses » en en ajoutant une à cette internaute/pub sélectionné.
    prize = df.values[n,ad]
    if prize == 1:
        rewards[ad] = rewards[ad] + 1
    else:
        penalties[ad] = penalties[ad] + 1
    
    total_prize = total_prize + prize
    
print("Total Prize:" , total_prize)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 310} id="PYOmaqWS3SwO" executionInfo={"status": "ok", "timestamp": 1635960841445, "user_tz": -60, "elapsed": 1002, "user": {"displayName": "Florian H.", "photoUrl": "https://lh3.googleusercontent.com/a/default-user=s64", "userId": "05323848775778397032"}} outputId="1e1c9b45-13e4-47c1-b953-11631e9b468a"
# Visualisation des résultats
plt.hist(selected)
plt.title('Histogramme des sélections d\'annonces')
plt.xlabel('Annonces')
plt.ylabel('Nombre de fois que chaque annonce a été sélectionnée')
plt.show()
```

```python id="t4pViTMo3VqX"

```
