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

<!-- #region id="jHuBprGFbBeX" -->
#Manipulation des Variables
<!-- #endregion -->

<!-- #region id="vRgpeu5Pa9zA" -->
####Convetir le nom d'une variable en sa string
<!-- #endregion -->

```python id="ldbrACeGa5fR" colab={"base_uri": "https://localhost:8080/", "height": 52} executionInfo={"status": "ok", "timestamp": 1723100461788, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="710ea810-bf0d-4c21-b6e8-5963653140df"
#Convetir le nom d'une variable en sa string

# Python 3 code
x = 7
x = [ i for i, a in locals().items() if a == x][0]
print("The variable name:", x)
x
```

<!-- #region id="jTMuAoFCpFtr" -->
Chercher dans une liste une dictionnaire
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uv7YuFmJpF3t" executionInfo={"status": "ok", "timestamp": 1722932175146, "user_tz": -120, "elapsed": 327, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="38da459e-0a9d-4edd-cc76-9db5a2c2f155"
ma_liste = [1, 2, 3, 2, 4, 2]
val = 2
index = [i for i, x in enumerate(ma_liste) if x == val]
print(index)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} id="vkG8OgqPqEGD" executionInfo={"status": "ok", "timestamp": 1723100716158, "user_tz": -120, "elapsed": 375, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="fafea981-34bd-4678-f6ed-0b64a1267161"
import pandas as pd
df = pd.DataFrame()
df["happy"] = [1,3,5,7,9,11]
df["surprise"] = [0,2,4,6,8,1]
df["happy"] = df["happy"].astype(str)
df.happy

df["surprise"]
```

```python id="A93jSGoiQwMK" colab={"base_uri": "https://localhost:8080/", "height": 35} executionInfo={"status": "ok", "timestamp": 1723100728557, "user_tz": -120, "elapsed": 313, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f464feaf-3363-44c5-e378-39b1b591e521"
df["happy"].max()
```

```python id="fMRHzo4OzlrF"
df.dtypes()
```
