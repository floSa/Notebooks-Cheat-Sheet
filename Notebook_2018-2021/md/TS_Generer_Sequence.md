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

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="184b_JGrahki" executionInfo={"status": "ok", "timestamp": 1697792888121, "user_tz": -120, "elapsed": 358, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c3d05118-6f3b-48a4-f25a-994c9cfdef5b"
import pandas as pd
import numpy as np
# Création du DataFrame
df = pd.DataFrame({
    'X1': list(range(30)),
    'X2': list(range(3)) * 10,
    'y': list(range(29, -1, -1))
})

# Affichage du DataFrame
df.head(2)
```

```python id="VvGxJOUCAxzW"
seq_length = 4
overlap = 2
pos = 0

def time_seq_generator(df, seq_length, seq_cols, target , overlap, pos):
    sequences, labels = [], []
    df = df.iloc[::-1].reset_index(drop=True)
    for i in range(len(df) - seq_length):
        if not i % overlap:
            X_values , y_labels= df[seq_cols].values, df[target].values
            sequences.append(X_values[i:i+seq_length])
            labels.append(y_labels[i-pos] )
    X_result = np.array(sequences)
    X_result = X_result[::-1, ::-1, ...]
    y_result = np.array(labels).reshape(-1, 1)
    y_result = y_result[::-1, ::-1, ...]
    return X_result, y_result

X, y  =  time_seq_generator(df, seq_length, ['X1', 'X2'], "y" , 2, pos)
```

```python colab={"base_uri": "https://localhost:8080/"} id="ly11zkd_bdOb" executionInfo={"status": "ok", "timestamp": 1697792890021, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2473eb6b-6c72-4b43-a457-64455f60b7ec"
X[0:2]
```

```python colab={"base_uri": "https://localhost:8080/"} id="aalNIyuwSZAF" executionInfo={"status": "ok", "timestamp": 1697792890598, "user_tz": -120, "elapsed": 5, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="acc23963-156b-4a49-8543-02627da6c12f"
y[0:2]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 676} id="pXa-LLu3GWV0" executionInfo={"status": "ok", "timestamp": 1697789138178, "user_tz": -120, "elapsed": 274, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="28774702-413c-481a-8628-fde963faa443"
df.head(20)
```

```python id="arLTRCgDGlIk"

```
