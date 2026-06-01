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

<!-- #region id="view-in-github" -->
<a href="https://colab.research.google.com/github/MachineLearnia/Deep-Learning-Youtube/blob/main/Tensorflow_MNIST_pour_d%C3%A9butants.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

<!-- #region id="kLrzSZ1Cfv6L" -->
Ce Notebook permet aux débutants d'apprendre à développer un premier modèle de classification sur le dataset MNIST, en utilisant l'API Keras.
<!-- #endregion -->

```python id="AIUwgqwP5hkp"
#!pip install tensorflow
```

```python id="cYvnJ6A8lmHK"
import warnings
warnings.filterwarnings('ignore')
```

<!-- #region id="oMi_RWW8Fle2" -->
# Implémentation de Différent type de Layers
<!-- #endregion -->

```python id="oa63KejBfog8"
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
```

<!-- #region id="UvrLumwPhZrN" -->
**1. Chargement des données et Normalisation**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="oPZCV7XTfucK" outputId="d7e7192a-2e70-4e97-fdad-44a9eea71c14" executionInfo={"status": "ok", "timestamp": 1685429705769, "user_tz": -120, "elapsed": 453, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# Chargement des données MNIST
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

print('trainset:', X_train.shape) # 60,000 images
print('testset:', X_test.shape) # 10,000 images

# Normalisation des données
X_train = X_train / 255
X_test = X_test / 255
```

<!-- #region id="JKEjEqbOhdh0" -->
**2. Visualisation des données**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 130} id="BmTNBz7bgeZG" outputId="7df5015d-0713-4ee4-a141-a241f67b2554" executionInfo={"status": "ok", "timestamp": 1685428496683, "user_tz": -120, "elapsed": 2252, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# visualisation de quelques images
fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 4))
for i in range(10):
  ax[i].imshow(X_train[i], cmap='gray')

plt.tight_layout()
plt.show()
```

<!-- #region id="Ugi5iYgBe3Vm" -->
## ANN
<!-- #endregion -->

<!-- #region id="FInfYGsMhflz" -->
**3. Configuration des Couches du Réseau de Neurones**
<!-- #endregion -->

```python id="NlOiNL7LoA56"
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
```

```python id="L6tJGIC1nbXb"
# Configuration des couches du réseau

Out_classes = 10
Inputs= (28,28)

model = Sequential()
model.add(Flatten(input_shape=Inputs))
model.add(Dense(64, activation='relu', input_shape=Inputs))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(Out_classes, activation='softmax'))
```

<!-- #region id="YfdDh4_zhkL7" -->
**4. Entrainement du Réseau de Neurones**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="tW4uOQEVw_QH" executionInfo={"status": "ok", "timestamp": 1684231785772, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b5b49d1a-c83f-4e59-9428-72a4803f902d"
# Compilation du modele
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="TvrDQgcTfeDx" outputId="a7b44937-4301-42aa-a704-602b350580a1" executionInfo={"status": "ok", "timestamp": 1684232049989, "user_tz": -120, "elapsed": 264223, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# Entrainement du modele
history = model.fit(X_train, y_train, validation_split=0.10, epochs=10, batch_size= 8, verbose=1)
```

<!-- #region id="EnifmnaOhodj" -->
**5. Évaluation du réseau de neurone sur les données de Test**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3LdmrOwKg8nK" outputId="2c378ef2-85fd-4fe2-dca8-10bf9a4b2cc8" executionInfo={"status": "ok", "timestamp": 1684232051758, "user_tz": -120, "elapsed": 1782, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# Evaluation du modele
test_loss, test_acc = model.evaluate(X_test,  y_test)
print('Test accuracy:', test_acc)
```

<!-- #region id="9a8vswFYhuGL" -->
**6. Création d'un modele prédictif**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="TFv035qwg-Q6" outputId="5bf1e485-4c47-4d02-9179-ff49d3a4a248" executionInfo={"status": "ok", "timestamp": 1684232051759, "user_tz": -120, "elapsed": 18, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# modele prédictif (softmax)
prediction_model = keras.Sequential([model, keras.layers.Softmax()])
predict_proba = prediction_model.predict(X_test)
predictions = np.argmax(predict_proba, axis=1)

print(predictions[:30])
print(y_test[:30])
```

<!-- #region id="8QwArnrxTk_2" -->
**7. History**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="xohRkq1LvAEH" executionInfo={"status": "ok", "timestamp": 1684232051760, "user_tz": -120, "elapsed": 12, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="01add2a7-df18-4a65-a9df-67d2489cad52"
# summarize history for accuracy
plt.plot(history.history['accuracy'][0:])
plt.plot(history.history['val_accuracy'][0:])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="2tmcAK8HvM6_" executionInfo={"status": "ok", "timestamp": 1684232052347, "user_tz": -120, "elapsed": 595, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="92864a7b-dea6-4139-fa29-2d0ee19ef4b6"
# summarize history for loss
plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

<!-- #region id="CxD7XZKMvudw" -->
### Autre Exemple
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Ze-5_ZtJVVYn" executionInfo={"status": "ok", "timestamp": 1684239571626, "user_tz": -120, "elapsed": 7519284, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2b555b44-cb06-4c24-dc48-865256b84c68"
# google.colab
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/MyDrive/Machine_Learning_Cheat_Sheet/Data/"
```

```python id="Q3ePFoigwDtz"
import pandas as pd
import numpy as np
```

```python id="wd16sp_ATsvt"
# load pima indians dataset
dataset = pd.read_csv(path+"diabetes.csv")
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
```

```python id="DEFrQcFrTxVT"
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

```python id="LG8mgAtATzIL"
# Fit the model
history = model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
```

```python colab={"base_uri": "https://localhost:8080/"} id="v2KrCWaUXeTt" executionInfo={"status": "ok", "timestamp": 1684239611767, "user_tz": -120, "elapsed": 28, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="39592f2d-31f0-4f77-9129-6700aaa79aa9"
# list all data in history
print(history.history.keys())
```

```python colab={"base_uri": "https://localhost:8080/"} id="stSBy7VbXgKF" executionInfo={"status": "ok", "timestamp": 1684239611768, "user_tz": -120, "elapsed": 25, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e3aa034a-aa20-4a15-a80c-c54545093d08"
# summarize history for accuracy
plt.plot(history.history['accuracy'][2:])
plt.plot(history.history['val_accuracy'][2:])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="WBzZKSxNTmmW" executionInfo={"status": "ok", "timestamp": 1684239611768, "user_tz": -120, "elapsed": 24, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7138d383-a943-467f-87a1-d215d19f88eb"
# summarize history for loss
plt.plot(history.history['loss'][2:])
plt.plot(history.history['val_loss'][2:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

<!-- #region id="QchtNzavy6Xj" -->
##RNN
<!-- #endregion -->

```python id="yL_pKTMPXhic"
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
```

```python id="40xiP74K0g3c"
# Configuration des couches du réseau
Out_classes = 10
Inputs= (28,28)

model = Sequential()
#model.add(Flatten(input_shape=Inputs))
model.add(SimpleRNN(64,input_shape=Inputs ,  return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(Out_classes, activation='softmax'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="oXhlaXyj01iQ" executionInfo={"status": "ok", "timestamp": 1684239611769, "user_tz": -120, "elapsed": 21, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2773a8ac-7f23-4e22-92f7-dda128ad42d3"
# Compilation du modele
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="cFDPpykp3Iul" executionInfo={"status": "ok", "timestamp": 1684241975493, "user_tz": -120, "elapsed": 2363739, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3ea28aa0-705c-41f4-ef47-21a0a701067e"
# Entrainement du modele
history = model.fit(X_train, y_train, validation_split=0.10, epochs=10, batch_size= 8, verbose=1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="vs0elVzo3OTZ" executionInfo={"status": "ok", "timestamp": 1684241977855, "user_tz": -120, "elapsed": 2373, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3e2a27f7-9378-45fb-924f-019ccca912a0"
# Evaluation du modele
test_loss, test_acc = model.evaluate(X_test,  y_test)
print('Test accuracy:', test_acc)
```

```python colab={"base_uri": "https://localhost:8080/"} id="e9Ag01zs300l" executionInfo={"status": "ok", "timestamp": 1684241981526, "user_tz": -120, "elapsed": 3676, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0da185fd-769a-4a9e-9a75-a600248b7635"
# modele prédictif (softmax)
prediction_model = keras.Sequential([model, keras.layers.Softmax()])
predict_proba = prediction_model.predict(X_test)
predictions = np.argmax(predict_proba, axis=1)

print(predictions[:30])
print(y_test[:30])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="OzpQxxfl34md" executionInfo={"status": "ok", "timestamp": 1684241981532, "user_tz": -120, "elapsed": 29, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="4b9ef6af-bf0f-4aa0-869d-e3b5d91841ac"
# summarize history for accuracy
plt.plot(history.history['accuracy'][0:])
plt.plot(history.history['val_accuracy'][0:])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 472} id="7n5gTQpu36r5" executionInfo={"status": "ok", "timestamp": 1684241981533, "user_tz": -120, "elapsed": 25, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e9a751ad-c907-4b2d-f2ec-1e669f07afc6"
# summarize history for loss
plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

<!-- #region id="CPxJSFrb7ntA" -->
##LSTM
<!-- #endregion -->

```python id="ALG5AuJS9haQ"
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
```

```python id="NGuQngvP6_G7" executionInfo={"status": "ok", "timestamp": 1684241982041, "user_tz": -120, "elapsed": 528, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} colab={"base_uri": "https://localhost:8080/"} outputId="d4958fb2-9453-43c7-9f88-38cd863bf909"
# Configuration des couches du réseau
Out_classes = 10
Inputs= (28,28)

model = Sequential()
#model.add(Flatten(input_shape=Inputs))
model.add(LSTM(64, return_sequences=True, input_shape=Inputs))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(Out_classes, activation='softmax'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="38KbvG8I92X8" executionInfo={"status": "ok", "timestamp": 1684241982041, "user_tz": -120, "elapsed": 8, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="673a6ebc-ecd4-4330-d163-96b1be2911f8"
# Compilation du modele
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
```

```python id="25RcxeKl99w7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1684243974745, "user_tz": -120, "elapsed": 1992709, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3e1d054a-52fa-452d-977d-aadbc3e9a182"
# Entrainement du modele
history = model.fit(X_train, y_train, validation_split=0.10, epochs=10, batch_size= 8, verbose=1)
```

```python id="A1mx4oza-B3l" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1684243977115, "user_tz": -120, "elapsed": 2390, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b8eab0f6-c2f5-49c7-b6e0-a59554859ee8"
# Evaluation du modele
test_loss, test_acc = model.evaluate(X_test,  y_test)
print('Test accuracy:', test_acc)
```

```python id="FMkPmk4W-JPy" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1684243981214, "user_tz": -120, "elapsed": 4107, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e358e13c-beb5-46a8-ee81-f5a1dd4ed840"
# modele prédictif (softmax)
prediction_model = keras.Sequential([model, keras.layers.Softmax()])
predict_proba = prediction_model.predict(X_test)
predictions = np.argmax(predict_proba, axis=1)

print(predictions[:30])
print(y_test[:30])
```

```python id="x-R1qsbO-Ji1" colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"status": "ok", "timestamp": 1684243981215, "user_tz": -120, "elapsed": 29, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="67a6acad-b80c-4c4c-d239-981c8c3d74a6"
# summarize history for accuracy
plt.plot(history.history['accuracy'][0:])
plt.plot(history.history['val_accuracy'][0:])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

```python id="dnwkHMh1-J1O" colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"status": "ok", "timestamp": 1684243981215, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a45de329-6aa2-467d-8c20-497a8cf24ae3"
# summarize history for loss
plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

<!-- #region id="av0yjSFlAqN-" -->
##CNN
<!-- #endregion -->

```python id="tDuR7vbiApYb"
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```

```python id="NXYKfSKW-KKB"
Out_classes = 10
Inputs = (28, 28, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=Inputs, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(Out_classes, activation='softmax'))
```

```python colab={"base_uri": "https://localhost:8080/"} id="JIvv-KgtBEL_" executionInfo={"status": "ok", "timestamp": 1684243981216, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="01adaafb-5c57-4206-d415-dfecbc36fd17"
# Compilation du modele
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="g_AIqNkVBGEF" executionInfo={"status": "ok", "timestamp": 1684244244730, "user_tz": -120, "elapsed": 263523, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="2e67e6ac-3cde-4fad-9a11-47f27fc93353"
# Entrainement du modele
history = model.fit(X_train, y_train, validation_split=0.10, epochs=10, batch_size= 8, verbose=1)
```

```python id="-vYavY4HBI9L" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1684244245295, "user_tz": -120, "elapsed": 593, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="cc61a9c7-2a3c-45e0-aba8-de01f1323ed0"
# Evaluation du modele
test_loss, test_acc = model.evaluate(X_test,  y_test)
print('Test accuracy:', test_acc)
```

```python id="nOMLiRB3BLyD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1684244247265, "user_tz": -120, "elapsed": 1972, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="5569a8e5-dc6b-4bc5-a2f7-85b2bfc3b3e6"
# modele prédictif (softmax)
prediction_model = keras.Sequential([model, keras.layers.Softmax()])
predict_proba = prediction_model.predict(X_test)
predictions = np.argmax(predict_proba, axis=1)

print(predictions[:30])
print(y_test[:30])
```

```python id="BuNjy_zjBP6f" colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"status": "ok", "timestamp": 1684244248115, "user_tz": -120, "elapsed": 854, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="1cf3f23d-1417-4437-8463-6456dbad2c0d"
# summarize history for accuracy
plt.plot(history.history['accuracy'][0:])
plt.plot(history.history['val_accuracy'][0:])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

```python id="WsPGQ1KQBSDW" colab={"base_uri": "https://localhost:8080/", "height": 472} executionInfo={"status": "ok", "timestamp": 1684244248115, "user_tz": -120, "elapsed": 27, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="58ae950e-272d-463b-fe7d-5b51cbbd8f97"
# summarize history for loss
plt.plot(history.history['loss'][0:])
plt.plot(history.history['val_loss'][0:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

<!-- #region id="0XbFKkvw7e2k" -->
# Batch Manuel pour classification
<!-- #endregion -->

<!-- #region id="ORrY9UJxz2PN" -->
####Fonction Down Samples equilibre
<!-- #endregion -->

```python id="daMQgDiw5MGO"
import numpy as np

def DownSampleBalanced(y, k):
    unique_elements, counts = np.unique(y, return_counts=True)
    n_min = np.min(counts)
    indices = [np.where(y==elem)[0] for elem in unique_elements]
    samples_idx = []
    for i in range(k):
        sample_indices = []
        for idxs in indices:
            sample_idxs = np.random.choice(idxs, n_min, replace=False)
            sample_indices.extend(sample_idxs)
        np.random.shuffle(sample_indices)
        samples_idx.append(sample_indices)
    return samples_idx
```

<!-- #region id="K_e6yNl98CNR" -->
**Test de la Fonction Samples égalitaire**


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="pNsY3Bkr-UHL" executionInfo={"status": "ok", "timestamp": 1684850281809, "user_tz": -120, "elapsed": 12, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9ac4d003-435b-4198-8e13-45444ff24d6e"
X = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0] ,[2, 1], [1, 3] , [2, 2], [2, 3], [1, 4] ])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 2 , 1 ])

k = 3
samples_idx = DownSampleBalanced(y, k)
for i , idx in enumerate(samples_idx):
    print("fold " + str(i)+ ':')
    print("indexs:",idx)
    print("labels:",y[idx])
```

<!-- #region id="Eo3xgi3ZX0t1" -->
####Fonction Up Samples equilibre
<!-- #endregion -->

```python id="cF3CeXhHZqa1"
def array_shuffled(lst1, lst2):
    assert len(lst1) == len(lst2)
    p = np.random.permutation(len(lst2))
    return lst1[p], lst2[p]
```

<!-- #region id="oaybpHYna-JG" -->
**Test de la Fonction Shuffle Array**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="hYRjdN7ja3VK" executionInfo={"status": "ok", "timestamp": 1684850281814, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b44608ae-b1aa-40d8-e41a-781e80bfce58"
X = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0] ,[2, 1], [1, 3] , [2, 2], [2, 3], [1, 4] ])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 2 , 1 ])

X, y = array_shuffled(X, y)

for i in range(len(y)):
    print("y:",y[i],"- X:",X[i])
```

<!-- #region id="edPm-cfSciYJ" -->
####Fonction Storted modulo Flex
<!-- #endregion -->

```python id="mVR7u7IhxXg5"
def Stort_modulo(ar_data, ar_labels):
    unique_elts = set(ar_labels.flatten())
    ar_labelSorted, ar_dataSorted = [] , []
    while len(ar_labels) != 0 :
        for elt in unique_elts:
            try:
                position = np.where(ar_labels == elt)[0][0]
                ar_labelSorted.append(ar_labels[position]  )
                ar_dataSorted.append(ar_data[position])
                ar_labels = np.delete(ar_labels, position)
                ar_data = np.delete(ar_data, position,0)
            except:
                pass
    return  np.array(ar_dataSorted) ,  np.array(ar_labelSorted)
```

<!-- #region id="2rKd2GvwcyLo" -->
**Test de La Fonction**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kDBqm1_lc0yV" executionInfo={"status": "ok", "timestamp": 1684850281817, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c83e9b63-d207-47dc-aa32-3796456fa9d0"
X = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0] ,[2, 1], [1, 3] , [2, 2], [2, 3], [1, 4] ])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 2 , 1 ])
X, y = Stort_modulo(X, y)
for i in range(len(y)):
    print("y:",y[i],"- X:",X[i])
```

<!-- #region id="BMlTCKy90HJM" -->
### Génération du set de données
<!-- #endregion -->

```python id="alcrtqYHDbD6"
from sklearn.datasets import make_classification

Features = 10 # Nombre de Features Explicatives
classes = 3 # Nombre de modalité
weights =[0.5,0.3,0.2]

X, y = make_classification(n_samples=5000,
                            n_features=Features,
                            n_informative=2,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=classes,
                            n_clusters_per_class=1,
                            weights=weights)
```

<!-- #region id="ov7TnvaHERSU" -->
## **Avec Nombre de Batch défini**
<!-- #endregion -->

<!-- #region id="Swld2qeS1gTm" -->
### Equilibré non trié
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="AfpxdzL_vFZK" executionInfo={"status": "ok", "timestamp": 1684850285234, "user_tz": -120, "elapsed": 3432, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="199e8fc9-992a-49be-f805-83e208a3d780"
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Paramètres pour la sélection des données
epochs = 10 # Taille du jeu de données
n_batches = 9 # Nombre de batchs par epoch

X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.33, random_state=42)

# Sample Equilibré par Down Sampling
samples_idx = DownSampleBalanced(y_train, epochs)

# Définition du modèle
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=Features, activation='relu'),
    keras.layers.Dense(32, input_dim=64, activation='relu'),
    keras.layers.Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du modèle
for idx ,epoch in zip(samples_idx , range(epochs)):
    # Sélection manuelle des données aléatoires
    indices = np.random.choice(len(X_train), epochs , replace=False) #epochs
    X_batch = X_train[indices]
    y_batch = to_categorical(y_train[indices], num_classes=classes)
    batch_losses = []
    batch_accs = []
    # Entrainement du modèle sur chaque batch
    for batch in range(n_batches):
        batch_loss, batch_acc = model.train_on_batch(X_batch, y_batch)
        # Ajouter la loss et l'accuracy de chaque batch à leurs listes respectives
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)

    # Calculer la moyenne de la loss et de l'accuracy de chaque batch
    mean_batch_loss = np.mean(batch_losses)
    mean_batch_acc = np.mean(batch_accs)
    # Evaluation du modèle sur les données d'entrainement
    val_loss, val_acc = model.evaluate(X_val, to_categorical(y_val, num_classes=classes), verbose=0)
    {print(f"Epoch {epoch+1}/{10} - Train loss: {mean_batch_loss:.4f} - Train Acc: {mean_batch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")}
```

<!-- #region id="HV7cH1E42DV6" -->
### Trié non equilibré
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="NMrOJE6M2Gj5" executionInfo={"status": "ok", "timestamp": 1684850292219, "user_tz": -120, "elapsed": 6987, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="019352e4-51bb-4b74-ba8b-09fb678f905c"
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Paramètres pour la sélection des données
epochs = 10 # Taille du jeu de données
n_batches = 9 # Nombre de batchs par epoch

X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.33, random_state=42)

# Définition du modèle
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=Features, activation='relu'),
    keras.layers.Dense(32, input_dim=64, activation='relu'),
    keras.layers.Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du modèle
for epoch in range(epochs):
    # Sélection manuelle des données aléatoires
    indices = np.random.choice(len(X_train), epochs , replace=False) #epochs

    # Melange le dataset d'entrainement
    X_batch , y_batch = array_shuffled(X_train, y_train)
    # Ordonnance le dataset par les classe du Label
    X_batch , y_batch = Stort_modulo(X_batch, y_batch)

    y_batch = to_categorical(y_batch, num_classes=classes)
    batch_losses = []
    batch_accs = []
    # Entrainement du modèle sur chaque batch
    for batch in range(n_batches):
        batch_loss, batch_acc = model.train_on_batch(X_batch, y_batch)
        # Ajouter la loss et l'accuracy de chaque batch à leurs listes respectives
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)

    # Calculer la moyenne de la loss et de l'accuracy de chaque batch
    mean_batch_loss = np.mean(batch_losses)
    mean_batch_acc = np.mean(batch_accs)
    # Evaluation du modèle sur les données d'entrainement
    val_loss, val_acc = model.evaluate(X_val, to_categorical(y_val, num_classes=classes), verbose=0)
    {print(f"Epoch {epoch+1}/{10} - Train loss: {mean_batch_loss:.4f} - Train Acc: {mean_batch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")}
```

<!-- #region id="e-97uSPYEDns" -->
## **Avec Taille du batch défini**
<!-- #endregion -->

<!-- #region id="e17HTRg5gTh6" -->
### Equilibré non trié
<!-- #endregion -->

```python id="94nY_e9cna3c"
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Paramètres pour la sélection des données
epochs = 10 # Taille du jeu de données
t_batch = 9 # Taille des batchs

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Sample Equilibré par Down Sampling
samples_idx = DownSampleBalanced(y_train, epochs)

# Définition du modèle
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=Features, activation='relu'),
    keras.layers.Dense(32, input_dim=64, activation='relu'),
    keras.layers.Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du modèle
for idx ,epoch in zip(samples_idx , range(epochs)):
    # Sélection manuelle des données aléatoires

    X_epoch, y_epoch = X_train[idx], y_train[idx]
    batch_losses = []
    batch_accs = []
    # Entrainement du modèle sur chaque batch
    for i in range(0, len(X_epoch), t_batch):
        X_batch = X_epoch[i:i+t_batch]
        y_batch = to_categorical(y_epoch[i:i+t_batch], num_classes=classes)
        batch_loss, batch_acc = model.train_on_batch(X_batch, y_batch)
        # Ajouter la loss et l'accuracy de chaque batch à leurs listes respectives
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)
    # Calculer la moyenne de la loss et de l'accuracy de chaque batch
    mean_batch_loss = np.mean(batch_losses)
    mean_batch_acc = np.mean(batch_accs)

    # Evaluation du modèle sur les données d'entrainement
    val_loss, val_acc = model.evaluate(X_val, to_categorical(y_val, num_classes=classes), verbose=0)

    print(f"Epoch {epoch+1}/{epochs} - Train loss: {mean_batch_loss:.4f} - Train Acc: {mean_batch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

```

<!-- #region id="Kg2YHnEIgUGo" -->
### Trié non equilibré
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Dju7QHJ7tSnm" executionInfo={"status": "ok", "timestamp": 1684850347902, "user_tz": -120, "elapsed": 25292, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a6b5ec81-ee58-4cb8-93aa-833645feb0e4"
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Paramètres pour la sélection des données
epochs = 10 # Taille du jeu de données
t_batch = 9 # Taille des batchs

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

# Définition du modèle
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=Features, activation='relu'),
    keras.layers.Dense(32, input_dim=64, activation='relu'),
    keras.layers.Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du modèle
for epoch in range(epochs):
    # Sélection manuelle des données aléatoires
    indices = np.random.choice(len(X_train), epochs , replace=False) #epochs

    # Melange le dataset d'entrainement
    X_batch , y_batch = array_shuffled(X_train, y_train)
    # Ordonnance le dataset par les classe du Label
    X_batch , y_batch = Stort_modulo(X_batch, y_batch)

    y_batch = to_categorical(y_batch, num_classes=classes)
    batch_losses = []
    batch_accs = []

    # Entrainement du modèle sur chaque batch
    for i in range(0, len(X_epoch), t_batch):
        X_batch = X_epoch[i:i+t_batch]
        y_batch = to_categorical(y_epoch[i:i+t_batch], num_classes=classes)
        batch_loss, batch_acc = model.train_on_batch(X_batch, y_batch)
        # Ajouter la loss et l'accuracy de chaque batch à leurs listes respectives
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)
    # Calculer la moyenne de la loss et de l'accuracy de chaque batch
    mean_batch_loss = np.mean(batch_losses)
    mean_batch_acc = np.mean(batch_accs)

    # Evaluation du modèle sur les données d'entrainement
    val_loss, val_acc = model.evaluate(X_val, to_categorical(y_val, num_classes=classes), verbose=0)

    print(f"Epoch {epoch+1}/{epochs} - Train loss: {mean_batch_loss:.4f} - Train Acc: {mean_batch_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

```

<!-- #region id="uQ5GRQaN6nUh" -->
### Class Weight
<!-- #endregion -->

<!-- #region id="okrZcaoDhK3I" -->
- **compute_class_weight :**  
Cette fonction est utilisée pour calculer les poids de classe afin de compenser les déséquilibres de classes dans un ensemble de données d'apprentissage. Elle prend en compte les étiquettes de classe et retourne les poids correspondants pour chaque classe. Ces poids de classe peuvent ensuite être utilisés pour attribuer plus d'importance aux classes sous-représentées lors de l'apprentissage du modèle. Cela permet de résoudre les problèmes de déséquilibre de classe et d'améliorer les performances du modèle.  
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="panLz5V2iYPI" executionInfo={"status": "ok", "timestamp": 1684850347903, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="75bf7d36-89c6-483d-80be-db90f4a45c4e"
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Calcul des poids de classe
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights = dict(zip(np.unique(y), class_weights))

unique, counts = np.unique(y, return_counts=True)
freqs = dict(zip(unique, counts))
total_count = sum(freqs.values())
freqs = {key: count / total_count for key, count in freqs.items()}
print("frequence par classe:\n",freqs)
print("class weights:\n",class_weights)

# Création du modèle
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=Features, activation='relu'),
    keras.layers.Dense(32, input_dim=64, activation='relu'),
    keras.layers.Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_cat = to_categorical(y, num_classes=classes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="eYIneWmDirbY" executionInfo={"status": "ok", "timestamp": 1684850360934, "user_tz": -120, "elapsed": 13040, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f85062f4-56ec-4c56-fd84-c96eafacf2a6"
# Entraînement du modèle avec les coefficients de poids en utilisant model.fit
model.fit(X, y_cat, class_weight=class_weights, validation_split=0.33 ,epochs=10, batch_size=8)
```

<!-- #region id="sjSc_qwu6rnX" -->
### Sample Weight
<!-- #endregion -->

<!-- #region id="Yfkuglry1wph" -->
- **compute_sample_weight :**  
Cette fonction est utilisée pour calculer les poids d'échantillon pour prendre en compte des facteurs autres que les classes, tels que des pondérations spécifiques à chaque échantillon ou des attributs individuels des échantillons. Elle prend en compte des paramètres tels que les étiquettes d'échantillon ou les attributs d'échantillon spécifiques, et elle retourne les poids correspondants pour chaque échantillon. Ces poids d'échantillon peuvent ensuite être utilisés pour attribuer des coefficients de poids différents à chaque échantillon lors de l'entraînement du modèle.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aFoiED_50XJj" executionInfo={"status": "ok", "timestamp": 1684850360935, "user_tz": -120, "elapsed": 27, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="7f6d64e4-5966-4f8a-ad85-69daa1edef6a"
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_sample_weight

epochs = 10

unique, counts = np.unique(y, return_counts=True)
freqs = dict(zip(unique, counts))
total_count = sum(freqs.values())
freqs = {key: count / total_count for key, count in freqs.items()}

# Création du modèle
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=Features, activation='relu'),
    keras.layers.Dense(32, input_dim=64, activation='relu'),
    keras.layers.Dense(classes, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] , weighted_metrics=[])

# Calcul des poids d'échantillon
sample_weights = compute_sample_weight(class_weight='balanced', y=y)
print("frequence par classe:\n",freqs)
y_cat = to_categorical(y, num_classes=classes)
```

```python colab={"base_uri": "https://localhost:8080/"} id="CFeElpAnEjDU" executionInfo={"status": "ok", "timestamp": 1684850382289, "user_tz": -120, "elapsed": 21377, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="90224cf9-d7fb-4eac-d4a7-17adcbb9e719"
model.fit(X, y_cat, sample_weight=sample_weights, validation_split=0.33 ,epochs=10, batch_size=9)
```

<!-- #region id="nOWF8us5MuRF" -->
## Eplicaion des modèles
<!-- #endregion -->

```python id="PPttW5QZ9Y3l"
pip install shap
```

```python id="Xwc1J4ZRlMOR" executionInfo={"status": "ok", "timestamp": 1685428632177, "user_tz": -120, "elapsed": 5851, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} colab={"base_uri": "https://localhost:8080/"} outputId="37daa16f-64db-4e9a-c45c-bc8800260127"
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

# Configuration des couches du réseau

Out_classes = 10
Inputs= (28,28)

model = Sequential()
model.add(Flatten(input_shape=Inputs))
model.add(Dense(64, activation='relu', input_shape=Inputs))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(Out_classes, activation='softmax'))

# Compilation du modele
model.compile(optimizer='adam',
              loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6qdyZ6UPUWyS" executionInfo={"status": "ok", "timestamp": 1685428709645, "user_tz": -120, "elapsed": 77473, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="d8fe55d3-edf6-49d9-acbe-38a9b3a793d2"
# Entrainement du modele
history = model.fit(X_train, y_train, validation_split=0.10, epochs=3, batch_size= 8, verbose=1)
```

```python colab={"base_uri": "https://localhost:8080/"} id="p8U9yJKFY_WG" executionInfo={"status": "ok", "timestamp": 1685429322289, "user_tz": -120, "elapsed": 85936, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="1019a46d-675f-46ef-b36a-56a9e62ef2f6"
# this is the code from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 422} id="h0TmSV53YnKI" executionInfo={"status": "ok", "timestamp": 1685429376617, "user_tz": -120, "elapsed": 6647, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="23fc52fb-ee00-427f-cd5f-aa9ed384e41d"
# ...include code from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

import shap
import numpy as np

# select a set of background examples to take an expectation over
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

# explain predictions of the model on three images
e = shap.DeepExplainer(model, background)
# ...or pass tensors directly
# e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
shap_values = e.shap_values(x_test[1:5])

# plot the feature attributions
shap.image_plot(shap_values, -x_test[1:5])
```

```python colab={"base_uri": "https://localhost:8080/"} id="Upwoqp80FzcF" executionInfo={"status": "ok", "timestamp": 1685428709647, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b5f7dcee-6919-4e54-e640-643fd53206c9"
X_train[np.random.choice(X_train.shape[0], 500, replace=False)].shape
```

```python colab={"base_uri": "https://localhost:8080/"} id="dHsw0EKRYEBR" executionInfo={"status": "ok", "timestamp": 1685428776176, "user_tz": -120, "elapsed": 481, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="35cecdef-3f2b-4a06-f303-297a788772e3"
X_train.shape
```

```python id="9FfLa4_uPUJR"
import shap
background = X_train[:100]
test_images = X_test[0:4]

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)
```

```python id="0Ln8N0BNSATG"
shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
```

```python id="ohidEE5qUlZd"
import shap

# Sélection d'un ensemble d'exemples d'arrière-plan pour obtenir une estimation.
background = X_train[np.random.choice(X_train.shape[0], 500, replace=False)]

# Expliquer les prédictions du modèle sur trois images

e = shap.DeepExplainer(model, background)
#e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)

#shap_values = e.shap_values(X_test[0].reshape(1,28,28)) #.reshape(1,28,28)
shap_values = e.shap_values(X_test[0:28]) #.reshape(1,28,28)
```

<!-- #region id="5sdVxVoQbNnU" -->
Le graphique ci-dessus montre les explications pour chaque classe sur quatre prédictions. Notez que les explications sont ordonnées pour les classes 0-9 de gauche à droite le long des lignes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3sSsJE2vNO8B" executionInfo={"status": "ok", "timestamp": 1685107051659, "user_tz": -120, "elapsed": 3, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="dafd2a34-6ee1-4a55-91cb-96c041b1c811"
X_test[0:2].shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="laZkP5ATLo_A" executionInfo={"status": "ok", "timestamp": 1685107067904, "user_tz": -120, "elapsed": 618, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="f151b731-d684-4952-f790-11cf5a5cec64"
plt.imshow(X_test[0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="JCYq07dzNZfe" executionInfo={"status": "ok", "timestamp": 1685107092930, "user_tz": -120, "elapsed": 433, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8182e292-ae4d-42ae-91f8-efdcafb85f03"
plt.imshow(X_test[1])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 447} id="y9dCVsYBKx_H" executionInfo={"status": "ok", "timestamp": 1685106837707, "user_tz": -120, "elapsed": 320, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="af0a01b0-80eb-4b95-e38f-e04dbd73f63e"
import matplotlib.pyplot as plt

plt.imshow(shap_values[1][0])
```

```python colab={"base_uri": "https://localhost:8080/"} id="-DC7Xo33d3UY" executionInfo={"status": "ok", "timestamp": 1685111405566, "user_tz": -120, "elapsed": 218, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="3c2caf6b-e091-4109-9efe-b31b60be4534"
shap_values
```

```python colab={"base_uri": "https://localhost:8080/", "height": 182} id="NxiVnlcY_MRq" executionInfo={"status": "ok", "timestamp": 1685111602809, "user_tz": -120, "elapsed": 504, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a82708b6-af43-499f-b747-db3ade1fbf28"
shap.image_plot(shap_values, X_test[0:28] )
```

```python id="5rSw4JaWcagA"
µ
```
