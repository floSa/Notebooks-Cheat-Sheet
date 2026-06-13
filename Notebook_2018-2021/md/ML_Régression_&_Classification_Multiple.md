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

```python id="mN8S5C74Zl1X" executionInfo={"status": "ok", "timestamp": 1780317083547, "user_tz": -120, "elapsed": 2653, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
```

<!-- #region id="-mVPPLBwXlIz" -->
# Régression
<!-- #endregion -->

<!-- #region id="lHxGDSduiboH" -->
**Construction du DataFrame**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 132} id="MGhJVO2EOWip" executionInfo={"status": "ok", "timestamp": 1780317083846, "user_tz": -120, "elapsed": 296, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="bbb5269f-ad2a-45cc-bdc9-cb7703c19126"
# Fixons la graine aléatoire pour la reproductibilité
np.random.seed(42)

# Génération de données
n_samples = 10000
n_features = 10
n_targets = 3

X, y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=1.0)
# Conversion en DataFrame
columns = [f'Feature_{i}' for i in range(1, n_features + 1)] + [f'Target_{i}' for i in range(1, n_targets + 1)]
df = pd.DataFrame(np.concatenate([X, y], axis=1), columns=columns)
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 132} id="hHwW-9MR5ds1" executionInfo={"status": "ok", "timestamp": 1780317084021, "user_tz": -120, "elapsed": 174, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="5ea6f4e2-649f-4218-89e9-312ecc2aa799"
np.random.seed(42)

n_samples = 10000
X = np.random.rand(n_samples, 10)*10
y = np.column_stack([
    X[:, 0] + 2* X[:, 1] -  X[:, 2] + np.random.randn(n_samples)  ,
    np.sin(X[:, 2]) + np.cos(X[:, 3]) + np.random.randn(n_samples),
    2*(X[:, 2])**2 - 1/(X[:, 0]) + np.random.randn(n_samples)
])
# Création d'un DataFrame avec les tableaux X et Y
df = pd.DataFrame(np.column_stack([X, y]), columns= ['Feature_'+str(i) for i in range(X.shape[1])] + ['Target_'+str(j) for j in range(y.shape[1])])
df.head(2)
```

<!-- #region id="62M7cPlSihP-" -->
## Machine Learning Benchmark
<!-- #endregion -->

```python id="sxl6y1FDXuZL" executionInfo={"status": "ok", "timestamp": 1780317097173, "user_tz": -120, "elapsed": 13150, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
```

```python colab={"base_uri": "https://localhost:8080/"} id="_XTQOeOHXJQc" executionInfo={"status": "ok", "timestamp": 1780317149345, "user_tz": -120, "elapsed": 52167, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="9b3bc321-111b-4b68-9bc4-32261acbd397"

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :n_features], df.iloc[:, n_features:], test_size=0.2, random_state=42)

# Modèles compatibles avec la régression multiple

# Création du modèle
models = {
    'Elastic Net': ElasticNet(),
    'KNN': KNeighborsRegressor(5),
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'MLPRegressor': MLPRegressor(),
    'SVR': MultiOutputRegressor(SVR()),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': MultiOutputRegressor(XGBRegressor()),
    'LightGBM': MultiOutputRegressor(LGBMRegressor(learning_rate=0.05, n_estimators=50 , force_col_wise=True , verbosity=-1))
}

# Entraînement des modèles et affichage des scores
for name, model in models.items():
    model.fit(X_train, y_train)
    #score = model.score(X_test, y_test)
    score = r2_score(y_test, model.predict(X_test))
    print(f"{name}: R-squared score : {score:.4f}")
```

<!-- #region id="Dd5kJMwTrf3w" -->
## Deep Learning
<!-- #endregion -->

<!-- #region id="RcoU-VKAr7Uk" -->
### Tensorflow
<!-- #endregion -->

```python id="7iwolcy9tLP2" executionInfo={"status": "ok", "timestamp": 1780317153212, "user_tz": -120, "elapsed": 3863, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import tensorflow as tf
```

```python colab={"base_uri": "https://localhost:8080/"} id="UMY1HF05rgJ4" executionInfo={"status": "ok", "timestamp": 1780317160459, "user_tz": -120, "elapsed": 7245, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="c5edbdb0-6ebc-4241-cc19-36c2f321b2c2"
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :n_features], df.iloc[:, n_features:], test_size=0.2, random_state=42)

# Création du modèle
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Couche d'entrée
    tf.keras.layers.Dense(64, activation='relu'),  # Couche cachée
    tf.keras.layers.Dropout(0.1),  # Dropout
    tf.keras.layers.Dense(32, activation='relu'),  # Couche cachée
    tf.keras.layers.Dropout(0.1),  # Dropout
    tf.keras.layers.Dense(y_train.shape[1])
])

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2 ,verbose=0)

# Évaluation du modèle sur l'ensemble de test
loss = model.evaluate(X_test, y_test, verbose=0)
print("Loss on test data:",  np.round(loss,3) )

# Calculez le R-squared
r2 = r2_score(y_test, model.predict(X_test))
print("R-squared on test data:", np.round(r2,3))
```

<!-- #region id="lMRumaIcsCWR" -->
### Pytorch
<!-- #endregion -->

```python id="gM96oZOPsPN3" executionInfo={"status": "ok", "timestamp": 1780317163626, "user_tz": -120, "elapsed": 3161, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
```

```python colab={"base_uri": "https://localhost:8080/"} id="s3MsvondsB32" executionInfo={"status": "ok", "timestamp": 1780317174176, "user_tz": -120, "elapsed": 10533, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="8baf6a0b-f674-4ca6-ed82-bd98fdc8dcb0"
# Séparation des données en ensembles d'entraînement et de test
n_features = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :n_features], df.iloc[:, n_features:], test_size=0.2, random_state=42)

# Conversion des données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Création des DataLoaders pour l'entraînement et le test
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Définition du modèle avec des couches de dropout
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = RegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

# Évaluation du modèle sur l'ensemble de test
model.eval()
with torch.no_grad():
    y_pred = []
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        y_pred.append(outputs)
    y_pred = torch.cat(y_pred, dim=0)

# Calcul de la perte et du R-squared
test_loss = criterion(y_pred, y_test_tensor).item()
print("Loss on test data:", np.round(test_loss, 3))

r2 = r2_score(y_test_tensor.numpy(), y_pred.numpy())
print("R-squared on test data:", np.round(r2, 3))
```

<!-- #region id="5nZZbgZ8XwTa" -->
# Classification
<!-- #endregion -->

<!-- #region id="3pHzbbv9MF6F" -->
**Construction du DataFrame**
<!-- #endregion -->

<!-- #region id="iGhkHLDo1YYh" -->
Création d'un dataframe avec deux labels avec:  
label1 = 2 classes  
label2 = 4 classes
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 132} id="D-u3b3VjCrFw" executionInfo={"status": "ok", "timestamp": 1780317175127, "user_tz": -120, "elapsed": 948, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="fb6252c2-f828-49f0-9b8b-54fab06b9d29"
from sklearn.datasets import make_multilabel_classification
n_samples = 10000
# Générer un ensemble de données synthétiques pour la classification multiclasse
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2, random_state=42)

df = pd.DataFrame(np.column_stack([X, y]), columns= ['Feature_'+str(i) for i in range(X.shape[1])] + ['Target_'+str(j) for j in range(y.shape[1])])
# Créer une nouvelle colonne 'nouvelle_colonne' en attribuant un numéro unique à chaque combinaison
# Création d'une nouvelle colonne 'nouvelle_colonne' avec des numéros uniques
df['temp'] = df.groupby(['Target_1', 'Target_2']).ngroup()
# Obtenir le nombre de groupes
N = df['temp'].nunique()
# Réaffecter les valeurs de 'nouvelle_colonne' pour qu'elles soient entre 0 et N-1
df['Target_1'] = df['temp'].rank(method='dense').astype(int) - 1
df = df.drop(columns=['Target_2','temp'])
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 132} id="9hu3-_MWBAoT" executionInfo={"status": "ok", "timestamp": 1780317176078, "user_tz": -120, "elapsed": 949, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="b39811c4-468d-4aa3-966a-5dcb229a8c66"
from sklearn.datasets import make_multilabel_classification
n_samples = 10000
# Générer un ensemble de données synthétiques pour la classification multiclasse
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2, random_state=42)

df = pd.DataFrame(np.column_stack([X, y]), columns= ['Feature_'+str(i) for i in range(X.shape[1])] + ['Target_'+str(j) for j in range(y.shape[1])])
# Créer une nouvelle colonne 'nouvelle_colonne' en attribuant un numéro unique à chaque combinaison
df['Target_1'], _ = pd.factorize(df[['Target_1', 'Target_2']].apply(tuple, axis=1), sort=True)
df.drop('Target_2', axis=1, inplace=True)
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 132} id="RjsjBftrBYMh" executionInfo={"status": "ok", "timestamp": 1780317177043, "user_tz": -120, "elapsed": 963, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0d1a635f-e985-49f2-c0c1-ee3321d09f62"
from sklearn.datasets import make_multilabel_classification
n_samples = 10000
# Générer un ensemble de données synthétiques pour la classification multiclasse
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2, random_state=42)
df = pd.DataFrame(np.column_stack([X, y]), columns= ['Feature_'+str(i) for i in range(X.shape[1])] + ['Target_'+str(j) for j in range(y.shape[1])])
#y[:, 0] = np.sum(y[:, :2], axis=1)
#y = np.delete(y, 1, axis=1)
# Isoler toutes les lignes uniques de col1 et col2
unique_rows = df[['Target_1', 'Target_2']].drop_duplicates().sort_values(['Target_1', 'Target_2'])

#Création du dictionnaire
compteur = 0
mapping = {}
for index, row in unique_rows.iterrows():
    mapping[(row['Target_1'], row['Target_2'])] = compteur
    compteur += 1
# Créer la nouvelle colonne en utilisant le mapping
df['Target_1'] = df.apply(lambda row: mapping[(row['Target_1'], row['Target_2'])], axis=1)
df.drop('Target_2', axis=1, inplace=True)
df.head(2)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 241} id="f1v67MHoB_Gg" executionInfo={"status": "ok", "timestamp": 1780317177053, "user_tz": -120, "elapsed": 8, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="0b01f6f7-0418-4e76-decd-29df66adfd85"
df.Target_1.value_counts()
```

<!-- #region id="tqAlGQ4QYdi9" -->
## Machine Learning Benchmark
<!-- #endregion -->

```python id="bMQn252vYZXr" executionInfo={"status": "ok", "timestamp": 1780317177102, "user_tz": -120, "elapsed": 47, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
```

```python colab={"base_uri": "https://localhost:8080/"} id="OiYrwc79Mg8-" executionInfo={"status": "ok", "timestamp": 1780317214780, "user_tz": -120, "elapsed": 37680, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="111df179-6f5c-42f8-f6c0-663847e0bf9e"
# Supposons que df soit votre DataFrame contenant les caractéristiques et les étiquettes multilabel

# Remplacez ces valeurs par vos propres valeurs
n_features = 10
random_state = 42

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :n_features], df.iloc[:, n_features:], test_size=0.2, random_state=random_state)

# Modèles compatibles avec la classification multilabel
models = {
    'Logistic Regression': MultiOutputClassifier(LogisticRegression(random_state=random_state)),
    'KNN': MultiOutputClassifier(KNeighborsClassifier()),
    'Gradient Boosting': MultiOutputClassifier(GradientBoostingClassifier(random_state=random_state)),
    'Random Forest': MultiOutputClassifier(RandomForestClassifier(random_state=random_state)),
    'MLPClassifier': MultiOutputClassifier(MLPClassifier(random_state=random_state)),
    'SVM': MultiOutputClassifier(SVC(probability=True, random_state=random_state)),
}

# Entraînement des modèles et affichage des scores
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Calcul de l'accuracy pour chaque label
    accuracy_label1 = accuracy_score(y_test.iloc[:, 0], y_pred[:, 0])
    accuracy_label2 = accuracy_score(y_test.iloc[:, 1], y_pred[:, 1])
    print(f"{name}: Accuracy:\nLabel 1: {accuracy_label1:.3f}, - Label 2: {accuracy_label2:.3f}")
```

<!-- #region id="knOl-WD4XQok" -->
## Deep Learning
<!-- #endregion -->

<!-- #region id="h6XGvZ2avpEF" -->
### Tensorflow
<!-- #endregion -->

<!-- #region id="Z7T2iN3Dy6ns" -->
**Préparation des données**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CPC1wZ-f-oLl" executionInfo={"status": "ok", "timestamp": 1780317214780, "user_tz": -120, "elapsed": 23, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="ae9cb3b5-ac77-4189-a861-f688a7c10c6b"
from sklearn.preprocessing import OneHotEncoder

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :n_features], df.iloc[:, n_features:], test_size=0.2, random_state=random_state)

encoder = OneHotEncoder()
y_Target_0_onehot = encoder.fit_transform(y_train.values[:,0].reshape(-1, 1)).toarray()
print('y_Target_0_onehot ordre:',encoder.categories_[0])
y_Target_0_onehot_t = encoder.fit_transform(y_test.values[:,0].reshape(-1, 1)).toarray()

encoder = OneHotEncoder()
y_Target_1_onehot = encoder.fit_transform(y_train.values[:,1].reshape(-1, 1)).toarray()
print('y_Target_1_onehot ordre:', encoder.categories_[0])
y_Target_1_onehot_t = encoder.fit_transform(y_test.values[:,1].reshape(-1, 1)).toarray()
```

<!-- #region id="bWFNf6HyzAlV" -->
**Définition et Entraiement du modèle**
<!-- #endregion -->

```python id="xB7HlHsbAN4p" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1780317224398, "user_tz": -120, "elapsed": 9632, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="a4d04d35-2d74-4369-ade5-b9e02b8afdb3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

# Créez un modèle séquentiel
model = keras.Sequential()

# Ajoutez une couche d'entrée
model.add(Input(shape=(X_train.shape[1],)))

# Ajoutez une couche dense
model.add(Dense(64, activation='relu'))  # Vous pouvez ajuster le nombre de neurones selon vos besoins

# Ajoutez une couche pour la classification multiclass (Target_0)
Target_0_output = Dense(2, activation='softmax', name='Target_0_output')(model.layers[-1].output)

# Ajoutez une couche pour la classification multilabel (Target_1)
Target_1_output = Dense(4, activation='sigmoid', name='Target_1_output')(model.layers[-1].output)

# Créez un modèle qui prend l'entrée et produit les deux sorties
full_model = keras.Model(inputs=model.inputs, outputs=[Target_0_output, Target_1_output])

# Compilez le modèle avec des fonctions de perte appropriées pour chaque sortie
full_model.compile(optimizer='adam',
                  loss={'Target_0_output': 'categorical_crossentropy', 'Target_1_output': 'binary_crossentropy'},
                  metrics={'Target_0_output': 'accuracy', 'Target_1_output': 'accuracy'})

# Entraînez le modèle avec vos données
full_model.fit(X_train, [y_Target_0_onehot, y_Target_1_onehot], epochs=10, batch_size=32)

```

```python colab={"base_uri": "https://localhost:8080/"} id="CutOIFv4bkVJ" executionInfo={"status": "ok", "timestamp": 1780317224660, "user_tz": -120, "elapsed": 250, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="e5006b44-f9e6-4435-e865-536d641fd599"
test_predict = full_model.predict(X_test)
print(test_predict[0].shape , test_predict[1].shape)
```

```python colab={"base_uri": "https://localhost:8080/"} id="Q0qLQvSSYwNr" executionInfo={"status": "ok", "timestamp": 1780317224759, "user_tz": -120, "elapsed": 83, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="59fdbb03-0b00-4500-edb1-ce4b3b541eb4"
full_model.predict(np.array([6.0, 8.0, 3.0, 7.0, 2.0, 6.0, 2.0, 10.0, 8.0, 4.0]).reshape(1, -1))
```

<!-- #region id="F-8EyB2EzXqM" -->
#### Compréhension du modèles
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 54} id="DV-qTSMEjfnE" executionInfo={"status": "ok", "timestamp": 1780317225710, "user_tz": -120, "elapsed": 948, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="532b51a8-5435-46c9-9687-7158eb2f9cef"
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Specify the figure size
plt.figure(figsize=(15, 5))

# Assume that you have two labels (lab1 and lab2)
for lab in range(2):
    label_t = y_test.values[:, lab].reshape(-1, 1)
    label_t_pred = test_predict[lab]

    cf_matrix = confusion_matrix(label_t, np.argmax(label_t_pred, axis=1))

    # Calculate the subplot position based on the number of columns (here, 2 columns)
    subplot_pos = 120 + lab + 1

    # Calculate accuracy
    accuracy = accuracy_score(label_t, np.argmax(label_t_pred, axis=1))

    plt.subplot(subplot_pos)
    labs = [str(i) for i in range(lab)]
    sns.heatmap(cf_matrix / cf_matrix.sum(axis=1)[:, None], annot=True, fmt='.2%', cmap='Blues', xticklabels=labs, yticklabels=labs)
    plt.title(f'Label {lab + 1} - Accuracy: {accuracy:.3f}')  # Add accuracy to the title
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} id="QFlUzGb1jfqc" executionInfo={"status": "ok", "timestamp": 1780317225878, "user_tz": -120, "elapsed": 166, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="96cf7c7a-d300-429e-e01e-9f0bf6c639a2"
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assume you have two labels (lab1 and lab2)
plt.figure(figsize=(4, 3))

for lab in range(2):
    label_t = y_test.values[:, lab].reshape(-1, 1)
    label_t_pred = test_predict[lab]

    fpr, tpr, _ = roc_curve(label_t,  np.argmax(label_t_pred, axis=1), pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'Label {lab + 1} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Labels')
plt.legend(loc='lower right')

plt.show()
```

<!-- #region id="B9-S_439vvC2" -->
### Pytorch
<!-- #endregion -->

```python id="X4MAMUtmTKoQ" executionInfo={"status": "ok", "timestamp": 1780317225880, "user_tz": -120, "elapsed": 4, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
```

<!-- #region id="5ezweWYMyKHd" -->
**Préparation des données**
<!-- #endregion -->

```python id="ic-3-5jayAwd" executionInfo={"status": "ok", "timestamp": 1780317225894, "user_tz": -120, "elapsed": 9, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# One-hot encode the target variables
encoder = OneHotEncoder()
y_Target_0_onehot = encoder.fit_transform(y_train.values[:, 0].reshape(-1, 1)).toarray()
y_Target_0_onehot_t = encoder.fit_transform(y_test.values[:, 0].reshape(-1, 1)).toarray()

encoder = OneHotEncoder()
y_Target_1_onehot = encoder.fit_transform(y_train.values[:, 1].reshape(-1, 1)).toarray()
y_Target_1_onehot_t = encoder.fit_transform(y_test.values[:, 1].reshape(-1, 1)).toarray()
```

<!-- #region id="he6Eia3NyNeN" -->
**Définition du modèles**
<!-- #endregion -->

```python id="-3yw1XXKyE4h" executionInfo={"status": "ok", "timestamp": 1780317225914, "user_tz": -120, "elapsed": 18, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# Define the neural network model
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2_target_0 = nn.Linear(hidden_size, 2)
        self.fc2_target_1 = nn.Linear(hidden_size, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        out_target_0 = torch.softmax(self.fc2_target_0(x), dim=1)
        out_target_1 = torch.sigmoid(self.fc2_target_1(x))
        return out_target_0, out_target_1

# Initialize the model
input_size = X_train_tensor.shape[1]
hidden_size = 64
model = CustomModel(input_size, hidden_size)

# Define loss function and optimizer
criterion_target_0 = nn.CrossEntropyLoss()
criterion_target_1 = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
```

<!-- #region id="7VgThAaxyUxV" -->
**Entrainement du modèle**
<!-- #endregion -->

```python id="DycpyjwKySpH" executionInfo={"status": "ok", "timestamp": 1780317228946, "user_tz": -120, "elapsed": 3047, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
# Train the model
epochs = 10
batch_size = 32
for epoch in range(epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets_0 = torch.tensor(y_Target_0_onehot[i:i+batch_size], dtype=torch.float32)
        targets_1 = torch.tensor(y_Target_1_onehot[i:i+batch_size], dtype=torch.float32)

        optimizer.zero_grad()
        outputs_0, outputs_1 = model(inputs)
        loss_0 = criterion_target_0(outputs_0, torch.argmax(targets_0, dim=1))
        loss_1 = criterion_target_1(outputs_1, targets_1)
        loss = loss_0 + loss_1
        loss.backward()
        optimizer.step()
```

<!-- #region id="7TnEukN5ycgN" -->
**Evaluation du modèle**
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ZziZ-DOhyYCN" executionInfo={"status": "ok", "timestamp": 1780317228961, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="32d4139a-c8f6-4c8b-c5f4-94e63184ee9c"
# Evaluate the model
model.eval()
with torch.no_grad():
    test_predict = model(X_test_tensor)
    print(test_predict[0].shape, test_predict[1].shape)

# Make predictions on a new data point
new_data_point = torch.tensor([6.0, 8.0, 3.0, 7.0, 2.0, 6.0, 2.0, 10.0, 8.0, 4.0], dtype=torch.float32).reshape(1, -1)
prediction = model(new_data_point)
print(prediction)
```

```python id="53ebc1gy2xIZ" colab={"base_uri": "https://localhost:8080/", "height": 108} executionInfo={"status": "error", "timestamp": 1780317228977, "user_tz": -120, "elapsed": 15, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}} outputId="41840543-8af1-41d7-bb79-d54042990b7f"
data.head()
data.value_counts(index_train_set).reindex().sort_value(by:data.iloc[0:20, 2:-1])
```

<!-- #region id="6LslF4duzM0d" -->
#### Compréhension du modèles
<!-- #endregion -->

```python id="Osrp3fv-yq2U" executionInfo={"status": "aborted", "timestamp": 1780317229075, "user_tz": -120, "elapsed": 2, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Specify the figure size
plt.figure(figsize=(15, 5))

# Assume that you have two labels (lab1 and lab2)
for lab in range(2):
    label_t = y_test.values[:, lab].reshape(-1, 1)
    label_t_pred = test_predict[lab]

    cf_matrix = confusion_matrix(label_t, np.argmax(label_t_pred, axis=1))

    # Calculate the subplot position based on the number of columns (here, 2 columns)
    subplot_pos = 120 + lab + 1

    # Calculate accuracy
    accuracy = accuracy_score(label_t, np.argmax(label_t_pred, axis=1))

    plt.subplot(subplot_pos)
    labs = [str(i) for i in range(lab)]
    sns.heatmap(cf_matrix / cf_matrix.sum(axis=1)[:, None], annot=True, fmt='.2%', cmap='Blues', xticklabels=labs, yticklabels=labs)
    plt.title(f'Label {lab + 1} - Accuracy: {accuracy:.3f}')  # Add accuracy to the title
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.show()
```

```python id="fPD1eC38yv3t" executionInfo={"status": "aborted", "timestamp": 1780317229077, "user_tz": -120, "elapsed": 148428, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assume you have two labels (lab1 and lab2)
plt.figure(figsize=(4, 3))

for lab in range(2):
    label_t = y_test.values[:, lab].reshape(-1, 1)
    label_t_pred = test_predict[lab]

    fpr, tpr, _ = roc_curve(label_t,  np.argmax(label_t_pred, axis=1), pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'Label {lab + 1} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Labels')
plt.legend(loc='lower right')

plt.show()
```

```python id="kqsNhvXC2KQP" executionInfo={"status": "aborted", "timestamp": 1780317229078, "user_tz": -120, "elapsed": 148429, "user": {"displayName": "Florian H.", "userId": "05323848775778397032"}}
data.head()


```
