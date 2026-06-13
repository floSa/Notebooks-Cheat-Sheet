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

```python id="HLqCaSz4q2Vk"
from sklearn.metrics import accuracy_score

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
```

```python id="wF4cued0a7x2"
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

# create dataset

models = [
    #SGDRegressor(max_iter=1000, tol=1e-3 ),
    RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0),
    GradientBoostingRegressor(   n_estimators=200, learning_rate=0.1, max_depth=5,  loss='squared_error'),
    AdaBoostRegressor(n_estimators=200),
    MLPRegressor(random_state=1, max_iter=1000),
    svm.SVR()
]

X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
kf = KFold(n_splits=3)

results = dict()

List_Scores ,List_Models  = [] , []

for i, (train_index, test_index) in enumerate(tqdm(kf.split(X)) ):
    X_train, y_train = X[train_index] , y[train_index] 
    X_test , y_test = X[test_index] , y[test_index] 
    Mods  ,  Scores= [] , []
    for model in models:
        # instantiation 
        model_name = model.__class__.__name__
        # training
        current_model = model.fit(X_train,y_train)
        # mean square error 
        Scores.append(mean_squared_error(y_test,current_model.predict(X_test)) )
        Mods.append(copy.deepcopy(current_model))
    List_Models.append(Mods)
    List_Scores.append(Scores)
```

```python id="bGmsN6xXb0Cb"
df_result = pd.DataFrame(List_Scores, columns =[ name.__class__.__name__ for name in models ]).transpose()
df_result.reset_index(inplace=True)
df_result = df_result.rename( columns={'index': 'model_name'})
df_result = pd.melt(df_result, id_vars='model_name', value_vars=[0,1,2], value_name='MSE' , var_name='fold_idx') 

plt.figure(figsize=(20, 10))
sns.boxplot(x='model_name', y='MSE', data=df_result)
sns.stripplot(x='model_name', y='MSE', data=df_result, 
              size=12, jitter=True, edgecolor="gray", linewidth=2   )
plt.show()
```
