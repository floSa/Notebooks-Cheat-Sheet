---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 🔬 EDA — Analyse multivariée
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki** sur les méthodes d'**analyse multivariée** : explorer la structure d'un dataset avec plus de 2 variables, réduire la dimensionnalité, segmenter.

Couvre :

1. **PCA** — variables quantitatives (la plus utilisée).
2. **MCA** — variables qualitatives (catégorielles).
3. **FAMD** — variables mixtes (quanti + quali).
4. **CA** — table de contingence (2 variables qualitatives).
5. **UMAP / t-SNE** — réduction non-linéaire (visualisation).
6. **Clustering multivarié** — KMeans + dendrogramme (renvoi vers ML_).
7. **Régression multivariée** — linéaire + logistique (rappel).

> Compagnion : `EDA_Stats_Analyse_Desc_Visual` (univariée + bivariée), `Detection_Outliers`.

Dataset : **Titanic** + **Iris** pour les démos numériques pures.
<!-- #endregion -->

<!-- #region -->
## 1. PCA — Analyse en composantes principales
<!-- #endregion -->

<!-- #region -->
### 1.1 Idée
<!-- #endregion -->

<!-- #region -->
Trouver les **axes orthogonaux** qui maximisent la **variance** des données. Le premier axe (PC1) explique le plus de variance, le deuxième (PC2) le plus restant orthogonalement, etc.

**Mathématiquement** : décomposition en valeurs/vecteurs propres de la matrice de covariance `Σ = X^T X / n` (X centré).

- Les **valeurs propres** = variances expliquées par chaque PC.
- Les **vecteurs propres** = directions des PCs.

**Pré-requis** : standardiser les variables (`StandardScaler`) sinon les variables à grande échelle écrasent les autres.

**Usages** :

- **Visualisation** d'un dataset > 2D en 2D/3D.
- **Réduction de dim** comme préprocessing ML (compresse l'info).
- **Décorrélation** des features.
- **Compression** d'image (truc célèbre PCA d'eigenfaces).
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
labels = np.array(iris.target_names)[iris.target]

pca = PCA(n_components=4)
proj = pca.fit_transform(X)

# Variance expliquée
explained = pca.explained_variance_ratio_
cumul = np.cumsum(explained)
print(f"Variance expliquée : {explained.round(3)}")
print(f"Cumulée            : {cumul.round(3)}")
```

<!-- #region -->
### 1.2 Scree plot + projection 2D
<!-- #endregion -->

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scree plot
axes[0].bar(range(1, 5), explained, alpha=0.7, label="Indiv")
axes[0].step(range(1, 5), cumul, where="mid", color="red", label="Cumulée")
axes[0].axhline(0.9, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Composante")
axes[0].set_ylabel("Variance expliquée")
axes[0].set_title("Scree plot")
axes[0].legend()

# Projection 2D
for sp in iris.target_names:
    mask = labels == sp
    axes[1].scatter(proj[mask, 0], proj[mask, 1], label=sp, alpha=0.7)
axes[1].set_xlabel(f"PC1 ({explained[0]:.1%})")
axes[1].set_ylabel(f"PC2 ({explained[1]:.1%})")
axes[1].set_title("Iris projeté en 2D")
axes[1].legend()

plt.tight_layout()
```

<!-- #region -->
### 1.3 Contributions des variables (cercle de corrélation)
<!-- #endregion -->

```python
# Loadings = vecteurs propres
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

fig, ax = plt.subplots(figsize=(7, 7))
for i, feat in enumerate(iris.feature_names):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, color="red")
    ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feat, color="red")

circle = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--")
ax.add_patch(circle)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xlabel(f"PC1 ({explained[0]:.1%})"); ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
ax.set_title("Cercle de corrélation")
ax.grid(alpha=0.3)
```

<!-- #region -->
## 2. MCA — Multiple Correspondence Analysis
<!-- #endregion -->

<!-- #region -->
**Idée** : équivalent de la PCA pour les variables **qualitatives**. Décompose la matrice indicatrice (one-hot) et donne une projection des modalités ET des observations dans le même espace.

**Quand l'utiliser** : datasets avec **beaucoup de variables catégorielles** (enquêtes, profils utilisateurs, données socio-démographiques).

**Lib 2026** : `prince` (anciennement `fanalysis`) — `pip install prince`. Maintenu, supporte PCA/MCA/FAMD/CA.

```python
# Pseudo-code MCA
# import prince
# mca = prince.MCA(n_components=2)
# mca = mca.fit(df[cat_cols])
# coords = mca.transform(df[cat_cols])      # projection des observations
# modalités = mca.column_coordinates(df[cat_cols])  # projection des modalités
# mca.plot()
```

**Lecture** : modalités proches dans le plan = associées (souvent ensemble dans la data). Les axes capturent les oppositions principales entre profils.
<!-- #endregion -->

<!-- #region -->
## 3. FAMD — Mixed numeric & categorical
<!-- #endregion -->

<!-- #region -->
**FAMD** (Factor Analysis of Mixed Data) combine PCA (sur les quanti centrées-réduites) + MCA (sur les indicatrices des quali). Permet d'analyser un dataset **mixte** en une seule analyse.

Disponible dans `prince.FAMD`. Très utile pour les datasets business réels (souvent mixtes).

```python
# Pseudo-code
# famd = prince.FAMD(n_components=2)
# famd = famd.fit(df_mixed)
# coords = famd.transform(df_mixed)
```
<!-- #endregion -->

<!-- #region -->
## 4. CA — Correspondence Analysis
<!-- #endregion -->

<!-- #region -->
**CA** = cas particulier à **2 variables qualitatives**. Projette les lignes et colonnes d'une table de contingence dans un plan factoriel. Très utilisé en marketing (analyse de profils consommateurs vs marques).

```python
# import prince
# ca = prince.CA(n_components=2)
# ca = ca.fit(contingency_table)
# row_coords = ca.row_coordinates(contingency_table)
# col_coords = ca.column_coordinates(contingency_table)
```
<!-- #endregion -->

<!-- #region -->
## 5. UMAP et t-SNE — Réduction non-linéaire
<!-- #endregion -->

<!-- #region -->
**Quand la PCA ne suffit pas** (structure non-linéaire — clusters courbes, varietés), utiliser **UMAP** ou **t-SNE**.

| Méthode | Forces | Faiblesses |
|---|---|---|
| **t-SNE** | Sépare très bien les clusters | Lent, non déterministe, ne préserve PAS les distances globales, pas projetable sur de nouvelles données |
| **UMAP** | Plus rapide que t-SNE, préserve mieux la structure globale, projetable sur new data | Hyperparams plus sensibles (`n_neighbors`, `min_dist`) |
| **PaCMAP** (2021) | Améliore UMAP sur la structure globale | Moins connu, moins mature |
| **TriMAP** | Triplet loss, alternative récente | Moins testé |

**Standard 2026** : **UMAP** pour 95 % des cas (rapide, projetable, bonne qualité).

```python
# Pseudo-code UMAP
# import umap
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# proj_umap = reducer.fit_transform(X)
```

**Important** : ces méthodes sont uniquement pour la **visualisation** (et parfois préprocessing). Elles **dégradent** typiquement la perf d'un classifieur en aval — utiliser la PCA pour ça.
<!-- #endregion -->

<!-- #region -->
## 6. Clustering multivarié (rappel)
<!-- #endregion -->

<!-- #region -->
Une fois projeté en 2-3D, on peut **clusteriser** :

| Méthode | Quand |
|---|---|
| **KMeans** | Clusters convexes, taille balanced, K fixé |
| **DBSCAN / HDBSCAN** | Densité variable, K inconnu, bruit |
| **Hierarchical** (dendrogramme) | Vouloir voir la hiérarchie |
| **GaussianMixture** | Mélanges probabilistes, soft assignment |
| **Spectral** | Clusters non-convexes |

Voir notebook `ML_Regression_Classification_CV_GridSearch` section clustering pour le détail.

```python
# Pseudo-code
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=3, random_state=42, n_init=10)
# clusters = km.fit_predict(proj)
```
<!-- #endregion -->

<!-- #region -->
## 7. Régression multivariée (rappel pédagogique)
<!-- #endregion -->

<!-- #region -->
Au-delà de l'analyse exploratoire, on peut faire de la **modélisation multivariée** :

- **Régression linéaire** multivariée : `y = β₀ + Σ βᵢ xᵢ + ε`. `statsmodels.OLS` pour les tests d'hypothèse, p-values, IC.
- **Régression logistique** : pour la classification binaire avec coefficients interprétables.

```python
# import statsmodels.api as sm
# X = sm.add_constant(df[['feat1', 'feat2']])
# model = sm.OLS(df['target'], X).fit()
# print(model.summary())  # coefficients, p-values, R², F-stat, AIC, BIC
```

Pour les modèles **prédictifs** modernes (RF, XGBoost, ...) → notebook `ML_Regression_Classification_CV_GridSearch`.
<!-- #endregion -->

<!-- #region -->
## 8. Workflow recommandé 2026
<!-- #endregion -->

<!-- #region -->
1. **Toujours standardiser** avant PCA/MCA/FAMD.
2. **Scree plot** + critère du coude (ou variance cumulée 80-90 %) pour choisir n_components.
3. **Cercle de corrélation** pour comprendre les loadings.
4. Si data quali pure → **MCA**. Mixte → **FAMD**. Contingence 2 vars → **CA**.
5. PCA linéaire **ne capture pas** les structures courbes → **UMAP** en complément pour la viz.
6. Toujours **valider visuellement** la projection avec les classes/labels connus si dispo.
7. **Ne pas confondre** réduction pour viz (UMAP/t-SNE) et réduction pour ML downstream (PCA).
<!-- #endregion -->

<!-- #region -->
## 9. Sources
<!-- #endregion -->

<!-- #region -->
- [sklearn — Decomposition](https://scikit-learn.org/stable/modules/decomposition.html)
- [prince — PCA/MCA/FAMD/CA en Python](https://github.com/MaxHalford/prince)
- [UMAP docs](https://umap-learn.readthedocs.io/)
- [statsmodels — Regression](https://www.statsmodels.org/stable/regression.html)
- Husson, Lê, Pagès — *Analyse de données avec R* (référence en français)
- Notebooks liés : `EDA_Stats_Analyse_Desc_Visual`, `EDA_Visualisation_Introduction`, `Detection_Outliers`, `ML_Regression_Classification_CV_GridSearch`.
<!-- #endregion -->
