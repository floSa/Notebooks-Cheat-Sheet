---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python (notebooks-refonte)
    name: notebooks-refonte
---

<!-- #region -->
# Détection d'outliers
<!-- #endregion -->

<!-- #region -->
Panorama pratique et théorique de la **détection de valeurs aberrantes** : des
statistiques univariées robustes aux modèles modernes (Isolation Forest, PyOD), en
passant par le deep learning et les séries temporelles.

Ce notebook joue trois rôles : **cheat-sheet** (snippets typés réutilisables),
**tutoriel** (quand/pourquoi chaque méthode) et **wiki** (les maths sous-jacentes).

**Jeux de données** (tous chargés programmatiquement, rien à committer) :

- un **nuage gaussien 2D synthétique** avec outliers injectés et **labels de vérité
  terrain** → permet de *mesurer* objectivement la qualité de détection ;
- **Titanic** (`seaborn`) pour les cas réels (stats univariées, Isolation Forest
  multivarié, distance de Cook) ;
- **Air Passengers** pour l'anomalie en série temporelle.
<!-- #endregion -->

<!-- #region -->
## 0. Setup
<!-- #endregion -->

<!-- #region -->
Imports centralisés, palette de couleurs du projet et graine aléatoire globale pour
la reproductibilité. La palette suit la convention : une seule couleur (`PRIMARY_1`)
pour une variable, le rouge (`MAUVAIS`) pour mettre en évidence les outliers.
<!-- #endregion -->

```python
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Palette projet
PRIMARY_1 = "#00798c"
MAUVAIS = "#d1495b"
MOYEN = "#edae49"
ACCENT = "#66a182"
ACCENT_DARK = "#2e4057"
LAVENDER = "#9d83b8"
DUSTY_ROSE = "#b8848e"
BEIGE = "#c9b78b"
PALETTE = [PRIMARY_1, MAUVAIS, MOYEN, ACCENT, ACCENT_DARK, LAVENDER, DUSTY_ROSE, BEIGE]

RANDOM_STATE = 1
RNG = np.random.default_rng(RANDOM_STATE)
sns.set_style("darkgrid")
```

<!-- #region -->
## 1. Cadrer le problème
<!-- #endregion -->

<!-- #region -->
Avant de choisir un algorithme, il faut nommer ce que l'on cherche. Trois notions
souvent confondues :

- **Outlier detection** : le jeu de données *contient déjà* des aberrations ; on les
  repère a posteriori (non supervisé, données « contaminées »).
- **Novelty detection** : on apprend la normalité sur des données *propres*, puis on
  juge si de **nouveaux** points sont anormaux (semi-supervisé).
- **Anomaly detection** : terme générique, fréquent en monitoring / séries temporelles.

Deux axes structurent le choix de méthode :

- **Global vs local** : un point est-il aberrant par rapport à *tout* le nuage
  (global : Z-score, Elliptic Envelope) ou seulement à son *voisinage* (local : LOF) ?
- **Univarié vs multivarié** : raisonne-t-on variable par variable (IQR sur `fare`) ou
  sur la structure jointe des features (Mahalanobis, Isolation Forest) ?

Presque toutes les méthodes prennent un paramètre **`contamination`** : la proportion
attendue d'outliers, qui fixe le seuil de décision.
<!-- #endregion -->

<!-- #region -->
| Méthode | Type | Hypothèse | Quand l'utiliser |
|---|---|---|---|
| Z-score | univarié, global | normalité | distribution ~gaussienne, 1 variable |
| IQR / Tukey | univarié, global | unimodale | robuste, exploratoire |
| MAD | univarié, global | aucune (médiane) | présence de fortes aberrations |
| Elliptic Envelope | multivarié, global | gaussien | nuage elliptique, peu de dimensions |
| LOF | multivarié, local | densité variable | clusters de densités différentes |
| Isolation Forest | multivarié, global | aucune | **défaut go-to**, haute dimension |
| One-Class SVM | multivarié | aucune | frontières non linéaires, novelty |
| Autoencoder | multivarié | variété latente | haute dimension, beaucoup de données |
| Matrix profile | temporel | sous-séquences | discords dans une série |
<!-- #endregion -->

<!-- #region -->
## 2. Jeux de données
<!-- #endregion -->

<!-- #region -->
### 2.1 Synthétique 2D avec vérité terrain
<!-- #endregion -->

<!-- #region -->
On génère un nuage gaussien (inliers) auquel on **injecte** des points tirés
uniformément loin du centre (outliers). Disposer des labels `y_true` est précieux :
cela permettra, en section 6, de comparer les détecteurs avec une vraie métrique
(ROC-AUC) plutôt qu'à l'œil.
<!-- #endregion -->

```python
from sklearn.datasets import make_blobs


def make_synthetic_outliers(
    n_inliers: int = 300,
    n_outliers: int = 15,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère un nuage gaussien 2D + outliers injectés uniformément.

    Returns
    -------
    X : (n_inliers + n_outliers, 2) features.
    y_true : (n,) labels — 1 = outlier (vérité terrain), 0 = inlier.
    """
    rng = np.random.default_rng(random_state)
    X_in, _ = make_blobs(
        n_samples=n_inliers, n_features=2, centers=1, cluster_std=1.0,
        shuffle=True, random_state=random_state,
    )
    lo, hi = X_in.min(axis=0) - 6, X_in.max(axis=0) + 6
    X_out = rng.uniform(low=lo, high=hi, size=(n_outliers, 2))
    X = np.vstack([X_in, X_out])
    y_true = np.concatenate([np.zeros(n_inliers, int), np.ones(n_outliers, int)])
    order = rng.permutation(len(X))
    return X[order], y_true[order]


X, y_true = make_synthetic_outliers()
print("Synthétique :", X.shape, "| outliers injectés :", int(y_true.sum()))
```

<!-- #region -->
### 2.2 Titanic (réel, mutualisé)
<!-- #endregion -->

<!-- #region -->
On garde les colonnes numériques et on retire les lignes incomplètes (`age` contient
des `NaN`). Le tarif `fare` présente des valeurs extrêmes réelles (cabines de luxe) :
parfait pour les méthodes univariées.
<!-- #endregion -->

```python
TITANIC_NUM = ["age", "fare", "sibsp", "parch", "pclass"]
titanic = sns.load_dataset("titanic")
titanic_num = titanic[TITANIC_NUM].dropna().reset_index(drop=True)
print("Titanic numérique :", titanic_num.shape)
```

<!-- #region -->
## 3. Statistiques univariées
<!-- #endregion -->

<!-- #region -->
On travaille **une variable à la fois** sur `fare`. Ces trois règles sont les briques
de base de tout pipeline de nettoyage : simples, rapides, interprétables.
<!-- #endregion -->

```python
fare = titanic_num["fare"].to_numpy(dtype=float)
```

<!-- #region -->
### 3.1 Z-score
<!-- #endregion -->

<!-- #region -->
Le **Z-score** standardise : $z_i = \dfrac{x_i - \mu}{\sigma}$. Sous hypothèse de
normalité, $\approx 99.7\%$ des points vérifient $|z| \le 3$ — d'où la règle empirique
« $|z| > 3$ = outlier ». **Limite** : $\mu$ et $\sigma$ sont eux-mêmes *tirés vers le
haut* par les aberrations (effet de masquage).
<!-- #endregion -->

```python
def zscore_outliers(x: np.ndarray, thresh: float = 3.0) -> np.ndarray:
    """Masque booléen des outliers via |z| > thresh (suppose normalité)."""
    z = (x - x.mean()) / x.std(ddof=0)
    return np.abs(z) > thresh
```

<!-- #region -->
### 3.2 IQR — bornes de Tukey
<!-- #endregion -->

<!-- #region -->
La détection par **IQR** (InterQuartile Range) est robuste car basée sur les quartiles.
On calcule $\text{IQR} = Q_3 - Q_1$, puis les bornes de Tukey :

$$\text{borne basse} = Q_1 - 1.5\,\text{IQR}, \qquad \text{borne haute} = Q_3 + 1.5\,\text{IQR}.$$

Tout point hors de ces bornes est aberrant. C'est exactement ce que dessinent les
« moustaches » d'un boxplot.
<!-- #endregion -->

<!-- #region -->
![Schéma IQR : Q1, Q2, Q3, IQR et bornes 1.5·IQR délimitant les outliers](images/outliers_orig_L113.png)
<!-- #endregion -->

```python
def iqr_bounds(x: np.ndarray, k: float = 1.5) -> tuple[float, float]:
    """Bornes de Tukey : (Q1 - k.IQR, Q3 + k.IQR)."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outliers(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Masque booléen des outliers hors bornes de Tukey (robuste)."""
    lo, hi = iqr_bounds(x, k)
    return (x < lo) | (x > hi)
```

<!-- #region -->
### 3.3 MAD — Median Absolute Deviation
<!-- #endregion -->

<!-- #region -->
La **MAD** est l'estimateur de dispersion le plus robuste : $\text{MAD} = \text{med}(|x_i - \tilde{x}|)$
où $\tilde{x}$ est la médiane. Le **z-score modifié** s'écrit :

$$z_i^{\text{mod}} = \dfrac{0.6745\,(x_i - \tilde{x})}{\text{MAD}}.$$

La constante $0.6745$ (et son inverse $1.4826$) rend la MAD **cohérente avec $\sigma$**
pour une loi normale. Seuil usuel : $|z^{\text{mod}}| > 3.5$. Médiane et MAD ayant un
point de rupture de 50 %, cette méthode résiste à de très nombreuses aberrations.
<!-- #endregion -->

```python
def mad_outliers(x: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """Masque via z-score modifié basé sur la MAD (ultra robuste).

    z_i = 0.6745 (x_i - med) / MAD ; MAD = median(|x - med|).
    Constante 1.4826 = 1/0.6745 rend la MAD cohérente avec sigma pour une gaussienne.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x, dtype=bool)
    z_mod = 0.6745 * (x - med) / mad
    return np.abs(z_mod) > thresh
```

<!-- #region -->
### 3.4 Comparaison des trois règles
<!-- #endregion -->

<!-- #region -->
Sur `fare`, les trois méthodes ne flaguent pas le même nombre de points : le Z-score
est le plus conservateur (il est lui-même biaisé par les extrêmes), MAD le plus
sensible. Le boxplot matérialise la borne de Tukey haute.
<!-- #endregion -->

```python
m_z = zscore_outliers(fare)
m_iqr = iqr_outliers(fare)
m_mad = mad_outliers(fare)

compare_univar = pd.DataFrame(
    {"methode": ["Z-score", "IQR", "MAD"],
     "n_outliers": [m_z.sum(), m_iqr.sum(), m_mad.sum()]}
)
print(compare_univar.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 2.5))
sns.boxplot(x=fare, ax=ax, color=PRIMARY_1, fliersize=4)
lo, hi = iqr_bounds(fare)
ax.axvline(hi, color=MAUVAIS, ls="--", lw=1, label=f"borne Tukey haute = {hi:.0f}")
ax.set_title("Distribution de fare (Titanic) — boxplot + borne IQR")
ax.legend()
plt.show()
```

<!-- #region -->
### 3.5 Supprimer les outliers et visualiser l'effet
<!-- #endregion -->

<!-- #region -->
Une fois les aberrations repérées, on peut **nettoyer** la variable en retirant les
points hors bornes de Tukey, puis comparer la distribution **avant / après** par
boxplot. C'est le réflexe de base en pré-traitement. *(Note : on supprime ici via un
**masque booléen** ; supprimer par index brut — comme le faisait la version d'origine
de ce notebook — produit des suppressions incohérentes dès que l'index n'est plus
0..n−1.)*
<!-- #endregion -->

```python
mask_fare = iqr_outliers(fare)
fare_clean = fare[~mask_fare]
print(f"fare : {mask_fare.sum()} outliers retirés ({len(fare)} -> {len(fare_clean)})")

fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=True)
sns.boxplot(x=fare, ax=axes[0], color=PRIMARY_1, fliersize=4)
axes[0].set_title("fare AVEC outliers")
sns.boxplot(x=fare_clean, ax=axes[1], color=ACCENT, fliersize=4)
axes[1].set_title("fare SANS outliers (bornes IQR)")
plt.show()
```

<!-- #region -->
## 4. Détection multivariée
<!-- #endregion -->

<!-- #region -->
On passe à la structure *jointe* des variables, sur le nuage synthétique 2D (la 2D
permet de visualiser, et on a la vérité terrain). On définit deux helpers réutilisés
par toutes les méthodes : `fit_flag` (entraîne un détecteur et renvoie un masque
d'outliers) et `plot_detection` (scatter inliers vs outliers).
<!-- #endregion -->

```python
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

CONTAM = 0.05


def fit_flag(detector, X: np.ndarray) -> np.ndarray:
    """Fit un détecteur sklearn (fit_predict -1/1) -> masque booléen (True=outlier)."""
    pred = detector.fit_predict(X)
    return pred == -1


def plot_detection(X: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Scatter inliers (primary) vs outliers détectés (rouge)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[~mask, 0], X[~mask, 1], s=18, color=PRIMARY_1, label="inliers")
    ax.scatter(X[mask, 0], X[mask, 1], s=45, color=MAUVAIS, edgecolor="k",
               linewidth=0.5, label="outliers")
    ax.set_title(title)
    ax.legend()
    plt.show()
```

<!-- #region -->
### 4.1 Elliptic Envelope
<!-- #endregion -->

<!-- #region -->
On suppose les données **gaussiennes** : on ajuste une ellipse de covariance robuste
(estimateur **MCD**, Minimum Covariance Determinant) et on classe outlier tout point
dont la **distance de Mahalanobis** au centre dépasse un seuil :

$$D_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}.$$

La covariance robuste $\Sigma$ évite que les outliers ne « gonflent » l'ellipse.
**Limite** : suppose un nuage unimodal elliptique.
<!-- #endregion -->

<!-- #region -->
![Intuition de l'Elliptic Envelope : une ellipse englobe le cœur du nuage, les points hors ellipse sont aberrants](images/outliers_orig_L44.png)
<!-- #endregion -->

```python
m_elliptic = fit_flag(
    EllipticEnvelope(contamination=CONTAM, random_state=RANDOM_STATE), X
)
plot_detection(X, m_elliptic, "Elliptic Envelope")
```

<!-- #region -->
### 4.2 Local Outlier Factor (LOF)
<!-- #endregion -->

<!-- #region -->
LOF compare la densité locale d'un point à celle de ses voisins. On définit la
*reachability distance*, la **densité locale d'accessibilité** (LRD) puis le score :

$$\text{LOF}_k(p) = \frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}.$$

$\text{LOF} \approx 1$ : densité comparable aux voisins (inlier). $\text{LOF} \gg 1$ :
beaucoup moins dense que son voisinage (outlier **local**). C'est sa force : il détecte
des aberrations que des méthodes globales ratent dans des clusters de densités
différentes.
<!-- #endregion -->

```python
m_lof = fit_flag(LocalOutlierFactor(n_neighbors=20, contamination=CONTAM), X)
plot_detection(X, m_lof, "Local Outlier Factor")
```

<!-- #region -->
**Outlier vs novelty.** Avec `novelty=False`, LOF étiquette le jeu fourni. Avec
`novelty=True`, on apprend la frontière sur `X` puis on juge de *nouveaux* points : ici
on évalue une grille pour visualiser la région considérée comme « normale ».
<!-- #endregion -->

```python
lof_nov = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=CONTAM)
lof_nov.fit(X)
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200),
    np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = lof_nov.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, Z, levels=[-1.5, 0, 1.5], colors=[MAUVAIS, PRIMARY_1], alpha=0.15)
ax.scatter(X[:, 0], X[:, 1], s=15, color=ACCENT_DARK)
ax.set_title("LOF novelty — région normale (bleu) vs nouveauté (rouge)")
plt.show()
```

<!-- #region -->
### 4.3 Isolation Forest — le go-to 2026
<!-- #endregion -->

<!-- #region -->
Idée : un outlier est **facile à isoler**. On construit des arbres en coupant
aléatoirement l'espace ; un point aberrant se retrouve seul après peu de coupes (chemin
court). Le score d'anomalie agrège la profondeur moyenne $E[h(x)]$ :

$$s(x, n) = 2^{-\,E[h(x)] / c(n)},$$

où $c(n)$ normalise la profondeur moyenne d'une recherche infructueuse dans un BST.
$s \to 1$ : anomalie ; $s \to 0.5$ : normal. **Sans hypothèse de distribution**, robuste
en haute dimension, quasi linéaire — c'est le **choix par défaut** moderne.
<!-- #endregion -->

```python
m_iforest = fit_flag(
    IsolationForest(n_estimators=200, contamination=CONTAM, random_state=RANDOM_STATE), X
)
plot_detection(X, m_iforest, "Isolation Forest")
```

<!-- #region -->
**Extended Isolation Forest (EIF).** L'IForest standard coupe selon des axes
horizontaux/verticaux, ce qui crée des artefacts (« fantômes » de score) sur des nuages
diagonaux. L'EIF tire des **hyperplans à pente aléatoire**, supprimant ce biais. Dispo
via le package `eif` ou `pyod.models.iforest` (variantes).
<!-- #endregion -->

<!-- #region -->
### 4.4 One-Class SVM
<!-- #endregion -->

<!-- #region -->
Le One-Class SVM apprend une **frontière non linéaire** (noyau RBF) englobant les
données normales. Le paramètre $\nu \in (0, 1]$ borne à la fois la fraction d'outliers
et la fraction de vecteurs supports. **Piège fréquent** (présent dans la version
d'origine de ce notebook) : entraîner sur `X` puis ne tester qu'**un seul** point
choisi à la main ne montre rien — on flague ici directement les outliers *de X*.
Sensible au scaling et au choix de $\nu$.
<!-- #endregion -->

```python
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAM)
m_ocsvm = ocsvm.fit_predict(X) == -1
plot_detection(X, m_ocsvm, "One-Class SVM (nu=0.05)")
```

<!-- #region -->
## 5. Autoencoder (PyTorch)
<!-- #endregion -->

<!-- #region -->
Un **autoencodeur** compresse puis reconstruit l'entrée. Entraîné sur des données
majoritairement normales, il apprend la *variété* des points usuels : un outlier, hors
de cette variété, se **reconstruit mal**. Le score d'anomalie est l'erreur de
reconstruction $\lVert x - \text{AE}(x) \rVert^2$. On standardise les données et on
seuille le score au quantile correspondant à `contamination`.
<!-- #endregion -->

```python
import torch
from torch import nn

torch.manual_seed(RANDOM_STATE)


class AutoEncoder(nn.Module):
    """Autoencodeur dense minimal : 2 -> 4 -> 1 -> 4 -> 2.

    Le score d'anomalie est l'erreur de reconstruction ||x - AE(x)||^2 :
    un point éloigné de la variété apprise se reconstruit mal.
    """

    def __init__(self, d_in: int, d_latent: int = 1) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 4), nn.ReLU(), nn.Linear(4, d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, 4), nn.ReLU(), nn.Linear(4, d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def train_autoencoder(X: np.ndarray, epochs: int = 300, lr: float = 1e-2) -> AutoEncoder:
    """Entraîne un AE non supervisé (MSE) sur des données standardisées."""
    Xt = torch.tensor(X, dtype=torch.float32)
    model = AutoEncoder(d_in=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), Xt)
        loss.backward()
        opt.step()
    return model


def reconstruction_scores(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    """Erreur de reconstruction par point (score d'anomalie, + haut = + anormal)."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        err = ((model(Xt) - Xt) ** 2).mean(dim=1)
    return err.numpy()
```

<!-- #region -->
On entraîne l'autoencodeur, on calcule les scores et on seuille pour obtenir le masque
d'outliers. Sur 2D l'AE est volontairement minimal : son intérêt réel apparaît en haute
dimension (images, capteurs), mais le principe se visualise bien ici.
<!-- #endregion -->

```python
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
ae = train_autoencoder(X_std)
ae_scores = reconstruction_scores(ae, X_std)
ae_thresh = np.quantile(ae_scores, 1 - CONTAM)
m_ae = ae_scores > ae_thresh
plot_detection(X, m_ae, "Autoencoder — erreur de reconstruction")
print(f"AE — outliers détectés : {m_ae.sum()} (seuil quantile {1 - CONTAM:.0%})")
```

<!-- #region -->
## 6. Comparer objectivement les détecteurs
<!-- #endregion -->

<!-- #region -->
Grâce aux labels `y_true`, on compare les méthodes avec de vraies métriques :

- **ROC-AUC** : capacité à *classer* les outliers avant les inliers (seuil-indépendant) ;
- **precision@k** : parmi les $k$ points les plus suspects ($k$ = nombre réel
  d'outliers), combien en sont vraiment.

On normalise tous les scores pour que « plus haut = plus anormal » (via
`-score_samples` pour les estimateurs sklearn).
<!-- #endregion -->

```python
from sklearn.metrics import roc_auc_score


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    """Fraction de vrais outliers parmi les k scores les plus élevés."""
    if k is None:
        k = int(y_true.sum())
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].sum()) / k


def anomaly_scores_sklearn(detector, X: np.ndarray) -> np.ndarray:
    """Score d'anomalie (+ haut = + anormal) via -score_samples."""
    return -detector.fit(X).score_samples(X)


rows = []
detectors = {
    "Elliptic Envelope": EllipticEnvelope(contamination=CONTAM, random_state=RANDOM_STATE),
    "Isolation Forest": IsolationForest(n_estimators=200, contamination=CONTAM,
                                        random_state=RANDOM_STATE),
    "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAM),
}
for name, det in detectors.items():
    t0 = time.perf_counter()
    scores = anomaly_scores_sklearn(det, X)
    dt = time.perf_counter() - t0
    rows.append((name, roc_auc_score(y_true, scores), precision_at_k(y_true, scores), dt))

# LOF : score via negative_outlier_factor_ après fit_predict
t0 = time.perf_counter()
lof_cmp = LocalOutlierFactor(n_neighbors=20, contamination=CONTAM)
lof_cmp.fit_predict(X)
lof_scores = -lof_cmp.negative_outlier_factor_
rows.append(("LOF", roc_auc_score(y_true, lof_scores),
             precision_at_k(y_true, lof_scores), time.perf_counter() - t0))

# Autoencoder
rows.append(("Autoencoder", roc_auc_score(y_true, ae_scores),
             precision_at_k(y_true, ae_scores), float("nan")))

comparatif = (
    pd.DataFrame(rows, columns=["detecteur", "roc_auc", "precision@k", "temps_s"])
    .sort_values("roc_auc", ascending=False)
    .reset_index(drop=True)
)
print(comparatif.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(comparatif["detecteur"], comparatif["roc_auc"], color=PRIMARY_1)
ax.set_xlim(0, 1)
ax.set_xlabel("ROC-AUC")
ax.set_title("Comparatif des détecteurs (ROC-AUC, plus haut = mieux)")
ax.invert_yaxis()
plt.show()
```

<!-- #region -->
## 7. PyOD : 60+ détecteurs unifiés
<!-- #endregion -->

<!-- #region -->
[PyOD](https://pyod.readthedocs.io/) est la librairie de référence pour la détection
d'anomalies : plus de **60 détecteurs** sous une API unique (`.fit`,
`.decision_scores_`, `.labels_`), de la proximité aux modèles profonds, plus une
orchestration *benchmark-backed* (ADEngine) et un workflow agentique (2026).

Deux détecteurs modernes **sans hyperparamètre** s'y distinguent :

- **ECOD** (2022) — estime la distribution par fonction de répartition empirique (ECDF)
  dimension par dimension, puis agrège les probabilités de queue. Surclasse 11 baselines
  sur 30 jeux de données.
- **COPOD** (2020) — modélise la dépendance via une **copule** empirique ; interprétable,
  efficace, ROC-AUC moyen ~82 % sur benchmark.

Être *param-free* est un atout majeur : en non supervisé, on n'a généralement pas de
labels pour régler les hyperparamètres.
<!-- #endregion -->

```python
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD

ecod = ECOD(contamination=CONTAM)
ecod.fit(X)
ecod_auc = roc_auc_score(y_true, ecod.decision_scores_)

copod = COPOD(contamination=CONTAM)
copod.fit(X)
copod_auc = roc_auc_score(y_true, copod.decision_scores_)

print(f"PyOD param-free — ECOD ROC-AUC={ecod_auc:.3f} | COPOD ROC-AUC={copod_auc:.3f}")

m_ecod = ecod.labels_.astype(bool)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[~m_ecod, 0], X[~m_ecod, 1], s=18, color=PRIMARY_1, label="inliers")
ax.scatter(X[m_ecod, 0], X[m_ecod, 1], s=45, color=MAUVAIS, edgecolor="k",
           linewidth=0.5, label="outliers")
ax.set_title(f"PyOD ECOD (param-free) — AUC={ecod_auc:.3f}")
ax.legend()
plt.show()
```

<!-- #region -->
## 8. Cas réel : Isolation Forest multivarié + PCA
<!-- #endregion -->

<!-- #region -->
Sur des données réelles à plus de 2 dimensions, on ne peut plus tracer directement le
nuage : on applique une **PCA** pour projeter en 2D *après* la détection (la détection,
elle, se fait dans l'espace complet standardisé). Ici Isolation Forest sur les features
numériques du Titanic ; on s'attend à voir flagués les passagers aux tarifs extrêmes et
aux configurations familiales atypiques.
<!-- #endregion -->

```python
from sklearn.decomposition import PCA

X_titanic = StandardScaler().fit_transform(titanic_num.to_numpy(dtype=float))
iforest_t = IsolationForest(n_estimators=200, contamination=CONTAM, random_state=RANDOM_STATE)
m_titanic = iforest_t.fit_predict(X_titanic) == -1

X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_titanic)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X_pca[~m_titanic, 0], X_pca[~m_titanic, 1], s=15, color=PRIMARY_1, label="inliers")
ax.scatter(X_pca[m_titanic, 0], X_pca[m_titanic, 1], s=45, color=MAUVAIS,
           edgecolor="k", linewidth=0.5, label="outliers")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Isolation Forest sur Titanic (projection PCA 2D)")
ax.legend()
plt.show()

print(f"Passagers flagués : {m_titanic.sum()} / {len(titanic_num)}")
print(titanic_num[m_titanic].sort_values("fare", ascending=False).head().to_string())
```

<!-- #region -->
## 9. Points influents en régression
<!-- #endregion -->

<!-- #region -->
En contexte de régression, un outlier n'est pas seulement « loin » : il peut être
**influent**, c'est-à-dire modifier fortement le modèle. Le **graphe des résidus**
(résidus en fonction d'une variable) est un premier diagnostic : une structure ou des
points très éloignés signalent des observations suspectes.
<!-- #endregion -->

```python
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

reg_df = titanic_num[["fare", "age"]].copy()

fig, ax = plt.subplots(figsize=(7, 4))
sns.residplot(x="age", y="fare", data=reg_df, scatter_kws=dict(s=25, color=PRIMARY_1), ax=ax)
ax.set_title("Graphe des résidus — OLS fare ~ age (Titanic)")
plt.show()
```

<!-- #region -->
### 9.1 Distance de Cook
<!-- #endregion -->

<!-- #region -->
La **distance de Cook** (Dennis Cook, 1977) mesure l'effet du retrait d'une observation
sur l'ensemble des prédictions :

$$D_i = \frac{\sum_{j=1}^{n} (\hat{y}_j - \hat{y}_{j(i)})^2}{p \cdot \text{MSE}},$$

où $\hat{y}_{j(i)}$ est la prédiction sans l'observation $i$ et $p$ le nombre de
paramètres. C'est le produit du **levier** (position extrême en $x$) et de
l'**aberration** (résidu élevé). Règle empirique : on signale $D_i > 4/n$.
<!-- #endregion -->

<!-- #region -->
![Formule de la distance de Cook](images/outliers_orig_L409.png)
<!-- #endregion -->

```python
model = smf.ols("fare ~ age", data=reg_df).fit()
cook_distance, _ = OLSInfluence(model).cooks_distance
n = len(reg_df)
threshold = 4 / n

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(reg_df["age"], reg_df["fare"], c=cook_distance, cmap="Reds",
           s=40, edgecolor="k", linewidth=0.3)
ax.set_xlabel("age")
ax.set_ylabel("fare")
ax.set_title("Distance de Cook (couleur = influence)")
plt.show()
```

<!-- #region -->
Un *stem plot* visualise les distances par observation et le seuil $4/n$ ; on annote
les points au-dessus.
<!-- #endregion -->

```python
influential = np.where(cook_distance > threshold)[0]

fig, ax = plt.subplots(figsize=(8, 4))
ax.stem(cook_distance, basefmt=" ")
ax.axhline(threshold, color=MAUVAIS, ls="--", label=f"seuil 4/n = {threshold:.4f}")
ax.set_xlabel("index observation")
ax.set_ylabel("distance de Cook")
ax.set_title(f"Points influents : {len(influential)} au-dessus du seuil")
ax.legend()
plt.show()

print(f"Points influents (> {threshold:.4f}) : {len(influential)}")
```

<!-- #region -->
## 10. Séries temporelles : matrix profile
<!-- #endregion -->

<!-- #region -->
Sur une série temporelle, un point isolé n'est pas le bon objet : on cherche des
**sous-séquences** inhabituelles (*discords*). Le **matrix profile** (STUMPY) calcule,
pour chaque fenêtre de longueur `m`, la distance à sa plus proche voisine ailleurs dans
la série. Un **pic** du profil = une sous-séquence sans équivalent = anomalie. On
applique cela à **Air Passengers** (fenêtre = 1 saison annuelle, `m=12`).
<!-- #endregion -->

```python
import statsmodels.api as sm
import stumpy

ap = sm.datasets.get_rdataset("AirPassengers", "datasets").data
series = ap.select_dtypes("number").iloc[:, -1].to_numpy(dtype=float)

m = 12  # fenêtre = 1 saison annuelle
mp = stumpy.stump(series, m)
profile = mp[:, 0].astype(float)
discord_idx = int(np.argmax(profile))

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(series, color=PRIMARY_1)
axes[0].axvspan(discord_idx, discord_idx + m, color=MAUVAIS, alpha=0.3)
axes[0].set_title("Air Passengers — discord (sous-séquence la plus inhabituelle)")
axes[1].plot(profile, color=ACCENT_DARK)
axes[1].axvline(discord_idx, color=MAUVAIS, ls="--")
axes[1].set_title("Matrix profile (pic = anomalie)")
plt.show()

print(f"Discord à l'index {discord_idx} (fenêtre m={m})")
```

<!-- #region -->
Pour aller plus loin en temporel : **River** (`HalfSpaceTrees`) pour la détection *en
streaming* / en ligne, **Merlion** (Salesforce) pour un pipeline complet, ou la
détection par **résidu de prévision** (un point très mal prédit par un modèle
ARIMA/Prophet est anormal).
<!-- #endregion -->

<!-- #region -->
## 11. Quelle méthode choisir ?
<!-- #endregion -->

<!-- #region -->
Arbre de décision rapide :

- **1 seule variable, exploratoire** → IQR ou MAD (MAD si beaucoup d'aberrations).
- **Données ~gaussiennes, peu de dimensions** → Elliptic Envelope.
- **Clusters de densités différentes** → LOF (notion *locale*).
- **Cas général, tabulaire, haute dimension** → **Isolation Forest** (défaut) ; pour du
  *param-free*, **ECOD/COPOD** via PyOD.
- **Frontière non linéaire / novelty sur données propres** → One-Class SVM.
- **Très haute dimension, beaucoup de données** (images, capteurs) → Autoencoder.
- **Série temporelle** → matrix profile (discords) ou détection par résidu de prévision.
- **Régression : observations qui faussent le modèle** → distance de Cook.

En production, on combine souvent : une règle univariée rapide pour le nettoyage
grossier, puis un modèle multivarié (Isolation Forest / PyOD) pour le reste.
<!-- #endregion -->

<!-- #region -->
## 12. Évaluer et fixer le seuil
<!-- #endregion -->

<!-- #region -->
Le paramètre `contamination` *fixe le seuil* : il transforme un score continu en
décision binaire. **Avec labels**, on optimise via ROC-AUC / precision@k (section 6).
**Sans labels** (cas usuel), on s'appuie sur : l'inspection visuelle du *tri* des
scores, un seuil sur un quantile du score, et une **analyse de sensibilité** —
faire varier `contamination` et observer la stabilité. Ci-dessous, le nombre de points
flagués par Isolation Forest croît mécaniquement avec `contamination` ; la vérité
terrain (15 outliers) tombe autour de `contamination ≈ 0.05`.
<!-- #endregion -->

```python
contams = [0.01, 0.02, 0.05, 0.1, 0.15]
n_flagged = []
for c in contams:
    det = IsolationForest(n_estimators=200, contamination=c, random_state=RANDOM_STATE)
    n_flagged.append(int((det.fit_predict(X) == -1).sum()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(contams, n_flagged, "o-", color=PRIMARY_1)
ax.axhline(int(y_true.sum()), color=ACCENT_DARK, ls="--", label="vrais outliers")
ax.set_xlabel("contamination")
ax.set_ylabel("nb points flagués")
ax.set_title("Sensibilité d'Isolation Forest au paramètre contamination")
ax.legend()
plt.show()

print("contamination sweep :", dict(zip(contams, n_flagged)))
```

<!-- #region -->
## 13. Sources
<!-- #endregion -->

<!-- #region -->
- scikit-learn — [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- PyOD — [documentation](https://pyod.readthedocs.io/) · [dépôt](https://github.com/yzhao062/pyod)
- ECOD — [arXiv 2201.00382](https://arxiv.org/abs/2201.00382)
- COPOD — [arXiv 2009.09463](https://arxiv.org/abs/2009.09463)
- STUMPY (matrix profile) — [documentation](https://stumpy.readthedocs.io/)
- Distance de Cook — Cook, R. D. (1977), *Detection of Influential Observations in Linear Regression*.
- Z-score modifié / MAD — Iglewicz & Hoaglin (1993), *How to Detect and Handle Outliers*.
<!-- #endregion -->
