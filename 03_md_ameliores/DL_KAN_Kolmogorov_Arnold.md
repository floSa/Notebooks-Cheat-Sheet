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
# 🌐 KAN — Kolmogorov-Arnold Networks
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur les **KAN** (Kolmogorov-Arnold Networks), proposés par Liu et al. en mai 2024 comme alternative aux MLPs classiques.

L'enjeu principal : **interprétabilité** + parfois **moins de paramètres** pour la même expressivité.

Couverture :

1. **Théorème de Kolmogorov-Arnold** (1957) — la fondation théorique.
2. **De MLP à KAN** — apprendre les fonctions d'activation au lieu des poids.
3. **B-splines** — l'implémentation pratique des fonctions apprenables.
4. **Architecture KAN** vs MLP.
5. **Avantages et limites** — état des benchmarks 2024-2026.
6. **Implémentation** (pykan, efficient-kan).
7. **Cas d'usage** — quand essayer un KAN.
<!-- #endregion -->

<!-- #region -->
## 1. Théorème de Kolmogorov-Arnold (1957)
<!-- #endregion -->

<!-- #region -->
**Énoncé** : toute fonction continue multivariée `f: [0,1]^n → ℝ` peut être représentée comme une **somme finie de compositions de fonctions à 1 variable** :

$$
f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q\!\left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)
$$

avec `Φ_q` et `φ_{q,p}` des fonctions continues d'une seule variable.

**Lecture** : un calcul "compliqué" sur N variables est équivalent à une **somme de fonctions 1D** composées. Toute la difficulté est cachée dans le choix de ces fonctions 1D.

**Note historique** : Arnold a prouvé en 1957 ce que Hilbert pensait impossible (son 13e problème). Théorème souvent considéré comme **inutilisable en pratique** parce que les `φ` peuvent être **pathologiques** (non-différentiables, fractales).

**L'idée de KAN** : utiliser ce théorème comme **squelette d'architecture neuronale**, en apprenant des fonctions `φ` **smooth** (B-splines).
<!-- #endregion -->

<!-- #region -->
## 2. De MLP à KAN
<!-- #endregion -->

<!-- #region -->
**MLP classique** :

- Sur chaque arête : **un poids `w`** (multiplication scalaire).
- Sur chaque neurone : **une activation `σ` fixe** (ReLU, GELU, ...).

**KAN** :

- Sur chaque arête : **une fonction apprenable `φ(x)`** (paramétrée par B-spline).
- Sur chaque neurone : **une simple sommation** (pas d'activation fixe).

Visualisation conceptuelle :

```
MLP :  x ── w1 ──► neuron(σ) ── w2 ──► out
KAN :  x ── φ1(x) ──► sum ── φ2(x) ──► out
```

Côté **expressivité** : on échange `(poids + activation fixe)` contre `(activation apprenable)`. Surprisingly, un KAN peut représenter beaucoup de fonctions avec **moins de paramètres** qu'un MLP équivalent.
<!-- #endregion -->

<!-- #region -->
## 3. B-splines — implémentation pratique des `φ`
<!-- #endregion -->

<!-- #region -->
Pour rendre les `φ` apprenables et différentiables, KAN les paramétrise comme des **B-splines** : combinaison linéaire de fonctions de base spline.

$$
\phi(x) = w_s \cdot \text{spline}(x) + w_b \cdot \text{silu}(x)
$$

où `spline(x) = Σᵢ cᵢ Bᵢ(x)` avec `Bᵢ` des B-splines de degré 3 sur une grille fixe, et `cᵢ` les coefficients **appris**.

Le terme `silu(x)` (résiduel) accélère le training en début.

Paramètres par arête : `G + k` (G = nombre de points de grille, k = degré spline, typique `G=5, k=3` → 8 paramètres par arête).
<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Démo : une B-spline de degré 3 sur une grille
t = np.linspace(0, 10, 12)  # knots
c = np.random.randn(8)       # coefficients (à apprendre dans un KAN)
spl = BSpline(t, c, 3)

x = np.linspace(0, 10, 200)
plt.figure(figsize=(8, 3))
plt.plot(x, spl(x), label="φ(x) — B-spline avec 8 coefs random")
plt.xlabel("x"); plt.ylabel("φ(x)"); plt.title("Fonction apprenable type KAN")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
```

<!-- #region -->
## 4. Architecture KAN vs MLP
<!-- #endregion -->

<!-- #region -->
**Forme générique d'un KAN** :

```
KAN([n_in, h_1, h_2, ..., n_out])
```

Entre chaque paire de couches `(L_i, L_{i+1})`, on a `L_i × L_{i+1}` **fonctions apprenables**.

Nombre total de paramètres :

$$
N_\text{KAN} = \sum_{l=0}^{L-1} n_l \cdot n_{l+1} \cdot (G + k)
$$

vs MLP : `Σ nl · nl+1` poids + biais.

**Exemple** : KAN[2, 5, 1] avec G=5, k=3 :
- (2 × 5 + 5 × 1) × 8 = **120 params**
- MLP équivalent [2, 5, 1] : (2×5 + 5×1) + 6 biais = **21 params**

**Mais** : KAN peut résoudre certaines tâches symboliques avec **moins de neurones** → parfois moins de params au total à expressivité égale.
<!-- #endregion -->

<!-- #region -->
## 5. Avantages et limites (état 2026)
<!-- #endregion -->

<!-- #region -->
### Avantages claimés

- **Interprétabilité** : on peut **visualiser** chaque fonction `φ` apprise et parfois la "lire" comme une formule symbolique (mode `prune + symbolize` de pykan).
- **Continual learning** : KAN serait moins sujet au **catastrophic forgetting** car les B-splines sont locales (modifier un coefficient n'affecte qu'une région).
- **Convergence sur tâches symboliques / scientific ML** : retrouve parfois la formule analytique d'une PDE.

### Limites (consensus 2024-2026)

- **Lent à entraîner** : 5-10× plus lent qu'un MLP équivalent (calcul des splines coûteux).
- **Pas SOTA sur les tâches DL standards** (images, NLP, RL) — les MLP/Transformers gagnent toujours.
- **Hyperparamètres délicats** : taille grille G, ordre k, mise à jour adaptive de la grille.
- **Adoption modeste** : peu de papers majeurs utilisent KAN en backbone (vs Mamba qui a explosé).
- **Bibliothèques moins matures** que torch.nn.

### Verdict 2026

KAN est **intéressant pour la recherche en interprétabilité et scientific ML** (découverte de lois physiques), mais **pas un remplaçant général** des MLPs en production. À considérer si :

- Tu fais du **symbolic regression** (retrouver une formule depuis des données).
- Tu travailles en **physics-informed neural networks** (PINN) ou **scientific computing**.
- L'**interprétabilité** est critique (médical, légal).

Sinon, MLP / Transformer restent les choix sûrs.
<!-- #endregion -->

<!-- #region -->
## 6. Implémentation — pykan et efficient-kan
<!-- #endregion -->

<!-- #region -->
### Bibliothèques

- **`pykan`** — la lib officielle des auteurs (Liu et al.). PyTorch-based. Features : visualisation des `φ`, pruning, symbolisation (extraire une formule symbolique).
- **`efficient-kan`** — réimplémentation optimisée (open source). Beaucoup plus rapide que pykan.
- **`Convolutional KAN`** — extension aux couches conv (KAN-CNN).
- **`KAN-Transformers`** — remplacer les MLP des Transformer par des KAN.

### Exemple pykan (pseudo-code)

```python
"""
from kan import KAN, create_dataset
import torch

# 1. Dataset (symbolic : y = sin(πx + y))
dataset = create_dataset(
    f=lambda x: torch.sin(torch.pi * x[:, [0]] + x[:, [1]]),
    n_var=2, train_num=1000, test_num=200,
)

# 2. KAN architecture
model = KAN(width=[2, 5, 1], grid=5, k=3)  # 2→5→1, G=5, spline order 3

# 3. Train
results = model.train(dataset, opt="LBFGS", steps=50)

# 4. Visualiser les φ apprises
model.plot()

# 5. Tenter de symboliser (retrouver la formule)
model.auto_symbolic()
formula = model.symbolic_formula()[0][0]
print(formula)  # Espère obtenir quelque chose comme : sin(π*x_1 + x_2)
"""
```
<!-- #endregion -->

<!-- #region -->
## 7. Quand essayer un KAN
<!-- #endregion -->

<!-- #region -->
| Situation | KAN ? | Alternative MLP |
|---|---|---|
| Symbolic regression / découvrir une formule | **Oui** | SymbolicRegression.jl, PySR |
| PINN (physics-informed) | **Oui** | MLP standard reste compétitif |
| Petit dataset tabulaire (<1000 ex) avec besoin d'interprétabilité | À tester | Decision Tree, GBM + SHAP |
| Image classification | Non | CNN / Transformer |
| NLP / texte | Non | Transformer |
| RL | Non | MLP / TBN-style |
| Tabulaire 10k+ exemples, perf max | Non (souvent) | XGBoost / LightGBM / CatBoost |

**Conseil pratique** : si tu veux essayer KAN, commence par un problème **symbolique** ou **scientifique** où l'**interprétabilité de la solution** est précieuse. Pour le reste, MLP est plus rapide à entraîner.
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- [Liu et al. (2024) — KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- [pykan — GitHub officiel](https://github.com/KindXiaoming/pykan)
- [efficient-kan — alternative perf](https://github.com/Blealtan/efficient-kan)
- [Awesome KAN — list de papers et impls](https://github.com/mintisan/awesome-kan)
- [Notebook DL_PyTorch — MLP pour comparaison](DL_PyTorch.ipynb)
- [Notebook DL_Deep_Learning_Maths — théorie générale des NN](DL_Deep_Learning_Maths.ipynb)
<!-- #endregion -->
