---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python (notebooks-refonte)
    language: python
    name: notebooks-refonte
---

<!-- #region -->
# KAN — Kolmogorov-Arnold Networks
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur les **Kolmogorov-Arnold Networks (KAN)**, proposés par Liu et al. (arXiv 2404.19756, mai 2024 ; ICLR 2025) comme alternative aux MLP classiques.

L'idée centrale : au lieu d'apprendre des **poids scalaires** sur les arêtes et de poser des **activations fixes** sur les neurones, un KAN apprend une **fonction d'activation** sur chaque arête (paramétrée par une B-spline) et se contente de **sommer** sur les neurones. Les promesses : **interprétabilité** (on peut lire les fonctions apprises) et **efficience paramétrique** sur les tâches structurées / scientifiques.

Fil du notebook :

1. Le **théorème de Kolmogorov-Arnold** (1957) — la fondation théorique.
2. De **MLP à KAN** — la différence d'architecture, et le piège « théorème littéral vs réseau pratique ».
3. Les **B-splines** — comment rendre les fonctions apprenables.
4. Une **couche KAN PyTorch** implémentée à la main (style *efficient-kan*).
5. **Entraînement** sur une tâche de régression symbolique.
6. **Benchmark** chiffré KAN vs MLP (params, erreur, temps).
7. **Interprétabilité** — visualiser les fonctions apprises.
8. **pykan** — extraction automatique d'une formule symbolique.
9. **Classification** 2D (`make_moons`) — frontières de décision.
10. **État 2026** — variantes, consensus, quand essayer un KAN.
<!-- #endregion -->

<!-- #region -->
## 0. Setup
<!-- #endregion -->

<!-- #region -->
Imports, graine aléatoire globale et choix du *device*. On fixe les seeds (`numpy` et `torch`) pour des résultats reproductibles.
<!-- #endregion -->

```python
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"torch {torch.__version__} | device={DEVICE}")
```

<!-- #region -->
## 1. Théorème de Kolmogorov-Arnold (1957)
<!-- #endregion -->

<!-- #region -->
**Énoncé.** Toute fonction continue multivariée $f:[0,1]^n \to \mathbb{R}$ peut s'écrire comme une **somme finie de compositions de fonctions à une seule variable** :

$$
f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q\!\left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)
$$

où les $\Phi_q$ (fonctions **externes**) et les $\phi_{q,p}$ (fonctions **internes**) sont continues et **univariées**.

**Lecture intuitive.** Un calcul « compliqué » sur $n$ variables se ramène à une **somme de fonctions 1D** composées. Toute la richesse multivariée est encapsulée dans le choix de ces fonctions 1D — il n'y a *aucune* multiplication entre variables, seulement des additions et des fonctions d'une variable.

**Note historique.** Ce résultat (Arnold, 1957) a **réfuté** la conjecture du 13ᵉ problème de Hilbert, qui supposait que certaines fonctions de 3 variables ne pouvaient pas se décomposer ainsi. Longtemps jugé **inutilisable en pratique** : les fonctions internes $\phi_{q,p}$ garanties par le théorème peuvent être **pathologiques** (non-différentiables, voire fractales), donc inapprenables par descente de gradient.
<!-- #endregion -->

<!-- #region -->
## 2. De MLP à KAN
<!-- #endregion -->

<!-- #region -->
**Le piège à éviter.** Le théorème *littéral* décrit un réseau de **profondeur 2** et de largeur exactement $2n+1$, avec des fonctions arbitraires. C'est ce que décrivait l'implémentation naïve d'origine — et ça ne mène nulle part car les $\phi$ optimales sont pathologiques.

**L'apport de Liu et al. (2024)** est de **relâcher la contrainte** : on garde l'*idée* (fonctions apprenables sur les arêtes, sommes sur les noeuds) mais on autorise des **profondeurs et largeurs arbitraires** et on impose des $\phi$ **lisses** (B-splines). On perd la garantie d'exactitude du théorème, on gagne un réseau **entraînable**.

Comparaison structurelle :

| | MLP | KAN |
|---|---|---|
| Sur chaque **arête** | un poids scalaire $w$ | une fonction apprenable $\phi(x)$ |
| Sur chaque **noeud** | une activation **fixe** $\sigma$ (ReLU, GELU…) | une simple **somme** |
| Ce qu'on apprend | les poids $w$ | les coefficients des fonctions $\phi$ |

Schéma conceptuel (un seul chemin) :

```
MLP :  x ──[× w1]──► neurone(σ) ──[× w2]──► sortie
KAN :  x ──[φ1(x)]──►    Σ      ──[φ2(x)]──► sortie
```

On échange `(poids + activation fixe)` contre `(activation apprenable)`. Pour certaines fonctions, un KAN atteint la même expressivité avec **moins de neurones** qu'un MLP.
<!-- #endregion -->

<!-- #region -->
## 3. B-splines — la brique apprenable
<!-- #endregion -->

<!-- #region -->
Pour rendre les $\phi$ **lisses et apprenables**, KAN les paramétrise comme des **B-splines** : une combinaison linéaire de fonctions de base, à support local, dont seuls les **coefficients** sont appris.

$$
\phi(x) = \underbrace{w_b\,\mathrm{silu}(x)}_{\text{résiduel}} + \underbrace{w_s \sum_{i} c_i\, B_i(x)}_{\text{spline apprenable}}
$$

- Les $B_i(x)$ sont des B-splines de degré $k$ (typiquement $k=3$, cubiques) sur une grille fixe de $G$ intervalles. Elles se calculent par la **récurrence de Cox-de Boor**.
- Les $c_i$ sont les **coefficients appris** (un par fonction de base).
- Le terme résiduel $\mathrm{silu}(x)$ stabilise et accélère l'entraînement au démarrage.

Nombre de paramètres par arête : $G + k$ (pour $G=5, k=3$ → 8 coefficients de spline).

La fonction ci-dessous implémente la récurrence de Cox-de Boor de façon vectorisée.
<!-- #endregion -->

```python
def cox_de_boor_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """Évalue les fonctions de base B-spline de degré ``k`` (récurrence de Cox-de Boor).

    Args:
        x: points d'évaluation, shape (N,).
        grid: noeuds (knots) croissants, shape (G_pts,).
        k: degré de la spline (3 = cubique).

    Returns:
        Bases B-spline, shape (N, G_pts - k - 1) : une colonne par fonction de base.
    """
    x_col = x.unsqueeze(-1)  # (N, 1)
    # Degré 0 : indicatrices d'intervalle [grid[i], grid[i+1])
    bases = ((x_col >= grid[:-1]) & (x_col < grid[1:])).to(x.dtype)  # (N, G_pts-1)
    for d in range(1, k + 1):
        left = (x_col - grid[: -(d + 1)]) / (grid[d:-1] - grid[: -(d + 1)]) * bases[:, :-1]
        right = (grid[d + 1 :] - x_col) / (grid[d + 1 :] - grid[1:-d]) * bases[:, 1:]
        bases = left + right
    return bases
```

<!-- #region -->
Visualisons : à gauche, les fonctions de base B-spline cubiques sur $[-1, 1]$ ; à droite, une fonction $\phi(x) = \sum_i c_i B_i(x)$ obtenue avec des coefficients aléatoires (dans un KAN, ces coefficients seraient **appris**).
<!-- #endregion -->

```python
k = 3
grid_size = 6
h = 2.0 / grid_size
knots = torch.arange(-k, grid_size + k + 1) * h - 1.0  # grille étendue sur [-1, 1]
xx = torch.linspace(-1, 1, 400)
B = cox_de_boor_basis(xx, knots, k)  # (400, n_bases)

torch.manual_seed(SEED)
coefs = torch.randn(B.shape[1])
phi = B @ coefs

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for i in range(B.shape[1]):
    axes[0].plot(xx, B[:, i])
axes[0].set_title(f"Bases B-spline degré {k} ({B.shape[1]} fonctions)")
axes[0].set_xlabel("x"); axes[0].grid(alpha=0.3)
axes[1].plot(xx, phi, color="#00798c", lw=2)
axes[1].set_title(r"$\phi(x)=\sum_i c_i B_i(x)$ (coefficients aléatoires)")
axes[1].set_xlabel("x"); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

<!-- #region -->
## 4. Une couche KAN en PyTorch
<!-- #endregion -->

<!-- #region -->
On implémente une couche KAN à la manière de *efficient-kan* : chaque arête $(i \to j)$ porte $\phi_{j,i}(x) = w_b\,\mathrm{silu}(x) + \sum_c \text{spline\_weight}_{j,i,c} B_c(x)$.

- `base_weight` (shape `out × in`) : le coefficient du terme résiduel SiLU.
- `spline_weight` (shape `out × in × n_coef`) : les coefficients de spline appris.
- `grid` : un buffer non-entraînable (les noeuds de la spline).

La méthode `edge_functions` (utilisée plus loin pour l'interprétabilité) évalue directement chaque fonction d'arête sur un balayage scalaire.
<!-- #endregion -->

```python
class KANLayer(nn.Module):
    """Couche Kolmogorov-Arnold : une fonction apprenable sur chaque arête.

    Args:
        in_features: dimension d'entrée.
        out_features: dimension de sortie.
        grid_size: nombre d'intervalles de la grille de splines.
        spline_order: degré des B-splines (3 = cubique).
        grid_range: bornes de la grille (doit couvrir le domaine des activations).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)

        n_coef = grid_size + spline_order  # nb de fonctions de base B-spline
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, n_coef))
        self.base_activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Init : base façon Kaiming, splines petites (proche d'un résiduel identité)."""
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        with torch.no_grad():
            self.spline_weight.normal_(0.0, 0.1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Bases B-spline par dimension. x: (batch, in) -> (batch, in, n_coef)."""
        x = x.unsqueeze(-1)  # (batch, in, 1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        for d in range(1, self.spline_order + 1):
            left = (x - grid[: -(d + 1)]) / (grid[d:-1] - grid[: -(d + 1)]) * bases[..., :-1]
            right = (grid[d + 1 :] - x) / (grid[d + 1 :] - grid[1:-d]) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) -> (batch, out_features)."""
        base = self.base_activation(x) @ self.base_weight.t()  # (batch, out)
        spline = torch.einsum("bic,oic->bo", self.b_splines(x), self.spline_weight)
        return base + spline

    def edge_functions(self, x_1d: torch.Tensor) -> torch.Tensor:
        """Évalue chaque fonction d'arête sur un balayage scalaire (pour la viz).

        x_1d: (N,) -> (out_features, in_features, N).
        """
        x = x_1d.unsqueeze(-1).expand(-1, self.in_features)  # (N, in)
        spline = torch.einsum("nic,oic->oin", self.b_splines(x), self.spline_weight)
        base = self.base_activation(x_1d).view(1, 1, -1) * self.base_weight.unsqueeze(-1)
        return base + spline
```

<!-- #region -->
Le réseau complet empile des `KANLayer` selon une liste `width = [n_in, h_1, …, n_out]`. On ajoute un compteur de paramètres.
<!-- #endregion -->

```python
class KAN(nn.Module):
    """Réseau Kolmogorov-Arnold : empilement de ``KANLayer`` selon ``width``."""

    def __init__(
        self,
        width: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            KANLayer(width[i], width[i + 1], grid_size, spline_order, grid_range)
            for i in range(len(width) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def count_params(model: nn.Module) -> int:
    """Nombre de paramètres apprenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

<!-- #region -->
Vérification rapide : on instancie un `KAN[2, 5, 1]`, on propage un batch factice et on affiche la shape de sortie + le nombre de paramètres.
<!-- #endregion -->

```python
torch.manual_seed(SEED)
kan_demo = KAN([2, 5, 1], grid_size=5, spline_order=3).to(DEVICE)
dummy = torch.randn(4, 2, device=DEVICE)
out = kan_demo(dummy)
print(f"KAN[2,5,1] -> sortie {tuple(out.shape)} | {count_params(kan_demo)} paramètres")
```

<!-- #region -->
## 5. Entraînement sur une régression symbolique
<!-- #endregion -->

<!-- #region -->
Tâche-phare du papier KAN : approcher une fonction analytique lisse

$$
f(x_1, x_2) = \exp\!\big(\sin(\pi x_1) + x_2^2\big)
$$

sur $[-1, 1]^2$. C'est un terrain idéal pour les splines (fonction lisse, peu de variables) et ça permet une comparaison nette avec un MLP.
<!-- #endregion -->

```python
def make_symbolic_dataset(
    n_train: int = 1000, n_test: int = 200, seed: int = SEED
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Génère f(x1, x2) = exp(sin(pi*x1) + x2^2) sur [-1, 1]^2."""
    g = torch.Generator().manual_seed(seed)
    n = n_train + n_test
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = torch.exp(torch.sin(torch.pi * X[:, [0]]) + X[:, [1]] ** 2)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


X_tr, y_tr, X_te, y_te = make_symbolic_dataset()
X_tr, y_tr = X_tr.to(DEVICE), y_tr.to(DEVICE)
X_te, y_te = X_te.to(DEVICE), y_te.to(DEVICE)
print(f"train={tuple(X_tr.shape)} | test={tuple(X_te.shape)}")
```

<!-- #region -->
Boucle d'entraînement générique (Adam, MSE, *full-batch*). En pratique, KAN converge souvent mieux avec un optimiseur de second ordre (L-BFGS) ; on garde Adam ici pour la simplicité et la comparabilité avec le MLP.
<!-- #endregion -->

```python
def train_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int = 400, lr: float = 1e-2
) -> tuple[list[float], float]:
    """Entraîne un modèle de régression. Returns (historique loss train, temps en s)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history, time.perf_counter() - t0
```

<!-- #region -->
On entraîne le KAN et on trace la courbe de convergence (MSE en échelle log).
<!-- #endregion -->

```python
torch.manual_seed(SEED)
kan = KAN([2, 5, 1], grid_size=5, spline_order=3).to(DEVICE)
kan_hist, kan_time = train_model(kan, X_tr, y_tr, steps=400)
with torch.no_grad():
    kan_mse = nn.functional.mse_loss(kan(X_te), y_te).item()
print(f"KAN : MSE test={kan_mse:.4e} | {kan_time:.2f}s | {count_params(kan)} params")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.semilogy(kan_hist, color="#00798c")
ax.set_title("KAN — convergence (régression symbolique)")
ax.set_xlabel("step"); ax.set_ylabel("MSE train (log)"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

<!-- #region -->
## 6. Benchmark KAN vs MLP
<!-- #endregion -->

<!-- #region -->
Pour situer le KAN, on entraîne un **MLP** (Linear + SiLU) sur **exactement la même tâche**, avec la même boucle d'entraînement, puis on compare trois métriques : nombre de paramètres, MSE de test, et temps d'entraînement.
<!-- #endregion -->

```python
class MLP(nn.Module):
    """MLP de référence (Linear + SiLU)."""

    def __init__(self, width: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

<!-- #region -->
On entraîne le MLP et on dresse le tableau comparatif + un barplot.
<!-- #endregion -->

```python
torch.manual_seed(SEED)
mlp = MLP([2, 16, 16, 1]).to(DEVICE)
mlp_hist, mlp_time = train_model(mlp, X_tr, y_tr, steps=400)
with torch.no_grad():
    mlp_mse = nn.functional.mse_loss(mlp(X_te), y_te).item()

bench = {
    "KAN[2,5,1]": (count_params(kan), kan_mse, kan_time),
    "MLP[2,16,16,1]": (count_params(mlp), mlp_mse, mlp_time),
}
print(f"{'modèle':<16}{'params':>8}{'MSE test':>14}{'temps (s)':>12}")
for name, (p, m, t) in bench.items():
    print(f"{name:<16}{p:>8}{m:>14.3e}{t:>12.2f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
names = list(bench.keys())
colors = ["#00798c", "#d1495b"]
for ax, idx, title in zip(axes, range(3), ["params", "MSE test", "temps (s)"]):
    ax.bar(names, [bench[n][idx] for n in names], color=colors)
    ax.set_title(title); ax.tick_params(axis="x", rotation=15)
    if idx == 1:
        ax.set_yscale("log")
plt.tight_layout(); plt.show()
```

<!-- #region -->
**Lecture.** Sur cette tâche lisse, le KAN atteint typiquement une **MSE plus basse avec nettement moins de paramètres** que le MLP — c'est l'argument d'efficience paramétrique des KAN. **Mais** : le KAN est **plus lent** à entraîner (calcul des splines plus coûteux qu'un produit matriciel). Ce compromis (meilleure expressivité par paramètre, mais plus lent) est le **consensus 2024-2026** : avantageux sur les petites tâches structurées / scientifiques, désavantageux quand le débit prime.
<!-- #endregion -->

<!-- #region -->
## 7. Interprétabilité — lire les fonctions apprises
<!-- #endregion -->

<!-- #region -->
L'argument phare des KAN : on peut **visualiser chaque fonction d'arête** $\phi_{j,i}$. Traçons celles de la **première couche** du KAN entraîné — une grille (sortie × entrée) où chaque case montre ce que le réseau a appris à faire de l'entrée correspondante.
<!-- #endregion -->

```python
layer0 = kan.layers[0]
x_sweep = torch.linspace(-1, 1, 200, device=DEVICE)
with torch.no_grad():
    phis = layer0.edge_functions(x_sweep).cpu()  # (out, in, 200)

fig, axes = plt.subplots(
    layer0.out_features, layer0.in_features,
    figsize=(6, 1.6 * layer0.out_features), sharex=True,
)
for j in range(layer0.out_features):
    for i in range(layer0.in_features):
        ax = axes[j, i]
        ax.plot(x_sweep.cpu(), phis[j, i], color="#2e4057")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_title(f"entrée x{i + 1}")
        if i == 0:
            ax.set_ylabel(f"vers h{j + 1}", fontsize=8)
fig.suptitle(r"Fonctions d'arête $\phi_{j,i}(x)$ apprises (couche 1)")
plt.tight_layout(); plt.show()
```

<!-- #region -->
On peut souvent reconnaître des formes interprétables (une bosse de type sinus pour $x_1$, une parabole pour $x_2$), ce qui reflète la structure $\sin(\pi x_1) + x_2^2$ de la cible. C'est cette lisibilité qui motive l'étape suivante : extraire **automatiquement** une formule symbolique.
<!-- #endregion -->

<!-- #region -->
## 8. pykan — extraction symbolique automatique
<!-- #endregion -->

<!-- #region -->
La bibliothèque officielle **`pykan`** (Liu et al.) va plus loin : elle peut **remplacer** chaque fonction apprise par la fonction analytique candidate la plus proche (`sin`, `x^2`, `exp`, …) puis recoller le tout en une **formule symbolique**. C'est le coeur de **KAN 2.0** (arXiv 2408.10205) avec MultKAN (noeuds de multiplication), le *kanpiler* et le *tree converter*.

Workflow : `create_dataset → KAN → fit → auto_symbolic → symbolic_formula`. On l'exécute sur la cible plus simple $\sin(\pi x_1) + x_2^2$ (sans l'exponentielle externe) pour faciliter la lecture. Le `try/except` rend la cellule robuste : si `pykan` n'est pas installé ou échoue, on l'indique sans casser le notebook.
<!-- #endregion -->

```python
try:
    from kan import KAN as PyKAN
    from kan.utils import create_dataset

    f_sym = lambda x: torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2
    ds = create_dataset(f_sym, n_var=2, train_num=1000, test_num=200, device="cpu")

    torch.manual_seed(SEED)
    pmodel = PyKAN(width=[2, 1, 1], grid=5, k=3, seed=SEED, device="cpu")
    pmodel.fit(ds, opt="LBFGS", steps=20)
    pmodel.auto_symbolic(lib=["sin", "x^2", "x", "exp"], verbose=0)
    formula = pmodel.symbolic_formula()[0][0]
    print("Formule récupérée :")
    print(formula)
except Exception as exc:
    print(f"pykan indisponible / a échoué ({type(exc).__name__}: {exc})")
```

<!-- #region -->
**Honnêteté sur le résultat.** Avec un budget court (20 pas de L-BFGS), la formule récupérée est **approximative** : `auto_symbolic` choisit parmi une bibliothèque de fonctions candidates et la qualité dépend fortement du fit, de la résolution de grille et du raffinement. En pratique on augmente les pas, on raffine la grille (`refine`) et on élague (`prune`) avant de symboliser. L'objectif ici est de **montrer le mécanisme**, pas d'obtenir la formule exacte du premier coup.

> Remarque : `pykan` écrit un dossier de checkpoints `./model/` dans le répertoire courant — sans incidence sur le notebook.
<!-- #endregion -->

<!-- #region -->
## 9. Classification 2D — make_moons
<!-- #endregion -->

<!-- #region -->
Pour sortir de la régression, on entraîne un petit KAN de classification binaire sur `make_moons` (deux croissants entrelacés) et on compare sa **frontière de décision** à celle d'un MLP. On élargit la grille du KAN (`grid_range=(-3, 3)`) car les coordonnées dépassent $[-1, 1]$.
<!-- #endregion -->

```python
Xm, ym = make_moons(n_samples=1000, noise=0.2, random_state=SEED)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.2, random_state=SEED)
Xm_tr_t = torch.tensor(Xm_tr, dtype=torch.float32, device=DEVICE)
ym_tr_t = torch.tensor(ym_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
Xm_te_t = torch.tensor(Xm_te, dtype=torch.float32, device=DEVICE)


def train_classifier(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int = 300, lr: float = 1e-2
) -> nn.Module:
    """Entraîne un classifieur binaire (BCEWithLogits, Adam, full-batch)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    return model


def accuracy(model: nn.Module, X: torch.Tensor, y_np: np.ndarray) -> float:
    """Accuracy d'un classifieur binaire à seuil 0.5."""
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)).cpu().numpy().ravel() > 0.5).astype(int)
    return float((pred == y_np).mean())


torch.manual_seed(SEED)
kan_clf = train_classifier(KAN([2, 5, 1], grid_range=(-3.0, 3.0)).to(DEVICE), Xm_tr_t, ym_tr_t)
torch.manual_seed(SEED)
mlp_clf = train_classifier(MLP([2, 16, 16, 1]).to(DEVICE), Xm_tr_t, ym_tr_t)

kan_acc = accuracy(kan_clf, Xm_te_t, ym_te)
mlp_acc = accuracy(mlp_clf, Xm_te_t, ym_te)
print(f"make_moons — accuracy test : KAN={kan_acc:.3f} | MLP={mlp_acc:.3f}")
```

<!-- #region -->
On trace les frontières de décision côte à côte (probabilité prédite en fond, points colorés par vraie classe).
<!-- #endregion -->

```python
xx, yy = np.meshgrid(
    np.linspace(Xm[:, 0].min() - 0.5, Xm[:, 0].max() + 0.5, 200),
    np.linspace(Xm[:, 1].min() - 0.5, Xm[:, 1].max() + 0.5, 200),
)
grid_t = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=DEVICE)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, model, name, acc in [
    (axes[0], kan_clf, "KAN", kan_acc),
    (axes[1], mlp_clf, "MLP", mlp_acc),
]:
    with torch.no_grad():
        zz = torch.sigmoid(model(grid_t)).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu_r", edgecolor="k", s=12)
    ax.set_title(f"{name} — acc test {acc:.2f}")
plt.tight_layout(); plt.show()
```

<!-- #region -->
Les deux séparent correctement les deux croissants. Sur ce type de tâche, l'intérêt du KAN n'est **pas** la performance brute (le MLP fait aussi bien, plus vite) mais l'**interprétabilité** des fonctions apprises.
<!-- #endregion -->

<!-- #region -->
## 10. État 2026 — variantes, consensus, quand essayer un KAN
<!-- #endregion -->

<!-- #region -->
**Variantes plus rapides.** Le goulot des KAN est le calcul des B-splines. Plusieurs réimplémentations l'attaquent :

- **`efficient-kan`** — réécriture optimisée de la formulation B-spline (la base de notre couche).
- **FastKAN** — remplace les B-splines par des **fonctions de base radiales gaussiennes** (RBF), ~3.3× plus rapide ; insight clé : *« les KAN sont des réseaux RBF »* (arXiv 2405.06721).
- **FasterKAN** — bases RSWAF, encore plus rapide.
- **P-KAN** (2025) — réduit l'espace de paramètres jusqu'à −83 % à précision équivalente.
<!-- #endregion -->

<!-- #region -->
**KAN 2.0 & scientific ML.** Le prolongement *KAN 2.0* (arXiv 2408.10205, publié dans *Phys. Rev. X*) oriente les KAN vers la **découverte scientifique** : identifier les variables pertinentes, révéler des structures modulaires, extraire des formules symboliques (lois de conservation, symétries, lagrangiens). C'est là que les KAN brillent vraiment — couplés à des PINN (physics-informed neural networks) et au *symbolic regression*.
<!-- #endregion -->

<!-- #region -->
**Consensus 2024-2026 (limites).**

- **Lents à entraîner** vs MLP équivalent (le compromis vu au §6).
- **Pas SOTA** sur les tâches DL standards (vision, NLP, RL) — MLP / Transformers gardent l'avantage.
- **Forts** sur le **tabulaire structuré**, les **séries temporelles** (performance comparable voire légèrement meilleure) et le **scientific ML**.
- **Hyperparamètres sensibles** (taille de grille, ordre des splines, raffinement adaptatif).
- Écosystème **moins mature** que `torch.nn`, mais en croissance rapide (cf. *A Survey on KAN*, ACM Computing Surveys 2025).
<!-- #endregion -->

<!-- #region -->
**Quand essayer un KAN ?**

| Situation | KAN ? | Alternative |
|---|---|---|
| Régression symbolique / découvrir une formule | **Oui** | PySR, SymbolicRegression.jl |
| PINN / physics-informed / scientific ML | **Oui** | MLP standard reste compétitif |
| Petit tabulaire avec besoin d'**interprétabilité** | À tester | Arbres + SHAP, GBM |
| Séries temporelles | À tester | Modèles TS dédiés |
| Vision / image | Non | CNN / ViT |
| NLP / texte | Non | Transformer |
| RL | Non | MLP |
| Tabulaire massif, perf max | Souvent non | XGBoost / LightGBM / CatBoost |

**En résumé** : essaie un KAN quand l'**interprétabilité** ou la **structure symbolique** comptent autant que la précision. Sinon, MLP / Transformer restent les choix sûrs et rapides.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Liu et al. (2024) — KAN: Kolmogorov-Arnold Networks (arXiv 2404.19756, ICLR 2025)](https://arxiv.org/abs/2404.19756)
- [Liu et al. (2024) — KAN 2.0: Kolmogorov-Arnold Networks Meet Science (arXiv 2408.10205)](https://arxiv.org/abs/2408.10205)
- [Li (2024) — Kolmogorov-Arnold Networks are Radial Basis Function Networks / FastKAN (arXiv 2405.06721)](https://arxiv.org/abs/2405.06721)
- [Somvanshi et al. (2025) — A Survey on Kolmogorov-Arnold Network (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3743128)
- [pykan — dépôt officiel](https://github.com/KindXiaoming/pykan)
- [efficient-kan — réimplémentation rapide](https://github.com/Blealtan/efficient-kan)
- [awesome-kan — recensement des papiers et implémentations](https://github.com/mintisan/awesome-kan)
- Notebooks liés : [DL_PyTorch](DL_PyTorch.ipynb) (MLP de référence), [DL_Deep_Learning_Maths](DL_Deep_Learning_Maths.ipynb) (théorie générale des réseaux).
<!-- #endregion -->
