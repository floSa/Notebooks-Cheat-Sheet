---
jupyter:
  kernelspec:
    display_name: Python (notebooks-refonte)
    language: python
    name: notebooks-refonte
---

<!-- #region -->
# Apprentissage par Renforcement
<!-- #endregion -->

<!-- #region -->
L'**apprentissage par renforcement** (RL) est le paradigme où un *agent* apprend à agir dans un *environnement* par essai-erreur, guidé par un signal de **récompense** — sans qu'on lui montre la bonne action, contrairement à l'apprentissage supervisé.

Ce notebook construit l'arc complet du domaine, du plus simple au plus moderne :

1. **Bandits manchots** — le RL *sans état* : on y pose le dilemme exploration/exploitation et la notion de regret (c'est le sujet du notebook d'origine, ici corrigé et étendu).
2. **Le cadre formel** — processus de décision markovien (MDP), fonctions de valeur, équations de Bellman.
3. **Méthodes tabulaires** — programmation dynamique (modèle connu), puis Q-learning et SARSA (sans modèle).
4. **Deep RL** — un DQN sur `CartPole`, quand les états deviennent continus.
5. **Écosystème et tendances 2026** — Gymnasium, Stable-Baselines3, et le RL appliqué aux LLMs (RLHF → DPO → GRPO → RLVR).

> Rôle : **wiki technique + tutoriel**. Les sections mathématiques sont assumées.
<!-- #endregion -->

<!-- #region -->
## 0. Setup et conventions
<!-- #endregion -->

<!-- #region -->
On centralise les imports, une palette de couleurs cohérente et une fonction `set_seeds` pour la reproductibilité (numpy, `random`, PyTorch). Tous les jeux de données sont **générés programmatiquement** ou fournis par Gymnasium — rien à télécharger.
<!-- #endregion -->

```python
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette cohérente pour les comparaisons de stratégies
PALETTE: list[str] = [
    "#00798c", "#d1495b", "#edae49", "#66a182",
    "#2e4057", "#9d83b8", "#b8848e", "#c9b78b",
]


def set_seeds(seed: int = 42) -> None:
    """Fixe les graines numpy / random / torch pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


set_seeds(42)
```

<!-- #region -->
## 1. Le problème du bandit manchot
<!-- #endregion -->

<!-- #region -->
Un **bandit manchot à $K$ bras** (*multi-armed bandit*) modélise un agent face à $K$ leviers. À chaque pas $t$, il en actionne un et reçoit une récompense stochastique. Il n'y a **pas d'état** : le monde ne change pas avec les actions — c'est le cas le plus simple du RL, et le point de départ idéal.

**Cas d'usage (fil rouge de l'original) :** $K = 10$ versions d'une annonce publicitaire ; la récompense vaut $1$ si le visiteur clique, $0$ sinon. On cherche la version au meilleur taux de clic (CTR).

**Le dilemme exploration / exploitation.** Pour maximiser le cumul, l'agent doit *exploiter* le bras qui semble le meilleur, mais aussi *explorer* les autres au cas où son estimation serait fausse. Toute la difficulté est là.

**La métrique : le regret.** Si $\mu^\star = \max_i \mu_i$ est le meilleur CTR et $a_t$ le bras choisi au pas $t$, le **regret cumulé** sur $T$ pas est
$$R_T = T\,\mu^\star - \sum_{t=1}^{T} \mu_{a_t}.$$
Plus il est bas, mieux c'est. Le mesurer suppose de connaître les vrais $\mu_i$ — ce que le CSV original (perdu) ne permettait pas : on le remplace donc par un **bandit Bernoulli synthétique à CTR connus**.
<!-- #endregion -->

<!-- #region -->
On modélise l'environnement par une `dataclass` : un vecteur de vraies probabilités et une méthode `pull` qui échantillonne une récompense binaire.
<!-- #endregion -->

```python
@dataclass
class BernoulliBandit:
    """Bandit à K bras de Bernoulli, à probabilités (CTR) connues.

    Connaître les vraies probas permet de mesurer le *regret*. Chaque bras
    renvoie 1 (clic) ou 0 (ignoré).
    """

    probs: np.ndarray  # vraies probabilités de récompense, shape (K,)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @property
    def n_arms(self) -> int:
        return len(self.probs)

    @property
    def optimal_arm(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def optimal_p(self) -> float:
        return float(np.max(self.probs))

    def pull(self, arm: int) -> int:
        """Tire le bras `arm` et renvoie une récompense binaire."""
        return int(self.rng.random() < self.probs[arm])
```

<!-- #region -->
On encapsule le résultat d'une stratégie (bras choisis, récompenses, regret cumulé) dans une seconde `dataclass`, construite par un helper qui calcule le regret à partir des vraies probas.
<!-- #endregion -->

```python
@dataclass
class BanditResult:
    """Résultat d'une stratégie de bandit sur N pas."""

    name: str
    selected: np.ndarray            # bras choisi à chaque pas, shape (N,)
    rewards: np.ndarray             # récompense obtenue à chaque pas, shape (N,)
    cumulative_regret: np.ndarray   # regret cumulé, shape (N,)

    @property
    def total_reward(self) -> int:
        return int(self.rewards.sum())


def _make_result(name: str, bandit: BernoulliBandit,
                 selected: list[int], rewards: list[int]) -> BanditResult:
    """Construit un BanditResult et calcule le regret cumulé."""
    sel = np.asarray(selected)
    rew = np.asarray(rewards)
    instant_regret = bandit.optimal_p - bandit.probs[sel]
    return BanditResult(name, sel, rew, np.cumsum(instant_regret))
```

<!-- #region -->
On instancie 10 annonces avec des CTR réalistes (2 % à 12 %) et un budget de 10 000 visiteurs. Le bras optimal est l'annonce n°7 (CTR 12 %).
<!-- #endregion -->

```python
set_seeds(42)
TRUE_CTR = np.array([0.03, 0.07, 0.02, 0.11, 0.05, 0.09, 0.04, 0.12, 0.06, 0.08])
bandit = BernoulliBandit(TRUE_CTR, rng=np.random.default_rng(42))
N_STEPS = 10_000

pd.DataFrame({"annonce": range(bandit.n_arms), "CTR_vrai": TRUE_CTR}).set_index("annonce").T
```

<!-- #region -->
### 1.1 Stratégie de référence : sélection aléatoire
<!-- #endregion -->

<!-- #region -->
La baseline naïve (héritée de l'original) : à chaque pas, tirer un bras **uniformément au hasard**. Elle n'apprend rien et sert de plancher de comparaison — son CTR moyen tend vers la moyenne des $\mu_i$.
<!-- #endregion -->

```python
def run_random(bandit: BernoulliBandit, n_steps: int) -> BanditResult:
    """Baseline : tire un bras uniformément au hasard à chaque pas."""
    selected, rewards = [], []
    for _ in range(n_steps):
        arm = random.randrange(bandit.n_arms)
        selected.append(arm)
        rewards.append(bandit.pull(arm))
    return _make_result("Aléatoire", bandit, selected, rewards)
```

<!-- #region -->
### 1.2 ε-greedy
<!-- #endregion -->

<!-- #region -->
La stratégie **ε-greedy** est le chaînon manquant de l'original. Elle maintient une estimation $\hat{\mu}_i$ de chaque bras (moyenne empirique mise à jour de façon incrémentale) et, à chaque pas :

- avec probabilité $1 - \varepsilon$ : **exploite** le meilleur bras estimé, $\arg\max_i \hat{\mu}_i$ ;
- avec probabilité $\varepsilon$ : **explore** un bras au hasard.

Un $\varepsilon$ fixe (ici $0{,}1$) garantit une exploration permanente, au prix d'un regret qui croît linéairement (on continue d'explorer même quand on sait).
<!-- #endregion -->

```python
def run_epsilon_greedy(bandit: BernoulliBandit, n_steps: int,
                       epsilon: float = 0.1) -> BanditResult:
    """ε-greedy : exploite le meilleur bras estimé (proba 1-ε), explore sinon."""
    k = bandit.n_arms
    counts = np.zeros(k, dtype=int)
    values = np.zeros(k)  # moyenne empirique de récompense par bras
    selected, rewards = [], []
    for _ in range(n_steps):
        if random.random() < epsilon:
            arm = random.randrange(k)
        else:
            arm = int(np.argmax(values))
        r = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]  # moyenne incrémentale
        selected.append(arm)
        rewards.append(r)
    return _make_result("ε-greedy", bandit, selected, rewards)
```

<!-- #region -->
### 1.3 Upper Confidence Bound (UCB1)
<!-- #endregion -->

<!-- #region -->
**UCB1** applique le principe d'**optimisme face à l'incertitude** : on choisit le bras dont la *borne supérieure de confiance* est la plus haute,
$$\text{UCB}_i(t) = \underbrace{\hat{\mu}_i}_{\text{exploitation}} + \underbrace{\sqrt{\frac{2 \ln t}{n_i}}}_{\text{exploration}},$$
où $\hat{\mu}_i$ est le CTR empirique et $n_i$ le nombre de fois où le bras $i$ a été tiré. Moins un bras est tiré, plus son bonus d'exploration est large. Chaque bras est tiré une fois au départ (borne infinie) — ce qui corrige proprement le hack `ucb = N*10` de l'original. UCB1 a une borne de regret **logarithmique** $O(\ln T)$, bien meilleure qu'ε-greedy.
<!-- #endregion -->

```python
def run_ucb1(bandit: BernoulliBandit, n_steps: int) -> BanditResult:
    """UCB1 : optimisme face à l'incertitude.

    Score = moyenne empirique + sqrt(2 ln t / n_i). Chaque bras est tiré une
    fois au départ (borne infinie).
    """
    k = bandit.n_arms
    counts = np.zeros(k, dtype=int)
    values = np.zeros(k)
    selected, rewards = [], []
    for t in range(n_steps):
        ucb = np.empty(k)
        for i in range(k):
            if counts[i] == 0:
                ucb[i] = math.inf  # force l'exploration initiale de chaque bras
            else:
                ucb[i] = values[i] + math.sqrt(2 * math.log(t + 1) / counts[i])
        arm = int(np.argmax(ucb))
        r = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        selected.append(arm)
        rewards.append(r)
    return _make_result("UCB1", bandit, selected, rewards)
```

<!-- #region -->
### 1.4 Thompson Sampling
<!-- #endregion -->

<!-- #region -->
**Thompson Sampling** adopte un point de vue **bayésien**. Pour des récompenses de Bernoulli, le conjugué naturel est la loi **Beta** : on maintient pour chaque bras un posterior $\text{Beta}(\alpha_i, \beta_i)$ où $\alpha_i = \text{succès}+1$ et $\beta_i = \text{échecs}+1$. À chaque pas :

1. on **échantillonne** $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$ pour chaque bras ;
2. on **joue** $\arg\max_i \theta_i$ ;
3. on **met à jour** le posterior du bras joué.

L'exploration émerge naturellement de l'incertitude du posterior : un bras peu tiré a une distribution large, donc une chance d'être échantillonné haut. En pratique, c'est souvent la stratégie la plus performante.
<!-- #endregion -->

```python
def run_thompson(bandit: BernoulliBandit, n_steps: int) -> BanditResult:
    """Thompson Sampling : approche bayésienne avec posterior Beta(α, β).

    Pour chaque bras on échantillonne θ_i ~ Beta(succès+1, échecs+1) et on
    joue argmax θ_i. L'exploration émerge de l'incertitude du posterior.
    """
    k = bandit.n_arms
    alpha = np.ones(k)  # succès + 1
    beta = np.ones(k)   # échecs + 1
    selected, rewards = [], []
    for _ in range(n_steps):
        theta = np.random.beta(alpha, beta)
        arm = int(np.argmax(theta))
        r = bandit.pull(arm)
        if r == 1:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        selected.append(arm)
        rewards.append(r)
    return _make_result("Thompson", bandit, selected, rewards)
```

<!-- #region -->
### 1.5 Comparaison des stratégies
<!-- #endregion -->

<!-- #region -->
On lance les quatre stratégies sur le **même** bandit et on trace le regret cumulé (plus bas = mieux) ainsi que la distribution des sélections de Thompson. On attend l'ordre : aléatoire $\gg$ ε-greedy / UCB1 $>$ Thompson.
<!-- #endregion -->

```python
set_seeds(42)
strategies = [
    run_random(bandit, N_STEPS),
    run_epsilon_greedy(bandit, N_STEPS, epsilon=0.1),
    run_ucb1(bandit, N_STEPS),
    run_thompson(bandit, N_STEPS),
]
for res in strategies:
    print(f"{res.name:10s} | reward total = {res.total_reward:5d} "
          f"| regret final = {res.cumulative_regret[-1]:7.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for res, color in zip(strategies, PALETTE):
    ax1.plot(res.cumulative_regret, label=res.name, color=color, lw=1.8)
ax1.set_title("Regret cumulé (plus bas = mieux)")
ax1.set_xlabel("Pas de temps")
ax1.set_ylabel("Regret cumulé")
ax1.legend()

best = strategies[-1]  # Thompson
ax2.bar(range(bandit.n_arms),
        np.bincount(best.selected, minlength=bandit.n_arms), color=PALETTE[0])
ax2.axvline(bandit.optimal_arm, color=PALETTE[1], ls="--", label="bras optimal")
ax2.set_title(f"Sélections — {best.name}")
ax2.set_xlabel("Annonce")
ax2.set_ylabel("Nombre de sélections")
ax2.legend()
plt.show()
```

<!-- #region -->
Le regret d'ε-greedy et de Thompson plafonne vite, tandis que celui de la sélection aléatoire croît linéairement : ces stratégies **convergent** vers le meilleur bras et concentrent leurs tirages dessus (pic net sur l'annonce optimale). Mais un bandit n'a qu'un seul état — pour des problèmes où les actions **changent le monde**, il faut le cadre général du RL.
<!-- #endregion -->

<!-- #region -->
## 2. Le cadre du RL : processus de décision markovien (MDP)
<!-- #endregion -->

<!-- #region -->
Le RL général se déroule en boucle : à chaque pas $t$, l'agent observe un **état** $s_t$, choisit une **action** $a_t$, reçoit une **récompense** $r_{t+1}$ et **transite** vers un nouvel état $s_{t+1}$. L'objectif n'est pas la récompense immédiate mais le **retour** cumulé à long terme. L'agent agit selon sa **politique**, l'environnement répond par un nouvel état et une récompense, et la boucle se répète.
<!-- #endregion -->

<!-- #region -->
### 2.1 Définition d'un MDP
<!-- #endregion -->

<!-- #region -->
Un **processus de décision markovien** est un quintuplet $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ :

- $\mathcal{S}$ : ensemble des états ; $\mathcal{A}$ : ensemble des actions ;
- $P(s' \mid s, a)$ : probabilité de transition ;
- $R(s, a)$ : récompense espérée ;
- $\gamma \in [0, 1[$ : facteur d'actualisation (privilégie le présent).

**Hypothèse de Markov :** le futur ne dépend du passé qu'à travers l'état présent. Une **politique** $\pi(a \mid s)$ décrit le comportement de l'agent. On cherche à maximiser le **retour actualisé**
$$G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1}.$$
<!-- #endregion -->

<!-- #region -->
### 2.2 Fonctions de valeur et équations de Bellman
<!-- #endregion -->

<!-- #region -->
On évalue une politique $\pi$ par deux **fonctions de valeur** :

- la valeur d'état $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$,
- la valeur d'action $Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$.

Elles vérifient les **équations de Bellman**, qui décomposent récursivement la valeur :
$$V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a)\,\big[R(s,a) + \gamma V^\pi(s')\big].$$

La politique optimale $\pi^\star$ vérifie l'**équation d'optimalité de Bellman** :
$$Q^\star(s,a) = R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\,\max_{a'} Q^\star(s',a'), \qquad \pi^\star(s) = \arg\max_a Q^\star(s,a).$$

> Un bandit manchot est simplement un MDP à **un seul état** : $V$ disparaît et $Q(a)$ se réduit au CTR $\mu_a$ de la partie 1.
<!-- #endregion -->

<!-- #region -->
## 3. Méthodes tabulaires
<!-- #endregion -->

<!-- #region -->
Quand $|\mathcal{S}|$ et $|\mathcal{A}|$ sont petits, on stocke $Q$ (ou $V$) dans un **tableau** indexé par état/action. On distingue deux régimes : le **modèle connu** (on connaît $P$ et $R$ → programmation dynamique) et le **modèle inconnu** (il faut apprendre par interaction → méthodes par différence temporelle).
<!-- #endregion -->

<!-- #region -->
### 3.1 Programmation dynamique : value iteration (modèle connu)
<!-- #endregion -->

<!-- #region -->
La **value iteration** applique l'équation d'optimalité de Bellman comme une mise à jour, jusqu'à convergence :
$$V_{k+1}(s) \leftarrow \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\, V_k(s') \Big].$$
On l'illustre sur un **GridWorld** déterministe $4\times4$ : un but ($+1$), un piège ($-1$), un mur, et une petite pénalité par pas qui pousse à finir vite. Le modèle $(P, R)$ est connu.
<!-- #endregion -->

```python
@dataclass
class GridWorld:
    """Petit MDP grille déterministe (modèle connu).

    Cases : but (+1) et piège (-1) terminaux, un mur bloquant. 4 actions
    (haut, droite, bas, gauche). Un déplacement hors grille ou dans le mur
    laisse sur place.
    """

    n_rows: int = 4
    n_cols: int = 4
    goal: tuple[int, int] = (0, 3)
    trap: tuple[int, int] = (1, 3)
    walls: tuple[tuple[int, int], ...] = ((1, 1),)
    step_penalty: float = -0.04

    ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # haut, droite, bas, gauche

    @property
    def n_states(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_actions(self) -> int:
        return 4

    def to_state(self, rc: tuple[int, int]) -> int:
        return rc[0] * self.n_cols + rc[1]

    def to_rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.n_cols)

    def is_terminal(self, rc: tuple[int, int]) -> bool:
        return rc == self.goal or rc == self.trap

    def step_model(self, rc: tuple[int, int], a: int) -> tuple[tuple[int, int], float]:
        """Modèle déterministe : renvoie (état suivant, récompense)."""
        if rc == self.goal or rc == self.trap:
            return rc, 0.0
        dr, dc = self.ACTIONS[a]
        nr, nc = rc[0] + dr, rc[1] + dc
        nxt = (nr, nc)
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or nxt in self.walls:
            nxt = rc  # mouvement bloqué → on reste
        if nxt == self.goal:
            return nxt, 1.0
        if nxt == self.trap:
            return nxt, -1.0
        return nxt, self.step_penalty
```

<!-- #region -->
La fonction `value_iteration` balaie tous les états jusqu'à ce que la plus grande variation passe sous un seuil `theta`, puis extrait la politique gloutonne par rapport à $V^\star$.
<!-- #endregion -->

```python
def value_iteration(env: GridWorld, gamma: float = 0.95,
                    theta: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration : balayages de Bellman jusqu'à convergence.

    Renvoie (V, policy), V de shape (n_states,) et policy de shape (n_states,).
    policy[s] = -1 pour les états terminaux / murs.
    """
    V = np.zeros(env.n_states)
    while True:
        delta = 0.0
        for s in range(env.n_states):
            rc = env.to_rc(s)
            if env.is_terminal(rc) or rc in env.walls:
                continue
            best = -math.inf
            for a in range(env.n_actions):
                nxt, r = env.step_model(rc, a)
                best = max(best, r + gamma * V[env.to_state(nxt)])
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        rc = env.to_rc(s)
        if env.is_terminal(rc) or rc in env.walls:
            policy[s] = -1
            continue
        q = []
        for a in range(env.n_actions):
            nxt, r = env.step_model(rc, a)
            q.append(r + gamma * V[env.to_state(nxt)])
        policy[s] = int(np.argmax(q))
    return V, policy
```

<!-- #region -->
On résout le GridWorld et on visualise $V^\star$ (couleur) avec la politique optimale (flèches) : depuis chaque case, la flèche indique le déplacement qui mène le plus vite au but en évitant le piège.
<!-- #endregion -->

```python
grid = GridWorld()
V_star, policy_star = value_iteration(grid, gamma=0.95)

arrows = {0: "↑", 1: "→", 2: "↓", 3: "←", -1: ""}
fig, ax = plt.subplots(figsize=(5, 5))
Vgrid = V_star.reshape(grid.n_rows, grid.n_cols)
im = ax.imshow(Vgrid, cmap="RdBu_r", vmin=-1, vmax=1)
for s in range(grid.n_states):
    r, c = grid.to_rc(s)
    rc = (r, c)
    if rc == grid.goal:
        txt = "BUT"
    elif rc == grid.trap:
        txt = "PIÈGE"
    elif rc in grid.walls:
        txt = "█"
    else:
        txt = arrows[policy_star[s]]
    ax.text(c, r, txt, ha="center", va="center", fontsize=14)
ax.set_title("V* (couleur) et politique optimale (flèches)")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046)
plt.show()
```

<!-- #region -->
### 3.2 Sans modèle : Q-learning et SARSA sur FrozenLake
<!-- #endregion -->

<!-- #region -->
Quand on ne connaît **pas** le modèle, on apprend $Q$ par interaction via une mise à jour par **différence temporelle (TD)**. Deux algorithmes de référence diffèrent par leur *cible* :

- **Q-learning** (*off-policy*) vise la meilleure action future, indépendamment de ce qui sera réellement joué :
$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\big].$$
- **SARSA** (*on-policy*) vise l'action $a'$ que la politique va effectivement prendre :
$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma\,Q(s',a') - Q(s,a)\big].$$

On les entraîne sur **`FrozenLake-v1`** de Gymnasium (grille $4\times4$, atteindre le cadeau sans tomber dans un trou). On utilise la version **déterministe** (`is_slippery=False`) pour une courbe d'apprentissage lisible, avec un $\varepsilon$ qui décroît linéairement.
<!-- #endregion -->

```python
import gymnasium as gym


def epsilon_by_episode(ep: int, n_eps: int,
                       eps_start: float = 1.0, eps_end: float = 0.05) -> float:
    """ε décroît linéairement de eps_start à eps_end sur 80 % des épisodes."""
    frac = min(1.0, ep / (0.8 * n_eps))
    return eps_start + frac * (eps_end - eps_start)


def q_learning(env_id: str, n_episodes: int = 2000, alpha: float = 0.8,
               gamma: float = 0.95, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Q-learning tabulaire (off-policy : cible = max_a' Q(s', a'))."""
    env = gym.make(env_id, is_slippery=False)
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_s, n_a))
    history = np.zeros(n_episodes)
    rng = random.Random(seed)
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        eps = epsilon_by_episode(ep, n_episodes)
        done = False
        total = 0.0
        while not done:
            a = rng.randrange(n_a) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            s, total = s2, total + r
        history[ep] = total
    env.close()
    return Q, history
```

<!-- #region -->
SARSA partage la structure mais met à jour avec l'action $a'$ réellement sélectionnée par la politique $\varepsilon$-greedy (d'où le caractère *on-policy*).
<!-- #endregion -->

```python
def sarsa(env_id: str, n_episodes: int = 2000, alpha: float = 0.8,
          gamma: float = 0.95, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """SARSA tabulaire (on-policy : cible = Q(s', a') avec a' la politique)."""
    env = gym.make(env_id, is_slippery=False)
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_s, n_a))
    history = np.zeros(n_episodes)
    rng = random.Random(seed)

    def pick(state: int, eps: float) -> int:
        return rng.randrange(n_a) if rng.random() < eps else int(np.argmax(Q[state]))

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        eps = epsilon_by_episode(ep, n_episodes)
        a = pick(s, eps)
        done = False
        total = 0.0
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            a2 = pick(s2, eps)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a, total = s2, a2, total + r
        history[ep] = total
    env.close()
    return Q, history
```

<!-- #region -->
On entraîne les deux sur 2000 épisodes et on trace le taux de succès lissé (moyenne glissante). Les deux convergent vers une politique qui atteint le but de façon quasi systématique.
<!-- #endregion -->

```python
def smooth(x: np.ndarray, w: int = 50) -> np.ndarray:
    """Moyenne glissante pour lisser une courbe d'apprentissage."""
    return np.convolve(x, np.ones(w) / w, mode="valid")


set_seeds(0)
Q_ql, hist_ql = q_learning("FrozenLake-v1", n_episodes=2000, seed=0)
Q_sa, hist_sa = sarsa("FrozenLake-v1", n_episodes=2000, seed=0)
print(f"Taux de succès (200 derniers épisodes) : "
      f"Q-learning {hist_ql[-200:].mean():.2f} | SARSA {hist_sa[-200:].mean():.2f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(smooth(hist_ql), color=PALETTE[0], label="Q-learning")
ax.plot(smooth(hist_sa), color=PALETTE[3], label="SARSA")
ax.set_title("FrozenLake (déterministe) — taux de succès lissé")
ax.set_xlabel("Épisode")
ax.set_ylabel("Récompense (moy. glissante 50)")
ax.legend()
plt.show()
```

<!-- #region -->
### 3.3 Q-learning vs SARSA
<!-- #endregion -->

<!-- #region -->
| Aspect | Q-learning | SARSA |
|---|---|---|
| Type | **off-policy** | **on-policy** |
| Cible TD | $\max_{a'} Q(s', a')$ | $Q(s', a')$ (action jouée) |
| Politique apprise | gloutonne optimale, même en explorant | tient compte de l'exploration |
| Face au risque | **agressif** (suppose un jeu parfait ensuite) | **prudent** (intègre les erreurs d'exploration) |
| Exemple typique | chemin le plus court | chemin sûr loin des pièges (*Cliff Walking*) |

Sur un environnement déterministe sans danger, les deux convergent vers la même politique optimale. Sur un environnement risqué (récompenses négatives proches du chemin optimal), SARSA tend à apprendre une trajectoire plus sûre.
<!-- #endregion -->

<!-- #region -->
## 4. Deep Reinforcement Learning
<!-- #endregion -->

<!-- #region -->
Le tabulaire explose dès que l'espace d'états est grand ou **continu** (impossible d'avoir une ligne par état). On remplace alors le tableau $Q$ par un **approximateur de fonction** $Q(s, a; \theta)$ — typiquement un réseau de neurones. Le **DQN** (*Deep Q-Network*) repose sur deux ingrédients qui stabilisent l'apprentissage :

- le **replay buffer** : on stocke les transitions $(s, a, r, s')$ et on en ré-échantillonne des lots aléatoires, ce qui décorrèle les données ;
- le **réseau cible** : une copie lente du réseau, utilisée pour calculer la cible TD, qui évite que la cible « bouge » à chaque pas.
<!-- #endregion -->

<!-- #region -->
### 4.1 DQN sur CartPole
<!-- #endregion -->

<!-- #region -->
**`CartPole-v1`** : équilibrer un pendule sur un chariot. L'état est continu (position, vitesse, angle, vitesse angulaire $\in \mathbb{R}^4$), 2 actions (pousser gauche/droite). La récompense vaut $+1$ par pas tenu, jusqu'à 500. On définit d'abord le réseau $Q$, la configuration d'hyperparamètres et le replay buffer.

> Budget volontairement borné (350 épisodes, réseau modeste, graines fixées) : l'objectif est d'**illustrer l'apprentissage**, pas de battre un record. Le critère de validation est « la récompense progresse », pas un score parfait.
<!-- #endregion -->

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """Réseau Q : état continu -> valeur Q de chaque action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    """Hyperparamètres du DQN (volontairement bornés)."""

    n_episodes: int = 350
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 10_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    target_tau: float = 0.01  # mise à jour douce (soft update) du réseau cible
    warmup: int = 500         # pas avant de commencer à apprendre


class ReplayBuffer:
    """Tampon d'expérience circulaire (s, a, r, s', done)."""

    def __init__(self, capacity: int) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self) -> int:
        return len(self.buf)
```

<!-- #region -->
La boucle d'entraînement combine tout : politique $\varepsilon$-greedy décroissante, stockage dans le buffer, mise à jour de Bellman sur un lot (perte de Huber), clipping du gradient et *soft update* du réseau cible.
<!-- #endregion -->

```python
def train_dqn(config: DQNConfig, seed: int = 0) -> np.ndarray:
    """Entraîne un DQN sur CartPole-v1. Renvoie la récompense par épisode."""
    set_seeds(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = QNet(obs_dim, n_actions)
    target = QNet(obs_dim, n_actions)
    target.load_state_dict(online.state_dict())
    optim = torch.optim.Adam(online.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size)

    rewards_hist = np.zeros(config.n_episodes)
    global_step = 0
    for ep in range(config.n_episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            eps = max(config.eps_end,
                      config.eps_start - global_step / config.eps_decay_steps
                      * (config.eps_start - config.eps_end))
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(online(torch.tensor(s, dtype=torch.float32)).argmax())
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buffer.push(s, a, r, s2, float(term))  # term seul = vrai état final
            s, total = s2, total + r
            global_step += 1

            if len(buffer) >= config.warmup:
                bs, ba, br, bs2, bd = buffer.sample(config.batch_size)
                q = online(bs).gather(1, ba)
                with torch.no_grad():
                    q_next = target(bs2).max(1, keepdim=True)[0]
                    target_q = br + config.gamma * q_next * (1 - bd)
                loss = F.smooth_l1_loss(q, target_q)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optim.step()
                with torch.no_grad():  # soft update du réseau cible
                    for tp, op in zip(target.parameters(), online.parameters()):
                        tp.mul_(1 - config.target_tau).add_(config.target_tau * op)
        rewards_hist[ep] = total
    env.close()
    return rewards_hist
```

<!-- #region -->
On entraîne et on trace la courbe d'apprentissage (récompense par épisode + moyenne glissante). L'`assert` vérifie le critère de cohérence : la récompense moyenne de fin doit dépasser celle du début (l'agent a bien appris).
<!-- #endregion -->

```python
dqn_hist = train_dqn(DQNConfig(), seed=0)
first_window = dqn_hist[:50].mean()
last_window = dqn_hist[-50:].mean()
print(f"DQN CartPole — récompense moyenne : début {first_window:.1f} "
      f"-> fin {last_window:.1f}")
assert last_window > first_window, "le DQN n'a pas appris"

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(dqn_hist, color=PALETTE[7], alpha=0.4, label="récompense / épisode")
ax.plot(np.arange(19, len(dqn_hist)), smooth(dqn_hist, 20),
        color=PALETTE[4], lw=2, label="moy. glissante 20")
ax.set_title("DQN sur CartPole-v1 — courbe d'apprentissage")
ax.set_xlabel("Épisode")
ax.set_ylabel("Récompense (durée de l'épisode)")
ax.legend()
plt.show()
```

<!-- #region -->
### 4.2 Au-delà de DQN
<!-- #endregion -->

<!-- #region -->
DQN est *value-based* et limité aux actions discrètes. Trois grandes familles complètent le paysage :

- **Policy gradient (REINFORCE)** : on paramétrise directement la politique $\pi_\theta(a\mid s)$ et on monte le gradient de la récompense espérée. Gère les actions **continues**, mais forte variance.
- **Actor-critic (A2C, PPO)** : combine une politique (*actor*) et une fonction de valeur (*critic*) qui réduit la variance. **PPO** (*Proximal Policy Optimization*) est devenu le cheval de bataille par sa stabilité.
- **SAC** (*Soft Actor-Critic*) : actor-critic *off-policy* à entropie maximale, état de l'art pour le contrôle continu (robotique).

Règle empirique : **value-based** (DQN) pour des actions discrètes et un budget d'échantillons serré ; **policy-based** (PPO/SAC) pour le contrôle continu ou les grands espaces d'action.
<!-- #endregion -->

<!-- #region -->
## 5. Écosystème et tendances 2026
<!-- #endregion -->

<!-- #region -->
### 5.1 Bibliothèques
<!-- #endregion -->

<!-- #region -->
On n'implémente plus ces algorithmes à la main en production. L'écosystème s'est standardisé autour de **Gymnasium** (le successeur maintenu d'OpenAI Gym), l'interface d'environnements qu'acceptent toutes les bibliothèques :

| Bibliothèque | Pour quoi | Particularité |
|---|---|---|
| **Gymnasium** | interface d'environnements | standard de facto ; `reset()` / `step()` |
| **Stable-Baselines3** | algos clés en main (PPO, DQN, A2C, SAC) | PyTorch, fiable, ~95 % de couverture de tests |
| **CleanRL** | apprendre / chercher | implémentations *single-file*, sans abstraction |
| **Ray RLlib** | passage à l'échelle | distribué multi-CPU/GPU, multi-agents |

Avec Stable-Baselines3, l'entraînement de la partie 4 tient en quelques lignes. On entraîne un **PPO** sur CartPole et on l'évalue.
<!-- #endregion -->

```python
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

set_seeds(0)
ppo_model = PPO("MlpPolicy", "CartPole-v1", seed=0, verbose=0)
ppo_model.learn(total_timesteps=25_000)  # entraînement court
mean_reward, std_reward = evaluate_policy(ppo_model, ppo_model.get_env(),
                                          n_eval_episodes=10)
print(f"SB3 PPO CartPole (25k steps) — récompense d'évaluation = "
      f"{mean_reward:.1f} ± {std_reward:.1f}")
```

<!-- #region -->
### 5.2 Le RL pour les LLMs
<!-- #endregion -->

<!-- #region -->
Le RL a connu une seconde jeunesse comme étage de **post-entraînement des grands modèles de langage**. La trajectoire 2022 → 2026 :

- **RLHF** (*RL from Human Feedback*) : on entraîne un *reward model* sur des préférences humaines, puis on optimise le LLM par **PPO**. C'est la recette historique (InstructGPT, ChatGPT).
- **DPO** (*Direct Preference Optimization*) : se passe du reward model et de PPO — on optimise directement une perte de classification sur les paires de préférences. Plus simple et stable, très adopté.
- **GRPO** (*Group Relative Policy Optimization*, introduit par DeepSeek pour R1) : supprime le réseau *critic*. Pour chaque prompt on génère un **groupe** de $K$ réponses, on les score, et l'avantage est la normalisation par la moyenne/écart-type du groupe. ~25 % de mémoire en moins que PPO.
- **RLVR** (*RL with Verifiable Rewards*) : la récompense n'est plus une préférence humaine mais un **vérificateur programmatique** (le code compile, la preuve math est correcte). C'est le moteur des modèles de **raisonnement** 2025-2026, et il s'étend au-delà des maths/code (domaine médical, etc.).

> Le fil conducteur : on remplace progressivement la récompense *humaine et coûteuse* par une récompense *automatique et vérifiable*, ce qui rend le RL scalable pour l'entraînement des LLMs.

**Sources :** [Gymnasium (arXiv 2407.17032)](https://arxiv.org/pdf/2407.17032) · [DPO vs PPO (arXiv 2404.10719)](https://arxiv.org/pdf/2404.10719) · [RLVR (arXiv 2506.14245)](https://arxiv.org/abs/2506.14245)
<!-- #endregion -->

<!-- #region -->
## Conclusion
<!-- #endregion -->

<!-- #region -->
On a parcouru l'arc complet du RL : du **bandit** sans état (exploration/exploitation, regret) au cadre formel des **MDP** (Bellman), puis aux méthodes **tabulaires** (value iteration, Q-learning, SARSA), au **Deep RL** (DQN, PPO) et enfin à l'écosystème **2026** et au RL pour LLMs.

Pour aller plus loin : PPO sur des environnements Atari ou MuJoCo, implémentation d'un RLVR maison avec un vérificateur custom, ou exploration des environnements multi-agents de RLlib.
<!-- #endregion -->
