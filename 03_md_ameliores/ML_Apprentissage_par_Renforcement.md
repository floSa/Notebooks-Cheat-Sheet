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
# 🎮 Reinforcement Learning
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur le **Reinforcement Learning** (RL) : du multi-armed bandits aux algos modernes (DQN, PPO).

Couverture :

1. **Cadre RL** : agent, environnement, états, actions, récompenses.
2. **Multi-Armed Bandits** : ε-greedy, **UCB**, **Thompson Sampling**.
3. **MDP** & **équation de Bellman**.
4. **Q-Learning tabulaire**.
5. **Deep Q-Learning (DQN)** — bridges vers le DL.
6. **Policy gradient** : REINFORCE, **PPO** (le standard 2026).
7. **Actor-Critic** : A2C, SAC.
8. **RLHF** (Reinforcement Learning from Human Feedback) — base de l'alignement des LLMs.
9. **Stack 2026** : Gymnasium + Stable-Baselines3 + Ray RLlib.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre RL
<!-- #endregion -->

<!-- #region -->
Un **agent** interagit avec un **environnement** :

```
État sₜ ── action aₜ ──► environnement
                              │
                              ▼
                  ◄── récompense rₜ, état suivant sₜ₊₁
```

**Objectif** : apprendre une **politique** `π(a|s)` qui maximise la **récompense cumulée espérée** sur le long terme :

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

avec `γ ∈ [0, 1]` le **facteur d'actualisation** (discount).

**Spécificités vs ML supervisé** :

- Pas de "vraie réponse" à chaque exemple — on n'a que des récompenses, parfois retardées.
- **Exploration vs exploitation** : essayer de nouvelles actions vs exploiter ce qui marche.
- L'agent **génère** ses propres données (politique d'exploration).
- Non-stationnaire : la distribution des données change avec la politique.
<!-- #endregion -->

<!-- #region -->
## 2. Multi-Armed Bandits — RL minimal
<!-- #endregion -->

<!-- #region -->
**Le problème** : `K` machines à sous, chacune renvoyant une récompense aléatoire d'espérance inconnue. Trouver la meilleure en minimisant le **regret** (manque-à-gagner vs choisir toujours la meilleure depuis le début).

**Application réelle** : A/B testing dynamique, recommandation, CTR optimization, allocation de budget marketing.
<!-- #endregion -->

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

# 5 bras avec probas de récompense inconnues
true_p = np.array([0.1, 0.4, 0.3, 0.7, 0.5])
n_arms = len(true_p)
n_steps = 2000


def random_strategy() -> np.ndarray:
    rewards = []
    for _ in range(n_steps):
        a = np.random.randint(n_arms)
        r = np.random.random() < true_p[a]
        rewards.append(r)
    return np.array(rewards)


print(f"Bras optimal : {true_p.argmax()} (p={true_p.max()})")
print(f"Random strategy reward avg = {random_strategy().mean():.3f}")
```

<!-- #region -->
### 2.1 ε-greedy
<!-- #endregion -->

```python
def epsilon_greedy(epsilon: float = 0.1) -> np.ndarray:
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    rewards = []
    for _ in range(n_steps):
        if np.random.random() < epsilon:
            a = np.random.randint(n_arms)
        else:
            a = int(np.argmax(values))
        r = int(np.random.random() < true_p[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]   # moyenne incrémentale
        rewards.append(r)
    return np.array(rewards)


print(f"ε-greedy (0.1) reward avg = {epsilon_greedy(0.1).mean():.3f}")
```

<!-- #region -->
### 2.2 UCB (Upper Confidence Bound)
<!-- #endregion -->

<!-- #region -->
**Idée** : choisir le bras qui maximise une **borne supérieure de confiance** sur sa valeur estimée, qui encourage l'exploration des bras peu essayés.

$$
\text{UCB}(a) = \hat{\mu}_a + c \sqrt{\frac{\ln t}{N_a}}
$$

avec `c` un hyperparamètre (souvent √2).
<!-- #endregion -->

```python
def ucb(c: float = np.sqrt(2)) -> np.ndarray:
    counts = np.zeros(n_arms)
    values = np.zeros(n_arms)
    rewards = []
    for t in range(1, n_steps + 1):
        if counts.min() == 0:
            a = int(np.argmin(counts))   # essai initial chaque bras
        else:
            ucb_vals = values + c * np.sqrt(np.log(t) / counts)
            a = int(np.argmax(ucb_vals))
        r = int(np.random.random() < true_p[a])
        counts[a] += 1
        values[a] += (r - values[a]) / counts[a]
        rewards.append(r)
    return np.array(rewards)


print(f"UCB reward avg = {ucb().mean():.3f}")
```

<!-- #region -->
### 2.3 Thompson Sampling (Bayésien)
<!-- #endregion -->

<!-- #region -->
**Idée** : maintenir une **distribution postérieure** Beta sur la proba de succès de chaque bras (initialement uniforme). À chaque pas, on **échantillonne** un point dans chaque postérieure et on prend le max. Très performant et théoriquement optimal en régime asymptotique.
<!-- #endregion -->

```python
def thompson() -> np.ndarray:
    alpha = np.ones(n_arms)
    beta = np.ones(n_arms)
    rewards = []
    for _ in range(n_steps):
        samples = np.random.beta(alpha, beta)
        a = int(np.argmax(samples))
        r = int(np.random.random() < true_p[a])
        if r == 1: alpha[a] += 1
        else:      beta[a] += 1
        rewards.append(r)
    return np.array(rewards)


print(f"Thompson reward avg = {thompson().mean():.3f}")
```

<!-- #region -->
## 3. MDP et équation de Bellman
<!-- #endregion -->

<!-- #region -->
Au-delà des bandits, le formalisme du **Markov Decision Process** (MDP) ajoute la notion d'**état évolutif** :

`(S, A, P, R, γ)` :

- `S` : espace d'états.
- `A` : espace d'actions.
- `P(s'|s,a)` : dynamique de transition.
- `R(s,a)` : récompense.
- `γ` : discount.

**Fonction de valeur** `V^π(s)` = récompense cumulée espérée en partant de `s` et suivant `π`. **Q-function** `Q^π(s,a)` = idem en imposant la 1ère action.

**Équation de Bellman** (récursive) :

$$
Q^*(s, a) = \mathbb{E}\!\left[ r + \gamma \max_{a'} Q^*(s', a') \mid s, a \right]
$$

C'est l'équation à résoudre pour trouver la **politique optimale**.
<!-- #endregion -->

<!-- #region -->
## 4. Q-Learning tabulaire
<!-- #endregion -->

<!-- #region -->
Pour des espaces **finis et petits**, on stocke `Q` comme une **table** `(|S|, |A|)`. À chaque transition, mise à jour temporal-difference :

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]
$$

Exploration via ε-greedy. Converge vers `Q*` sous condition de visite infinie de chaque `(s,a)`.

```python
# Pseudo-code Q-learning sur env Gymnasium
"""
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha, gamma, eps = 0.1, 0.99, 0.1
for ep in range(5000):
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample() if np.random.random() < eps else int(Q[s].argmax())
        s2, r, done, _, _ = env.step(a)
        Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
        s = s2
"""
```
<!-- #endregion -->

<!-- #region -->
## 5. Deep Q-Learning (DQN)
<!-- #endregion -->

<!-- #region -->
Quand `|S|` est trop grand (pixels d'écran d'Atari, configurations possibles d'un jeu), on **approxime** `Q(s,a;θ)` par un réseau de neurones.

**Tricks DQN (Mnih 2015, Atari)** :

- **Experience replay** : buffer des `(s, a, r, s')` passées, on échantillonne au hasard pour briser la corrélation temporelle.
- **Target network** : un second réseau, mis à jour périodiquement, sert à calculer `max Q(s', a')` (stabilise l'optim).
- **Frame stacking** : 4 frames consécutifs en input pour capturer la dynamique.

**Variantes** : Double DQN, Dueling DQN, Prioritized Experience Replay, Rainbow (combinaison de tout).
<!-- #endregion -->

<!-- #region -->
## 6. Policy gradient — REINFORCE & PPO
<!-- #endregion -->

<!-- #region -->
Au lieu d'apprendre `Q` puis dériver `π`, on apprend **directement** `π(a|s; θ)` (réseau de neurones). Gradient :

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\pi}\!\left[ \nabla_\theta \log \pi(a|s) \cdot R \right]
$$

(REINFORCE, Williams 1992).

**Problèmes** : haute variance, instabilité (un mauvais update peut détruire la politique).

**PPO** (Proximal Policy Optimization, Schulman 2017) résout ça avec une fonction objectif **clippée** qui empêche les updates trop drastiques. **Le standard 2026** pour la plupart des cas RL en deep, notamment pour le **RLHF des LLMs** (alignement par feedback humain).
<!-- #endregion -->

<!-- #region -->
## 7. Actor-Critic
<!-- #endregion -->

<!-- #region -->
Combine policy gradient (acteur) avec une **fonction de valeur** apprise (critique) qui réduit la variance du gradient.

- **A2C / A3C** : Advantage Actor-Critic (synchronisé / asynchronisé).
- **PPO** : version stabilisée de A2C.
- **SAC** (Soft Actor-Critic) : algo off-policy pour les actions continues (robotique, simulation).
- **TD3** : alternative récente pour les actions continues.
<!-- #endregion -->

<!-- #region -->
## 8. RLHF — l'alignement des LLMs
<!-- #endregion -->

<!-- #region -->
**Reinforcement Learning from Human Feedback** : pipeline d'alignement utilisé pour ChatGPT, Claude, Llama-Instruct :

1. **SFT** (Supervised Fine-Tuning) : fine-tune le LLM de base sur des prompts/réponses humaines.
2. **Reward Model** : entraîne un modèle qui prédit la **préférence humaine** entre deux réponses.
3. **PPO** : optimise le LLM pour maximiser le score donné par le Reward Model, sous contrainte de **KL-divergence** par rapport au SFT (évite que le modèle dérive trop loin).

**Variantes 2024-2026** :

- **DPO** (Direct Preference Optimization, 2023) : élimine le Reward Model, optimise directement via une fonction objectif élégante. Plus stable, plus simple, dominant en 2025-2026.
- **ORPO**, **IPO**, **KTO** : extensions de DPO.
- **RLAIF** : remplace les feedbacks humains par des feedbacks LLM.
<!-- #endregion -->

<!-- #region -->
## 9. Stack 2026
<!-- #endregion -->

<!-- #region -->
- **`gymnasium`** (fork moderne de OpenAI Gym) — environnements standardisés (Atari, MuJoCo, Box2D, custom).
- **`stable-baselines3`** — implémentations propres et benchmarkées de PPO, SAC, DQN, A2C, TD3.
- **`ray rllib`** — RL distribué à grande échelle, prod-ready.
- **`pettingzoo`** — multi-agent RL.
- **`trl`** (HuggingFace) — RLHF et DPO sur LLMs avec transformers.
- **`cleanrl`** — implémentations minimalistes single-file (pédagogique).
- **`open_spiel`** (DeepMind) — jeux multi-agents.

```python
# Pseudo-code PPO avec stable-baselines3
"""
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=20_000)

# Test
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done: break
"""
```
<!-- #endregion -->

<!-- #region -->
## 10. Quand utiliser le RL en 2026
<!-- #endregion -->

<!-- #region -->
- **A/B testing dynamique / personnalisation** → bandits (Thompson Sampling).
- **Optimisation de policy** (logistique, supply chain, energy management) → PPO / SAC sur un simulateur.
- **Robotique / jeux / simulations** → DRL (DQN, PPO, SAC).
- **Alignement de LLMs** → DPO (préférable à RLHF/PPO en 2025-2026).
- **Recommendation systems sequencielle** → contextual bandits ou off-policy RL.

**Quand NE PAS l'utiliser** : si le problème peut être formulé en supervisé, le supervisé est presque toujours **plus simple, plus rapide, plus stable**.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Sutton & Barto — Reinforcement Learning: An Introduction (livre de référence)](http://incompleteideas.net/book/the-book-2nd.html)
- [Gymnasium](https://gymnasium.farama.org/) · [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Schulman et al. (2017) — PPO](https://arxiv.org/abs/1707.06347)
- [Rafailov et al. (2023) — DPO](https://arxiv.org/abs/2305.18290)
- [Hugging Face TRL](https://huggingface.co/docs/trl)
- [CleanRL — minimal RL implementations](https://github.com/vwxyzjn/cleanrl)
- [Spinning Up — OpenAI tutoriel](https://spinningup.openai.com/)
<!-- #endregion -->
