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
# 🔗 Causal Inference
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Scientist · Wiki + Tutoriel

**Dataset(s)** : Lalonde (NSW job training), STAR (éducation)

Inférence causale : passer de la corrélation à la causalité, base nécessaire pour tout impact mesurement, A/B testing, économétrie ou recommandation rigoureuse.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Causalité vs Corrélation
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Paradoxe de Simpson (avec exemple chiffré)
- Paradoxe de Berkson (sélection sur collider)
- Confondants observés vs non observés
- Pourquoi un fit ML supervisé n'est PAS de la causalité

**Code visé** : exemple Simpson sur dataset jouet + graphe.

<!-- #endregion -->

<!-- #region -->
## 2. Causal DAGs et Do-Calculus (Pearl)
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Notation DAG : `X → Y`, `Z → X, Z → Y` (confounder)
- Backdoor criterion (quels Z conditionner)
- Frontdoor criterion (alternative quand backdoor impossible)
- Intervention `do(X=x)` vs observation `P(Y|X=x)`

**Code visé** : `dowhy` ou `causal-learn` — construire un DAG, identifier l'effet.
<!-- #endregion -->

<!-- #region -->
## 3. A/B testing rigoureux
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Hypothèse nulle, test bilatéral/unilatéral
- Puissance statistique, MDE (Minimum Detectable Effect)
- Sample size calculator (binary, continuous)
- Test séquentiel vs fixed horizon
- Peeking problem

**Code visé** : `statsmodels` ou `scipy.stats` pour le test + calculator.
<!-- #endregion -->

<!-- #region -->
## 4. DiD — Differences-in-Differences
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Quasi-expérience : `effect = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)`
- Hypothèse de tendances parallèles
- Two-way fixed effects regression

**Code visé** : régression OLS avec interactions, `linearmodels`.
<!-- #endregion -->

<!-- #region -->
## 5. Propensity Score Matching et IPW
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Estimer `e(X) = P(T=1|X)` (LogReg ou GBM)
- Matching : 1-NN, k-NN, caliper
- Inverse Probability Weighting (IPW)
- Doubly Robust estimators

**Code visé** : `causalml` ou `econml`.
<!-- #endregion -->

<!-- #region -->
## 6. Synthetic Control
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Construire une 'unité contrôle synthétique' par pondération
- Cas d'usage : effet d'une loi sur un pays (Abadie)
- Hypothèses

**Code visé** : `pysyncon` ou `causalpy`.
<!-- #endregion -->

<!-- #region -->
## 7. DoubleML et Causal Forest
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Athey & Imbens — apprentissage statistique pour causalité
- Double ML : enlever le bias via 2 modèles ML (nuisance functions)
- Causal Forest : RF qui partitionne par hétérogénéité du traitement
- Heterogeneous Treatment Effects (HTE / CATE)

**Code visé** : `EconML.dml`, `econml.grf`.
<!-- #endregion -->

<!-- #region -->
## 8. Instrumental Variables (2SLS)
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Quand : confondants non observés, IV satisfait exclusion restriction
- Two-Stage Least Squares
- Force de l'instrument (F-stat)
- Cas réel : RCT vs natural experiment

**Code visé** : `linearmodels.IV2SLS`.
<!-- #endregion -->

<!-- #region -->
## 9. Mediation analysis
<!-- #endregion -->

<!-- #region -->
**Notions** :
- `X → M → Y` : effet direct vs indirect
- Decomposition : NDE, NIE (Natural Direct / Indirect Effect)
- Pearl mediation formula

**Code visé** : `causaleffect` ou implémentation manuelle.
<!-- #endregion -->

<!-- #region -->
## 10. Frameworks 2026 et pièges
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Libs : `dowhy` (Microsoft), `econml` (Microsoft), `causalml` (Uber), `causal-learn` (CMU)
- Pièges : confounders cachés, sensitivity analysis, p-hacking
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Pearl — Causality (livre)](http://bayes.cs.ucla.edu/BOOK-2K/)
- [DoWhy docs](https://www.pywhy.org/dowhy/)
- [EconML docs](https://econml.azurewebsites.net/)
- [Causal Inference: The Mixtape (Cunningham)](https://mixtape.scunning.com/)
- Notebooks liés : `DS_Bayesian`, `MLE_AB_Testing`.
<!-- #endregion -->
