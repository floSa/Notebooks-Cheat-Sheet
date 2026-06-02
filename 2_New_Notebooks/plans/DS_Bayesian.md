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
# 🎲 Bayesian Data Science
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Scientist · Wiki + Tutoriel

**Dataset(s)** : Eight Schools (classique), Cal Housing

Approche bayésienne : modélisation avec incertitude explicite, priors informatifs, hiérarchies. Indispensable quand petit échantillon, besoin d'IC, ou modèle interprétable structurellement.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi bayésien en 2026
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Incertitude paramétrique vs prédictive
- Priors : injection de connaissance experte
- Régularisation naturelle (vs L1/L2 ad-hoc)
- Hiérarchies : partial pooling (sweet spot biais/variance)
- Quand préférer fréquentiste : très grandes data, baseline ML
<!-- #endregion -->

<!-- #region -->
## 2. Bayes en un paragraphe
<!-- #endregion -->

<!-- #region -->
**Notions** :
- `P(θ|D) ∝ P(D|θ) P(θ)` (Bayes rule)
- Posterior = likelihood × prior / evidence
- Posterior predictive : `P(y_new|D) = ∫ P(y_new|θ) P(θ|D) dθ`
- Pourquoi MCMC : intégrales hautes dim impossibles à calculer analytiquement
<!-- #endregion -->

<!-- #region -->
## 3. Conjugate priors — cas simples
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Beta-Binomial (taux de conversion)
- Normal-Normal (moyenne d'une mesure)
- Gamma-Poisson
- Inverse-Gamma — Normal

**Code visé** : calculer le posterior analytiquement pour 2 exemples.
<!-- #endregion -->

<!-- #region -->
## 4. MCMC — Metropolis, Gibbs, HMC, NUTS
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Metropolis-Hastings : règle d'acceptation
- Gibbs : sampling conditionnel
- HMC : Hamiltonian Monte Carlo (gradient-based)
- NUTS : No-U-Turn Sampler (HMC adaptatif)
- Convergence : Rhat, ESS, trace plots
<!-- #endregion -->

<!-- #region -->
## 5. PyMC v5+ — workflow complet
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Définition du modèle (`with pm.Model()`)
- Priors, likelihood, posterior
- `pm.sample()` (NUTS par défaut)
- `arviz` pour diagnostics et viz (trace, posterior plots, energy plot)

**Code visé** : régression linéaire bayésienne + interprétation.
<!-- #endregion -->

<!-- #region -->
## 6. NumPyro (JAX-based)
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Plus rapide que PyMC (JIT JAX)
- Modèle = fonction Python
- API stochastique avec `numpyro.sample`
- `MCMC(NUTS, ...).run(...)`

**Code visé** : même modèle reproduit, comparaison vitesse.
<!-- #endregion -->

<!-- #region -->
## 7. Variational Inference (ADVI)
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Quand MCMC est trop lent
- ADVI : Automatic Differentiation VI
- Mean-field vs full-rank approximation
- Trade-off : rapide mais biais sur les queues
<!-- #endregion -->

<!-- #region -->
## 8. Hierarchical models — Eight Schools
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Le modèle classique de Gelman
- Partial pooling vs no pooling vs complete pooling
- Funnel-shaped posterior (problème + reparam)
- Cas d'usage : effets multi-niveaux (élèves dans classes dans écoles)
<!-- #endregion -->

<!-- #region -->
## 9. Posterior predictive checks
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Comparer données simulées vs réelles
- p-values bayésiennes
- Détection de model misspecification
<!-- #endregion -->

<!-- #region -->
## 10. Bayesian A/B testing
<!-- #endregion -->

<!-- #region -->
**Notions** :
- Posterior sur `θ_A - θ_B`
- `P(θ_B > θ_A | data)` directement interprétable
- Loss function : risque attendu, stop rule
- Lien avec `MLE_AB_Testing`
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Statistical Rethinking — McElreath (livre + lectures YouTube)](https://xcelab.net/rm/statistical-rethinking/)
- [PyMC docs](https://www.pymc.io/)
- [NumPyro docs](https://num.pyro.ai/)
- [ArviZ docs](https://python.arviz.org/)
- Notebooks liés : `DS_Causal_Inference`, `MLE_AB_Testing`.
<!-- #endregion -->
