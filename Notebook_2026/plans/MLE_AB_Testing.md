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
# 🧪 A/B Testing — ML production
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : ML Engineer + Data Scientist · Wiki + Tutoriel

**Dataset(s)** : Synthetic conversion data

A/B testing rigoureux pour valider l'impact business d'un nouveau modèle.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre A/B
<!-- #endregion -->

<!-- #region -->
Hypothèse nulle H0 (les variantes ont même perf). Test bilatéral. p-value, puissance, MDE.
<!-- #endregion -->

<!-- #region -->
## 2. Sample size calculator
<!-- #endregion -->

<!-- #region -->
Pour binary outcome (conversion) et continuous (revenue). Formules. Outils : Optimizely, Statsig.
<!-- #endregion -->

<!-- #region -->
## 3. Sequential testing
<!-- #endregion -->

<!-- #region -->
Permet d'arrêter dès qu'on a un signal (vs fixed horizon). Always Valid Inference (mSPRT). Évite peeking.
<!-- #endregion -->

<!-- #region -->
## 4. CUPED
<!-- #endregion -->

<!-- #region -->
Controlled-experiment Using Pre-Experiment Data (Microsoft 2013). Réduit la variance en utilisant un covariate pré-expé.
<!-- #endregion -->

<!-- #region -->
## 5. Multi-arm bandits
<!-- #endregion -->

<!-- #region -->
Quand on veut optimiser online plutôt que tester rigoureusement. Lien `ML_Apprentissage_par_Renforcement`.
<!-- #endregion -->

<!-- #region -->
## 6. Bayesian A/B testing
<!-- #endregion -->

<!-- #region -->
Posterior sur Δθ. `P(B > A | data)` directement interprétable business. Lien `DS_Bayesian`.
<!-- #endregion -->

<!-- #region -->
## 7. Stratification, blocking
<!-- #endregion -->

<!-- #region -->
Garantir équilibre des covariates entre A et B (sex, geo, device). Réduit la variance.
<!-- #endregion -->

<!-- #region -->
## 8. Pièges
<!-- #endregion -->

<!-- #region -->
Peeking (multiple comparisons), Simpson's paradox, network effects (mes amis voient B et m'influencent), novelty effects.
<!-- #endregion -->

<!-- #region -->
## 9. Outils 2026
<!-- #endregion -->

<!-- #region -->
GrowthBook (OSS), Statsig, Optimizely, LaunchDarkly, Eppo. Compare : OSS vs SaaS, coût, fonctionnalités.
<!-- #endregion -->

<!-- #region -->
## 10. Cas spéciaux
<!-- #endregion -->

<!-- #region -->
Long-term effects (carryover), holdout group permanent, expérimentation continue (rolling rollouts).
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Trustworthy Online Controlled Experiments — Kohavi et al. (livre)](https://www.cambridge.org/)
- [GrowthBook docs](https://docs.growthbook.io/)
- [Statsig docs](https://docs.statsig.com/)
- Notebooks liés : `DS_Causal_Inference`, `DS_Bayesian`, `ML_Apprentissage_par_Renforcement`.
<!-- #endregion -->
