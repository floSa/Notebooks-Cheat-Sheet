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
# ⏳ Survival Analysis
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Scientist · Wiki + Tutoriel

**Dataset(s)** : Telco churn, ROSSi recidivism, NCCTG lung cancer

Analyse de survie : prédire le temps avant un événement (churn, défaillance, décès) avec censure.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi survival (vs classification binaire)
<!-- #endregion -->

<!-- #region -->
Notion clé : **censure à droite** (l'event n'a pas eu lieu à la fin du suivi). Ignorer la censure = biais. Cas : qui churne et **quand**.
<!-- #endregion -->

<!-- #region -->
## 2. Kaplan-Meier curves
<!-- #endregion -->

<!-- #region -->
Estimateur non-paramétrique. Survival function S(t). Cas : comparer 2 groupes (homme/femme, traitement A/B).
<!-- #endregion -->

<!-- #region -->
## 3. Log-rank test
<!-- #endregion -->

<!-- #region -->
Tester si 2 courbes KM diffèrent significativement. H0 = même hazard.
<!-- #endregion -->

<!-- #region -->
## 4. Cox Proportional Hazards
<!-- #endregion -->

<!-- #region -->
Modèle semi-paramétrique : `h(t|X) = h_0(t) exp(β^T X)`. Coefficients interprétables comme hazard ratios.
<!-- #endregion -->

<!-- #region -->
## 5. Vérifier l'hypothèse PH
<!-- #endregion -->

<!-- #region -->
Schoenfeld residuals. Que faire si non respectée : stratification, time-varying covariates, AFT.
<!-- #endregion -->

<!-- #region -->
## 6. AFT models
<!-- #endregion -->

<!-- #region -->
Accelerated Failure Time : `log(T) = β^T X + σε`. Plus simple à interpréter (effet sur le temps de survie).
<!-- #endregion -->

<!-- #region -->
## 7. Random Survival Forest
<!-- #endregion -->

<!-- #region -->
RF adapté à la censure. `scikit-survival`. Importance des features.
<!-- #endregion -->

<!-- #region -->
## 8. DeepSurv et DeepHit
<!-- #endregion -->

<!-- #region -->
Neural Cox. Permet interactions non-linéaires + features riches. `pycox`.
<!-- #endregion -->

<!-- #region -->
## 9. XGBoost / LightGBM Survival
<!-- #endregion -->

<!-- #region -->
`XGBoostSurvivalEmbeddings`, `LightGBM` avec `objective='regression:cox'`.
<!-- #endregion -->

<!-- #region -->
## 10. Évaluation
<!-- #endregion -->

<!-- #region -->
C-index (concordance), Brier score time-dependent, Integrated Brier Score, calibration.
<!-- #endregion -->

<!-- #region -->
## 11. Cas d'usage
<!-- #endregion -->

<!-- #region -->
Churn prediction (telco), équipement / défaillance (lien `TS_Maintenance_Predictive`), médical (cohortes).
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [lifelines docs](https://lifelines.readthedocs.io/)
- [scikit-survival docs](https://scikit-survival.readthedocs.io/)
- [Statistical methods for survival data — Klein & Moeschberger (livre)](https://www.springer.com/)
<!-- #endregion -->
