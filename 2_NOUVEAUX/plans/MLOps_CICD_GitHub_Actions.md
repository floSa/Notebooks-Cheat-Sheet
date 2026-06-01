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
# 🔁 CI/CD pour ML avec GitHub Actions
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : MLOps · Tutoriel

**Dataset(s)** : Repo ML demo

CI/CD adapté au ML : datasets, modèles, gates de qualité.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Spécificités CI/CD ML
<!-- #endregion -->

<!-- #region -->
vs CI/CD classique : datasets (versioner, large), modèles (binary, lourd), reproductibilité (seed, deps), eval (pas juste tests unitaires).
<!-- #endregion -->

<!-- #region -->
## 2. GitHub Actions — workflows
<!-- #endregion -->

<!-- #region -->
Fichier `.github/workflows/ml.yml`. Triggers : push, PR, schedule, manual.
<!-- #endregion -->

<!-- #region -->
## 3. Steps types pipeline
<!-- #endregion -->

<!-- #region -->
Lint (ruff/black) → tests unitaires (pytest) → data tests (Great Expectations) → train (sur runner GPU si dispo) → eval → register MLflow → deploy si meilleur.
<!-- #endregion -->

<!-- #region -->
## 4. CML — Continuous Machine Learning
<!-- #endregion -->

<!-- #region -->
Iterative.ai lib qui ajoute des comments dans PR avec metrics, graphes, comparaison vs baseline.
<!-- #endregion -->

<!-- #region -->
## 5. DVC integration
<!-- #endregion -->

<!-- #region -->
Versioner datasets et modèles dans le repo via pointers. `dvc pull` dans CI pour récupérer.
<!-- #endregion -->

<!-- #region -->
## 6. Self-hosted runners GPU
<!-- #endregion -->

<!-- #region -->
Cluster perso ou cloud (CML provisione automatiquement EC2 GPU).
<!-- #endregion -->

<!-- #region -->
## 7. Secrets & OIDC
<!-- #endregion -->

<!-- #region -->
GitHub OIDC → assume role AWS sans long-lived credentials. Best practice 2024+.
<!-- #endregion -->

<!-- #region -->
## 8. Cache
<!-- #endregion -->

<!-- #region -->
Cache pip dependencies, modèles téléchargés, datasets DVC. Gain énorme.
<!-- #endregion -->

<!-- #region -->
## 9. Tests de modèle
<!-- #endregion -->

<!-- #region -->
Behavioral tests (CheckList), robustness (perturbations), fairness (Fairlearn), regression (vs previous best).
<!-- #endregion -->

<!-- #region -->
## 10. Comparatif
<!-- #endregion -->

<!-- #region -->
GitHub Actions vs GitLab CI vs Jenkins vs CircleCI vs Buildkite. Pour ML : GitHub Actions + CML = standard 2026.
<!-- #endregion -->

<!-- #region -->
## 11. Deployment gates
<!-- #endregion -->

<!-- #region -->
Reviewers requis pour merger en main. Environments protected (prod). Manual approval avant deploy.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [GitHub Actions docs](https://docs.github.com/en/actions)
- [CML — Iterative](https://cml.dev/)
- [DVC docs](https://dvc.org/)
- Notebooks liés : `ML_MLFlow_Bench`, `MLOps_Pipelines_Airflow`.
<!-- #endregion -->
