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
# 🔄 Pipelines ML end-to-end (Airflow / Prefect)
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : MLOps · Tutoriel

**Dataset(s)** : Cal Housing pipeline ELT + ML

Construire un pipeline ML production-grade orchestré.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Workflow ML standard
<!-- #endregion -->

<!-- #region -->
Ingestion → validation → preprocessing → train → eval → register → deploy → monitor. Boucle continue.
<!-- #endregion -->

<!-- #region -->
## 2. DAG Airflow / Prefect concret
<!-- #endregion -->

<!-- #region -->
Code complet d'un pipeline mensuel : ingest from S3 → validate with GE → train via TaskFlow → register MLflow → deploy via KServe.
<!-- #endregion -->

<!-- #region -->
## 3. Idempotence + retry
<!-- #endregion -->

<!-- #region -->
Toute task doit pouvoir être rejouée sans dommage. Patterns : INSERT IGNORE, upsert, checkpoint.
<!-- #endregion -->

<!-- #region -->
## 4. Branching
<!-- #endregion -->

<!-- #region -->
Si nouvelles données détectées → train. Sinon → skip et notify.
<!-- #endregion -->

<!-- #region -->
## 5. Backfill
<!-- #endregion -->

<!-- #region -->
Rejouer le passé sur les N derniers jours. Crucial après bug fix.
<!-- #endregion -->

<!-- #region -->
## 6. Secrets management
<!-- #endregion -->

<!-- #region -->
Vault (HashiCorp), AWS Secrets Manager, Doppler. Jamais en clair dans le code.
<!-- #endregion -->

<!-- #region -->
## 7. Notifications
<!-- #endregion -->

<!-- #region -->
Slack on failure, success summary daily. PagerDuty pour critiques.
<!-- #endregion -->

<!-- #region -->
## 8. Combine MLflow + Airflow
<!-- #endregion -->

<!-- #region -->
Chaque DAG task logue dans la run MLflow appropriée. Mapping DAG_RUN_ID ↔ MLflow run_id.
<!-- #endregion -->

<!-- #region -->
## 9. CI/CD du pipeline
<!-- #endregion -->

<!-- #region -->
Tester les DAGs avant déploiement. `airflow dags test`. Pre-commit hooks.
<!-- #endregion -->

<!-- #region -->
## 10. Alternatives
<!-- #endregion -->

<!-- #region -->
Kubeflow Pipelines (K8s natif), Metaflow (Netflix), ZenML, Flyte. Lequel quand.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Airflow docs](https://airflow.apache.org/docs/)
- [Prefect docs](https://docs.prefect.io/)
- [Kubeflow docs](https://www.kubeflow.org/)
- Notebook lié : `ML_MLFlow_Bench`.
<!-- #endregion -->
