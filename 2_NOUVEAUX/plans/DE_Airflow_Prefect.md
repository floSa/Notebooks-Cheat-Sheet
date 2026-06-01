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
# 🌬️ Orchestration — Airflow / Prefect / Dagster
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Engineer + MLOps · Wiki + Tutoriel

**Dataset(s)** : Pipeline ELT démo sur Cal Housing

Orchestration de workflows : comparatif Airflow vs Prefect vs Dagster, patterns et bonnes pratiques 2026.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre orchestration
<!-- #endregion -->

<!-- #region -->
DAG (Directed Acyclic Graph), task, scheduler, executor. Pourquoi pas juste cron : retry, dépendances, observabilité, backfill.
<!-- #endregion -->

<!-- #region -->
## 2. Apache Airflow 2.x
<!-- #endregion -->

<!-- #region -->
DAG file Python, Operators (BashOperator, PythonOperator, KubernetesPodOperator), Sensors (attendre un évent), XCom (passage de data), Pools (concurrence).
<!-- #endregion -->

<!-- #region -->
## 3. TaskFlow API (Airflow 2+)
<!-- #endregion -->

<!-- #region -->
Décorateur `@task` qui rend les DAGs beaucoup plus pythoniques. Type-checked. XCom implicite.
<!-- #endregion -->

<!-- #region -->
## 4. Prefect 2/3
<!-- #endregion -->

<!-- #region -->
Flow + tasks, deployment, blocks, work pools. Plus moderne mais moins mature qu'Airflow.
<!-- #endregion -->

<!-- #region -->
## 5. Dagster
<!-- #endregion -->

<!-- #region -->
Software Defined Assets (SDA) : pense en termes de *données produites* plutôt que de tâches. Type-safe, asset materialization.
<!-- #endregion -->

<!-- #region -->
## 6. Argo Workflows
<!-- #endregion -->

<!-- #region -->
Kubernetes-native, YAML DAG. Pour stack K8s.
<!-- #endregion -->

<!-- #region -->
## 7. Comparatif 2026
<!-- #endregion -->

<!-- #region -->
Airflow : éprouvé, communauté massive, UI vieillissante. Prefect : moderne, cloud-first. Dagster : pour data engineers pointus. Tableau : adoption, scalabilité, courbe d'apprentissage.
<!-- #endregion -->

<!-- #region -->
## 8. Patterns
<!-- #endregion -->

<!-- #region -->
Retry exponentiel, backfill (rejouer le passé), branching (skip si condition), dynamic mapping (N tâches en parallèle).
<!-- #endregion -->

<!-- #region -->
## 9. Monitoring & alerting
<!-- #endregion -->

<!-- #region -->
Slack/PagerDuty integration, SLA misses, métriques Prometheus.
<!-- #endregion -->

<!-- #region -->
## 10. ML pipelines
<!-- #endregion -->

<!-- #region -->
Combine MLflow + orchestrateur : training pipeline = Airflow DAG qui appelle MLflow log_model.
<!-- #endregion -->

<!-- #region -->
## 11. Best practices
<!-- #endregion -->

<!-- #region -->
Idempotence (tâche rejouable), atomic checkpoints, secrets en KMS, tests des DAGs.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Airflow docs](https://airflow.apache.org/docs/)
- [Prefect docs](https://docs.prefect.io/)
- [Dagster docs](https://docs.dagster.io/)
- Notebook lié : `MLOps_Pipelines_Airflow`.
<!-- #endregion -->
