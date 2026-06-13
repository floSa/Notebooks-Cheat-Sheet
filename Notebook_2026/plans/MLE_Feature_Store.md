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
# 🗄️ Feature Store — Feast et alternatives
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : ML Engineer + MLOps · Wiki + Tutoriel

**Dataset(s)** : Cal Housing batch + features streaming

Feature stores : éliminer le training-serving skew, gouverner les features, point-in-time correctness.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi un feature store
<!-- #endregion -->

<!-- #region -->
Training-serving skew (features calculées différemment en train et en prod = bugs subtils). DRY across teams. Online vs offline.
<!-- #endregion -->

<!-- #region -->
## 2. Feast (open-source)
<!-- #endregion -->

<!-- #region -->
Entities, FeatureView, FeatureService. Materialization batch (offline → online). Retrieval API.
<!-- #endregion -->

<!-- #region -->
## 3. Alternatives managées
<!-- #endregion -->

<!-- #region -->
Tecton (ex-Uber), Hopsworks, AWS SageMaker Feature Store, Databricks Feature Store, GCP Vertex AI Feature Store.
<!-- #endregion -->

<!-- #region -->
## 4. Online vs offline store
<!-- #endregion -->

<!-- #region -->
Online : Redis/DynamoDB/Cassandra (read latency ms). Offline : Parquet/BigQuery/Snowflake (batch analytics).
<!-- #endregion -->

<!-- #region -->
## 5. Point-in-time correctness
<!-- #endregion -->

<!-- #region -->
Le pire bug ML : utiliser une feature future au moment t. Le feature store enforce que toute query donne la valeur 'as-of' un timestamp.
<!-- #endregion -->

<!-- #region -->
## 6. Feature transformations
<!-- #endregion -->

<!-- #region -->
Batch (Spark/dbt), on-demand (au moment du request, calculé live), streaming (Flink/Kafka).
<!-- #endregion -->

<!-- #region -->
## 7. Streaming features
<!-- #endregion -->

<!-- #region -->
Kafka → Flink → online store. Cas : nb_clicks_last_5min mis à jour en temps réel.
<!-- #endregion -->

<!-- #region -->
## 8. Discovery & gouvernance
<!-- #endregion -->

<!-- #region -->
Feature registry, documentation, owner, tests. Comme un data catalog mais pour features ML.
<!-- #endregion -->

<!-- #region -->
## 9. Integration MLflow
<!-- #endregion -->

<!-- #region -->
Logger les features utilisées par chaque run pour reproductibilité.
<!-- #endregion -->

<!-- #region -->
## 10. Quand pas de feature store
<!-- #endregion -->

<!-- #region -->
< 10 modèles en prod, équipe < 5 personnes : surestimation. Préférer un dbt + bonnes pratiques.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Feast docs](https://docs.feast.dev/)
- [Tecton docs](https://docs.tecton.ai/)
- [Feature Stores for ML (Hopsworks blog)](https://www.hopsworks.ai/)
<!-- #endregion -->
