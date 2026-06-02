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
# 🔥 PySpark — Big Data Processing
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Engineer · Tutoriel

**Dataset(s)** : NYC Taxi parquet (gros volume)

PySpark pour le traitement distribué de gros volumes (≥ 100 GB). Comparatif avec Polars / DuckDB.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Architecture Spark
<!-- #endregion -->

<!-- #region -->
Driver + executors + cluster manager (YARN/Kubernetes/Standalone). RDD vs DataFrame vs Dataset. Lazy evaluation + DAG.
<!-- #endregion -->

<!-- #region -->
## 2. PySpark DataFrame API
<!-- #endregion -->

<!-- #region -->
select, filter, groupBy, agg, join, window. Idiome : tout chaîner avant `.show()` ou `.collect()`.
<!-- #endregion -->

<!-- #region -->
## 3. SQL via spark.sql
<!-- #endregion -->

<!-- #region -->
Registrer un DataFrame comme view temporaire puis SQL. Bonnes pratiques.
<!-- #endregion -->

<!-- #region -->
## 4. UDFs
<!-- #endregion -->

<!-- #region -->
Python UDF (lent — serialization), Pandas UDF (vectorisé, beaucoup plus rapide), SQL UDF.
<!-- #endregion -->

<!-- #region -->
## 5. Partitioning et bucketing
<!-- #endregion -->

<!-- #region -->
Partition par colonne pour pushdown filter. Bucketing pour optim joins.
<!-- #endregion -->

<!-- #region -->
## 6. Catalyst optimizer
<!-- #endregion -->

<!-- #region -->
EXPLAIN, PARSED plan vs OPTIMIZED. Comment lire un plan.
<!-- #endregion -->

<!-- #region -->
## 7. Performance
<!-- #endregion -->

<!-- #region -->
Broadcast joins (petite table), shuffle vs broadcast, skew handling, salt key, coalesce vs repartition.
<!-- #endregion -->

<!-- #region -->
## 8. Structured Streaming
<!-- #endregion -->

<!-- #region -->
Streaming sur Kafka, micro-batch, exactly-once, watermark.
<!-- #endregion -->

<!-- #region -->
## 9. Spark ML — pyspark.ml
<!-- #endregion -->

<!-- #region -->
Pipeline + Estimator + Transformer. Fit / transform sur DataFrame.
<!-- #endregion -->

<!-- #region -->
## 10. Delta Lake
<!-- #endregion -->

<!-- #region -->
ACID sur S3, time travel, schema evolution, CDC. Alternative : Iceberg, Hudi.
<!-- #endregion -->

<!-- #region -->
## 11. Spark vs alternatives 2026
<!-- #endregion -->

<!-- #region -->
Spark reste roi pour > 1 TB / cluster. Polars / DuckDB gagnent en single-machine. Daft (Python-native distributed).
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Spark docs](https://spark.apache.org/docs/latest/)
- [Learning Spark 2nd ed (Databricks free)](https://www.databricks.com/p/ebook/learning-spark-from-oreilly)
- [Delta Lake docs](https://docs.delta.io/)
<!-- #endregion -->
