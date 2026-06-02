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
# ✅ Data Quality — tests, monitoring, contracts
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Engineer + MLOps · Wiki + Tutoriel

**Dataset(s)** : Titanic + Cal Housing

Garantir la qualité des données : profiling, tests, contracts, observability.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. 6 dimensions de la qualité
<!-- #endregion -->

<!-- #region -->
Completeness, uniqueness, validity, accuracy, consistency, timeliness. Examples concrets.
<!-- #endregion -->

<!-- #region -->
## 2. Great Expectations 1.x (2024+)
<!-- #endregion -->

<!-- #region -->
Expectations declaratives. Suites de tests. Data docs HTML. Integration Airflow.
<!-- #endregion -->

<!-- #region -->
## 3. Soda Core / Soda Cloud
<!-- #endregion -->

<!-- #region -->
Alternative à GE, plus léger. SodaCL — DSL de tests data.
<!-- #endregion -->

<!-- #region -->
## 4. Pandera — schema validation
<!-- #endregion -->

<!-- #region -->
Type-safe DataFrame schemas. Decorator pour valider à l'entrée/sortie de fonctions.
<!-- #endregion -->

<!-- #region -->
## 5. Pydantic pour structured data
<!-- #endregion -->

<!-- #region -->
Validation des configs, payloads API. Type coercion automatique.
<!-- #endregion -->

<!-- #region -->
## 6. Profiling automatique
<!-- #endregion -->

<!-- #region -->
`ydata-profiling` (HTML report), `dataprep.eda` (rapide via Dask), `sweetviz` (comparaison train/test).
<!-- #endregion -->

<!-- #region -->
## 7. Data observability
<!-- #endregion -->

<!-- #region -->
Monte Carlo, Anomalo, Datafold, Lightup. Détectent freshness, volume, distribution drift automatiquement.
<!-- #endregion -->

<!-- #region -->
## 8. Schema drift detection
<!-- #endregion -->

<!-- #region -->
Nouveau champ, type changé, valeurs nulles apparues. Tools : Datafold, dbt schema tests.
<!-- #endregion -->

<!-- #region -->
## 9. Data contracts
<!-- #endregion -->

<!-- #region -->
Concept : producer s'engage sur le schéma + SLA. Enforcement via CI dbt/Pandera. Initiatives : Atlan, Acryl.
<!-- #endregion -->

<!-- #region -->
## 10. Tests dans le pipeline
<!-- #endregion -->

<!-- #region -->
Pattern Airflow : task `validate_with_GE` qui FAIL la DAG si tests rouges. Quarantine flow.
<!-- #endregion -->

<!-- #region -->
## 11. Lineage
<!-- #endregion -->

<!-- #region -->
OpenLineage (CNCF standard), DataHub, Marquez, OpenMetadata. Suivi automatique des transformations.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Great Expectations docs](https://docs.greatexpectations.io/)
- [Soda docs](https://docs.soda.io/)
- [Pandera docs](https://pandera.readthedocs.io/)
- [OpenLineage](https://openlineage.io/)
<!-- #endregion -->
