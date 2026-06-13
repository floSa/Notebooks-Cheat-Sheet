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
# 🛠️ dbt — Transformation SQL versionnée
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Engineer · Wiki + Tutoriel

**Dataset(s)** : NYC Taxi parquet via DuckDB ou Postgres

dbt (data build tool) : SQL-first ELT, tests, documentation auto, version contrôlé. Standard 2024+ pour la transformation analytique.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi dbt
<!-- #endregion -->

<!-- #region -->
Tournant du data warehouse moderne : ELT (transformations dans le DW, pas en amont). Versioning, tests, docs auto. Communauté + écosystème énorme.
<!-- #endregion -->

<!-- #region -->
## 2. Pattern staging → intermediate → mart
<!-- #endregion -->

<!-- #region -->
Layers : `stg_*` (parsing brut), `int_*` (jointures et nettoyage), `fct_*` / `dim_*` (vue business). Modularité.
<!-- #endregion -->

<!-- #region -->
## 3. Materialization
<!-- #endregion -->

<!-- #region -->
`view` (rapide à créer), `table` (rapide à query), `incremental` (append/merge), `ephemeral` (CTE inline).
<!-- #endregion -->

<!-- #region -->
## 4. Tests
<!-- #endregion -->

<!-- #region -->
Built-in : `unique`, `not_null`, `accepted_values`, `relationships`. Custom tests via SQL. Singular vs generic tests.
<!-- #endregion -->

<!-- #region -->
## 5. Sources et freshness
<!-- #endregion -->

<!-- #region -->
Declarer les sources (tables raw). `dbt source freshness` pour alerter si data stale.
<!-- #endregion -->

<!-- #region -->
## 6. Macros + Jinja
<!-- #endregion -->

<!-- #region -->
Réutilisation de SQL. Macros utilitaires (`dbt_utils`, `dbt_expectations`).
<!-- #endregion -->

<!-- #region -->
## 7. Documentation auto-générée
<!-- #endregion -->

<!-- #region -->
`dbt docs generate && dbt docs serve` → site web avec lineage + descriptions.
<!-- #endregion -->

<!-- #region -->
## 8. dbt + DuckDB en local
<!-- #endregion -->

<!-- #region -->
`dbt-duckdb` : dev en local sans Snowflake/BQ. Migration vers prod sans changer le SQL.
<!-- #endregion -->

<!-- #region -->
## 9. Cloud DWs : Snowflake / BigQuery / Postgres
<!-- #endregion -->

<!-- #region -->
Adapter par moteur. Particularités : permissions, scaling, coût.
<!-- #endregion -->

<!-- #region -->
## 10. Comparatif 2026
<!-- #endregion -->

<!-- #region -->
dbt Core (OSS) vs dbt Cloud (SaaS) vs **SQLMesh** (nouvelle alternative — versioning automatique, plus rapide) vs **Dataform** (Google).
<!-- #endregion -->

<!-- #region -->
## 11. CI/CD pour dbt
<!-- #endregion -->

<!-- #region -->
GitHub Actions : `dbt build` sur PR, slim CI (`--state`).
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [dbt docs](https://docs.getdbt.com/)
- [Analytics Engineering with dbt — Madison Mae](https://www.oreilly.com/)
- [SQLMesh docs](https://sqlmesh.readthedocs.io/)
<!-- #endregion -->
