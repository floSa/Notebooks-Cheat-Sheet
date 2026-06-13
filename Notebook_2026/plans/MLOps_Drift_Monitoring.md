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
# 📉 Drift Monitoring — Evidently, NannyML
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : MLOps · Wiki + Tutoriel

**Dataset(s)** : Cal Housing train vs simulated prod (avec drift injecté)

Détecter dégradation modèle en prod : data drift, concept drift, performance.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi monitorer
<!-- #endregion -->

<!-- #region -->
Modèles dégradent avec le temps : data distribution change, monde change. Détection avant que ça pète.
<!-- #endregion -->

<!-- #region -->
## 2. Types de drift
<!-- #endregion -->

<!-- #region -->
**Covariate drift** : `P(X)` change. **Label drift** : `P(Y)` change. **Concept drift** : `P(Y|X)` change. Conséquences différentes.
<!-- #endregion -->

<!-- #region -->
## 3. Tests statistiques
<!-- #endregion -->

<!-- #region -->
Kolmogorov-Smirnov (continu), Chi² (catégoriel), PSI (Population Stability Index — seuils 0.1, 0.2), Wasserstein.
<!-- #endregion -->

<!-- #region -->
## 4. Evidently AI
<!-- #endregion -->

<!-- #region -->
Lib OSS référence 2024+. Reports HTML interactifs. Tests intégrés. Integration MLflow / Prefect / Airflow.
<!-- #endregion -->

<!-- #region -->
## 5. NannyML
<!-- #endregion -->

<!-- #region -->
Performance estimation **sans ground truth** (CBPE — Confidence-Based Performance Estimation). Quand les labels arrivent en retard ou jamais.
<!-- #endregion -->

<!-- #region -->
## 6. SaaS : WhyLabs / Arize / Fiddler
<!-- #endregion -->

<!-- #region -->
Observabilité ML cloud. Inclut feature attribution, fairness, root cause analysis.
<!-- #endregion -->

<!-- #region -->
## 7. Monitoring inference
<!-- #endregion -->

<!-- #region -->
Latency (P50/P95/P99), throughput, error rate, GPU utilization, memory. Prometheus + Grafana standard.
<!-- #endregion -->

<!-- #region -->
## 8. Business metrics
<!-- #endregion -->

<!-- #region -->
CTR, conversion, revenue. Le vrai but. Aligner ML metrics avec business metrics (parfois pas corrélés).
<!-- #endregion -->

<!-- #region -->
## 9. Alerting
<!-- #endregion -->

<!-- #region -->
Slack, PagerDuty, email. Thresholds adaptifs (z-score sur historique). Anti-spam : grouping, deduplication.
<!-- #endregion -->

<!-- #region -->
## 10. Stratégie de retraining
<!-- #endregion -->

<!-- #region -->
Trigger-based (drift détecté), scheduled (chaque semaine), adaptive (continual learning), shadow validation avant rollout.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Evidently AI](https://docs.evidentlyai.com/)
- [NannyML](https://nannyml.readthedocs.io/)
- [Arize AI](https://docs.arize.com/)
- [ML Drift Detection (Microsoft Learn)](https://learn.microsoft.com/)
<!-- #endregion -->
