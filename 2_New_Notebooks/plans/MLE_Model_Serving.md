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
# 📦 Model Serving — packaging et déploiement
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : ML Engineer + MLOps · Wiki + Tutoriel

**Dataset(s)** : Modèle ML pré-entrainé

Packager un modèle pour le déploiement : MLflow Models, BentoML, container, serverless.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Packaging
<!-- #endregion -->

<!-- #region -->
Modèle artifact + code d'inférence + deps + signature. Container Docker = standard reproductible.
<!-- #endregion -->

<!-- #region -->
## 2. MLflow Models (pyfunc)
<!-- #endregion -->

<!-- #region -->
Format universel cross-framework. `mlflow.pyfunc.load_model(uri)`. Sert le modèle via `mlflow models serve`.
<!-- #endregion -->

<!-- #region -->
## 3. BentoML vs Cog vs Truss
<!-- #endregion -->

<!-- #region -->
BentoML : Python-native, riche. Cog (Replicate) : YAML + Dockerfile auto. Truss (Baseten) : similaire à Cog.
<!-- #endregion -->

<!-- #region -->
## 4. KServe / Seldon
<!-- #endregion -->

<!-- #region -->
Kubernetes-native serving. InferenceService CRD. Auto-scaling, canary, transformers.
<!-- #endregion -->

<!-- #region -->
## 5. Cloud SaaS
<!-- #endregion -->

<!-- #region -->
AWS SageMaker, Vertex AI (GCP), Azure ML Endpoints, Databricks Model Serving. Pros : managed. Cons : lock-in, coût.
<!-- #endregion -->

<!-- #region -->
## 6. Serverless
<!-- #endregion -->

<!-- #region -->
AWS Lambda (cold start), GCP Cloud Run, Modal, Replicate, Beam. Pour low-throughput intermittent.
<!-- #endregion -->

<!-- #region -->
## 7. API design
<!-- #endregion -->

<!-- #region -->
Sync vs async, batch endpoint, streaming SSE pour LLMs. Versionning d'API (/v1, /v2).
<!-- #endregion -->

<!-- #region -->
## 8. Versioning
<!-- #endregion -->

<!-- #region -->
Modèles (MLflow), schemas (Pydantic), contrats (OpenAPI). Backward compatibility.
<!-- #endregion -->

<!-- #region -->
## 9. Rollback
<!-- #endregion -->

<!-- #region -->
Always keep N-1 deployed. Feature flags pour switch instantané. Database migrations reversible.
<!-- #endregion -->

<!-- #region -->
## 10. Multi-modèle serving
<!-- #endregion -->

<!-- #region -->
Gateway pattern : 1 API qui route vers N modèles selon header / payload. Bonne pratique pour A/B testing et fallback.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [BentoML docs](https://docs.bentoml.com/)
- [Cog (Replicate)](https://github.com/replicate/cog)
- [KServe](https://kserve.github.io/)
<!-- #endregion -->
