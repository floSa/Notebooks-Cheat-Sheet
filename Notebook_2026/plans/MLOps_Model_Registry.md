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
# 📚 Model Registry — gouvernance des modèles
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : MLOps · Wiki + Tutoriel

**Dataset(s)** : Cal Housing modèle

Gérer le cycle de vie des modèles en prod : versions, aliases, approval, lineage.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi un registry
<!-- #endregion -->

<!-- #region -->
Versioning, traçabilité, rollback, governance, compliance (AI Act 2024).
<!-- #endregion -->

<!-- #region -->
## 2. MLflow Registry (review)
<!-- #endregion -->

<!-- #region -->
Voir `ML_MLFlow_Bench`. Focus prod : aliases (@champion, @challenger), workflow promotion.
<!-- #endregion -->

<!-- #region -->
## 3. Weights & Biases Model Registry
<!-- #endregion -->

<!-- #region -->
UI plus polie, integration W&B Sweeps. SaaS.
<!-- #endregion -->

<!-- #region -->
## 4. SageMaker Model Registry
<!-- #endregion -->

<!-- #region -->
Tight integration avec SageMaker Endpoints. Approval workflow natif.
<!-- #endregion -->

<!-- #region -->
## 5. Workflow aliases détaillé
<!-- #endregion -->

<!-- #region -->
@staging → tests automatiques → @challenger → A/B test → @champion (si meilleur). Rollback en 1 commande.
<!-- #endregion -->

<!-- #region -->
## 6. Approval workflows
<!-- #endregion -->

<!-- #region -->
Pull request-like : un model doit être approuvé par data scientist + MLE + tech lead avant prod. Audit trail.
<!-- #endregion -->

<!-- #region -->
## 7. Lineage
<!-- #endregion -->

<!-- #region -->
Modèle → run → dataset version (DVC) → code commit (git SHA). Indispensable pour reproductibilité et audit.
<!-- #endregion -->

<!-- #region -->
## 8. Sécurité
<!-- #endregion -->

<!-- #region -->
RBAC (qui peut promote en prod), audit log (qui a fait quoi quand), modèles signés.
<!-- #endregion -->

<!-- #region -->
## 9. Compliance (AI Act EU 2024)
<!-- #endregion -->

<!-- #region -->
Model cards obligatoires, datasheet (datasets), SBOM ML, risk assessment.
<!-- #endregion -->

<!-- #region -->
## 10. Pratiques 2026
<!-- #endregion -->

<!-- #region -->
Versioner aussi : prompts (pour LLMs), tools (pour agents), datasets. Tout est immutable et versionné.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [MLflow Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [Model Cards (Google paper)](https://arxiv.org/abs/1810.03993)
- [AI Act EU](https://artificialintelligenceact.eu/)
- Notebook lié : `ML_MLFlow_Bench`.
<!-- #endregion -->
