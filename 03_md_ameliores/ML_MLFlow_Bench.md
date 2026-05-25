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
# 🧪 MLflow — Tracking, Registry, Deployment
<!-- #endregion -->

<!-- #region -->
Notebook **Wiki + Tutoriel** sur **MLflow 3.x (2026)** — la plateforme MLOps open-source de référence pour gérer le cycle de vie complet d'un modèle ML : **tracking d'expériences**, **model registry**, **deployment**, et désormais **agents/LLMs**.

Couverture :

1. Vue d'ensemble MLflow 3.x — pourquoi, écosystème, comparatif (W&B / DVC / ClearML).
2. **Tracking** — runs, params, metrics, artifacts, autolog.
3. **Model Registry** — versions, aliases (`@champion`/`@challenger`), staging.
4. **Bench multi-modèles** — comparer 5 algos sur **California Housing** (mutualisé).
5. **Deployment** — `mlflow models serve`, formats (pyfunc, MLmodel).
6. **MLflow LLM / Agents** (nouveauté 3.x) — tracing, evaluations.
7. **Production setup** — PostgreSQL backend, S3 artifacts, CI/CD.

> Dataset : **California Housing** (régression), mutualisé avec `ML_Optimisation_de_Modèles`, `ML_Bagging_Boosting`, etc.
<!-- #endregion -->

<!-- #region -->
## 1. Vue d'ensemble MLflow 3.x (2026)
<!-- #endregion -->

<!-- #region -->
**MLflow** est devenu en 2024-2026 le standard open-source pour le MLOps. Quatre piliers :

| Composant | Rôle |
|---|---|
| **Tracking** | Logger params, metrics, artifacts, modèles par run |
| **Models** | Format universel `pyfunc` qui encapsule code + deps + signature |
| **Model Registry** | Versionning + aliases (`@champion`, `@challenger`, `@staging`) |
| **Deployments / Serving** | Sert un modèle via REST/Docker/SageMaker/Databricks |
| **GenAI / Agents** (3.x) | Tracing LLMs, eval automatique de pipelines RAG/agentic |

**Comparatif rapide 2026** :

| Outil | Open-source | Tracking | Registry | LLM tracing | Notes |
|---|---|---|---|---|---|
| **MLflow** | ✅ | ✅ | ✅ | ✅ (3.x) | Le standard polyvalent |
| **Weights & Biases** | partiel | ✅✅ (UI top) | ✅ | ✅ | UI supérieure, SaaS coût $$ |
| **ClearML** | ✅ | ✅ | ✅ | ✅ | Bonne alternative open intégrée |
| **DVC + DVC Studio** | ✅ | ✅ (git-like) | ✅ | partiel | Très bon pour la **data versioning** |
| **Comet** | partiel | ✅ | ✅ | ✅ | UI propre, freemium |
| **Neptune.ai** | non | ✅✅ | ✅ | ✅ | SaaS, très bon pour les équipes |

**Quand choisir MLflow** : 80 % des cas. Self-hosted facile, intégré à PyTorch / sklearn / XGB / LightGBM / Transformers automatiquement, ouvert.
<!-- #endregion -->

```python
import mlflow
import sklearn
import numpy as np
import pandas as pd

print(f"MLflow : {mlflow.__version__}")
print(f"sklearn : {sklearn.__version__}")
```

<!-- #region -->
## 2. Tracking — le fondamental
<!-- #endregion -->

<!-- #region -->
Une **run** = un essai (un entraînement) qui logge :

- **Params** : hyperparamètres (typiquement <50, fixes).
- **Metrics** : valeurs numériques scalaires (avec steps possibles).
- **Artifacts** : fichiers (modèle sérialisé, plots, fichiers eval, données sample).
- **Tags** : métadonnées arbitraires (auteur, branch git, dataset version).
- **Model** : sérialisation + signature + requirements.

**Backend** : par défaut local `./mlruns/` (SQLite + filesystem). En prod : Postgres + S3.

> Toujours fixer un **experiment name** clair (1 par projet/tâche). Sinon tout part dans "Default" et c'est l'enfer.
<!-- #endregion -->

```python
from pathlib import Path
import tempfile

# Tracking URI local pour la démo. En prod : http://mlflow-server:5000
TRACKING_DIR = Path(tempfile.gettempdir()) / "mlflow_demo"
mlflow.set_tracking_uri(f"file:///{TRACKING_DIR.as_posix()}")
mlflow.set_experiment("housing-regression-bench")

print(f"Tracking URI : {mlflow.get_tracking_uri()}")
print(f"Experiment   : {mlflow.get_experiment_by_name('housing-regression-bench').experiment_id}")
```

<!-- #region -->
### 2.1 Une run minimale
<!-- #endregion -->

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Data (mutualisée avec d'autres notebooks ML)
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run MLflow
with mlflow.start_run(run_name="ridge-baseline") as run:
    alpha = 1.0
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    # Log : params, metrics, model
    mlflow.log_param("alpha", alpha)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.set_tag("model_family", "linear")
    mlflow.set_tag("dataset", "california_housing")

    # Inférer la signature (schémas d'entrée/sortie) — crucial pour la prod
    signature = mlflow.models.infer_signature(X_train, preds)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(2),
    )

    print(f"Run ID : {run.info.run_id}")
    print(f"RMSE   : {rmse:.4f}  R²: {r2:.4f}")
```

<!-- #region -->
### 2.2 Autolog — pour aller vite
<!-- #endregion -->

<!-- #region -->
`mlflow.autolog()` enregistre automatiquement params, metrics et le modèle pour **toutes** les libs supportées (sklearn, XGBoost, LightGBM, PyTorch, Keras, Transformers, ...). Pratique pour démarrer, à désactiver quand on veut un contrôle fin.
<!-- #endregion -->

```python
# Décommenter pour activer autolog
# mlflow.autolog()
# with mlflow.start_run(run_name="auto-ridge"):
#     Ridge(alpha=0.5).fit(X_train, y_train)  # log auto de tout
```

<!-- #region -->
## 3. Bench multi-modèles
<!-- #endregion -->

<!-- #region -->
Pattern canonique : une boucle sur N modèles, **une run par modèle**, comparaison ensuite dans l'UI MLflow.
<!-- #endregion -->

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, ElasticNet


def bench_model(name: str, model, X_tr, X_te, y_tr, y_te) -> dict:
    """Entraine un modèle, logge une run MLflow, renvoie les metrics."""
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        r2 = float(r2_score(y_te, preds))

        mlflow.log_params({f"clf__{k}": v for k, v in model.get_params().items()
                           if isinstance(v, (int, float, str, bool, type(None)))})
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("model_family", type(model).__name__)

        signature = mlflow.models.infer_signature(X_tr, preds)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", signature=signature)

        return {"name": name, "rmse": rmse, "r2": r2}


models = {
    "ridge":        Ridge(alpha=1.0),
    "elastic_net":  ElasticNet(alpha=0.1, l1_ratio=0.5),
    "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "gbm":          GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
}

results = [bench_model(n, m, X_train, X_test, y_train, y_test) for n, m in models.items()]
results_df = pd.DataFrame(results).sort_values("rmse")
print(results_df.to_string(index=False))
```

<!-- #region -->
### 3.1 Recherche programmatique du meilleur run
<!-- #endregion -->

```python
exp = mlflow.get_experiment_by_name("housing-regression-bench")
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.rmse > 0",
    order_by=["metrics.rmse ASC"],
)
print(runs[["tags.mlflow.runName", "metrics.rmse", "metrics.r2"]].head())
```

<!-- #region -->
## 4. Model Registry — versioning et aliases
<!-- #endregion -->

<!-- #region -->
**Idée** : un nom de modèle dans le Registry (ex: `housing-regressor`) → plusieurs versions (1, 2, 3, ...) → des **aliases** mobiles (`@champion`, `@challenger`, `@production`, `@staging`).

**Avantage des aliases** sur les "stages" historiques (`None/Staging/Production/Archived`, **deprecated dans MLflow 3.x**) :

- Un alias est juste un pointeur vers une version. On bascule sans déployer.
- Le code de prod référence `models:/housing-regressor@champion` — il n'a pas à connaître le numéro.
- Multiples aliases possibles simultanément (canary, blue/green).
<!-- #endregion -->

```python
# Enregistrer le meilleur modèle dans le registry
best_run_id = runs.iloc[0]["run_id"]
model_uri = f"runs:/{best_run_id}/model"
print(f"Best model URI : {model_uri}")

# Enregistrement (peut nécessiter un client MLflow ; en local marche directement)
try:
    mv = mlflow.register_model(model_uri=model_uri, name="housing-regressor")
    print(f"Registered as version {mv.version}")

    # Assigner l'alias @champion
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name="housing-regressor", alias="champion", version=mv.version)
    print(f"Alias @champion → version {mv.version}")
except Exception as e:
    print(f"Registry registration skipped (likely no DB-backed backend) : {type(e).__name__}: {e}")
```

<!-- #region -->
### 4.1 Charger depuis le registry
<!-- #endregion -->

```python
# Charger par alias (le pattern recommandé)
try:
    loaded = mlflow.pyfunc.load_model("models:/housing-regressor@champion")
    sample_preds = loaded.predict(X_test.head(3))
    print(f"Predictions sample : {sample_preds}")
except Exception as e:
    print(f"Load skipped : {type(e).__name__}: {e}")
```

<!-- #region -->
## 5. Deployment
<!-- #endregion -->

<!-- #region -->
Trois patterns :

| Pattern | Commande | Quand |
|---|---|---|
| **Local serve** | `mlflow models serve -m models:/housing-regressor@champion -p 5001` | Dev / debug local |
| **Docker** | `mlflow models build-docker -m models:/housing-regressor@champion -n housing-api` | Conteneur portable |
| **Cloud** | Plugins : SageMaker, Azure ML, Databricks Model Serving | Prod managé |

Le serving local lance un endpoint REST `/invocations` qui prend un JSON avec les inputs au format pandas `split` ou `records`.

```bash
# Sert le modèle sur localhost:5001
mlflow models serve -m models:/housing-regressor@champion -p 5001 --env-manager local

# Inference :
curl -X POST http://localhost:5001/invocations \
     -H "Content-Type: application/json" \
     -d '{"dataframe_split": {"columns": ["MedInc", "HouseAge", ...],
                               "data": [[8.32, 41, ...]]}}'
```

L'**input_example** loggué à `log_model` documente le payload attendu — visible dans l'UI MLflow.
<!-- #endregion -->

<!-- #region -->
## 6. MLflow LLM / Agents (3.x)
<!-- #endregion -->

<!-- #region -->
MLflow 3.x ajoute le support natif des **LLMs et agents** :

- **Tracing** auto via `mlflow.langchain.autolog()`, `mlflow.openai.autolog()`, `mlflow.dspy.autolog()`. Capture chaque appel LLM, ses inputs/outputs, sa latence, ses tokens.
- **Evaluation** automatique : `mlflow.evaluate()` avec `extra_metrics=[faithfulness, relevancy, toxicity]` (LLM-as-judge ou heuristiques).
- **Prompt management** : versionner les prompts comme des modèles.

```python
# Pseudo-code MLflow LLM
"""
import mlflow.langchain
mlflow.langchain.autolog()

# Toute chaîne LangChain exécutée est tracée automatiquement
chain = ...  # ta chaîne RAG
result = chain.invoke({"question": "..."})

# Eval RAG end-to-end
mlflow.evaluate(
    data=eval_df,                # DataFrame avec question, contexte, ground_truth
    targets="ground_truth",
    model_type="question-answering",
    extra_metrics=[mlflow.metrics.genai.faithfulness(), mlflow.metrics.genai.relevance()],
)
"""
```
<!-- #endregion -->

<!-- #region -->
## 7. Production setup
<!-- #endregion -->

<!-- #region -->
### 7.1 Backend recommandé
<!-- #endregion -->

<!-- #region -->
| Composant | Dev | Prod |
|---|---|---|
| **Backend store** (params, metrics) | SQLite local | **PostgreSQL** managé (RDS, CloudSQL, Supabase) |
| **Artifact store** (modèles, fichiers) | Filesystem local | **S3** / GCS / Azure Blob / MinIO |
| **Tracking server** | `mlflow ui` local | Service `mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...` derrière nginx + auth |
| **Auth** | aucune | Reverse proxy avec OIDC, ou MLflow auth plugin |

Lancement prod typique :

```bash
mlflow server \
  --backend-store-uri postgresql+psycopg2://user:pw@host/mlflow \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --host 0.0.0.0 --port 5000 \
  --workers 4
```
<!-- #endregion -->

<!-- #region -->
### 7.2 CI/CD intégration
<!-- #endregion -->

<!-- #region -->
Pattern 2026 (GitHub Actions / GitLab CI) :

1. **Sur PR** : pipeline d'entraînement → tagger la run avec `git_sha`, `pr_number`.
2. **Sur merge main** : promouvoir la run en `@challenger` du registry.
3. **Tests d'intégration** : challenger vs champion → si meilleur sur metrics business, déclencher revue manuelle pour promouvoir en `@champion`.
4. **Déploiement automatique** quand `@champion` change → CI re-build le container et redéploie.

> Le modèle de prod ne change **jamais** sans qu'une run trackée et validée soit derrière. Garantit la **lineage** : prod → run → code commit → data version.
<!-- #endregion -->

<!-- #region -->
### 7.3 Pièges courants
<!-- #endregion -->

<!-- #region -->
- ❌ Tout loguer en métrique scalaire au lieu de logger des artifacts (eval CSV, confusion matrix PNG).
- ❌ Pas de **signature** sur le modèle → impossible à déployer proprement (l'API ne sait pas valider l'input).
- ❌ Stages au lieu d'aliases (deprecated en 3.x).
- ❌ Pas de versioning du **code** (logger `git_sha` en tag) ni de la **data** (DVC ou tag dataset).
- ❌ Tracking server local → perte des runs au prochain reboot.
- ✅ Toujours `set_experiment` au début d'un script — éviter le dump dans "Default".
- ✅ Logger un `input_example` + signature → le serving Docker fonctionne out-of-the-box.
<!-- #endregion -->

<!-- #region -->
## 8. Sources
<!-- #endregion -->

<!-- #region -->
- [MLflow — docs officielles](https://mlflow.org/docs/latest/)
- [MLflow Model Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Production Guide — Chaos and Order 2026](https://www.youngju.dev/blog/ai-platform/2026-03-07-ai-platform-mlflow-experiment-tracking-model-registry.en)
- [Complete MLflow Guide — Chaos and Order 2026](https://www.youngju.dev/blog/ai-platform/2026-03-03-mlflow-experiment-tracking-guide.en)
- [MLflow vs W&B vs ClearML — comparison 2026](https://ml-digest.com/mlflow/)
<!-- #endregion -->
