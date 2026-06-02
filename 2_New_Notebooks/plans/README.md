# 🆕 Nouveaux notebooks — plans détaillés

24 notebooks de **plans détaillés** couvrant les rôles Data Scientist / Data Engineer / AI Engineer / ML Engineer / MLOps.

Chaque notebook ici est un **squelette structuré** (titre + description + sections + plan détaillé des notions + sources). Le **code reste à implémenter** — voir [`../00_critique.md`](../00_critique.md) pour la priorisation.

## Organisation

```
05_nouveaux_notebooks/
├── README.md                    # ce fichier
├── *.md                         # 24 plans .md jupytext (sources)
└── ipynb/                       # 24 .ipynb convertis
    ├── DS_*.ipynb               # 5 — Data Scientist
    ├── DE_*.ipynb               # 5 — Data Engineer
    ├── AI_*.ipynb               # 6 — AI Engineer
    ├── MLE_*.ipynb              # 4 — ML Engineer
    └── MLOps_*.ipynb            # 4 — MLOps
```

## Liste complète

### Data Scientist (5)
- `DS_Causal_Inference` — Pearl, DiD, PSM, IV, DoubleML
- `DS_Bayesian` — PyMC, NumPyro, MCMC, hierarchical models
- `DS_Geospatial` — GeoPandas, H3, raster, ML géo
- `DS_Survival_Analysis` — Kaplan-Meier, Cox, DeepSurv
- `DS_Recommender_Systems` — CF, MF, Two-Tower, LLM-based

### Data Engineer (5)
- `DE_PySpark` — DataFrame, Catalyst, Delta Lake, vs Polars/DuckDB
- `DE_Airflow_Prefect` — orchestration comparée (Airflow / Prefect / Dagster)
- `DE_Kafka_Streaming` — pub/sub, Schema Registry, Faust, Streams
- `DE_dbt_Modeling` — staging/intermediate/mart, tests, docs auto
- `DE_Data_Quality` — Great Expectations, Soda, Pandera, lineage

### AI Engineer (6)
- `AI_Local_LLMs` — Ollama, vLLM, SGLang, llama.cpp, quantization
- `AI_LLM_Finetuning_PEFT` — LoRA/QLoRA/DPO via TRL, Unsloth, Axolotl
- `AI_Agents_Tools` — LangGraph, AutoGen, MCP, computer use
- `AI_Prompt_Engineering` — CoT, ToT, DSPy, structured output
- `AI_Multimodal_VLM` — CLIP, LLaVA, Qwen-VL, ColPali, computer use
- `AI_Speech_Audio` — Whisper, XTTS, pyannote, MusicGen, voice clone

### ML Engineer (4)
- `MLE_Feature_Store` — Feast, Tecton, point-in-time correctness
- `MLE_Online_Inference` — Triton, BentoML, vLLM, quantization
- `MLE_Model_Serving` — packaging MLflow, BentoML/Cog/Truss, KServe
- `MLE_AB_Testing` — sample size, CUPED, sequential, Bayesian

### MLOps (4)
- `MLOps_Pipelines_Airflow` — pipeline ML end-to-end orchestré
- `MLOps_Model_Registry` — aliases, approval, lineage, AI Act compliance
- `MLOps_Drift_Monitoring` — Evidently, NannyML, performance estimation
- `MLOps_CICD_GitHub_Actions` — CML, DVC, self-hosted GPU runners

## Comment implémenter

Pour chaque plan :

1. **Choisir un sujet prioritaire** (recommandé : commencer par `AI_Local_LLMs`, `MLOps_Drift_Monitoring`, `DE_dbt_Modeling` — fort impact 2026).
2. **Lire le plan** dans le .md (sections + notions listées).
3. **Implémenter section par section** : ajouter le code Python entre les `<!-- #region -->` markers.
4. **Smoke test** : `uv run python scripts/_sandbox/smoke_<nom>.py`.
5. **Convertir** : `uv run jupytext --to ipynb --output ../04_notebooks_finaux/<nom>.ipynb ../05_nouveaux_notebooks/<nom>.md`.
6. **Mettre à jour le status** + commit + push.

Budget estimé : **2-5h par notebook** selon profondeur souhaitée.
