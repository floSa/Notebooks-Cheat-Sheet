# 🆕 Nouveaux notebooks proposés — couverture par rôle

Document de **propositions** de nouveaux notebooks pour couvrir les rôles que tu pratiques : **Data Scientist (DS)**, **Data Engineer (DE)**, **AI Engineer (AIE)**, **ML Engineer (MLE)**, **ML Ops (MLOps)**.

Format : titre, rôle(s) cible, dataset(s), **plan détaillé des sections à aborder**.

> Pour chacune des préconisations ci-dessous, un **notebook plan détaillé** est créé dans le dossier `Notebook_2026/plans/` (cf section "Implémentation" en fin de document).

---

## 🎯 Synthèse par rôle

| Rôle | Notebooks existants pertinents | Nouveaux proposés |
|---|---|---|
| **DS** | Tout EDA + ML + DL + NLP_Classification + Detection_Outliers | DS_Causal_Inference, DS_Bayesian, DS_Geospatial, DS_Survival, DS_Recommender |
| **DE** | DE_Docling, BDD_*, Structures_DataFrame, Structure_BDD | DE_PySpark, DE_Airflow_Prefect, DE_Kafka_Streaming, DE_dbt_Modeling, DE_Data_Quality |
| **AIE** | NLP_*, BDD_Vectorielles | AI_Local_LLMs, AI_LLM_Finetuning_PEFT, AI_Agents_Tools, AI_Prompt_Engineering, AI_Multimodal_VLM, AI_Speech_Audio |
| **MLE** | ML_*, DL_*, INRIA_MOOC | MLE_Feature_Store, MLE_Online_Inference, MLE_Model_Serving, MLE_AB_Testing |
| **MLOps** | ML_MLFlow_Bench | MLOps_Pipelines_Airflow, MLOps_Model_Registry, MLOps_Drift_Monitoring, MLOps_CICD_GitHub_Actions |

**Total nouveaux notebooks** : **24 sujets** (couverture multi-rôle).

---

## 1. Data Scientist (DS)

### DS_Causal_Inference
**Rôle** : DS · Wiki + Tutoriel. **Dataset** : Lalonde (NSW job training, classique), STAR (éducation).

**Sections** :
1. Pourquoi causal ≠ corrélation (Simpson, Berkson).
2. Causal DAGs (Pearl) — backdoor/frontdoor criterion.
3. **A/B testing** : design d'expérience, puissance statistique, MDE.
4. **DiD** (Differences-in-Differences) — quasi-expériences.
5. **Propensity Score Matching** (PSM) + IPW.
6. **Synthetic Control** (Abadie).
7. **DoubleML / Causal Forest** (Athey, Wager).
8. **Instrumental Variables** (2SLS).
9. **Mediation analysis**.
10. Libs : `dowhy`, `econml`, `causalml`, `causal-learn`.
11. Pièges : confounders cachés, robustness checks.
12. Sources.

### DS_Bayesian
**Rôle** : DS · Wiki + Tutoriel. **Dataset** : Eight Schools (classique), Cal Housing.

**Sections** :
1. Pourquoi bayésien : incertitude + priors + petit data.
2. Bayes en 1 paragraphe : `P(θ|D) ∝ P(D|θ) P(θ)`.
3. Conjugate priors (Beta-Binomial, Normal-Normal).
4. **MCMC** : Metropolis-Hastings, Gibbs, HMC, NUTS.
5. **PyMC** v5+ : modélisation, sampling, diagnostics (Rhat, ESS, trace plots).
6. **NumPyro** (JAX-based, plus rapide).
7. **Variational Inference** quand MCMC trop lent.
8. **Bayesian regression**, **hierarchical models** (multi-level / partial pooling).
9. **Posterior predictive checks**.
10. **Bayesian A/B testing** (lien avec Causal Inference notebook).
11. **PyMC + arviz** workflow.
12. Sources.

### DS_Geospatial
**Rôle** : DS · Wiki + Tutoriel. **Dataset** : NYC Taxi (lat/lon), Natural Earth, OpenStreetMap.

**Sections** :
1. Cadrer : projections, CRS, géodésie.
2. **GeoPandas** : Series + DataFrame géo.
3. **shapely** : Point, LineString, Polygon, opérations (intersect, buffer, union).
4. **Spatial joins** (within, intersects, nearest).
5. Indexation spatiale : R-tree, `rtree`, `pyrosm`.
6. **Visualisation** : folium (Leaflet), geoplot, plotly, kepler.gl.
7. **Géocodage** : `geopy` (Nominatim, Google).
8. Raster vs vector : `rasterio`, NetCDF, xarray.
9. **DuckDB spatial extension** + **PostGIS**.
10. ML géo : features (distance to X, density grids), GNN sur graphes routiers.
11. **Foursquare / Uber H3 hexagons** pour aggregation.
12. Sources.

### DS_Survival_Analysis
**Rôle** : DS · Wiki + Tutoriel. **Dataset** : Telco churn, ROSSi (recidivism).

**Sections** :
1. Pourquoi : time-to-event + censure (churn, défaillance, médical).
2. Kaplan-Meier survival curves.
3. Log-rank test (comparer 2 groupes).
4. **Cox Proportional Hazards** model.
5. Test de l'hypothèse PH (Schoenfeld residuals).
6. **AFT models** (Accelerated Failure Time).
7. **Random Survival Forest** (`scikit-survival`).
8. **DeepSurv** (NN-based Cox).
9. Compétitions : XGBoostSurvival, LightGBMLSS.
10. Évaluation : C-index, time-dependent AUC, calibration.
11. Lib : `lifelines`, `scikit-survival`.
12. Sources.

### DS_Recommender_Systems
**Rôle** : DS + AIE · Wiki + Tutoriel. **Dataset** : MovieLens, Amazon reviews.

**Sections** :
1. Cadre : explicit (ratings) vs implicit (clicks).
2. Baselines : popularité, item-item top similar.
3. **Collaborative filtering** : KNN-based.
4. **Matrix Factorization** : SVD, ALS, NMF.
5. **Implicit feedback** ALS (Hu et al.).
6. **Deep learning** : Neural CF, autoencoders.
7. **Two-tower models** (YouTube DNN, Cross-encoders).
8. **Sequential recommenders** : SASRec, BERT4Rec, Transformer4Rec.
9. **LLM-based recommenders** (2024-2026 : retrieve + rerank via LLM).
10. Évaluation : Precision@k, Recall@k, NDCG, MRR, Hit Rate.
11. Cold start, exploration/exploitation (bandits).
12. Libs : `implicit`, `lightfm`, `surprise`, `recbole`, `merlin`.
13. Sources.

---

## 2. Data Engineer (DE)

### DE_PySpark
**Rôle** : DE · Tutoriel. **Dataset** : NYC Taxi parquet (gros volume).

**Sections** :
1. Architecture Spark : driver, executors, RDD vs DataFrame vs Dataset.
2. **PySpark DataFrame API** : select, filter, groupBy, agg, join, window.
3. SQL via `spark.sql`.
4. UDFs (Python + Pandas UDF vectorisées).
5. **Partitioning** (parquet, bucketing).
6. **Catalyst** optimizer + EXPLAIN.
7. Performance : broadcast joins, shuffle, skew.
8. Streaming (Structured Streaming).
9. ML : `pyspark.ml` (Pipeline + Estimator/Transformer).
10. **Delta Lake** : ACID sur S3, time travel, schema evolution.
11. Comparatif Spark vs Polars vs DuckDB en 2026.
12. Sources.

### DE_Airflow_Prefect
**Rôle** : DE + MLOps · Wiki + Tutoriel. **Dataset** : pipeline ELT sur Cal Housing.

**Sections** :
1. Cadre orchestration : DAG, task, scheduler.
2. **Apache Airflow 2.x** : DAGs, Operators, Sensors, XCom.
3. **TaskFlow API** (Airflow 2+).
4. **Prefect 2/3** : flow & task, deployment, blocks.
5. **Dagster** : assets vs tasks, Software Defined Assets (SDA).
6. **Argo Workflows** (Kubernetes-native).
7. Comparatif Airflow vs Prefect vs Dagster en 2026.
8. Patterns : retry, backfill, branching, dynamic mapping.
9. Monitoring & alerting.
10. ML pipelines : combine MLflow + Airflow.
11. Best practices : idempotence, observabilité, secrets.
12. Sources.

### DE_Kafka_Streaming
**Rôle** : DE · Wiki + Tutoriel. **Dataset** : synthetic event stream.

**Sections** :
1. Cadre messaging : queues vs pub/sub.
2. Architecture **Kafka** : brokers, topics, partitions, consumers.
3. `confluent-kafka-python` ou `kafka-python` producer/consumer.
4. **Schema Registry** (Avro/Protobuf/JSON Schema).
5. **Kafka Streams** vs **ksqlDB**.
6. **Kafka Connect** : ingestion / sinks.
7. **Faust** (Python streaming framework).
8. Alternatives : Pulsar, Redpanda, RabbitMQ, NATS.
9. **Event-driven architecture** : CQRS, event sourcing.
10. Streaming ML : online learning + River.
11. Patterns : exactly-once, dedup, idempotence.
12. Sources.

### DE_dbt_Modeling
**Rôle** : DE · Wiki + Tutoriel. **Dataset** : NYC Taxi parquet via DuckDB ou Postgres.

**Sections** :
1. **dbt** (data build tool) : SQL-first ELT, version contrôlé.
2. Modèles : staging → intermediate → mart pattern.
3. Materialization : view, table, incremental, ephemeral.
4. Tests : unique, not null, accepted_values, relationships, custom.
5. Sources, freshness.
6. Macros + Jinja.
7. Documentation auto-générée.
8. **dbt** + DuckDB en local, dbt + Snowflake/BigQuery/Postgres prod.
9. **Snowplow / Segment** integration.
10. Comparatif dbt vs SQLMesh vs Dataform.
11. CI/CD pour dbt (GitHub Actions).
12. Sources.

### DE_Data_Quality
**Rôle** : DE + MLOps · Wiki + Tutoriel. **Dataset** : Titanic + Cal Housing.

**Sections** :
1. Définir la qualité : 6 dimensions (completeness, uniqueness, validity, accuracy, consistency, timeliness).
2. **Great Expectations 1.x** (2024+).
3. **Soda Core / Soda Cloud**.
4. **Pandera** : validation DataFrame typée par schema.
5. **Pydantic** pour structured data validation.
6. Profiling : `ydata-profiling`, `pandas-profiling` legacy.
7. **Data observability** : Monte Carlo, Anomalo, datafold.
8. **Schema drift** detection.
9. **Data contracts** : enforcement, versioning.
10. Tests dans le pipeline (Airflow tasks).
11. Lineage : OpenLineage, DataHub, Marquez.
12. Sources.

---

## 3. AI Engineer (AIE)

### AI_Local_LLMs
**Rôle** : AIE · Wiki + Tutoriel. **Dataset** : prompts test custom.

**Sections** :
1. Pourquoi local : confidentialité, coût, offline, latence.
2. Modèles 2026 : Llama 3.x, Mistral, Qwen 3, Gemma 3, Phi-4, SmolLM2.
3. **Ollama** : install + API REST + library Python.
4. **llama.cpp** + GGUF format.
5. **LM Studio** (UI desktop).
6. **vLLM** (serveur prod, batch + paged attention).
7. **SGLang** (alternative haute perf).
8. **MLX** (Apple Silicon natif).
9. **TGI** (Hugging Face Text Generation Inference).
10. Quantization : 4-bit GGUF Q4_K_M, AWQ, GPTQ.
11. Benchmarks throughput / latency / quality.
12. Sécurité prompts + guardrails (Guardrails AI, NeMo Guardrails).
13. Sources.

### AI_LLM_Finetuning_PEFT
**Rôle** : AIE + MLE · Wiki + Tutoriel. **Dataset** : Alpaca-fr ou Dolly-15k.

**Sections** :
1. Quand fine-tuner vs prompt engineering vs RAG.
2. **Types** : SFT (instruction), DPO (preference), reward modeling.
3. **PEFT** : LoRA / QLoRA / DoRA / AdaLoRA — théorie.
4. **Hugging Face TRL** : `SFTTrainer`, `DPOTrainer`.
5. **Unsloth** : 2-5× plus rapide que TRL pur, mémoire optimisée.
6. **Axolotl** : config YAML déclarative.
7. Pipeline complet : data prep → SFT → DPO → eval → quantize → deploy.
8. **Datasets** : format ChatML, ShareGPT, Alpaca.
9. Eval : LM Eval Harness, MT-Bench, AlpacaEval.
10. Pièges : catastrophic forgetting, mode collapse, length bias.
11. Synthetic data (auto-instruct, Evol-Instruct).
12. Sources.

### AI_Agents_Tools
**Rôle** : AIE · Wiki + Tutoriel. **Dataset** : tâches assistantes simulées.

**Sections** :
1. Cadre agent : LLM + tools + memory + planner + executor.
2. **Tool use** natif : OpenAI Function Calling, Anthropic Tools API, JSON Schema.
3. Frameworks : **LangGraph**, **LlamaIndex Agents**, **AutoGen**, **CrewAI**, **DSPy**.
4. **ReAct** pattern (Yao 2022).
5. **Reflexion** / self-critique.
6. **Multi-agent** : coordinator + workers.
7. **MCP** (Model Context Protocol) — 2024+ standard d'Anthropic pour les tools.
8. **Computer use** agents (Anthropic 2024).
9. Memory : short-term (context) vs long-term (vector DB).
10. Évaluation : task success rate, cost, latency.
11. Sécurité : sandbox, prompt injection, tool injection.
12. Cas d'usage : RAG agent, coding agent (Aider, Devin), web research agent.
13. Sources.

### AI_Prompt_Engineering
**Rôle** : AIE · Wiki + Tutoriel. **Dataset** : tâches NLP variées.

**Sections** :
1. Cadre : in-context learning, few-shot vs zero-shot.
2. Anatomie d'un bon prompt : role + task + examples + format + constraints.
3. **Chain-of-Thought** (Wei 2022).
4. **Self-consistency** + voting.
5. **Tree-of-Thought** (Yao 2023).
6. **Skeleton-of-Thought**.
7. **Constitutional AI** prompts.
8. **JSON mode** / structured output (Pydantic + Outlines + Instructor).
9. **Few-shot example selection** (semantic similarity).
10. **Meta-prompts** : prompt qui génère des prompts.
11. **DSPy** : programmer les prompts (compilation automatique).
12. Évaluation : `promptfoo`, `langfuse`, `langsmith`.
13. Anti-patterns : ambiguïté, contradictions, prompts trop longs.
14. Sources.

### AI_Multimodal_VLM
**Rôle** : AIE · Wiki + Tutoriel. **Dataset** : COCO captions, sample images.

**Sections** :
1. Cadre : vision + language joint (CLIP idée).
2. **CLIP / SigLIP / OpenCLIP** : embeddings alignés.
3. **VLMs génératifs** : LLaVA, Qwen-VL, Gemma 3, Pixtral, Molmo.
4. Document understanding : LayoutLM, Donut, ColPali.
5. **Video** : VideoLLaMA, InternVideo, Apollo.
6. Cas d'usage : captioning, VQA, OCR, table extraction, agent qui voit l'écran.
7. **Tool use multimodal** : Claude vision, GPT-4V.
8. Fine-tuning VLM (LoRA sur LLaVA).
9. Eval : MMMU, MMBench, ChartQA.
10. Libs : `transformers`, `unsloth`, `vllm` (support VLM).
11. Sources.

### AI_Speech_Audio
**Rôle** : AIE · Wiki + Tutoriel. **Dataset** : LibriSpeech samples.

**Sections** :
1. ASR (Automatic Speech Recognition) : **Whisper** (OpenAI), Conformer, NeMo.
2. **TTS** (Text-to-Speech) : XTTS, F5-TTS, OpenVoice, ElevenLabs.
3. **Speaker diarization** (qui parle quand) : pyannote.
4. **Voice cloning** (éthique).
5. Speech embeddings : wav2vec2, HuBERT, WavLM.
6. **Music generation** : MusicGen, AudioGen, Stable Audio.
7. **Speech-to-speech** : SeamlessM4T, GPT-4o voice mode.
8. Pipelines réels : meeting transcription, dubbing, audiobook.
9. Latence temps réel (vocal assistants).
10. Libs : `whisper`, `pyannote`, `coqui-tts`, `bark`.
11. Sources.

---

## 4. ML Engineer (MLE)

### MLE_Feature_Store
**Rôle** : MLE + MLOps · Wiki + Tutoriel. **Dataset** : Cal Housing batch + features streaming.

**Sections** :
1. Pourquoi un feature store : online vs offline, training-serving skew.
2. **Feast** (open-source) : entities, feature views, materialization.
3. **Tecton** / **Hopsworks** / **AWS Feature Store** (managés).
4. Online store (Redis, DynamoDB) vs offline (Parquet, BigQuery).
5. Point-in-time correctness (anti-leak).
6. Feature transformations : on-demand vs batch.
7. Streaming features (Kafka + Flink).
8. Discovery & gouvernance.
9. Integration MLflow.
10. Sources.

### MLE_Online_Inference
**Rôle** : MLE · Wiki + Tutoriel. **Dataset** : Cal Housing model.

**Sections** :
1. Latence cible : real-time (<100ms), near-real-time (<1s), batch.
2. Servers : **BentoML**, **Triton** (NVIDIA), **TorchServe**, **TF Serving**.
3. ONNX Runtime / TensorRT pour accélération.
4. Quantization int8 / fp16 / 4-bit (bnb).
5. Batching dynamique.
6. Caching de prédictions.
7. A/B testing modèles en prod (canary, shadow).
8. Monitoring latence (P50/P95/P99).
9. Auto-scaling Kubernetes (KServe, Seldon Core).
10. Cas LLMs : vLLM/SGLang spécifiques.
11. Sources.

### MLE_Model_Serving
**Rôle** : MLE + MLOps · Wiki + Tutoriel. **Dataset** : modèle ML pretrained.

**Sections** :
1. Packaging : container Docker, modèle artifact, dépendances.
2. **MLflow Models** format (pyfunc) : portabilité multi-framework.
3. **BentoML** vs **Cog** (Replicate) vs **Truss** (Baseten).
4. KServe / Seldon (Kubernetes).
5. Cloud SaaS : SageMaker, Vertex AI, Azure ML, Databricks Model Serving.
6. Serverless : AWS Lambda, GCP Cloud Run, Modal, Replicate.
7. **API design** : sync vs async, batching, streaming SSE.
8. Versioning : modèles, schemas, contrats.
9. Rollback strategies.
10. Multi-modèle serving (gateway pattern).
11. Sources.

### MLE_AB_Testing
**Rôle** : MLE + DS · Wiki + Tutoriel. **Dataset** : synthetic conversion data.

**Sections** :
1. Cadre A/B : hypothèse nulle, p-value, puissance, MDE.
2. Sample size calculator (binary, continuous outcomes).
3. **Sequential testing** vs **fixed horizon**.
4. **CUPED** (variance reduction).
5. Multi-arm bandits (cf `ML_Apprentissage_par_Renforcement`).
6. Bayesian A/B testing (cf DS_Bayesian).
7. Stratification, blocking.
8. Pièges : peeking, Simpson's paradox, network effects.
9. Outils : GrowthBook, Statsig, Optimizely.
10. Cas spéciaux : long-term effects, novelty, ad hoc.
11. Sources.

---

## 5. ML Ops (MLOps)

### MLOps_Pipelines_Airflow
**Rôle** : MLOps · Tutoriel. **Dataset** : Cal Housing pipeline ELT + ML.

**Sections** :
1. Workflow ML : ingestion → validation → preprocessing → train → eval → register → deploy.
2. DAG Airflow / Prefect flow d'un cycle complet.
3. Idempotence + retry strategy.
4. Conditional logic (branching).
5. Backfill historique.
6. Secrets management (Vault, AWS Secrets Manager).
7. Notifications (Slack on failure).
8. Combine avec MLflow (auto-register).
9. CI/CD du pipeline lui-même.
10. Comparatif avec **Kubeflow Pipelines**, **Metaflow**, **ZenML**.
11. Sources.

### MLOps_Model_Registry
**Rôle** : MLOps · Wiki + Tutoriel. **Dataset** : Cal Housing modèle.

**Sections** :
1. Pourquoi : versioning, traçabilité, rollback, governance.
2. **MLflow Registry** (déjà couvert dans `ML_MLFlow_Bench`) — focus production patterns.
3. **Weights & Biases Model Registry**.
4. **Sagemaker Model Registry**.
5. Aliases (@champion/@challenger/@staging) workflow détaillé.
6. Approval workflows.
7. Lineage : modèle → run → dataset → code commit.
8. Sécurité : RBAC, audit log.
9. Compliance : SBOM ML, datasheet, model card.
10. Sources.

### MLOps_Drift_Monitoring
**Rôle** : MLOps · Wiki + Tutoriel. **Dataset** : Cal Housing train vs simulated prod.

**Sections** :
1. Pourquoi monitorer : drift data, drift concept, dégradation perf.
2. Types de drift : covariate, label, concept.
3. Tests stat : KS test, Chi², PSI (Population Stability Index), Wasserstein.
4. **Evidently AI** : librairie 2024+ référence.
5. **NannyML** : performance estimation sans ground truth.
6. **WhyLabs / Arize / Fiddler** (SaaS).
7. Monitoring inference latency + business metrics.
8. Alerting (Slack, PagerDuty).
9. Stratégie de retraining : trigger-based vs scheduled.
10. Shadow deployment pour détecter avant rollout.
11. Sources.

### MLOps_CICD_GitHub_Actions
**Rôle** : MLOps · Tutoriel. **Dataset** : repo ML demo.

**Sections** :
1. CI/CD pour ML : différences vs CI/CD classique (datasets, modèles).
2. **GitHub Actions** workflows.
3. Steps types : lint → tests → train → eval → register → deploy.
4. **GitHub Actions** pour ML : CML (Continuous Machine Learning), DVC integration.
5. Self-hosted runners (GPU on demand).
6. Secrets & OIDC pour cloud.
7. Cache : pip, modèles, datasets.
8. Tests : data tests (Great Expectations), model tests (deepchecks).
9. **Comparatif** GitLab CI, Jenkins, CircleCI.
10. **Deployment gates** : reviewers requis, validation auto.
11. Sources.

---

## 6. Implémentation

Pour **chacun** des 24 nouveaux sujets ci-dessus, un fichier `Notebook_2026/plans/<NOM>.md` est créé contenant le **plan détaillé** (déjà rempli dans ce document) — prêt à être implémenté.

Les notebooks `Notebook_2026/plans/` sont des **squelettes** : titres + sections + descriptions + datasets à utiliser. Ils ne contiennent **pas encore le code complet** (volonté d'incremental delivery sans surcharger).

Pour exécuter chacun :

1. Choisir un sujet prioritaire (suggestion : commencer par AI_Local_LLMs ou MLOps_Drift_Monitoring — fort impact).
2. Implémenter selon le plan (idéalement 1-3h par notebook).
3. Smoke test + conversion ipynb + commit.

Voir [`00_status_notebooks_nouveaux.md`](00_status_notebooks_nouveaux.md) pour le suivi des 24 nouveaux.
