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
# ⚡ Online Inference — Triton, BentoML, vLLM
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : ML Engineer · Wiki + Tutoriel

**Dataset(s)** : Cal Housing model + LLM model

Servir des modèles en temps réel : latence, throughput, batching, quantization.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Latence cible
<!-- #endregion -->

<!-- #region -->
Real-time (<100ms typique), near-real-time (<1s), batch (>1s OK). Définir SLO.
<!-- #endregion -->

<!-- #region -->
## 2. Servers 2026
<!-- #endregion -->

<!-- #region -->
**BentoML** (Python-native, multi-model), **NVIDIA Triton** (perf max, multi-framework), **TorchServe**, **TensorFlow Serving**.
<!-- #endregion -->

<!-- #region -->
## 3. ONNX Runtime / TensorRT
<!-- #endregion -->

<!-- #region -->
Conversion modèle → ONNX → ONNX Runtime (CPU/GPU portable). TensorRT (NVIDIA-only, fastest). OpenVINO (Intel).
<!-- #endregion -->

<!-- #region -->
## 4. Quantization
<!-- #endregion -->

<!-- #region -->
int8 (4× plus rapide souvent), fp16, bf16, 4-bit (LLMs). Outils : `bitsandbytes`, `optimum`, `auto-gptq`.
<!-- #endregion -->

<!-- #region -->
## 5. Batching dynamique
<!-- #endregion -->

<!-- #region -->
Combiner N requests entrantes en une batch pour mieux utiliser le GPU. Triton dynamic batching, vLLM continuous batching.
<!-- #endregion -->

<!-- #region -->
## 6. Caching de prédictions
<!-- #endregion -->

<!-- #region -->
Hash des inputs, Redis. Énorme gain si même requête répétée (search query, image identique).
<!-- #endregion -->

<!-- #region -->
## 7. A/B testing en prod
<!-- #endregion -->

<!-- #region -->
Canary (1 % traffic), shadow deployment (vrai model + nouveau en parallèle, comparer), blue-green.
<!-- #endregion -->

<!-- #region -->
## 8. Monitoring latence
<!-- #endregion -->

<!-- #region -->
P50, P95, P99. Histogrammes Prometheus + Grafana. Alerting si dégradation.
<!-- #endregion -->

<!-- #region -->
## 9. Auto-scaling Kubernetes
<!-- #endregion -->

<!-- #region -->
KServe (CNCF), Seldon Core, Ray Serve. HPA sur métriques custom (request rate, GPU utilization).
<!-- #endregion -->

<!-- #region -->
## 10. Cas LLMs
<!-- #endregion -->

<!-- #region -->
vLLM / SGLang / TGI : paged attention, prefix caching, speculative decoding. KV cache management.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [BentoML docs](https://docs.bentoml.com/)
- [NVIDIA Triton docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [KServe](https://kserve.github.io/website/)
- Notebook lié : `MLE_Model_Serving`.
<!-- #endregion -->
