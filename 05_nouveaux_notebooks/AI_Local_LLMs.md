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
# 🏠 LLMs en local — Ollama, vLLM, llama.cpp
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : AI Engineer · Wiki + Tutoriel

**Dataset(s)** : Prompts test custom

Exécuter des LLMs en local : confidentialité, coût, offline. Stack 2026 : Ollama, llama.cpp, vLLM, SGLang.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Pourquoi local
<!-- #endregion -->

<!-- #region -->
Confidentialité (santé, juridique), coût (vs API par token), offline (terrain), latence (pas de network), customisation totale.
<!-- #endregion -->

<!-- #region -->
## 2. Modèles open-weights 2026
<!-- #endregion -->

<!-- #region -->
Llama 3.x/4.x (Meta), Mistral / Mixtral, Qwen 3 (Alibaba), Gemma 3 (Google), Phi-4 (Microsoft), SmolLM2 (HF). Tailles : 0.5B → 405B.
<!-- #endregion -->

<!-- #region -->
## 3. Ollama — install + Python
<!-- #endregion -->

<!-- #region -->
`ollama pull llama3.2`. API REST OpenAI-compatible. Library Python officielle. Modelfiles pour custom system prompts.
<!-- #endregion -->

<!-- #region -->
## 4. llama.cpp + GGUF
<!-- #endregion -->

<!-- #region -->
Format GGUF (binaire compact, quantizé). `llama-cpp-python`. Run CPU/GPU/Metal/CUDA.
<!-- #endregion -->

<!-- #region -->
## 5. LM Studio (UI desktop)
<!-- #endregion -->

<!-- #region -->
Pour les non-devs ou debug. Catalog de modèles.
<!-- #endregion -->

<!-- #region -->
## 6. vLLM — serveur prod
<!-- #endregion -->

<!-- #region -->
Paged attention, continuous batching. Throughput 10-100× supérieur. OpenAI-compatible API.
<!-- #endregion -->

<!-- #region -->
## 7. SGLang — alternative haute perf
<!-- #endregion -->

<!-- #region -->
RadixAttention (cache shared prefixes). Native function calling. Encore mieux que vLLM sur certains workloads.
<!-- #endregion -->

<!-- #region -->
## 8. MLX (Apple Silicon)
<!-- #endregion -->

<!-- #region -->
Framework Apple optimisé pour M-series. `mlx-lm` pour LLMs.
<!-- #endregion -->

<!-- #region -->
## 9. TGI — Hugging Face
<!-- #endregion -->

<!-- #region -->
Text Generation Inference. Production-grade. Bonne integration HF Hub.
<!-- #endregion -->

<!-- #region -->
## 10. Quantization
<!-- #endregion -->

<!-- #region -->
4-bit GGUF Q4_K_M (sweet spot), AWQ (vLLM-compatible), GPTQ. Trade-off qualité vs taille / vitesse.
<!-- #endregion -->

<!-- #region -->
## 11. Benchmarks 2026
<!-- #endregion -->

<!-- #region -->
Throughput (tokens/s), latency (TTFT - Time To First Token), batch size scaling, RAM/VRAM utilisation, quality on MT-Bench/IFEval.
<!-- #endregion -->

<!-- #region -->
## 12. Sécurité prompts + guardrails
<!-- #endregion -->

<!-- #region -->
Prompt injection. NeMo Guardrails (NVIDIA), Guardrails AI, LLM Guard. Detection + sanitization.
<!-- #endregion -->

<!-- #region -->
## 13. Sources
<!-- #endregion -->

<!-- #region -->
- [Ollama docs](https://ollama.com/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [vLLM docs](https://docs.vllm.ai/)
- [SGLang docs](https://docs.sglang.ai/)
<!-- #endregion -->
