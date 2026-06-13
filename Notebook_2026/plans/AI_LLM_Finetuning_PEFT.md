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
# 🎚️ LLM Fine-tuning — PEFT, LoRA, DPO
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : AI Engineer + ML Engineer · Wiki + Tutoriel

**Dataset(s)** : Alpaca-fr ou Dolly-15k

Fine-tuning de LLMs : quand, comment, avec quels outils 2026 (PEFT, TRL, Unsloth, Axolotl).

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Quand fine-tuner
<!-- #endregion -->

<!-- #region -->
Decision tree : prompt engineering → RAG → fine-tune. Cas qui justifient fine-tune : style spécifique, format strict, latence (modèle plus petit), confidentialité.
<!-- #endregion -->

<!-- #region -->
## 2. Types de fine-tuning
<!-- #endregion -->

<!-- #region -->
**SFT** (Supervised Fine-Tuning, instruction-tuning), **Reward Modeling**, **DPO** (Direct Preference Optimization), RLHF (PPO), distillation.
<!-- #endregion -->

<!-- #region -->
## 3. PEFT : LoRA / QLoRA / DoRA / AdaLoRA
<!-- #endregion -->

<!-- #region -->
LoRA : matrices low-rank `ΔW = B A`. QLoRA : LoRA + base model quantizé 4-bit. DoRA : amélioration récente (magnitude + direction). AdaLoRA : rang adaptatif.
<!-- #endregion -->

<!-- #region -->
## 4. Hugging Face TRL
<!-- #endregion -->

<!-- #region -->
Library officielle. `SFTTrainer`, `DPOTrainer`, `ORPOTrainer`. Integration `transformers`.
<!-- #endregion -->

<!-- #region -->
## 5. Unsloth — 2-5× plus rapide
<!-- #endregion -->

<!-- #region -->
Optimisations mémoire + kernels Triton custom. Permet de fine-tuner Llama 70B sur 1 GPU 48 GB.
<!-- #endregion -->

<!-- #region -->
## 6. Axolotl — config YAML
<!-- #endregion -->

<!-- #region -->
Wrapper TRL avec config déclarative. Standard pour les fine-tunes communautaires.
<!-- #endregion -->

<!-- #region -->
## 7. Pipeline complet
<!-- #endregion -->

<!-- #region -->
Data prep (ChatML format) → SFT → DPO → eval → quantize (GGUF/AWQ) → deploy (vLLM/Ollama).
<!-- #endregion -->

<!-- #region -->
## 8. Datasets formats
<!-- #endregion -->

<!-- #region -->
ChatML : `{messages: [{role, content}]}`. ShareGPT, Alpaca, OpenAssistant. Synthetic via Evol-Instruct.
<!-- #endregion -->

<!-- #region -->
## 9. Eval
<!-- #endregion -->

<!-- #region -->
LM Eval Harness (200+ tasks), MT-Bench (chat quality), AlpacaEval (head-to-head vs gpt-4), IFEval (instruction following), domain-specific.
<!-- #endregion -->

<!-- #region -->
## 10. Pièges
<!-- #endregion -->

<!-- #region -->
Catastrophic forgetting, mode collapse (toutes les réponses identiques), length bias (DPO favorise long), data leakage train/eval.
<!-- #endregion -->

<!-- #region -->
## 11. Synthetic data
<!-- #endregion -->

<!-- #region -->
Self-Instruct, Evol-Instruct, Distilabel (Argilla), generating from a strong teacher LLM.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [PEFT docs](https://huggingface.co/docs/peft)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- Notebook lié : `NLP_Transformers`.
<!-- #endregion -->
