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
# 👁️ Vision-Language Models (VLMs)
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : AI Engineer · Wiki + Tutoriel

**Dataset(s)** : COCO captions, sample images, screenshots

Modèles multimodaux vision+texte : CLIP, LLaVA, Qwen-VL. Cas d'usage 2026.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre
<!-- #endregion -->

<!-- #region -->
Vision Encoder (ViT) + LLM. Espace latent partagé (CLIP-style) ou cross-attention (LLaVA-style).
<!-- #endregion -->

<!-- #region -->
## 2. CLIP / SigLIP / OpenCLIP
<!-- #endregion -->

<!-- #region -->
Embeddings alignés image/texte par contrastive learning. Zero-shot image classification, retrieval cross-modal.
<!-- #endregion -->

<!-- #region -->
## 3. VLMs génératifs
<!-- #endregion -->

<!-- #region -->
**LLaVA**, **Qwen-VL** (Alibaba), **Gemma 3** (multimodal natif), **Pixtral** (Mistral), **Molmo** (Allen AI), **SmolVLM** (HF).
<!-- #endregion -->

<!-- #region -->
## 4. Document understanding
<!-- #endregion -->

<!-- #region -->
LayoutLM, Donut (sans OCR), **ColPali** (retrieval natif sur PDFs scannés). Liens avec `DE_Docling`.
<!-- #endregion -->

<!-- #region -->
## 5. Video
<!-- #endregion -->

<!-- #region -->
VideoLLaMA, InternVideo, Apollo. Plus de difficulté car séquence temporelle.
<!-- #endregion -->

<!-- #region -->
## 6. Cas d'usage
<!-- #endregion -->

<!-- #region -->
Captioning, VQA, OCR-free document parsing, table extraction, chart understanding, agent qui voit l'écran (Claude computer use).
<!-- #endregion -->

<!-- #region -->
## 7. Tool use multimodal
<!-- #endregion -->

<!-- #region -->
Claude vision + tools, GPT-4V + function calling. Agent qui prend une screenshot et clique.
<!-- #endregion -->

<!-- #region -->
## 8. Fine-tuning
<!-- #endregion -->

<!-- #region -->
LoRA sur LLaVA / Qwen-VL via TRL `SFTTrainer` (support multimodal 2024+). Datasets : LLaVA-Instruct, ShareGPT-4V.
<!-- #endregion -->

<!-- #region -->
## 9. Évaluation
<!-- #endregion -->

<!-- #region -->
MMMU (Massive Multimodal Understanding), MMBench, ChartQA, DocVQA, RefCOCO.
<!-- #endregion -->

<!-- #region -->
## 10. Libs / serveurs 2026
<!-- #endregion -->

<!-- #region -->
`transformers` (HF support VLMs), `vllm` (support multimodal), `unsloth` (fine-tune VLMs), Ollama (LLaVA local).
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [LLaVA project](https://llava-vl.github.io/)
- [ColPali](https://github.com/illuin-tech/colpali)
- [Qwen-VL](https://huggingface.co/Qwen)
- [MMMU benchmark](https://mmmu-benchmark.github.io/)
<!-- #endregion -->
