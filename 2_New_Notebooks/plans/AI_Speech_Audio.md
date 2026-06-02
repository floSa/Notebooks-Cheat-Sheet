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
# 🔊 Speech & Audio AI
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : AI Engineer · Wiki + Tutoriel

**Dataset(s)** : LibriSpeech samples, librosa builtins

ASR, TTS, diarization, embeddings audio, music generation. Stack open 2026.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. ASR — Automatic Speech Recognition
<!-- #endregion -->

<!-- #region -->
**Whisper** (OpenAI 2022, multi-langues, ouvert), **Whisper-large-v3-turbo** (2024). Alternatives : Conformer (NeMo), Distil-Whisper. Streaming ASR.
<!-- #endregion -->

<!-- #region -->
## 2. TTS — Text-to-Speech
<!-- #endregion -->

<!-- #region -->
**XTTS** (Coqui, voice cloning), **F5-TTS** (Microsoft 2024), **OpenVoice** (MyShell). SaaS : ElevenLabs, PlayHT.
<!-- #endregion -->

<!-- #region -->
## 3. Speaker diarization
<!-- #endregion -->

<!-- #region -->
Qui parle quand. **pyannote** (référence), Whisper-X (Whisper + pyannote).
<!-- #endregion -->

<!-- #region -->
## 4. Voice cloning
<!-- #endregion -->

<!-- #region -->
Zero-shot voice cloning (XTTS-v2). **Considérations éthiques** : consent, deepfakes.
<!-- #endregion -->

<!-- #region -->
## 5. Speech embeddings
<!-- #endregion -->

<!-- #region -->
wav2vec2 (Meta), HuBERT (Meta), WavLM (Microsoft). Pour speaker verification, accent classification, language ID.
<!-- #endregion -->

<!-- #region -->
## 6. Music generation
<!-- #endregion -->

<!-- #region -->
**MusicGen** (Meta, conditionnel sur texte), **AudioGen**, **Stable Audio** (Stability AI), **Suno** (SaaS).
<!-- #endregion -->

<!-- #region -->
## 7. Speech-to-speech
<!-- #endregion -->

<!-- #region -->
**SeamlessM4T** (Meta), GPT-4o voice mode, Moshi (Kyutai). Latence ultra-faible.
<!-- #endregion -->

<!-- #region -->
## 8. Pipelines réels
<!-- #endregion -->

<!-- #region -->
Meeting transcription (Whisper + pyannote), real-time dubbing, audiobook generation, voice assistants embedded.
<!-- #endregion -->

<!-- #region -->
## 9. Latence temps réel
<!-- #endregion -->

<!-- #region -->
VAD (Voice Activity Detection), streaming inference, model quantization, batching.
<!-- #endregion -->

<!-- #region -->
## 10. Libs / outils 2026
<!-- #endregion -->

<!-- #region -->
`whisper` (OpenAI), `whisper-cpp`, `faster-whisper` (CTranslate2), `pyannote`, `coqui-tts`, `bark`, `audiocraft` (Meta), `librosa`.
<!-- #endregion -->

<!-- #region -->
## 11. Sources
<!-- #endregion -->

<!-- #region -->
- [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
- [pyannote-audio](https://github.com/pyannote/pyannote-audio)
- [Coqui-TTS](https://github.com/coqui-ai/TTS)
- [audiocraft (Meta)](https://github.com/facebookresearch/audiocraft)
<!-- #endregion -->
