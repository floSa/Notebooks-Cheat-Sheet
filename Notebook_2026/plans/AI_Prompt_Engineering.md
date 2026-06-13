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
# 💬 Prompt Engineering — patterns 2026
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : AI Engineer · Wiki + Tutoriel

**Dataset(s)** : Tâches NLP variées (classif, extraction, raisonnement)

Concevoir des prompts efficaces : structures, CoT, few-shot, structured output, DSPy.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre
<!-- #endregion -->

<!-- #region -->
In-context learning : le LLM apprend sa tâche depuis le prompt. Zero-shot vs few-shot. Limites du context.
<!-- #endregion -->

<!-- #region -->
## 2. Anatomie d'un bon prompt
<!-- #endregion -->

<!-- #region -->
Role / persona + Task + Examples + Output format + Constraints. Template `you are X. Do Y. Format : Z`.
<!-- #endregion -->

<!-- #region -->
## 3. Chain-of-Thought (CoT)
<!-- #endregion -->

<!-- #region -->
Wei 2022 : 'Let's think step by step'. Augmente la perf sur raisonnement maths/logique.
<!-- #endregion -->

<!-- #region -->
## 4. Self-consistency
<!-- #endregion -->

<!-- #region -->
Wang 2022 : générer N CoT, voter sur la réponse majoritaire. Plus robuste mais plus cher.
<!-- #endregion -->

<!-- #region -->
## 5. Tree-of-Thought (ToT)
<!-- #endregion -->

<!-- #region -->
Yao 2023 : explorer plusieurs branches de raisonnement, pruning. Pour les problèmes combinatoires (jeu de 24, Sudoku).
<!-- #endregion -->

<!-- #region -->
## 6. Skeleton-of-Thought
<!-- #endregion -->

<!-- #region -->
Génère un plan, puis remplit chaque section en parallèle. Réduit la latence.
<!-- #endregion -->

<!-- #region -->
## 7. Constitutional AI
<!-- #endregion -->

<!-- #region -->
Anthropic : guide le LLM par des principes (constitution). Self-critique + révision.
<!-- #endregion -->

<!-- #region -->
## 8. Structured output
<!-- #endregion -->

<!-- #region -->
JSON mode (OpenAI), tool use (Anthropic). Libs : **Outlines**, **Instructor**, **Marvin** — forcent la sortie au schéma Pydantic via grammar-based decoding.
<!-- #endregion -->

<!-- #region -->
## 9. Few-shot example selection
<!-- #endregion -->

<!-- #region -->
Choisir les exemples par similarité sémantique au prompt (via embeddings).
<!-- #endregion -->

<!-- #region -->
## 10. Meta-prompts
<!-- #endregion -->

<!-- #region -->
Prompts qui génèrent des prompts. Auto-amélioration. APE (Automatic Prompt Engineer).
<!-- #endregion -->

<!-- #region -->
## 11. DSPy
<!-- #endregion -->

<!-- #region -->
Programmer les prompts comme du code. Signatures = type-system. Compilers optimisent automatiquement les prompts/few-shots.
<!-- #endregion -->

<!-- #region -->
## 12. Évaluation
<!-- #endregion -->

<!-- #region -->
`promptfoo` (regression tests), `langfuse` / `langsmith` (observabilité), `OpenAI Evals`, `LangSmith Datasets`.
<!-- #endregion -->

<!-- #region -->
## 13. Anti-patterns
<!-- #endregion -->

<!-- #region -->
Ambiguïté (« écris bien »), contradictions, prompts trop longs (lost in the middle), instructions négatives.
<!-- #endregion -->

<!-- #region -->
## 14. Sources
<!-- #endregion -->

<!-- #region -->
- [Prompt Engineering Guide — DAIR](https://www.promptingguide.ai/)
- [DSPy](https://dspy.ai/)
- [Instructor](https://python.useinstructor.com/)
- [Outlines](https://github.com/dottxt-ai/outlines)
<!-- #endregion -->
