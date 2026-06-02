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
# 🤖 LLM Agents et Tool Use
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : AI Engineer · Wiki + Tutoriel

**Dataset(s)** : Tâches assistantes simulées (web search, calcul, code)

Construire des agents LLM avec tool use : frameworks 2026, patterns, MCP, sécurité.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre agent
<!-- #endregion -->

<!-- #region -->
LLM (cerveau) + tools (mains) + memory + planner + executor. Loop : observe → think → act.
<!-- #endregion -->

<!-- #region -->
## 2. Tool use natif
<!-- #endregion -->

<!-- #region -->
OpenAI Function Calling, Anthropic Tools API, JSON Schema. Structured outputs. Limites et coûts.
<!-- #endregion -->

<!-- #region -->
## 3. Frameworks 2026
<!-- #endregion -->

<!-- #region -->
**LangGraph** (state machine), **LlamaIndex Agents**, **AutoGen** (Microsoft, multi-agent), **CrewAI** (rôles spécialisés), **DSPy** (programmatique).
<!-- #endregion -->

<!-- #region -->
## 4. ReAct pattern
<!-- #endregion -->

<!-- #region -->
Yao 2022 : reasoning + acting interleaved. Thought → Action → Observation → Thought... Le pattern fondateur.
<!-- #endregion -->

<!-- #region -->
## 5. Reflexion / self-critique
<!-- #endregion -->

<!-- #region -->
Shinn 2023 : agent qui critique sa propre réponse et itère. Améliore SWE-bench, AlfWorld.
<!-- #endregion -->

<!-- #region -->
## 6. Multi-agent
<!-- #endregion -->

<!-- #region -->
Coordinator + workers spécialisés. Patterns : assembly line, parliament, debate. CrewAI / AutoGen.
<!-- #endregion -->

<!-- #region -->
## 7. MCP — Model Context Protocol
<!-- #endregion -->

<!-- #region -->
Standard Anthropic 2024 pour les tools. Server exposant des fonctions, client (LLM) consommant. Découplage.
<!-- #endregion -->

<!-- #region -->
## 8. Computer use agents
<!-- #endregion -->

<!-- #region -->
Anthropic Claude computer use (oct 2024). L'agent voit l'écran et clique. GPT operator. Cas d'usage et limites.
<!-- #endregion -->

<!-- #region -->
## 9. Memory
<!-- #endregion -->

<!-- #region -->
Short-term (context window), long-term (vector DB, voir `BDD_Vectorielles`), episodic, semantic, procedural.
<!-- #endregion -->

<!-- #region -->
## 10. Évaluation
<!-- #endregion -->

<!-- #region -->
Task success rate (binary), cost ($/task), latency, robustness, tool selection accuracy. Benchmarks : SWE-bench, ToolBench, AgentBench, BFCL.
<!-- #endregion -->

<!-- #region -->
## 11. Sécurité
<!-- #endregion -->

<!-- #region -->
Sandbox tools, allowlist d'actions, validation outputs, prompt injection (jailbreak via tool output). LLM Guard, Prompt Shield.
<!-- #endregion -->

<!-- #region -->
## 12. Cas d'usage
<!-- #endregion -->

<!-- #region -->
RAG agent (retrieve dynamique), coding agent (Aider, Devin, Cursor), web research agent (Perplexity), data analyst agent.
<!-- #endregion -->

<!-- #region -->
## 13. Sources
<!-- #endregion -->

<!-- #region -->
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [MCP spec — Anthropic](https://modelcontextprotocol.io/)
- [AutoGen — Microsoft](https://github.com/microsoft/autogen)
- [DSPy](https://dspy.ai/)
- Notebook lié : `NLP_Recherche_d_informations`.
<!-- #endregion -->
