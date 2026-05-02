# Building AI Agent Systems That Keep Your Data Safe

**CABS AI Productivity Series** — Workshop on LangChain, LangGraph & Local LLM Deployment

A hands-on workshop designed for biologists who want to understand how AI agent
systems work, how to build them, and how to make informed decisions about data
privacy when using LLMs in research.

---

## What You'll Learn

1. **RAG (Retrieval-Augmented Generation)** — Make an LLM answer questions based on your own scientific literature, not its training data
2. **AI Agents with Tools** — Build an agent that autonomously queries UniProt and AlphaFold to look up gene and protein structure information
3. **Local LLM Deployment** — Run the same agent on your own machine with Ollama, so your data never leaves your computer

---

## Quick Start

Notebooks 01 and 02 run in Google Colab — no local setup needed. Click to open:

| Notebook | Topic                           | Run in Colab                                                                                                                                                                                                                              |
| -------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01       | RAG — Literature Q&A            | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lecaibio/cabs-workshop-llm-agents/blob/main/notebooks/01_rag_agent_ask_questions_about_scientific_literature.ipynb) |
| 02       | Agent — Gene & Structure Lookup | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lecaibio/cabs-workshop-llm-agents/blob/main/notebooks/02_gene_lookup_agent.ipynb)                                   |
| 03       | Local Deployment with Ollama    | [View guide](notebooks/03_local_llm_deployment_with_ollama.md)                                                                                                                                                                            |

---

## Repository Contents

```
cabs-workshop-llm-agents/
├── README.md
├── notebooks/
│ ├── 01_rag_literature_qa.ipynb ← RAG over organ-on-a-chip papers (Colab)
│ ├── 02_gene_lookup_agent.ipynb ← Agent with UniProt + AlphaFold tools (Colab)
│ ├── 03_local_ollama_setup.md ← Local deployment guide (read + follow)
│ └── local_agent.py ← Standalone local agent script
└── slides/
  └── cabs_workshop_llm_agents.pdf ← Workshop slide deck
```

---

## Prerequisites

**For Notebooks 01 & 02 (cloud, Colab):**

- A Google account
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com) (no credit card needed)

**For Notebook 03 (local):**

- A Mac, Linux, or Windows machine with at least 8 GB RAM
- [Ollama](https://ollama.com) installed
- A model pulled: `ollama pull llama3.1:8b`

No prior experience with LangChain, LangGraph, or Ollama is required. Basic Python familiarity is helpful.

---

## Workshop Slides

📎 [cabs_workshop_llm_agents.pdf](slides/cabs_workshop_llm_agents.pdf)

---

## Data Privacy: Cloud vs. Local

A central theme of this workshop is understanding where your data goes when you use an LLM.

|                       | Cloud API (Notebooks 01–02)       | Local LLM (Notebook 03)              |
| --------------------- | --------------------------------- | ------------------------------------ |
| Your prompts and data | Sent to Google's servers          | Stay on your machine                 |
| Suitable for          | Public data, published literature | Unpublished data, sensitive research |
| Model quality         | Higher (larger models)            | Lower (limited by your hardware)     |
| Internet required     | Yes                               | No (for LLM inference)               |

The workshop demonstrates that switching between cloud and local is a **one-line code change** — the architecture is identical. The decision is about your data sensitivity and hardware, not about rewriting your code.

---

## About

This workshop is part of the [CABS](https://www.cabsweb.org/) AI Productivity Series, created to help biologists build practical AI skills with an emphasis on data security and scientific applications.
