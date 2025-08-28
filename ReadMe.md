# 🔮 CompareLLM

A **multi-provider AI playground** for running, comparing, and embedding across different LLMs in real time. Built with **Next.js (frontend)** + **FastAPI (backend)** + **Docker** for experimentation with Anthropic, OpenAI, DeepSeek, Ollama, Gemini, Cerebras, Cohere, and more.

---

## ✨ Features

* **Multi-Model Chat Comparison**

  * Run prompts across multiple LLMs in parallel
  * Streaming NDJSON responses for real-time output
  * Retry, per-model chat continuation, fullscreen & resizable modals

* **Embeddings & Semantic Search**

  * Upload datasets (JSON) and embed with multiple models simultaneously
  * Single-model or multi-model similarity search & side-by-side comparison
  * Powered by **pgvector** for fast vector search

* **Rich Frontend UX**

  * Tailwind + shadcn components, resizable panels, draggable lists
  * Dark mode, dynamic provider badges, Markdown rendering

* **Flexible Provider Support**

  * **Anthropic (Claude)**, **OpenAI (GPT-5, GPT-4o)**, **DeepSeek**, **Ollama local models**, **Google Gemini**, **Cerebras**, **Cohere**, **Voyage AI**
  * Embedding models across providers for evaluation & retrieval
  * Users can easily extend support: just add the model or embedding model name in `config/models.yaml` under the relevant provider section.

* **Enterprise-Ready Architecture**

  * Modular adapters for each provider
  * Configurable via `models.yaml`
  * Dockerized for reproducibility & team onboarding

---

## 🛠️ Tech Stack

* **Frontend:** Next.js 14, React, TailwindCSS, shadcn/ui
* **Backend:** FastAPI, Python 3.11
* **Infra:** Docker Compose (multi-container), Node, Yarn
* **Providers:** Anthropic, OpenAI, DeepSeek, Gemini, Ollama, Cerebras, Cohere, Voyage

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/DimitarAtanassov/CompareLLM.git
cd CompareLLM
```

### 2. Edit `.env`

Open the newly created `.env` file and adjust the values to match your API keys and system setup:

```env
# ======================
# Core API Config
# ======================
MODELS_CONFIG=/config/models.yaml
LOG_LEVEL=INFO

# ======================
# OpenAI (Chat + Embeddings)
# ======================
# Example: sk-proj-...
OPENAI_API_KEY=sk-proj-...

# ======================
# DeepSeek (Chat)
# ======================
# Example: sk-...
DEEPSEEK_API_KEY=sk-...

# ======================
# Google Gemini (Chat)
# ======================
# Example: AIza...
GOOGLE_API_KEY=AIza...

# ======================
# Anthropic Claude (Chat)
# ======================
# Example: sk-ant-...
ANTHROPIC_API_KEY=sk-ant-...

# ======================
# Voyage AI (Embeddings - Premium)
# ======================
# Example: pa-...
VOYAGE_API_KEY=pa-...

# ======================
# Cohere (Chat + Embeddings)
# ======================
# Example: ...
COHERE_API_KEY=...

# ======================
# Cerebras (Chat)
# ======================
# Example: csk-...
CEREBRAS_API_KEY=csk-...

# ======================
# Ollama (Local Models - Chat + Embeddings)
# ======================
# No API key needed for local models, just ensure Ollama is running
# (ollama serve inside the container or on host)
```

⚠️ **Note:**

* Replace placeholder keys with your real provider API keys.
* Adjust Ollama platform under `docker-compose.yml` depending on your system:

  * `linux/arm64/v8` → Apple Silicon (M1/M2/M3)
  * `linux/amd64` → Intel/AMD

### 3. Run with Docker

```bash
docker compose up --build
```

This will start:

* `api` → FastAPI backend
* `ui` → Next.js frontend
* `ollama` → local LLM runner (if enabled)
* `model-puller` → automatically pulls Ollama models from `models.yaml`

### 4. Open the UI

Visit [http://localhost:3000](http://localhost:3000)

---

## 📊 Example Workflows

### Chat

1. Select multiple models in the **Model List**
2. Enter a prompt and hit **Run**
3. Compare responses side-by-side, expand any model, or open interactive chat

### Embeddings

1. Upload JSON documents with IDs and text
2. Run **single-model search** or **multi-model comparison**
3. Inspect cosine similarity, metadata, and raw JSON results

📥 **Expected Dataset Format (JSON):**

```json
[
  {
    "ticker": "AAPL",
    "title": "Apple Inc. Teases New Product Launch with AI and In-House Modem",
    "link": "https://finance.yahoo.com/news/apple-inc-aapl-teases-product-182528020.html",
    "summary": "Apple plans to launch a new product on Feb. 19 featuring Apple Intelligence and an in-house modem chip."
  },
  {
    "ticker": "MSFT",
    "title": "Microsoft-Backed OpenAI Introduces SWE-Lancer Benchmark",
    "link": "https://finance.yahoo.com/news/microsoft-backed-openai-introduces-swe-182754256.html",
    "summary": "OpenAI, supported by Microsoft, introduces SWE-Lancer benchmark with 1,400+ tasks for evaluating AI performance."
  }
]
```

**Dataset Structure Expectations**

* Each dataset must be a list of JSON objects.
* Each object can contain metadata fields (e.g., `ticker`, `title`, `link`).
* One of the keys must hold the text you want to embed (e.g., `summary`).

**Frontend Usage**

* When uploading a dataset, the **Embed Text** field must be set to the key that contains the text you want embedded (e.g., `summary`).
* If multiple keys exist, only the selected one will be used for embeddings.

---

## 📸 Screenshots

### 1. Prompt Input & Multi-Model Comparison

![Prompt Input](./prompt_in.png)

### 2. Model Completion Results

![Completion Results](./completion.png)

### 3. Interactive Single Model Chat

![Single Model Interaction](./singleModelInteraction.png)

### 4. Side-by-Side Embedding Comparison

![Side-by-Side Embedding Comparison](./side_by_side_embed.png)

### 5. Single-Model Embedding Search

![Single-Model Embedding Search](./single_embed_search.png)

---

## 🧩 Project Structure

```
backend/
  main.py              # FastAPI entry
  adapters/            # Provider adapters (OpenAI, Anthropic, etc.)
  services/            # Chat + embedding services
  config/              # models.yaml
frontend/
  app/components/      # React components (Chat, Embeddings, Modals)
  lib/                 # Types & utils
docker-compose.yml     # Multi-service stack
```

---

## 📜 License

MIT License © 2025 AskManyLLMs Contributors

---

⚡ Ready for experimentation, benchmarking, and enterprise AI workflows.
