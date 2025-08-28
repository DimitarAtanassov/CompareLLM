# 🔮 CompareLLM

A **multi-provider AI playground** for running, comparing, and embedding across different LLMs in real time.
Built with **Next.js (frontend)** + **FastAPI (backend)** + **Docker** for experimentation with Anthropic, OpenAI, DeepSeek, Ollama, Gemini, Cerebras, Cohere, and more.

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

Perfect 🔥 — I’ll update the **Quick Start** section of your README to include an explicit step about editing `.env` with those API keys and Ollama config. This makes onboarding crystal clear for any developer or recruiter browsing the repo.

Here’s the updated section:

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/your-org/askmanyllms.git
cd askmanyllms
cp .env.example .env
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
OPENAI_API_KEY=sk-proj-...

# ======================
# DeepSeek (Chat)
DEEPSEEK_API_KEY=sk-...

# ======================
# Google Gemini (Chat)
GOOGLE_API_KEY=AIza...

# ======================
# Anthropic Claude (Chat)
ANTHROPIC_API_KEY=sk-ant-...

# ======================
# Voyage AI (Embeddings)
VOYAGE_API_KEY=pa-...

# ======================
# Cohere (Chat + Embeddings)
COHERE_API_KEY=...

# ======================
# Ollama (Local Models - no key required)
# Ensure Ollama runs inside container or host
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