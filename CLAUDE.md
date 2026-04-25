# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Still_thinking** is a RAG (Retrieval-Augmented Generation) academic assistant. Users describe a research idea in any language and the system retrieves semantically similar arXiv papers from a local ChromaDB vector store, then uses an Ollama-hosted LLM (Qwen3:8b at UC3M) to generate a structured comparison: similarities, differences, novelty, and conclusion.

## Commands

### Backend (FastAPI)

```bash
cd backend-fastapi

# Install dependencies (note: langid and deep-translator are missing from requirements.txt)
pip install -r requirements.txt
pip install langid deep-translator

# Run dev server (from inside backend-fastapi/)
uvicorn main:app --reload --port 8000

# Docker build & run
docker build -t still-thinking-backend .
docker run -p 10000:10000 --env-file .env still-thinking-backend
```

Required `.env` in `backend-fastapi/`:
```
OLLAMA_API_KEY=<key>
OLLAMA_URL=https://yiyuan.tsc.uc3m.es   # optional
OLLAMA_MODEL=qwen3:8b                    # optional
```

### Frontend (Angular)

```bash
npm install
npm start          # dev server at http://localhost:4200
npm run build      # production build → dist/
npm test           # Vitest unit tests
```

### LLM / ChromaDB tooling

```bash
# Interactive CLI test against the live backend
cd backend-fastapi && python -m llm.test_llm

# Batch prompt experiments → notes/week1_prompt_tests.md
cd backend-fastapi && python -m llm.run_prompt_experiments

# Rebuild ChromaDB from papers_final.jsonl (one-time setup)
python backend-fastapi/scripts/build_chroma.py
```

## Architecture

### Request flow

```
User query (any language)
  → Angular ChatComponent → ChatService (HTTP POST /search)
  → FastAPI main.py:
      1. langid: detect language
      2. GoogleTranslator: translate to English if needed
      3. ChromaDB: semantic search (top-5 papers)
      4. guardrails.py: refuse if max similarity < 0.05
      5. prompts.py: build prompt from comparison_prompt.txt template
      6. UC3MClient (Ollama): Qwen3:8b inference (temperature=0.0)
      7. Return {answer (HTML), articles[], language}
  → Angular: strip <think>…</think> tags, render answer as innerHTML
             show article sidebar with "Ver resumen" button
  → optional: POST /summarize → per-article summary
```

### Key files

| File | Role |
|---|---|
| `backend-fastapi/main.py` | FastAPI app; `/search` and `/summarize` endpoints; full pipeline |
| `backend-fastapi/llm/client.py` | `UC3MClient` — Ollama wrapper with API key auth |
| `backend-fastapi/llm/prompts.py` | `build_comparison_prompt()` from template |
| `backend-fastapi/prompts/comparison_prompt.txt` | Prompt template; instructs LLM to output HTML |
| `backend-fastapi/scripts/search_chroma.py` | `search()` — ChromaDB query used by main.py |
| `backend-fastapi/llm/guardrails.py` | `should_refuse()` — relevance threshold check |
| `backend-fastapi/llm/language.py` | Language detection and name lookup |
| `src/app/components/chat/` | Main Angular UI: chat + article sidebar |
| `src/app/services/chat.service.ts` | HTTP calls to `/search` and `/summarize` |

### ChromaDB setup

- Collection name: `papers`
- Stored at `backend-fastapi/chroma_db/` (distributed as `chroma_db.zip`, gitignored)
- Document format: `"Title: {title}\nAbstract: {abstract}"`
- Uses ChromaDB's default sentence-transformers embedding (same for indexing and querying)
- Source data: `backend-fastapi/data/papers_final.jsonl` (~375k CS papers, gitignored)

## Known Issues

- `/summarize` endpoint uses a hardcoded inline prompt string; a proper template file (like `comparison_prompt.txt`) is still TODO.
- `langid` and `deep-translator` are not listed in `backend-fastapi/requirements.txt` — install manually.
- CORS in `main.py` only allows `http://localhost:4200`; update for any other deployment origin.
- `main.py` contains dead legacy functions (`call_llama_uc3m`, `assistant_logic`) that are never called.
- The Dockerfile targets port 10000 (Render deployment); local dev uses port 8000.
- The LLM outputs `<think>…</think>` chain-of-thought blocks — the Angular frontend strips these via regex before rendering.
- The frontend renders LLM responses as raw HTML (`[innerHTML]`) because the prompt instructs Qwen3 to use HTML formatting.
