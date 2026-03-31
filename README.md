# PageIndex RAG

Vectorless, reasoning-based RAG using the open-source [PageIndex](https://github.com/pageindex-ai/pageindex) library.
No PageIndex cloud API key required — all document indexing runs locally.

---

## How it works

1. Upload a PDF → backend indexes it locally using PageIndex (LLM builds a hierarchical tree)
2. Ask a question → LLM picks relevant sections from the tree → retrieves page text → generates answer
3. Indexed documents are cached in `./workspace/` so re-uploads are instant

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- An LLM API key: **OpenAI** (`OPENAI_API_KEY`) or **Anthropic** (`ANTHROPIC_API_KEY`)

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd pageIndex
```

### 2. Configure environment variables

Copy and edit the `.env` file:

```bash
cp .env .env.local   # optional, or edit .env directly
```

```env
# Provide ONE of these:
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional overrides (defaults shown):
# WORKSPACE_DIR=./workspace
# UPLOADS_DIR=./uploads
```

### 3. Install Python dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the project

You need **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend API

```bash
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## Usage

### Web UI

1. Open `http://localhost:5173`
2. Drag & drop or click to upload a PDF
3. Wait for indexing to complete (1–5 min depending on PDF size)
4. Ask questions in the chat

**Modes:**
- **Auto** — shows answer only
- **Manual** — shows retrieval reasoning + which sections were used

### CLI (no UI needed)

```bash
# One-shot question
python main.py report.pdf -q "What is the net revenue?"

# Interactive mode
python main.py report.pdf

# Manual mode (shows reasoning)
python main.py report.pdf --mode manual --verbose

# List all indexed documents
python main.py --list
```

---

## Project structure

```
pageIndex/
├── api.py              # FastAPI backend
├── main.py             # CLI interface
├── pyproject.toml      # Python dependencies
├── .env                # API keys (do not commit)
├── PageIndex/          # Local open-source PageIndex library
│   └── pageindex/      # Core indexing library
├── frontend/           # React + Vite UI
│   └── src/App.jsx
├── workspace/          # Auto-created: indexed document cache
├── uploads/            # Auto-created: uploaded PDFs
└── explain.md          # How PageIndex works internally
```

---

## Notes

- **First index is slow** — PageIndex makes many LLM calls to build the tree (20–50 calls per PDF). Subsequent runs use the cache.
- **Model for indexing** — configured in `PageIndex/pageindex/config.yaml` (default: `gpt-4o-2024-11-20`). Change to `anthropic/claude-sonnet-4-6` if using Anthropic.
- **Model for answering** — set by `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env`.
