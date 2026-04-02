#!/usr/bin/env python3
"""
PageIndex RAG API — local mode (no PageIndex cloud API key required).

All document indexing is done locally using the open-source PageIndex library
in ./PageIndex. LLM calls (retrieval reasoning + answer generation) use
ANTHROPIC_API_KEY or OPENAI_API_KEY from .env.
"""

import os
import json
import sys
import uuid
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Use the local open-source PageIndex library
sys.path.insert(0, str(Path(__file__).parent / "PageIndex"))

load_dotenv(Path(__file__).resolve().parent / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
WORKSPACE_DIR     = os.getenv("WORKSPACE_DIR", "./workspace")
UPLOADS_DIR       = os.getenv("UPLOADS_DIR", "./uploads")

app = FastAPI(title="PageIndex RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton PageIndex client + persistent job tracking
# ─────────────────────────────────────────────────────────────────────────────

_pi_client_instance = None
_jobs: dict = {}  # job_id → {status, real_doc_id, name, page_count}


def _jobs_file_path() -> Path:
    p = Path(WORKSPACE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p / "jobs.json"


def _load_jobs():
    global _jobs
    path = _jobs_file_path()
    if path.exists():
        try:
            with open(path) as f:
                _jobs = json.load(f)
        except (json.JSONDecodeError, OSError):
            _jobs = {}


def _save_jobs():
    try:
        with open(_jobs_file_path(), "w") as f:
            json.dump(_jobs, f, indent=2)
    except OSError:
        pass


def _get_pi_client():
    global _pi_client_instance
    if _pi_client_instance is None:
        from pageindex import PageIndexClient
        _pi_client_instance = PageIndexClient(workspace=WORKSPACE_DIR)
    return _pi_client_instance


def _find_existing_doc(file_path: str) -> str | None:
    """Return real doc_id if file is already indexed in workspace, else None."""
    abs_path = os.path.abspath(file_path)
    pi = _get_pi_client()
    for doc_id, doc in pi.documents.items():
        if doc.get("path") == abs_path:
            return doc_id
    return None


def _resolve_doc_id(job_or_doc_id: str) -> str:
    """Resolve a job_id to a real workspace doc_id (or return as-is)."""
    if job_or_doc_id in _jobs:
        job = _jobs[job_or_doc_id]
        if job["status"] != "completed":
            raise HTTPException(400, f"Document not ready yet (status: {job['status']})")
        return job["real_doc_id"]
    return job_or_doc_id


@app.on_event("startup")
async def startup():
    _load_jobs()
    _get_pi_client()  # eager init so first request isn't slow


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers (Claude or OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    if ANTHROPIC_API_KEY:
        import anthropic
        ac = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = ac.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    if OPENAI_API_KEY:
        import openai
        oc = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = oc.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    raise HTTPException(
        status_code=500,
        detail="ANTHROPIC_API_KEY or OPENAI_API_KEY required in .env"
    )


def _strip_tree_text(obj):
    if isinstance(obj, dict):
        return {k: _strip_tree_text(v) for k, v in obj.items() if k != "text"}
    if isinstance(obj, list):
        return [_strip_tree_text(i) for i in obj]
    return obj


def _build_node_map(tree) -> dict:
    mapping: dict = {}
    def walk(node):
        if isinstance(node, dict):
            if "node_id" in node:
                mapping[node["node_id"]] = node
            for child in node.get("nodes", []):
                walk(child)
    if isinstance(tree, list):
        for root in tree:
            walk(root)
    else:
        walk(tree)
    return mapping


RETRIEVE_PROMPT = """\
You are given a question and a hierarchical tree index of a document.
Each node has: node_id, title, summary, and optionally child nodes.

Identify ALL node IDs whose content is likely to help answer the question.
Include parent nodes if their children are relevant.

Question:
{query}

Document tree:
{tree_json}

Reply ONLY with valid JSON (no markdown fences):
{{
  "thinking": "<concise reasoning about which nodes are relevant>",
  "node_list": ["node_id_1", "node_id_2"]
}}"""

ANSWER_PROMPT = """\
Answer the following question using ONLY the context provided.
If the context is insufficient, say so explicitly.

Question: {query}

Context:
{context}

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
# Background indexing task
# ─────────────────────────────────────────────────────────────────────────────

def _run_index(job_id: str, file_path: str, filename: str, user_id: str = ""):
    """Synchronous indexing — called in a thread pool via BackgroundTasks."""
    try:
        existing = _find_existing_doc(file_path)
        if existing:
            pi = _get_pi_client()
            doc = pi.documents.get(existing, {})
            _jobs[job_id] = {
                "status": "completed",
                "real_doc_id": existing,
                "name": filename,
                "page_count": doc.get("page_count"),
                "user_id": user_id,
            }
            _save_jobs()
            return

        pi = _get_pi_client()
        real_doc_id = pi.index(file_path)
        doc_info = json.loads(pi.get_document(real_doc_id))
        _jobs[job_id] = {
            "status": "completed",
            "real_doc_id": real_doc_id,
            "name": filename,
            "page_count": doc_info.get("page_count"),
            "user_id": user_id,
        }
        _save_jobs()
    except Exception as e:
        _jobs[job_id] = {
            "status": "failed",
            "real_doc_id": None,
            "name": filename,
            "page_count": None,
            "error": str(e),
            "user_id": user_id,
        }
        _save_jobs()


# ─────────────────────────────────────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(default=""),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save to a user-specific subdirectory to prevent filename collisions between users
    uploads = Path(UPLOADS_DIR)
    user_dir = uploads / user_id if user_id else uploads
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "real_doc_id": None, "name": file.filename, "page_count": None, "user_id": user_id}
    _save_jobs()

    background_tasks.add_task(_run_index, job_id, str(dest), file.filename, user_id)
    return {"doc_id": job_id, "name": file.filename}


@app.get("/api/status/{doc_id}")
async def get_status(doc_id: str):
    # Check jobs tracking first
    if doc_id in _jobs:
        job = _jobs[doc_id]
        return {
            "doc_id": doc_id,
            "status": job["status"],
            "pageNum": job.get("page_count"),
            "name": job.get("name", ""),
        }

    # Fall back to workspace (pre-existing indexed doc used directly)
    pi = _get_pi_client()
    doc = pi.documents.get(doc_id)
    if doc:
        return {
            "doc_id": doc_id,
            "status": "completed",
            "pageNum": doc.get("page_count"),
            "name": doc.get("doc_name", ""),
        }

    raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")


@app.get("/api/documents")
async def list_documents(limit: int = 20, user_id: str = ""):
    """List indexed documents, filtered by user_id when provided."""
    docs = []
    seen_real_ids = set()

    # Documents tracked by jobs — filter by owner when user_id is given
    for job_id, job in _jobs.items():
        if user_id and job.get("user_id", "") != user_id:
            continue
        docs.append({
            "id": job_id,
            "status": job["status"],
            "name": job.get("name", ""),
            "pageNum": job.get("page_count"),
        })
        if job.get("real_doc_id"):
            seen_real_ids.add(job["real_doc_id"])

    # Pre-existing workspace docs with no job entry — only shown when no user filter
    if not user_id:
        pi = _get_pi_client()
        for doc_id, doc in pi.documents.items():
            if doc_id not in seen_real_ids:
                docs.append({
                    "id": doc_id,
                    "status": "completed",
                    "name": doc.get("doc_name", ""),
                    "pageNum": doc.get("page_count"),
                })

    docs = docs[:limit]
    return {"documents": docs, "total": len(docs)}


@app.get("/api/tree/{doc_id}")
async def get_tree(doc_id: str):
    """Fetch the hierarchical tree index for a document."""
    real_doc_id = _resolve_doc_id(doc_id)
    pi = _get_pi_client()
    try:
        tree = json.loads(pi.get_document_structure(real_doc_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"tree": tree}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str = ""):
    """Remove a document job entry. Verifies ownership when user_id is provided."""
    if doc_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    job = _jobs[doc_id]

    if user_id and job.get("user_id", "") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    del _jobs[doc_id]
    _save_jobs()
    return {"deleted": True, "doc_id": doc_id}


# ─────────────────────────────────────────────────────────────────────────────
# Chat  (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    doc_id: str
    messages: list[dict]    # [{role, content}, ...]
    mode: str = "auto"      # "auto" | "manual"
    cite: bool = False      # kept for API compatibility; unused in local mode
    user_id: str = ""       # when set, restricts retrieval to this user's documents


async def _rag_stream(real_doc_id: str, messages: list, mode: str) -> AsyncGenerator[str, None]:
    """Local RAG pipeline as SSE stream."""
    pi = _get_pi_client()
    loop = asyncio.get_event_loop()
    question = messages[-1]["content"] if messages else ""

    # 1. Fetch document structure (tree without page text)
    try:
        raw = await loop.run_in_executor(None, lambda: pi.get_document_structure(real_doc_id))
        tree = json.loads(raw)
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    node_map = _build_node_map(tree)
    compact = _strip_tree_text(tree)

    # 2. LLM reasoning-based retrieval
    try:
        raw_retrieval = await loop.run_in_executor(None, lambda: _call_llm(
            RETRIEVE_PROMPT.format(query=question, tree_json=json.dumps(compact, indent=2))
        ))
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if raw_retrieval.startswith("```"):
        raw_retrieval = raw_retrieval.split("```")[1].lstrip("json").strip()

    try:
        retrieval = json.loads(raw_retrieval)
    except json.JSONDecodeError:
        retrieval = {"thinking": raw_retrieval, "node_list": []}

    thinking = retrieval.get("thinking", "")
    node_ids = retrieval.get("node_list", [])

    # Stream reasoning trace (manual mode)
    if mode == "manual" and thinking:
        yield f"data: {json.dumps({'reasoning': thinking})}\n\n"

    if not node_ids:
        yield f"data: {json.dumps({'text': 'No relevant sections found in the document.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 3. Retrieve page content for each relevant node
    sections = []
    context_parts: list[str] = []
    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            continue
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        page = f"{start}–{end}" if start != "?" else "?"
        sections.append({"node_id": nid, "title": node.get("title", ""), "page": page})
        if start != "?" and end != "?":
            try:
                s, e = start, end  # capture for lambda
                raw_content = await loop.run_in_executor(
                    None, lambda s=s, e=e: pi.get_page_content(real_doc_id, f"{s}-{e}")
                )
                pages = json.loads(raw_content)
                text = "\n".join(p["content"] for p in pages if p.get("content"))
                if text:
                    context_parts.append(text)
            except Exception:
                pass

    # Stream section metadata (manual mode)
    if mode == "manual" and sections:
        yield f"data: {json.dumps({'sections': sections})}\n\n"

    if not context_parts:
        yield f"data: {json.dumps({'text': 'Retrieved sections have no page content available.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 4. Generate answer
    context = "\n\n---\n\n".join(context_parts)
    try:
        answer = await loop.run_in_executor(None, lambda: _call_llm(
            ANSWER_PROMPT.format(query=question, context=context)
        ))
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 5. Stream answer word by word for smooth UX
    words = answer.split(" ")
    for i, word in enumerate(words):
        text = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'text': text})}\n\n"
        await asyncio.sleep(0.01)

    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-document RAG
# ─────────────────────────────────────────────────────────────────────────────

def _get_all_real_doc_ids(user_id: str = "") -> list[str]:
    """Return real doc_ids that are fully indexed, filtered by user_id when provided."""
    ids: dict = {}
    for job in _jobs.values():
        if user_id and job.get("user_id", "") != user_id:
            continue
        if job["status"] == "completed" and job.get("real_doc_id"):
            ids[job["real_doc_id"]] = job.get("name", "")
    if not user_id:
        pi = _get_pi_client()
        for doc_id, doc in pi.documents.items():
            if doc_id not in ids:
                ids[doc_id] = doc.get("doc_name", "")
    return list(ids.keys())


async def _rag_stream_multi(real_doc_ids: list[str], messages: list, mode: str) -> AsyncGenerator[str, None]:
    """RAG across all documents: retrieve from each in parallel, merge context, answer once."""
    pi = _get_pi_client()
    loop = asyncio.get_event_loop()
    question = messages[-1]["content"] if messages else ""

    # 1. Fetch all trees in parallel
    async def fetch_tree(doc_id):
        try:
            raw = await loop.run_in_executor(None, lambda: pi.get_document_structure(doc_id))
            return doc_id, json.loads(raw)
        except Exception:
            return doc_id, None

    tree_results = await asyncio.gather(*[fetch_tree(did) for did in real_doc_ids])
    trees = {doc_id: tree for doc_id, tree in tree_results if tree is not None}

    if not trees:
        yield f"data: {json.dumps({'text': 'No documents available.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. Run retrieval LLM call for each document in parallel
    async def retrieve_from_doc(doc_id, tree):
        compact = _strip_tree_text(tree)
        try:
            raw = await loop.run_in_executor(None, lambda: _call_llm(
                RETRIEVE_PROMPT.format(query=question, tree_json=json.dumps(compact, indent=2))
            ))
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            return doc_id, tree, json.loads(raw)
        except Exception:
            return doc_id, tree, {"thinking": "", "node_list": []}

    retrievals = await asyncio.gather(*[retrieve_from_doc(did, t) for did, t in trees.items()])

    # 3. Collect page content from relevant nodes across all docs
    all_context_parts: list[str] = []
    all_sections: list[dict] = []

    for doc_id, tree, retrieval in retrievals:
        doc_name = pi.documents.get(doc_id, {}).get("doc_name", doc_id)
        node_map = _build_node_map(tree)
        for nid in retrieval.get("node_list", []):
            node = node_map.get(nid)
            if not node:
                continue
            start = node.get("start_index", "?")
            end = node.get("end_index", "?")
            page = f"{start}–{end}" if start != "?" else "?"
            all_sections.append({"doc": doc_name, "node_id": nid, "title": node.get("title", ""), "page": page})
            if start != "?" and end != "?":
                try:
                    s, e = start, end
                    raw_content = await loop.run_in_executor(
                        None, lambda s=s, e=e: pi.get_page_content(doc_id, f"{s}-{e}")
                    )
                    pages = json.loads(raw_content)
                    text = "\n".join(p["content"] for p in pages if p.get("content"))
                    if text:
                        all_context_parts.append(f"[Source: {doc_name}]\n{text}")
                except Exception:
                    pass

    if mode == "manual" and all_sections:
        yield f"data: {json.dumps({'sections': all_sections})}\n\n"

    if not all_context_parts:
        yield f"data: {json.dumps({'text': 'No relevant content found across the documents.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 4. Single answer from merged context
    context = "\n\n---\n\n".join(all_context_parts)
    try:
        answer = await loop.run_in_executor(None, lambda: _call_llm(
            ANSWER_PROMPT.format(query=question, context=context)
        ))
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    words = answer.split(" ")
    for i, word in enumerate(words):
        text = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'text': text})}\n\n"
        await asyncio.sleep(0.01)

    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE endpoint — streams answer chunks as JSON events."""
    _SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    if req.doc_id == "__all__":
        doc_ids = _get_all_real_doc_ids(req.user_id)
        if not doc_ids:
            raise HTTPException(400, "No indexed documents found")
        return StreamingResponse(
            _rag_stream_multi(doc_ids, req.messages, req.mode),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    # Verify ownership when user_id is provided
    if req.user_id and req.doc_id in _jobs:
        if _jobs[req.doc_id].get("user_id", "") != req.user_id:
            raise HTTPException(403, "Document not found or access denied")

    real_doc_id = _resolve_doc_id(req.doc_id)
    return StreamingResponse(
        _rag_stream(real_doc_id, req.messages, req.mode),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}
