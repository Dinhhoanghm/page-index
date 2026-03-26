#!/usr/bin/env python3

import os
import json
import tempfile
import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="PageIndex RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _pi_client():
    if not PAGEINDEX_API_KEY:
        raise HTTPException(status_code=500, detail="PAGEINDEX_API_KEY not configured")
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


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


def _call_llm(prompt: str) -> str:
    if ANTHROPIC_API_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    if OPENAI_API_KEY:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    raise HTTPException(
        status_code=500,
        detail="Manual mode requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
    )


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    client = _pi_client()

    # Save to a temp file, upload, then clean up
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        result = client.submit_document(tmp_path)
        doc_id = result["doc_id"]
    finally:
        os.unlink(tmp_path)

    return {"doc_id": doc_id, "name": file.filename}

@app.get("/api/status/{doc_id}")
async def get_status(doc_id: str):
    client = _pi_client()
    try:
        doc = client.get_document(doc_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "doc_id": doc_id,
        "status": doc.get("status", "unknown"),
        "pageNum": doc.get("pageNum"),
        "name": doc.get("name", ""),
    }


@app.get("/api/documents")
async def list_documents(limit: int = 20):
    """List all uploaded documents."""
    client = _pi_client()
    result = client.list_documents(limit=limit)
    docs = result.get("data") or result.get("documents") or []
    return {"documents": docs, "total": result.get("total", len(docs))}



@app.get("/api/tree/{doc_id}")
async def get_tree(doc_id: str):
    """Fetch the hierarchical tree index for a document."""
    client = _pi_client()
    try:
        result = client.get_tree(doc_id, node_summary=True)
        tree = result["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"tree": tree}


# ─────────────────────────────────────────────────────────────────────────────
# Chat  (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    doc_id: str
    messages: list[dict]          # [{role, content}, ...]
    mode: str = "auto"            # "auto" | "manual"
    cite: bool = False            # auto mode: include citations


async def _auto_stream(doc_id: str, messages: list, cite: bool) -> AsyncGenerator[str, None]:
    """Stream chunks from PageIndex chat_completions."""
    client = _pi_client()
    loop = asyncio.get_event_loop()

    def _run_sync():
        chunks = []
        for chunk in client.chat_completions(
            messages=messages,
            doc_id=doc_id,
            stream=True,
            enable_citations=cite,
        ):
            # SDK yields plain strings when streaming
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, dict):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content") or ""
            else:
                text = ""
            if text:
                chunks.append(text)
        return chunks

    # Run sync SDK in a thread so we don't block the event loop
    chunks = await loop.run_in_executor(None, _run_sync)
    for text in chunks:
        yield f"data: {json.dumps({'text': text})}\n\n"
    yield "data: [DONE]\n\n"


async def _manual_stream(doc_id: str, messages: list) -> AsyncGenerator[str, None]:
    """
    Manual mode: fetch tree, use LLM to pick nodes, stream answer token-by-token.
    Yields reasoning first, then the answer.
    """
    client = _pi_client()
    loop = asyncio.get_event_loop()

    # Fetch tree
    try:
        result = await loop.run_in_executor(
            None, lambda: client.get_tree(doc_id, node_summary=True)
        )
        tree = result["result"]
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    question = messages[-1]["content"] if messages else ""
    node_map = _build_node_map(tree)
    compact = _strip_tree_text(tree)

    # Step 1 — reasoning-based retrieval
    raw = await loop.run_in_executor(
        None,
        lambda: _call_llm(RETRIEVE_PROMPT.format(
            query=question,
            tree_json=json.dumps(compact, indent=2),
        ))
    )

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        retrieval = json.loads(raw)
    except json.JSONDecodeError:
        retrieval = {"thinking": raw, "node_list": []}

    thinking = retrieval.get("thinking", "")
    node_ids = retrieval.get("node_list", [])

    # Send reasoning trace
    if thinking:
        yield f"data: {json.dumps({'reasoning': thinking})}\n\n"

    if not node_ids:
        yield f"data: {json.dumps({'text': 'No relevant sections found in the document.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Send retrieved sections metadata
    sections = []
    context_parts: list[str] = []
    for nid in node_ids:
        node = node_map.get(nid)
        if node is None:
            continue
        page = (node.get("page_index")
                or f"{node.get('start_index','?')}–{node.get('end_index','?')}")
        sections.append({"node_id": nid, "title": node.get("title", ""), "page": page})
        if "text" in node:
            context_parts.append(node["text"])

    if sections:
        yield f"data: {json.dumps({'sections': sections})}\n\n"

    if not context_parts:
        yield f"data: {json.dumps({'text': 'Retrieved node summaries only — no full text available.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Step 2 — generate answer
    context = "\n\n---\n\n".join(context_parts)
    answer = await loop.run_in_executor(
        None,
        lambda: _call_llm(ANSWER_PROMPT.format(query=question, context=context))
    )

    # Stream answer word by word for a nicer UX
    words = answer.split(" ")
    for i, word in enumerate(words):
        text = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'text': text})}\n\n"
        await asyncio.sleep(0.01)

    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE endpoint — streams answer chunks as JSON events."""
    if req.mode == "auto":
        generator = _auto_stream(req.doc_id, req.messages, req.cite)
    else:
        generator = _manual_stream(req.doc_id, req.messages)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}
