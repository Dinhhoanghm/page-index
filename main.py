#!/usr/bin/env python3
"""
PageIndex RAG — Vectorless, Reasoning-based RAG
Ask questions about any PDF using PageIndex.

Two modes
─────────
auto   (default) — PageIndex runs the full RAG pipeline and streams the answer.
                   No extra LLM key needed; supports citations.
manual           — Fetches the tree index, uses Claude/OpenAI to pick relevant
                   nodes, then generates the answer. More transparent.

Usage
─────
  python main.py report.pdf                          # interactive, auto mode
  python main.py report.pdf --mode manual            # interactive, manual mode
  python main.py report.pdf -q "What is revenue?"   # one-shot
  python main.py report.pdf -q "..." --cite          # with page citations (auto mode)
  python main.py report.pdf -q "..." -v              # show retrieval reasoning (manual)
  python main.py --list                              # list all uploaded documents
"""

import os
import json
import time
import argparse
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# PageIndex client
# ─────────────────────────────────────────────────────────────────────────────

def _pi_client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY not set.\n"
            "Get your key at https://dash.pageindex.ai/api-keys\n"
            "Then add it to your .env file."
        )
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Document upload & caching
#
# Cache file stores { "doc_id": "pi-xxx", "tree": {...} }
# so we never re-upload the same PDF twice.
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(pdf_path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{Path(pdf_path).stem}.json")


def _load_cache(cache_file: str) -> dict:
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {}


def _save_cache(cache_file: str, data: dict) -> None:
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


def upload_and_wait(pdf_path: str, cache_dir: str) -> tuple[str, dict | None]:
    """
    Upload pdf_path to PageIndex (if not already done) and wait for processing.
    Returns (doc_id, tree_or_None).
    tree is only populated if it was already cached from a manual-mode run.
    """
    cache_file = _cache_path(pdf_path, cache_dir)
    cache = _load_cache(cache_file)

    if "doc_id" in cache:
        doc_id = cache["doc_id"]
        # Verify it's still valid on the server
        client = _pi_client()
        try:
            status = client.get_document(doc_id).get("status", "")
        except Exception:
            status = ""

        if status == "completed":
            print(f"[cache] Using existing doc_id: {doc_id}")
            return doc_id, cache.get("tree")
        elif status in ("processing", "queued"):
            print(f"[index] Document still processing ({doc_id}) …", end="", flush=True)
            _wait_for_completion(client, doc_id)
            return doc_id, cache.get("tree")
        else:
            # doc_id expired or invalid — re-upload
            print(f"[index] Cached doc_id expired, re-uploading …")

    client = _pi_client()
    print(f"[index] Uploading '{pdf_path}' to PageIndex …")
    result = client.submit_document(pdf_path)
    doc_id = result["doc_id"]
    print(f"[index] Document ID: {doc_id}")

    _wait_for_completion(client, doc_id)

    # Save doc_id to cache immediately (tree added later in manual mode)
    cache["doc_id"] = doc_id
    _save_cache(cache_file, cache)

    return doc_id, None


def _wait_for_completion(client, doc_id: str) -> None:
    print("       Waiting for processing", end="", flush=True)
    while True:
        status = client.get_document(doc_id).get("status", "processing")
        if status == "completed":
            print(" ✓")
            return
        if status == "failed":
            raise RuntimeError(f"PageIndex processing failed for {doc_id}")
        print(".", end="", flush=True)
        time.sleep(3)


def get_tree_cached(pdf_path: str, doc_id: str, cache_dir: str) -> dict:
    """Fetch the tree index, using the local cache when available."""
    cache_file = _cache_path(pdf_path, cache_dir)
    cache = _load_cache(cache_file)

    if "tree" in cache:
        print(f"[cache] Loading tree from cache")
        return cache["tree"]

    print(f"[index] Fetching tree index …")
    client = _pi_client()
    tree = client.get_tree(doc_id, node_summary=True)["result"]

    cache["tree"] = tree
    _save_cache(cache_file, cache)
    print(f"[index] Tree cached → {cache_file}")
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# AUTO mode — PageIndex Chat API (PageIndex runs the full RAG + LLM)
# ─────────────────────────────────────────────────────────────────────────────

def ask_auto(doc_id: str, question: str, cite: bool = False,
             history: list | None = None) -> list:
    """
    Use PageIndex's built-in Chat API.
    Returns updated conversation history (list of message dicts).
    Streams the answer to stdout.
    """
    client = _pi_client()
    messages = (history or []) + [{"role": "user", "content": question}]

    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")
    print("A: ", end="", flush=True)

    full_answer = ""
    for chunk in client.chat_completions(
        messages=messages,
        doc_id=doc_id,
        stream=True,
        enable_citations=cite,
    ):
        # SDK yields plain strings when streaming; fall back to dict format
        if isinstance(chunk, str):
            text = chunk
        elif isinstance(chunk, dict):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content") or ""
        else:
            text = ""
        print(text, end="", flush=True)
        full_answer += text

    print("\n")

    # Return updated history for multi-turn conversations
    return messages + [{"role": "assistant", "content": full_answer}]


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL mode — fetch tree, use Claude/OpenAI for retrieval + answer
# ─────────────────────────────────────────────────────────────────────────────

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

    raise RuntimeError(
        "Manual mode requires ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
    )


def _strip_tree_text(obj):
    """Remove 'text' fields recursively so tree fits in LLM context."""
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


def ask_manual(tree: dict, question: str, verbose: bool = False) -> None:
    """Retrieve relevant nodes from the tree with an LLM, then generate an answer."""
    node_map = _build_node_map(tree)
    compact = _strip_tree_text(tree)

    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")

    # Step 1 — Reasoning-based retrieval
    raw = _call_llm(RETRIEVE_PROMPT.format(
        query=question,
        tree_json=json.dumps(compact, indent=2),
    ))

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    result = json.loads(raw)

    if verbose:
        print("\n[reasoning]")
        for line in textwrap.wrap(result.get("thinking", ""), width=78):
            print("  " + line)
        print()

    node_ids = result.get("node_list", [])
    if not node_ids:
        print("[!] No relevant sections identified.\n")
        return

    # Step 2 — Show which sections were found
    print(f"Retrieved {len(node_ids)} section(s):")
    context_parts: list[str] = []
    for nid in node_ids:
        node = node_map.get(nid)
        if node is None:
            print(f"  • [{nid}] (not found in tree)")
            continue
        page = (node.get("page_index")
                or f"{node.get('start_index','?')}–{node.get('end_index','?')}")
        print(f"  • [{nid}] {node.get('title', 'Untitled')}  (page {page})")
        if "text" in node:
            context_parts.append(node["text"])

    if not context_parts:
        print("\n[!] Tree nodes have no text. The tree was built without node text.\n")
        return

    # Step 3 — Generate answer
    context = "\n\n---\n\n".join(context_parts)
    answer = _call_llm(ANSWER_PROMPT.format(query=question, context=context))
    print(f"\nA: {answer}\n")


def list_documents(limit: int = 20) -> None:
    client = _pi_client()
    result = client.list_documents(limit=limit)
    docs = result.get("data") or result.get("documents") or []
    total = result.get("total", len(docs))

    if not docs:
        print("No documents found.")
        return

    print(f"\n{'─'*60}")
    print(f"{'ID':<22}  {'Status':<12}  {'Pages':<6}  Name")
    print(f"{'─'*60}")
    for d in docs:
        print(
            f"{d.get('id',''):<22}  "
            f"{d.get('status',''):<12}  "
            f"{str(d.get('pageNum','?')):<6}  "
            f"{d.get('name','')}"
        )
    print(f"{'─'*60}")
    print(f"Showing {len(docs)} of {total} documents.\n")




def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ask questions about any PDF using PageIndex reasoning-based RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples:
              python main.py report.pdf
              python main.py report.pdf -q "What is the net revenue?"
              python main.py report.pdf -q "Summarise risks." --cite
              python main.py report.pdf --mode manual -q "..." --verbose
              python main.py --list
        """),
    )
    p.add_argument("pdf", nargs="?", help="Path to the PDF file")
    p.add_argument("-q", "--question", help="Question (omit for interactive mode)")
    p.add_argument(
        "--mode", choices=["auto", "manual"], default="auto",
        help=(
            "auto   — PageIndex runs the full RAG pipeline (default)\n"
            "manual — fetch tree + use Claude/OpenAI for retrieval & answering"
        ),
    )
    p.add_argument("--cite", action="store_true",
                   help="Enable page citations in answers (auto mode only)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show LLM retrieval reasoning (manual mode only)")
    p.add_argument("--cache-dir", default="./index_cache", metavar="DIR",
                   help="Cache directory for doc_id and tree (default: ./index_cache)")
    p.add_argument("--list", action="store_true",
                   help="List all uploaded documents and exit")
    return p


def main():
    args = build_parser().parse_args()

    if args.list:
        list_documents()
        return

    if not args.pdf:
        build_parser().print_help()
        return

    if not os.path.exists(args.pdf):
        print(f"[error] File not found: {args.pdf}")
        raise SystemExit(1)

    # Upload once; reuse doc_id from cache on subsequent runs
    doc_id, cached_tree = upload_and_wait(args.pdf, args.cache_dir)

    # ── AUTO mode ──────────────────────────────────────────────────────────
    if args.mode == "auto":
        if args.question:
            ask_auto(doc_id, args.question, cite=args.cite)
            return

        print(f"\nPageIndex Chat — {Path(args.pdf).name}  [{doc_id}]")
        print("Multi-turn conversation. Type 'quit' or Ctrl-C to exit.\n")
        history: list = []
        while True:
            try:
                question = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            history = ask_auto(doc_id, question, cite=args.cite, history=history)

    # ── MANUAL mode ────────────────────────────────────────────────────────
    else:
        tree = cached_tree or get_tree_cached(args.pdf, doc_id, args.cache_dir)

        if args.question:
            ask_manual(tree, args.question, verbose=args.verbose)
            return

        print(f"\nPageIndex Manual RAG — {Path(args.pdf).name}  [{doc_id}]")
        print("Type 'quit' or Ctrl-C to exit.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            ask_manual(tree, question, verbose=args.verbose)


if __name__ == "__main__":
    main()
