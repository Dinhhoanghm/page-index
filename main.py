#!/usr/bin/env python3
"""
PageIndex RAG — Vectorless, Reasoning-based RAG (Local Mode)
Ask questions about any PDF using the open-source PageIndex library.
No PageIndex cloud API key required.

Two modes
─────────
auto   (default) — Indexes locally; LLM picks sections and answers.
manual           — Same pipeline; --verbose shows retrieval reasoning.

Usage
─────
  python main.py report.pdf                          # interactive, auto mode
  python main.py report.pdf --mode manual            # manual mode
  python main.py report.pdf -q "What is revenue?"   # one-shot
  python main.py report.pdf -q "..." -v              # show retrieval reasoning
  python main.py --list                              # list all indexed documents
"""

import os
import json
import sys
import argparse
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# Use the local open-source PageIndex library
sys.path.insert(0, str(Path(__file__).parent / "PageIndex"))

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
WORKSPACE_DIR     = os.getenv("WORKSPACE_DIR", "./workspace")


# ─────────────────────────────────────────────────────────────────────────────
# Local PageIndex client
# ─────────────────────────────────────────────────────────────────────────────

def _pi_client():
    from pageindex import PageIndexClient
    return PageIndexClient(workspace=WORKSPACE_DIR)


def _find_existing_doc(client, file_path: str) -> str | None:
    """Return doc_id if file_path is already indexed in workspace, else None."""
    abs_path = os.path.abspath(os.path.expanduser(file_path))
    for doc_id, doc in client.documents.items():
        if doc.get("path") == abs_path:
            return doc_id
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Document indexing
# ─────────────────────────────────────────────────────────────────────────────

def index_document(pdf_path: str):
    """Index pdf_path locally (reuses cached entry if already indexed). Returns (doc_id, client)."""
    client = _pi_client()
    doc_id = _find_existing_doc(client, pdf_path)
    if doc_id:
        print(f"[cache] Using existing doc_id: {doc_id}")
        return doc_id, client

    print(f"[index] Indexing '{pdf_path}' locally … (this may take a few minutes)")
    doc_id = client.index(pdf_path)
    print(f"[index] Done. Document ID: {doc_id}")
    return doc_id, client


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers (Claude or OpenAI — for retrieval & answering)
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

    raise RuntimeError("ANTHROPIC_API_KEY or OPENAI_API_KEY required in .env")


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


# ─────────────────────────────────────────────────────────────────────────────
# Local RAG pipeline (shared by auto and manual modes)
# ─────────────────────────────────────────────────────────────────────────────

def _rag(pi_client, doc_id: str, question: str, verbose: bool = False) -> None:
    """Fetch relevant nodes via LLM, retrieve page content, generate answer."""
    raw_structure = pi_client.get_document_structure(doc_id)
    tree = json.loads(raw_structure)
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

    # Step 2 — Retrieve page content for relevant nodes
    print(f"Retrieved {len(node_ids)} section(s):")
    context_parts: list[str] = []
    for nid in node_ids:
        node = node_map.get(nid)
        if node is None:
            print(f"  • [{nid}] (not found in tree)")
            continue
        start = node.get("start_index", "?")
        end = node.get("end_index", "?")
        print(f"  • [{nid}] {node.get('title', 'Untitled')}  (pages {start}–{end})")
        if start != "?" and end != "?":
            raw_content = pi_client.get_page_content(doc_id, f"{start}-{end}")
            pages = json.loads(raw_content)
            text = "\n".join(p["content"] for p in pages if p.get("content"))
            if text:
                context_parts.append(text)

    if not context_parts:
        print("\n[!] Could not retrieve page content for identified sections.\n")
        return

    # Step 3 — Generate answer
    context = "\n\n---\n\n".join(context_parts)
    answer = _call_llm(ANSWER_PROMPT.format(query=question, context=context))
    print(f"\nA: {answer}\n")


def ask_auto(pi_client, doc_id: str, question: str, history: list | None = None) -> list:
    """Local RAG pipeline. Returns updated conversation history."""
    _rag(pi_client, doc_id, question, verbose=False)
    return (history or []) + [{"role": "user", "content": question}]


def ask_manual(pi_client, doc_id: str, question: str, verbose: bool = False) -> None:
    _rag(pi_client, doc_id, question, verbose=verbose)


def list_documents() -> None:
    client = _pi_client()
    docs = list(client.documents.values())
    if not docs:
        print("No documents found.")
        return

    print(f"\n{'─'*60}")
    print(f"{'ID':<38}  {'Type':<5}  {'Pages':<6}  Name")
    print(f"{'─'*60}")
    for d in docs:
        pages = d.get("page_count") or d.get("line_count") or "?"
        print(
            f"{d.get('id', ''):<38}  "
            f"{d.get('type', ''):<5}  "
            f"{str(pages):<6}  "
            f"{d.get('doc_name', '')}"
        )
    print(f"{'─'*60}")
    print(f"Total: {len(docs)} document(s).\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ask questions about any PDF using PageIndex local RAG (no cloud API required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples:
              python main.py report.pdf
              python main.py report.pdf -q "What is the net revenue?"
              python main.py report.pdf --mode manual -q "..." --verbose
              python main.py --list
        """),
    )
    p.add_argument("pdf", nargs="?", help="Path to the PDF file")
    p.add_argument("-q", "--question", help="Question (omit for interactive mode)")
    p.add_argument(
        "--mode", choices=["auto", "manual"], default="auto",
        help=(
            "auto   — Local RAG pipeline (default)\n"
            "manual — Same as auto; --verbose shows retrieval reasoning"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show LLM retrieval reasoning (manual mode)")
    p.add_argument("--workspace", default=None, metavar="DIR",
                   help=f"Workspace directory for indexed docs (default: {WORKSPACE_DIR})")
    p.add_argument("--list", action="store_true",
                   help="List all indexed documents and exit")
    return p


def main():
    args = build_parser().parse_args()

    if args.workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = args.workspace

    if args.list:
        list_documents()
        return

    if not args.pdf:
        build_parser().print_help()
        return

    if not os.path.exists(args.pdf):
        print(f"[error] File not found: {args.pdf}")
        raise SystemExit(1)

    doc_id, pi_client = index_document(args.pdf)

    if args.mode == "auto":
        if args.question:
            ask_auto(pi_client, doc_id, args.question)
            return

        print(f"\nPageIndex Local RAG — {Path(args.pdf).name}  [{doc_id}]")
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
            history = ask_auto(pi_client, doc_id, question, history=history)

    else:  # manual
        if args.question:
            ask_manual(pi_client, doc_id, args.question, verbose=args.verbose)
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
            ask_manual(pi_client, doc_id, question, verbose=args.verbose)


if __name__ == "__main__":
    main()
