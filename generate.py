"""
generate.py
--------------------
Answer generation using retrieved + reranked context.

This is the final node in the RAG pipeline:
  retrieve → rerank → generate

No LangGraph required — can be called directly or wired into a graph later.

Usage:
    from generate import generate_answer
    answer = generate_answer(query="...", docs=[...])
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from retrieve import RetrievedDoc

load_dotenv()

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 6000))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided context.

Rules:
- Only use information present in the context below.
- If the context does not contain enough information, say so clearly.
- Cite sources when available using [Source: filename, page N] format.
- Be concise but complete.
"""

def build_context(docs: list[RetrievedDoc], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Concatenate retrieved docs into a context block, respecting token budget."""
    parts = []
    total = 0
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        header = f"[{i}] Source: {source}, page {page}"
        block  = f"{header}\n{doc.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    docs: list[RetrievedDoc],
    stream: bool = False,
) -> str:
    """
    Generate an answer from the query + retrieved docs.

    Args:
        query:  The user's question.
        docs:   Reranked (or retrieved) documents to use as context.
        stream: If True, print tokens as they arrive (CLI mode).

    Returns:
        The generated answer string.
    """
    from helpers import get_llm

    llm = get_llm()
    context = build_context(docs)

    if not context.strip():
        return "I could not find relevant information to answer your question."

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )

    if stream:
        answer_parts = []
        for chunk in llm.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            print(token, end="", flush=True)
            answer_parts.append(token)
        print()
        return "".join(answer_parts)
    else:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


# ─────────────────────────────────────────────
# Full RAG pipeline (retrieve → rerank → generate)
# ─────────────────────────────────────────────
def rag(query: str, top_k_retrieve: int = 10, top_k_rerank: int = 4) -> dict:
    """
    End-to-end RAG: retrieve → rerank → generate.

    Returns:
        {
            "query": str,
            "answer": str,
            "sources": [{"text": ..., "source": ..., "page": ..., "score": ...}]
        }
    """
    from retrieve import retrieve
    from rerank import rerank

    docs     = retrieve(query, top_k=top_k_retrieve)
    reranked = rerank(query, docs, top_n=top_k_rerank)
    answer   = generate_answer(query, reranked)

    sources = [
        {
            "text":   d.text[:300] + "..." if len(d.text) > 300 else d.text,
            "source": d.metadata.get("source", "unknown"),
            "page":   d.metadata.get("page", "?"),
            "score":  round(d.score, 4),
        }
        for d in reranked
    ]

    return {"query": query, "answer": answer, "sources": sources}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query = input("Ask a question: ").strip()
    if not query:
        print("No query entered.")
        exit(1)

    print("\nRetrieving and generating...\n")
    result = rag(query)

    print("=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['score']:.4f}] {s['source']} (page {s['page']})")
        print(f"    {s['text'][:120]}...")
    print("=" * 60)
