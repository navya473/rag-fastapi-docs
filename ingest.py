"""
ingest.py
------------------
Ingests raw PDF / Markdown files from `data/` into Pinecone.

Pinecone index is created with:
  - Dense vector  : sentence-transformers embeddings  (dim=768)
  - Sparse vector : BM25 (pinecone-text)  → enables hybrid search

Usage:
    python -m src.core.ingest [--strategy recursive|paragraph|semantic]
"""

from __future__ import annotations
import os
import uuid
import argparse
from pathlib import Path
from typing import Generator

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR         = Path("docs")
INDEX_NAME       = os.environ["PINECONE_INDEX_NAME"]
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIMENSION", 768))
BATCH_SIZE       = 100
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")


# ─────────────────────────────────────────────
# Document loading
# ─────────────────────────────────────────────
def load_documents() -> Generator[tuple[str, dict], None, None]:
    """
    Yield (raw_text, metadata) for every PDF and Markdown file in DATA_DIR.
    """
    for path in sorted(DATA_DIR.rglob("*")):
        if path.suffix.lower() == ".pdf":
            yield from _load_pdf(path)
        elif path.suffix.lower() in (".md", ".mdx"):
            yield from _load_markdown(path)


def _load_pdf(path: Path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            yield text, {"source": str(path), "page": page_num + 1, "type": "pdf"}


def _load_markdown(path: Path):
    text = path.read_text(encoding="utf-8")
    if text.strip():
        yield text, {"source": str(path), "page": 1, "type": "markdown"}


# ─────────────────────────────────────────────
# Pinecone index creation
# ─────────────────────────────────────────────
def create_index_if_not_exists():
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' …")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",           # required for hybrid search
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1"),
            ),
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    return pc.Index(INDEX_NAME)


# ─────────────────────────────────────────────
# BM25 sparse encoder
# ─────────────────────────────────────────────
def _get_bm25_encoder(all_texts: list[str]):
    """
    Fit a BM25Encoder on the corpus and return it.
    Saves the fitted encoder to disk so repeated runs skip refitting.
    """
    from pinecone_text.sparse import BM25Encoder

    encoder_path = Path(".bm25_encoder.json")
    encoder = BM25Encoder()

    if encoder_path.exists():
        print("Loading existing BM25 encoder …")
        encoder = BM25Encoder().load(str(encoder_path))
    else:
        print(f"Fitting BM25 encoder on {len(all_texts)} chunks …")
        encoder.fit(all_texts)
        encoder.dump(str(encoder_path))
        print("BM25 encoder saved.")

    return encoder


# ─────────────────────────────────────────────
# Main ingestion pipeline
# ─────────────────────────────────────────────
def ingest(strategy: str = CHUNKING_STRATEGY):
    from chunking import chunk_text
    from helpers import embed_texts

    index = create_index_if_not_exists()

    # ── 1. Load & chunk ──────────────────────────────────────────
    all_chunks = []
    print(f"Loading documents from '{DATA_DIR}' with '{strategy}' chunking …")
    for raw_text, meta in load_documents():
        chunks = chunk_text(raw_text, strategy=strategy, metadata=meta)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError(f"No documents found in '{DATA_DIR}'. Add PDFs or Markdown files.")

    print(f"Total chunks: {len(all_chunks)}")
    corpus_texts = [c.text for c in all_chunks]

    # ── 2. Fit BM25 ──────────────────────────────────────────────
    bm25 = _get_bm25_encoder(corpus_texts)

    # ── 3. Embed (dense) ─────────────────────────────────────────
    print("Generating dense embeddings …")
    dense_vectors = embed_texts(corpus_texts)

    # ── 4. Sparse vectors ────────────────────────────────────────
    print("Generating sparse (BM25) vectors …")
    sparse_vectors = bm25.encode_documents(corpus_texts)

    # ── 5. Upsert in batches ─────────────────────────────────────
    print("Upserting vectors into Pinecone …")
    records = []
    for i, (chunk, dense, sparse) in enumerate(
        zip(all_chunks, dense_vectors, sparse_vectors)
    ):
        records.append(
            {
                "id": str(uuid.uuid4()),
                "values": dense,
                "sparse_values": sparse,
                "metadata": {
                    **chunk.metadata,
                    "text": chunk.text,
                    "chunk_index": i,
                },
            }
        )

    for batch_start in tqdm(range(0, len(records), BATCH_SIZE), desc="Upserting"):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        index.upsert(vectors=batch)

    print(f"\n✅  Ingestion complete. {len(records)} vectors upserted to '{INDEX_NAME}'.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs into Pinecone")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "paragraph", "semantic"],
        default=CHUNKING_STRATEGY,
        help="Chunking strategy to use",
    )
    args = parser.parse_args()
    ingest(strategy=args.strategy)

