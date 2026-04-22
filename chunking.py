"""
chunking.py
--------------------
Three chunking strategies:
  1. recursive  – LangChain RecursiveCharacterTextSplitter (default)
  2. paragraph  – split on blank lines, respect min/max sizes
  3. semantic   – embed sentences, cut on cosine-distance spikes
"""

from __future__ import annotations
import os
import re
from typing import Literal
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)   # source, page, strategy …


# ─────────────────────────────────────────────
# 1. Recursive character splitter
# ─────────────────────────────────────────────
def recursive_chunk(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Standard LangChain recursive splitter.
    Tries to split on [\\n\\n, \\n, ' ', ''] in order.
    """
    # from langchain.text_splitter import RecursiveCharacterTextSplitter

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    pieces = splitter.split_text(text)
    meta = metadata or {}
    return [Chunk(text=p, metadata={**meta, "strategy": "recursive"}) for p in pieces]


# ─────────────────────────────────────────────
# 2. Paragraph splitter
# ─────────────────────────────────────────────
def paragraph_chunk(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Split on blank lines (\\n\\n+).
    Paragraphs shorter than CHUNK_SIZE//4 are merged with the next one.
    Paragraphs longer than CHUNK_SIZE are further split recursively.
    """
    meta = metadata or {}
    raw_paras = re.split(r"\n{2,}", text.strip())
    chunks: list[Chunk] = []
    buffer = ""

    for para in raw_paras:
        para = para.strip()
        if not para:
            continue
        combined = (buffer + "\n\n" + para).strip() if buffer else para

        if len(combined) <= CHUNK_SIZE:
            buffer = combined
        else:
            # flush buffer
            if buffer:
                chunks.append(Chunk(text=buffer, metadata={**meta, "strategy": "paragraph"}))
            # if para itself is too long, recurse with recursive splitter
            if len(para) > CHUNK_SIZE:
                chunks.extend(recursive_chunk(para, {**meta, "strategy": "paragraph"}))
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append(Chunk(text=buffer, metadata={**meta, "strategy": "paragraph"}))

    return chunks


# ─────────────────────────────────────────────
# 3. Semantic splitter (embedding-based)
# ─────────────────────────────────────────────
def semantic_chunk(text: str, metadata: dict | None = None, threshold: float = 0.5) -> list[Chunk]:
    """
    Split text into semantically coherent chunks by:
      1. Splitting into sentences.
      2. Embedding each sentence.
      3. Computing cosine distance between consecutive sentences.
      4. Cutting where distance > threshold (local maximum strategy).
    Falls back to recursive chunking for very short texts.
    """
    import numpy as np
    from helpers import embed_texts

    meta = metadata or {}

    # ── sentence tokenisation (regex, no NLTK dependency) ──
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
    sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]

    if len(sentences) <= 2:
        return recursive_chunk(text, meta)

    embeddings = np.array(embed_texts(sentences))

    # cosine distance between consecutive sentences
    def cos_dist(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 1.0
        return 1 - float(np.dot(a, b) / denom)

    distances = [cos_dist(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    # adaptive threshold: mean + threshold * std
    mean_d, std_d = float(np.mean(distances)), float(np.std(distances))
    cut_threshold = mean_d + threshold * std_d

    # build chunks
    chunks: list[Chunk] = []
    current_sentences: list[str] = [sentences[0]]

    for i, dist in enumerate(distances):
        if dist > cut_threshold:
            chunk_text = " ".join(current_sentences)
            # keep chunks within size budget
            if len(chunk_text) > CHUNK_SIZE:
                chunks.extend(recursive_chunk(chunk_text, meta))
            else:
                chunks.append(Chunk(text=chunk_text, metadata={**meta, "strategy": "semantic"}))
            current_sentences = []
        current_sentences.append(sentences[i + 1])

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if len(chunk_text) > CHUNK_SIZE:
            chunks.extend(recursive_chunk(chunk_text, meta))
        else:
            chunks.append(Chunk(text=chunk_text, metadata={**meta, "strategy": "semantic"}))

    return chunks


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────
Strategy = Literal["recursive", "paragraph", "semantic"]


def chunk_text(
    text: str,
    strategy: Strategy = "recursive",
    metadata: dict | None = None,
) -> list[Chunk]:
    """Route to the correct chunking strategy."""
    dispatch = {
        "recursive": recursive_chunk,
        "paragraph": paragraph_chunk,
        "semantic":  semantic_chunk,
    }
    if strategy not in dispatch:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(dispatch)}")
    return dispatch[strategy](text, metadata)
