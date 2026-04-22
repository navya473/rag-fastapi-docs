"""
helpers.py
--------------------
Initializes shared clients (Pinecone, Groq, Cohere, Embeddings).
All functions are cached so clients are created only once per process.
"""

import os
import functools
from dotenv import load_dotenv

load_dotenv()


# helpers.py
import functools
import os

@functools.lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0):
    from langchain_google_genai import ChatGoogleGenerativeAI
    """
    Switching to Gemini to avoid Groq Rate Limits during evaluation.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=temperature,
    )

# ─────────────────────────────────────────────
# Pinecone
# ─────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_pinecone_index():
    """Return a connected Pinecone Index object."""
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    if index_name not in [i.name for i in pc.list_indexes()]:
        raise ValueError(
            f"Index '{index_name}' not found in Pinecone. "
            "Run `python -m src.core.ingest` first."
        )

    return pc.Index(index_name)


# ─────────────────────────────────────────────
# Groq LLM (via LangChain)
# ─────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_groq_llm(temperature: float = 0.0):
    """
    FORCING GEMINI: Renamed the internal logic but kept the function 
    name so nodes.py doesn't break.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Use the most recent stable 2026 model name
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview", 
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=temperature,
    )

# ─────────────────────────────────────────────
# Cohere rerank client
# ─────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_cohere_client():
    """Return a raw Cohere client for reranking."""
    import cohere

    return cohere.Client(api_key=os.environ["COHERE_API_KEY"])


# ─────────────────────────────────────────────
# Sentence-Transformers embedding model
# ─────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_embedding_model():
    """Return a SentenceTransformer model (cached)."""
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
    )
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience wrapper: embed a list of strings → list of vectors."""
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
