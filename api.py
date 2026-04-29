"""
api.py
------
FastAPI REST API for the RAG pipeline.

Endpoints:
  POST /ask          — full RAG (retrieve → rerank → generate)
  POST /retrieve     — retrieval only (no generation)
  GET  /health       — health check

Run:
    uvicorn api:app --reload --port 8000

Docs auto-available at: http://localhost:8000/docs
"""

from __future__ import annotations
import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="Hybrid retrieval-augmented generation over your docs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k_retrieve: int = Field(10, ge=1, le=50, description="Docs to retrieve")
    top_k_rerank: int   = Field(4,  ge=1, le=20, description="Docs after reranking")

class SourceDoc(BaseModel):
    text:   str
    source: str
    page:   int | str
    score:  float

class AskResponse(BaseModel):
    query:        str
    answer:       str
    sources:      list[SourceDoc]
    latency_ms:   float

class RetrieveRequest(BaseModel):
    query:  str = Field(..., min_length=1)
    top_k:  int = Field(10, ge=1, le=50)

class RetrieveResponse(BaseModel):
    query:       str
    results:     list[SourceDoc]
    latency_ms:  float

class HealthResponse(BaseModel):
    status:      str
    index_name:  str
    embedding_model: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Check that the pipeline is reachable."""
    return {
        "status":          "ok",
        "index_name":      os.getenv("PINECONE_INDEX_NAME", "unknown"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2"),
    }


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(body: AskRequest):
    """
    Full RAG pipeline: retrieve → rerank → generate.
    Returns the answer and the source documents used.
    """
    t0 = time.perf_counter()
    try:
        from generate import rag
        result = rag(
            query=body.query,
            top_k_retrieve=body.top_k_retrieve,
            top_k_rerank=body.top_k_rerank,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Pipeline not ready: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "query":      result["query"],
        "answer":     result["answer"],
        "sources":    result["sources"],
        "latency_ms": latency,
    }


@app.post("/retrieve", response_model=RetrieveResponse, tags=["rag"])
def retrieve_only(body: RetrieveRequest):
    """
    Retrieval only — returns the top-k chunks without generating an answer.
    Useful for debugging retrieval quality.
    """
    t0 = time.perf_counter()
    try:
        from retrieve import retrieve
        docs = retrieve(body.query, top_k=body.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Pipeline not ready: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "query": body.query,
        "results": [
            {
                "text":   d.text[:400],
                "source": d.metadata.get("source", "unknown"),
                "page":   d.metadata.get("page", "?"),
                "score":  round(d.score, 4),
            }
            for d in docs
        ],
        "latency_ms": latency,
    }


