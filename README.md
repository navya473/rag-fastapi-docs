# RAG — FastAPI Docs Q&A

A production-grade Retrieval-Augmented Generation system with an
hybrid Pinecone search, Cohere reranking, and full
evaluation tooling.


---

## Architecture

```
RAG/
├── data/                  # Raw PDFs / Markdown (FastAPI docs)
├── .env.example           # API keys
├── chunking.py            # recursive | paragraph | semantic
├── ingest.py              # Pinecone ingestion (dense + BM25 sparse)
├── rerank.py              # Cohere / cross-encoder reranking
├── retrieve.py            # Hybrid search + RRF fusion
├── deep_eval.py           # Faithfulness, Relevancy, Precision, Recall
├── eval_retrieval.py      # Precision@K, Recall@K, MRR
├── helpers.py             # Cached client initializers
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` and fill in your keys:



| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | Pinecone project API key |
| `PINECONE_INDEX_NAME` | Index name to create / use |
| `GROQ_API_KEY` | Groq API key (LLaMA 3.3 70B) |
| `GOOGLE_API_KEY` | Gemini API key (gemini-3.1-flash-lite-preview) |
| `COHERE_API_KEY` | Cohere API key (reranking) |
| `CHUNKING_STRATEGY` | `recursive` \| `paragraph` \| `semantic` |
| `TOP_K_RETRIEVE` | Candidates from hybrid search (default 10) |
| `TOP_K_RERANK` | Docs kept after reranking (default 4) |

### 3. Add documents

Place FastAPI docs (PDF or Markdown) into `data/`:

```bash
# Example: clone FastAPI docs
git clone --depth 1 https://github.com/tiangolo/fastapi.git /tmp/fastapi
cp -r /tmp/fastapi/docs/en/docs/*.md data/
```

### 4. Ingest into Pinecone

```bash
# Default (recursive chunking)
python ingest.py 

```

This will:
- Load all PDFs and Markdown files from `data/`
- Chunk with the chosen strategy
- Generate dense embeddings (sentence-transformers)
- Fit a BM25 encoder for sparse vectors
- Upsert hybrid vectors into Pinecone


## Chunking Strategies

| Strategy | Description | Best for |
|---|---|---|
| `recursive` | LangChain RecursiveCharacterTextSplitter | General use, safe default |
| `paragraph` | Split on blank lines, merge short paras | Structured docs, tutorials |
| `semantic` | Embed sentences, cut on cosine-distance spikes | Dense technical prose |

---

## Retrieval: Hybrid Search + RRF

1. **Dense search** — query embedded with `sentence-transformers/all-mpnet-base-v2`
2. **Sparse search** — query encoded with fitted BM25 (pinecone-text)
3. **RRF fusion** — Reciprocal Rank Fusion merges both result lists:
   `score(d) = Σ 1/(60 + rank_i)`
4. **Reranking** — top candidates re-scored with Cohere Rerank v3 (or local cross-encoder)

---

## Evaluation

### Retrieval metrics (Precision@K, Recall@K, MRR)

```bash
# Uses retrieval_ground_truth.json
python -m eval_retrieval --k 10
```

Edit `retrieval_ground_truth.json`:
```json
[
  {
    "query": "How do I use background tasks?",
    "relevant_texts": ["BackgroundTasks", "background_tasks.add_task"]
  }
]
```

### DeepEval metrics (Faithfulness, Answer Relevancy, Context Precision/Recall)

```bash
# Uses deep_eval_ground_truth.json
python -m eval_ragas
```

Edit `deep_eval_ground_truth.json`:
```json
[
  {
    "question": "How do I use background tasks?",
    "ground_truth": "Use BackgroundTasks parameter and call add_task()..."
  }
]
```

---

## Reranker backends

Switch via `.env`:

```bash
RERANKER=cohere          # Cohere Rerank v3 (cloud, best quality)
RERANKER=cross-encoder   # ms-marco-MiniLM-L-6-v2 (local, free)
```

---

## Testing

The test suite validates the text-chunking strategies, retrieval logic, and metadata consistency using `pytest`.

To run the unit and integration tests completely offline (disabling active LangChain/LangSmith tracing plugins):

```bash
LANGCHAIN_TRACING_V2=false pytest test_rag.py -p no:langsmith -p no:anyio
---

## Deployment

### Run FastAPI

uvicorn api:app --host 0.0.0.0 --port 8000