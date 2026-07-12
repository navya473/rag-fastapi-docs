"""
test_rag.py
--------------------
Lightweight unit tests for the RAG pipeline's core logic:

  * chunking strategies   (chunking.py)
  * retrieval / fusion     (retrieve.py)
  * reranker outputs       (rerank.py)

Everything external — Pinecone, Cohere, sentence-transformers, LangChain
LLMs — is mocked, so the suite runs offline with no API keys and no
network access. All heavy dependencies in the source modules are imported
lazily *inside* functions, so importing the modules here is cheap.

Run with:
    python -m pytest test_rag.py -v
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Make sure the repo root (where chunking/retrieve/rerank live) is importable
# regardless of the directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chunking
import retrieve
import rerank
from chunking import Chunk, chunk_text, paragraph_chunk, recursive_chunk, semantic_chunk
from retrieve import RetrievedDoc, reciprocal_rank_fusion


# ═════════════════════════════════════════════════════════════════════
# CHUNKING STRATEGIES
# ═════════════════════════════════════════════════════════════════════
class TestRecursiveChunk:
    def test_short_text_is_single_chunk(self):
        chunks = recursive_chunk("hello world", metadata={"source": "a.md"})
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == "hello world"
        assert chunks[0].metadata["source"] == "a.md"
        assert chunks[0].metadata["strategy"] == "recursive"

    def test_long_text_splits_into_many(self, monkeypatch):
        # Force a tiny chunk size so splitting is deterministic.
        monkeypatch.setattr(chunking, "CHUNK_SIZE", 50)
        monkeypatch.setattr(chunking, "CHUNK_OVERLAP", 0)
        chunks = recursive_chunk("word " * 200)  # ~1000 chars
        assert len(chunks) > 1
        assert all(c.metadata["strategy"] == "recursive" for c in chunks)
        assert all(len(c.text) <= 50 for c in chunks)

    def test_defaults_metadata_when_none(self):
        chunks = recursive_chunk("hi")
        assert chunks[0].metadata == {"strategy": "recursive"}


class TestParagraphChunk:
    def test_merges_small_paragraphs(self, monkeypatch):
        monkeypatch.setattr(chunking, "CHUNK_SIZE", 100)
        text = "First para.\n\nSecond para.\n\nThird para."
        chunks = paragraph_chunk(text, metadata={"source": "x"})
        # All three short paragraphs fit under CHUNK_SIZE → merged into one.
        assert len(chunks) == 1
        assert "First para." in chunks[0].text
        assert "Third para." in chunks[0].text
        assert chunks[0].metadata["strategy"] == "paragraph"
        assert chunks[0].metadata["source"] == "x"

    def test_flushes_when_combined_exceeds_size(self, monkeypatch):
        monkeypatch.setattr(chunking, "CHUNK_SIZE", 20)
        text = "aaaaaaaaaa\n\nbbbbbbbbbb\n\ncccccccccc"
        chunks = paragraph_chunk(text)
        assert len(chunks) >= 2
        joined = " ".join(c.text for c in chunks)
        assert "aaaaaaaaaa" in joined
        assert "cccccccccc" in joined

    def test_recurses_on_oversized_paragraph(self, monkeypatch):
        monkeypatch.setattr(chunking, "CHUNK_SIZE", 30)
        monkeypatch.setattr(chunking, "CHUNK_OVERLAP", 0)
        chunks = paragraph_chunk("z" * 200)  # single huge paragraph
        assert len(chunks) > 1
        assert all(c.metadata["strategy"] == "paragraph" for c in chunks)

    def test_blank_input_yields_nothing(self):
        assert paragraph_chunk("   \n\n   ") == []


class TestSemanticChunk:
    def test_short_text_falls_back_to_recursive(self):
        # <= 2 sentences → recursive fallback, so no embedding call happens.
        chunks = semantic_chunk("One sentence. Two sentences.")
        assert len(chunks) >= 1
        assert all(c.metadata["strategy"] == "recursive" for c in chunks)

    def test_cuts_on_embedding_distance_spike(self, monkeypatch):
        """
        Four sentences forming two clusters:
            [1,0], [1,0]  |  [0,1], [0,1]
        The cosine distance spikes in the middle → expect two chunks.
        embed_texts is mocked so the cut point is deterministic.
        """
        monkeypatch.setattr(chunking, "CHUNK_SIZE", 512)

        import helpers

        fake_embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

        def fake_embed_texts(texts):
            assert len(texts) == 4  # 4 sentences tokenised
            return fake_embeddings

        monkeypatch.setattr(helpers, "embed_texts", fake_embed_texts)

        text = "Cats sit here. Cats sit there. Dogs run fast. Dogs run far."
        chunks = semantic_chunk(text, metadata={"source": "s"})

        assert len(chunks) == 2
        assert all(c.metadata["strategy"] == "semantic" for c in chunks)
        assert all(c.metadata["source"] == "s" for c in chunks)
        joined = " ".join(c.text for c in chunks)
        for fragment in ("Cats sit here.", "Dogs run far."):
            assert fragment in joined


class TestChunkTextDispatcher:
    def test_routes_to_selected_strategy(self, monkeypatch):
        sentinel = [Chunk(text="stub", metadata={})]
        fake = MagicMock(return_value=sentinel)
        monkeypatch.setattr(chunking, "paragraph_chunk", fake)

        result = chunk_text("hello", strategy="paragraph")
        fake.assert_called_once()
        assert result is sentinel

    def test_defaults_to_recursive(self):
        chunks = chunk_text("plain text")
        assert chunks[0].metadata["strategy"] == "recursive"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_text("text", strategy="nonsense")


# ═════════════════════════════════════════════════════════════════════
# RETRIEVAL  (reciprocal rank fusion + hybrid query)
# ═════════════════════════════════════════════════════════════════════
def _doc(id_, text="t", score=0.0, metadata=None):
    return RetrievedDoc(id=id_, text=text, score=score, metadata=metadata or {})


class TestReciprocalRankFusion:
    def test_rewards_docs_ranked_high_in_both_lists(self):
        list_a = [_doc("A"), _doc("B"), _doc("C")]
        list_b = [_doc("B"), _doc("A"), _doc("D")]
        merged = reciprocal_rank_fusion(list_a, list_b, k=60)

        ids = [d.id for d in merged]
        # A and B appear in both lists → they beat C and D.
        assert set(ids[:2]) == {"A", "B"}
        assert set(ids) == {"A", "B", "C", "D"}

    def test_score_is_sum_of_reciprocal_ranks(self):
        merged = reciprocal_rank_fusion([_doc("A")], [_doc("A")], k=60)
        assert len(merged) == 1
        assert merged[0].score == pytest.approx(2 * (1.0 / 61))

    def test_deduplicates_by_id(self):
        merged = reciprocal_rank_fusion([_doc("A"), _doc("B")], [_doc("A"), _doc("B")])
        assert len(merged) == 2
        assert sorted(d.id for d in merged) == ["A", "B"]

    def test_output_sorted_descending(self):
        merged = reciprocal_rank_fusion([_doc("A"), _doc("B"), _doc("C")])
        scores = [d.score for d in merged]
        assert scores == sorted(scores, reverse=True)


class TestRetrieve:
    """`retrieve()` with Pinecone, embeddings, and BM25 all mocked."""

    def _make_index(self, matches):
        index = MagicMock()
        index.query.return_value = {"matches": matches}
        return index

    def _patch_all(self, monkeypatch, index):
        import helpers

        bm25 = MagicMock()
        bm25.encode_queries.return_value = {"indices": [1], "values": [0.5]}

        monkeypatch.setattr(helpers, "get_pinecone_index", lambda: index)
        monkeypatch.setattr(helpers, "embed_query", lambda q: [0.1, 0.2, 0.3])
        monkeypatch.setattr(retrieve, "_get_bm25", lambda: bm25)

    def test_builds_docs_and_strips_text_from_metadata(self, monkeypatch):
        matches = [
            {"id": "doc1", "score": 0.9,
             "metadata": {"text": "hello", "source": "a.pdf", "page": 3}},
            {"id": "doc2", "score": 0.7,
             "metadata": {"text": "world", "source": "b.md", "page": 1}},
        ]
        index = self._make_index(matches)
        self._patch_all(monkeypatch, index)

        docs = retrieve.retrieve("my question", top_k=5)

        assert [d.id for d in docs] == ["doc1", "doc2"]
        assert docs[0].text == "hello"
        assert docs[0].score == 0.9
        assert "text" not in docs[0].metadata
        assert docs[0].metadata == {"source": "a.pdf", "page": 3}

    def test_over_fetches_double_top_k(self, monkeypatch):
        index = self._make_index([])
        self._patch_all(monkeypatch, index)

        retrieve.retrieve("q", top_k=4)

        _, kwargs = index.query.call_args
        assert kwargs["top_k"] == 8               # candidates = top_k * 2
        assert kwargs["vector"] == [0.1, 0.2, 0.3]
        assert "sparse_vector" in kwargs
        assert kwargs["include_metadata"] is True

    def test_missing_text_defaults_to_empty(self, monkeypatch):
        index = self._make_index([{"id": "d", "score": 0.5, "metadata": {"source": "a"}}])
        self._patch_all(monkeypatch, index)

        docs = retrieve.retrieve("q")
        assert docs[0].text == ""

    def test_retrieve_as_dicts_flattens_metadata(self, monkeypatch):
        docs = [_doc("1", text="foo", score=0.42, metadata={"source": "s", "page": 2})]
        monkeypatch.setattr(retrieve, "retrieve", lambda query, top_k=10: docs)

        out = retrieve.retrieve_as_dicts("q")
        assert out == [{"id": "1", "text": "foo", "score": 0.42, "source": "s", "page": 2}]


# ═════════════════════════════════════════════════════════════════════
# RERANKER OUTPUTS
# ═════════════════════════════════════════════════════════════════════
def _rerank_docs():
    return [
        RetrievedDoc(id="a", text="alpha", score=0.1, metadata={"source": "1"}),
        RetrievedDoc(id="b", text="bravo", score=0.2, metadata={"source": "2"}),
        RetrievedDoc(id="c", text="charlie", score=0.3, metadata={"source": "3"}),
    ]


class TestRerankDispatch:
    def test_empty_docs_returns_empty(self):
        assert rerank.rerank("q", [], top_n=4) == []

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker backend"):
            rerank.rerank("q", _rerank_docs(), top_n=2, backend="bogus")

    def test_top_n_capped_at_doc_count(self, monkeypatch):
        captured = {}

        def fake_ce(query, docs, top_n):
            captured["top_n"] = top_n
            return docs[:top_n]

        monkeypatch.setattr(rerank, "_cross_encoder_rerank", fake_ce)
        rerank.rerank("q", _rerank_docs(), top_n=99, backend="cross-encoder")
        assert captured["top_n"] == 3


class TestCrossEncoderRerank:
    def _fake_ce(self, scores):
        """A stand-in CrossEncoder whose .predict returns numpy-like scores."""
        import numpy as np

        ce = MagicMock()
        ce.predict.return_value = np.array(scores)
        return ce

    def test_sorts_by_score_and_trims(self, monkeypatch):
        # raw scores 0.1, 0.9, 0.5 for a, b, c → order b, c with top_n=2
        monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: self._fake_ce([0.1, 0.9, 0.5]))
        out = rerank.rerank("q", _rerank_docs(), top_n=2, backend="cross-encoder")

        assert [d.id for d in out] == ["b", "c"]
        assert out[0].score == pytest.approx(0.9)
        assert out[1].score == pytest.approx(0.5)
        assert len(out) == 2

    def test_overwrites_original_scores(self, monkeypatch):
        monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: self._fake_ce([0.4, 0.4, 0.4]))
        out = rerank.rerank("q", _rerank_docs(), top_n=3, backend="cross-encoder")
        assert all(d.score == pytest.approx(0.4) for d in out)

    def test_builds_query_document_pairs(self, monkeypatch):
        ce = self._fake_ce([0.5, 0.5, 0.5])
        monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: ce)
        rerank.rerank("my query", _rerank_docs(), top_n=3, backend="cross-encoder")

        (pairs,), _ = ce.predict.call_args
        assert pairs == [["my query", "alpha"], ["my query", "bravo"], ["my query", "charlie"]]


class TestCohereRerank:
    def _fake_cohere(self, ranked):
        """ranked = list of (index, relevance_score) tuples."""
        results = [MagicMock(index=i, relevance_score=s) for i, s in ranked]
        client = MagicMock()
        client.rerank.return_value = MagicMock(results=results)
        return client

    def test_uses_returned_order_and_scores(self, monkeypatch):
        import helpers

        # Cohere ranks doc index 2 (0.95) first, then index 0 (0.40).
        client = self._fake_cohere([(2, 0.95), (0, 0.40)])
        monkeypatch.setattr(helpers, "get_cohere_client", lambda: client)

        out = rerank.rerank("q", _rerank_docs(), top_n=2, backend="cohere")

        assert [d.id for d in out] == ["c", "a"]
        assert out[0].score == pytest.approx(0.95)
        assert out[1].score == pytest.approx(0.40)

        _, kwargs = client.rerank.call_args
        assert kwargs["documents"] == ["alpha", "bravo", "charlie"]
        assert kwargs["top_n"] == 2
        assert kwargs["query"] == "q"


class TestRerankAsDicts:
    def test_flattens_reranked_docs(self, monkeypatch):
        # rerank_as_dicts() doesn't take a backend arg — it uses whatever the
        # default (RERANKER, "cohere") is. Rather than depend on that, stub the
        # underlying rerank() to return a known ordered list and check flattening.
        reranked = [_doc("b", text="bravo", score=0.9, metadata={"source": "2"})]
        monkeypatch.setattr(rerank, "rerank", lambda q, docs, top_n=4: reranked)

        out = rerank.rerank_as_dicts("q", _rerank_docs(), top_n=1)
        assert out == [{"id": "b", "text": "bravo", "score": 0.9, "source": "2"}]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
