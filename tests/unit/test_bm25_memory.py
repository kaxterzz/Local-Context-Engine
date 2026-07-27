"""Tests for BM25 corpus memory bounds and streaming build."""

from __future__ import annotations

from local_context_engine.retrieval.bm25_retriever import BM25Retriever, _tokenize


class TestBM25MemoryBounds:
    def test_add_batch_and_search(self) -> None:
        bm25 = BM25Retriever()
        bm25.add_batch(["c1", "c2"], ["user login handler", "invoice pdf export"])
        bm25.add_batch(["c3"], ["user profile settings page"])
        bm25.finalize()
        results = bm25.search("user login")
        assert results
        assert results[0].chunk_id == "c1"
        assert bm25.corpus_size == 3

    def test_max_docs_cap(self) -> None:
        bm25 = BM25Retriever(max_docs=5)
        added = bm25.add_batch([f"c{i}" for i in range(10)],
                               [f"text {i}" for i in range(10)])
        assert added == 5
        assert bm25.corpus_size == 5
        assert bm25.truncated
        # Further batches add nothing
        assert bm25.add_batch(["x"], ["y"]) == 0

    def test_tokens_per_doc_cap(self) -> None:
        long_text = " ".join(f"tok{i}" for i in range(1000))
        assert len(_tokenize(long_text, max_tokens=64)) == 64

        bm25 = BM25Retriever(max_tokens_per_doc=16)
        bm25.add_batch(["c1"], [long_text])
        assert bm25._acc_doc_lens[0] <= 16

    def test_raw_corpus_not_retained(self) -> None:
        """Neither raw texts nor accumulation buffers survive finalization."""
        bm25 = BM25Retriever()
        bm25.add_batch(
            ["c1", "c2", "c3"], ["alpha beta", "gamma delta", "epsilon zeta"]
        )
        assert bm25._acc_terms is not None
        bm25.finalize()
        assert bm25._acc_terms is None       # accumulation buffers freed
        assert bm25._acc_freqs is None
        assert not hasattr(bm25, "_corpus")  # raw texts never stored
        # Search still works from the built index
        assert bm25.search("alpha")[0].chunk_id == "c1"

    def test_add_documents_replaces_corpus(self) -> None:
        bm25 = BM25Retriever()
        bm25.add_documents(["c1"], ["first corpus"])
        bm25.add_documents(
            ["c2", "c3", "c4"],
            ["second corpus", "third things", "fourth items"],
        )
        assert bm25.corpus_size == 3
        assert bm25.search("second")[0].chunk_id == "c2"

    def test_reset_releases_everything(self) -> None:
        bm25 = BM25Retriever()
        bm25.add_documents(["c1"], ["some text"])
        bm25.finalize()
        bm25.reset()
        assert bm25.corpus_size == 0
        assert bm25.search("some") == []

    def test_add_after_finalize_starts_fresh(self) -> None:
        bm25 = BM25Retriever()
        bm25.add_documents(["c1"], ["old text"])
        bm25.finalize()
        bm25.add_batch(
            ["c2", "c3", "c4"], ["new text", "other words", "more stuff"]
        )
        bm25.finalize()
        assert bm25.corpus_size == 3
        assert bm25.search("new")[0].chunk_id == "c2"
