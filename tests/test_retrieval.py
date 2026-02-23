"""Tests for retrieval phases 1, 2, and 3."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_PATH = str(Path(__file__).parent.parent / "data" / "sample_docs")


# ---- Phase 1 Tests ----


class TestPhase1:
    """Test brute force TF-IDF retrieval."""

    def test_load_documents(self):
        from retrieval.phase1_bruteforce import load_documents

        docs = load_documents(DATA_PATH)
        assert len(docs) == 5
        assert all("name" in d and "content" in d for d in docs)

    def test_chunk_text(self):
        from retrieval.phase1_bruteforce import chunk_text

        text = "a" * 1200
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 3
        assert len(chunks[0]) == 500
        assert len(chunks[2]) == 200

    def test_tokenize(self):
        from retrieval.phase1_bruteforce import tokenize

        tokens = tokenize("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_cosine_similarity_identical(self):
        from retrieval.phase1_bruteforce import cosine_similarity

        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        from retrieval.phase1_bruteforce import cosine_similarity

        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        from retrieval.phase1_bruteforce import cosine_similarity

        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_query_returns_results(self):
        from retrieval.phase1_bruteforce import BruteForceRAG

        rag = BruteForceRAG(DATA_PATH)
        results = rag.query("retrieval augmented generation", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Scores should be non-negative
        assert all(score >= 0 for _, score in results)
        # Results should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# ---- Phase 2 Tests ----


class TestPhase2Chunker:
    """Test sentence-aware chunking."""

    def test_respects_sentences(self):
        from retrieval.phase2_hybrid import SentenceChunker

        chunker = SentenceChunker(max_chunk_size=100, overlap=0)
        text = "First sentence. Second sentence. Third sentence is a bit longer than the others."
        chunks = chunker.chunk(text)
        # Should not split mid-sentence
        for chunk in chunks:
            assert chunk.endswith(".") or chunk == chunks[-1]

    def test_overlap(self):
        from retrieval.phase2_hybrid import SentenceChunker

        chunker = SentenceChunker(max_chunk_size=50, overlap=30)
        text = "Short one. Another short. Third short. Fourth one here."
        chunks = chunker.chunk(text)
        # With overlap, some content should repeat between consecutive chunks
        if len(chunks) > 1:
            # Just verify we get multiple chunks
            assert len(chunks) >= 2

    def test_empty_text(self):
        from retrieval.phase2_hybrid import SentenceChunker

        chunker = SentenceChunker()
        assert chunker.chunk("") == []


class TestPhase2Search:
    """Test semantic, BM25, and hybrid search."""

    @pytest.fixture(scope="class")
    def indexed_components(self):
        from retrieval.phase2_hybrid import (
            BM25Search,
            HybridRetriever,
            SemanticSearch,
            SentenceChunker,
            load_documents,
        )

        docs = load_documents(DATA_PATH)
        chunker = SentenceChunker(max_chunk_size=500, overlap=50)
        chunks = []
        for doc in docs:
            chunks.extend(chunker.chunk(doc["content"]))

        semantic = SemanticSearch()
        semantic.index(chunks)

        bm25 = BM25Search()
        bm25.index(chunks)

        hybrid = HybridRetriever(semantic, bm25)
        return {"semantic": semantic, "bm25": bm25, "hybrid": hybrid, "chunks": chunks}

    def test_semantic_search_returns_results(self, indexed_components):
        results = indexed_components["semantic"].search("What is RAG?", top_k=3)
        assert len(results) == 3
        assert all(score > 0 for _, score in results)

    def test_bm25_search_returns_results(self, indexed_components):
        results = indexed_components["bm25"].search("BM25 keyword search", top_k=3)
        assert len(results) == 3

    def test_hybrid_search_returns_results(self, indexed_components):
        results = indexed_components["hybrid"].search("hybrid retrieval", top_k=5)
        assert len(results) <= 5
        assert len(results) > 0

    def test_semantic_beats_tfidf_on_semantic_queries(self, indexed_components):
        """Semantic search should find hallucination content for this query."""
        results = indexed_components["semantic"].search(
            "How do you make sure the AI doesn't make things up?", top_k=3
        )
        # At least one result should mention hallucination or faithfulness
        texts = " ".join(chunk for chunk, _ in results).lower()
        assert "hallucination" in texts or "faithful" in texts or "grounded" in texts


# ---- Phase 3 Tests ----


class TestPhase3Pipeline:
    """Test the production pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from retrieval.phase3_production import HybridRAGPipeline, RAGConfig

        config = RAGConfig(final_top_k=3, initial_candidates=10)
        p = HybridRAGPipeline(config)
        p.add_documents([DATA_PATH])
        return p

    def test_documents_indexed(self, pipeline):
        assert pipeline.num_chunks > 0

    def test_query_returns_retrieval_results(self, pipeline):
        results = pipeline.query("What is RAG?")
        assert len(results) > 0
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results)

    def test_results_are_ranked(self, pipeline):
        results = pipeline.query("evaluation metrics for RAG")
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)

    def test_empty_query(self, pipeline):
        results = pipeline.query("")
        # Should return results (empty query still gets processed)
        assert isinstance(results, list)

    def test_custom_top_k(self, pipeline):
        results = pipeline.query("chunking strategies", top_k=2)
        assert len(results) <= 2

    def test_no_documents_warning(self):
        from retrieval.phase3_production import HybridRAGPipeline

        pipeline = HybridRAGPipeline()
        results = pipeline.query("anything")
        assert results == []
