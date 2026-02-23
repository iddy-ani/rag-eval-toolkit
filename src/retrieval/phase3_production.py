"""
Phase 3: Production-Grade RAG Pipeline

This is the backup version of the code that Claude Code generates live on camera.
During recording, Claude Code will generate something similar. This file serves as
a safety net in case of issues during the recording session.

The key improvements over Phase 2:
- Configuration via dataclass (no magic constants)
- Proper logging instead of print statements
- Error handling for missing models, empty corpora, edge cases
- Clean public API via a single HybridRAGPipeline class
- Type hints throughout
- Designed to be importable and testable
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Chunking
    max_chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Re-ranker model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hybrid scoring
    semantic_weight: float = 0.7  # BM25 weight = 1 - semantic_weight

    # Retrieval
    initial_candidates: int = 20
    final_top_k: int = 5

    # Minimum score threshold (cross-encoder scores can be negative)
    min_score_threshold: float = -20.0


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""

    text: str
    score: float
    method: str  # "semantic", "bm25", "hybrid", "reranked"
    rank: int


class HybridRAGPipeline:
    """Production-grade hybrid RAG pipeline.

    Combines sentence-aware chunking, semantic search, BM25 keyword search,
    weighted hybrid scoring, and cross-encoder re-ranking.

    Usage:
        pipeline = HybridRAGPipeline()
        pipeline.add_documents(["path/to/docs/"])
        results = pipeline.query("What is RAG?")
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

        logger.info("Loading embedding model: %s", self.config.embedding_model)
        self._embedder = SentenceTransformer(self.config.embedding_model)

        logger.info("Loading re-ranker model: %s", self.config.reranker_model)
        self._reranker = CrossEncoder(self.config.reranker_model)

        self._indexed = False

    @property
    def num_chunks(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def add_documents(self, sources: list[str]) -> int:
        """Add documents from file paths or directory paths.

        Args:
            sources: List of file paths or directory paths containing .txt files.

        Returns:
            Number of chunks created.
        """
        texts: list[str] = []
        for source in sources:
            path = Path(source)
            if path.is_dir():
                for filepath in sorted(path.glob("*.txt")):
                    texts.append(filepath.read_text(encoding="utf-8"))
                    logger.info("Loaded: %s", filepath.name)
            elif path.is_file():
                texts.append(path.read_text(encoding="utf-8"))
                logger.info("Loaded: %s", path.name)
            else:
                logger.warning("Skipping invalid path: %s", source)

        if not texts:
            logger.warning("No documents loaded.")
            return 0

        # Chunk all documents
        new_chunks: list[str] = []
        for text in texts:
            new_chunks.extend(self._chunk_text(text))

        self._chunks.extend(new_chunks)
        logger.info("Created %d chunks from %d documents", len(new_chunks), len(texts))

        # Rebuild indices
        self._build_index()
        return len(new_chunks)

    def query(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Search for the most relevant chunks.

        Args:
            question: The search query.
            top_k: Number of results to return. Defaults to config.final_top_k.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        if not self._indexed:
            logger.warning("No documents indexed. Call add_documents() first.")
            return []

        top_k = top_k or self.config.final_top_k
        n_candidates = self.config.initial_candidates

        # Stage 1: Semantic search
        semantic_results = self._semantic_search(question, n_candidates)

        # Stage 2: BM25 keyword search
        bm25_results = self._bm25_search(question, n_candidates)

        # Stage 3: Hybrid fusion
        hybrid_results = self._hybrid_fusion(semantic_results, bm25_results, n_candidates)

        # Stage 4: Cross-encoder re-ranking
        reranked = self._rerank(question, hybrid_results, top_k)

        # Filter by minimum score
        reranked = [r for r in reranked if r.score >= self.config.min_score_threshold]

        return reranked

    def _chunk_text(self, text: str) -> list[str]:
        """Sentence-aware chunking with overlap."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for sentence in sentences:
            size = len(sentence)
            if current_size + size > self.config.max_chunk_size and current:
                chunks.append(" ".join(current))

                # Overlap
                overlap: list[str] = []
                overlap_size = 0
                for s in reversed(current):
                    if overlap_size + len(s) <= self.config.chunk_overlap:
                        overlap.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current = overlap
                current_size = overlap_size

            current.append(sentence)
            current_size += size

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _build_index(self) -> None:
        """Build semantic and BM25 indices."""
        if not self._chunks:
            return

        logger.info("Building semantic index for %d chunks...", len(self._chunks))
        self._embeddings = self._embedder.encode(
            self._chunks, convert_to_numpy=True, show_progress_bar=False
        )

        logger.info("Building BM25 index...")
        tokenized = [chunk.lower().split() for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        self._indexed = True
        logger.info("Indexing complete.")

    def _semantic_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Semantic similarity search."""
        query_emb = self._embedder.encode([query], convert_to_numpy=True)
        sims = np.dot(self._embeddings, query_emb.T).flatten()
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        scores = sims / np.maximum(norms, 1e-10)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """BM25 keyword search."""
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]

    def _hybrid_fusion(
        self,
        semantic: list[tuple[str, float]],
        bm25: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Weighted fusion of semantic and BM25 results."""
        sem_norm = self._min_max_normalise(dict(semantic))
        bm25_norm = self._min_max_normalise(dict(bm25))

        all_chunks = set(sem_norm.keys()) | set(bm25_norm.keys())
        alpha = self.config.semantic_weight

        combined = []
        for chunk in all_chunks:
            score = alpha * sem_norm.get(chunk, 0.0) + (1 - alpha) * bm25_norm.get(chunk, 0.0)
            combined.append((chunk, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _rerank(
        self, query: str, candidates: list[tuple[str, float]], top_k: int
    ) -> list[RetrievalResult]:
        """Cross-encoder re-ranking."""
        if not candidates:
            return []

        pairs = [(query, chunk) for chunk, _ in candidates]
        scores = self._reranker.predict(pairs)

        results = []
        for i, (chunk, _) in enumerate(candidates):
            results.append(
                RetrievalResult(
                    text=chunk,
                    score=float(scores[i]),
                    method="reranked",
                    rank=0,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for rank, result in enumerate(results):
            result.rank = rank + 1

        return results[:top_k]

    @staticmethod
    def _min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
        """Normalise scores to [0, 1]."""
        if not scores:
            return scores
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


# --- Demo ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_path = str(Path(__file__).parent.parent.parent / "data" / "sample_docs")

    config = RAGConfig(
        max_chunk_size=500,
        chunk_overlap=50,
        semantic_weight=0.7,
        initial_candidates=20,
        final_top_k=3,
    )

    pipeline = HybridRAGPipeline(config)
    num_chunks = pipeline.add_documents([data_path])

    print(f"\nIndexed {num_chunks} chunks. Running queries...\n")

    queries = [
        "What is retrieval augmented generation?",
        "How do you make sure the AI doesn't make things up?",
        "BM25 keyword search algorithm",
        "What chunking strategies work best for code files?",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = pipeline.query(query)
        for result in results:
            print(f"  [{result.rank}] Score: {result.score:.4f} | {result.text[:120]}...")
        print()
