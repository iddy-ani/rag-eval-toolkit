"""
Phase 2: Optimised RAG — Hybrid Retrieval with Semantic Search, BM25, and Re-ranking

This is closer to what runs in production serving thousands of engineers.
Proper chunking, real embeddings, hybrid scoring, and cross-encoder re-ranking.

Dependencies: sentence-transformers, rank-bm25, numpy
"""

import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_documents(folder_path: str) -> list[dict[str, str]]:
    """Load all .txt files from a folder."""
    docs = []
    for filepath in sorted(Path(folder_path).glob("*.txt")):
        docs.append({"name": filepath.name, "content": filepath.read_text(encoding="utf-8")})
    return docs


class SentenceChunker:
    """Split text into chunks that respect sentence boundaries with overlap."""

    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks respecting sentence boundaries."""
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Calculate overlap: keep last sentences that fit in overlap size
                overlap_chunk: list[str] = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break

                current_chunk = overlap_chunk
                current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class SemanticSearch:
    """Semantic search using sentence-transformer embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.chunks: list[str] = []

    def index(self, chunks: list[str]) -> None:
        """Encode and store chunk embeddings."""
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find the most semantically similar chunks to the query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        scores = similarities / np.maximum(norms, 1e-10)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


class BM25Search:
    """BM25 keyword search — catches exact matches that embeddings miss."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for BM25."""
        return text.lower().split()

    def index(self, chunks: list[str]) -> None:
        """Build BM25 index from chunks."""
        self.chunks = chunks
        tokenized = [self._tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find chunks matching the query keywords."""
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


class HybridRetriever:
    """Combine semantic search and BM25 with weighted fusion."""

    def __init__(
        self,
        semantic: SemanticSearch,
        bm25: BM25Search,
        alpha: float = 0.7,
    ):
        self.semantic = semantic
        self.bm25 = bm25
        self.alpha = alpha  # Weight for semantic, (1 - alpha) for BM25

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Hybrid search combining semantic and BM25 scores."""
        semantic_results = self.semantic.search(query, top_k=top_k * 2)
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # Normalise scores to [0, 1]
        semantic_scores = self._normalise(
            {chunk: score for chunk, score in semantic_results}
        )
        bm25_scores = self._normalise(
            {chunk: score for chunk, score in bm25_results}
        )

        # Combine all unique chunks
        all_chunks = set(semantic_scores.keys()) | set(bm25_scores.keys())
        combined = []
        for chunk in all_chunks:
            sem_score = semantic_scores.get(chunk, 0.0)
            bm_score = bm25_scores.get(chunk, 0.0)
            hybrid_score = self.alpha * sem_score + (1 - self.alpha) * bm_score
            combined.append((chunk, hybrid_score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _normalise(self, scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalise scores to [0, 1]."""
        if not scores:
            return scores
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


class CrossEncoderReranker:
    """Re-rank candidates using a cross-encoder for higher accuracy."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[tuple[str, float]], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Re-rank candidate chunks using cross-encoder scores."""
        if not candidates:
            return []

        # Prepare query-document pairs
        pairs = [(query, chunk) for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        # Combine with original chunks
        reranked = [
            (candidates[i][0], float(scores[i])) for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]


# --- Demo ---
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "sample_docs"

    # Load and chunk documents
    docs = load_documents(str(data_path))
    chunker = SentenceChunker(max_chunk_size=500, overlap=50)
    chunks: list[str] = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc["content"]))

    print(f"Loaded {len(docs)} documents, {len(chunks)} chunks (sentence-aware)")
    print()

    # Build indices
    print("Building semantic index...")
    semantic = SemanticSearch()
    semantic.index(chunks)

    print("Building BM25 index...")
    bm25 = BM25Search()
    bm25.index(chunks)

    # Hybrid retriever
    hybrid = HybridRetriever(semantic, bm25, alpha=0.7)

    # Re-ranker
    print("Loading cross-encoder re-ranker...")
    reranker = CrossEncoderReranker()
    print()

    # Test queries
    queries = [
        "What is retrieval augmented generation?",
        "How do you make sure the AI doesn't make things up?",
        "BM25 keyword search algorithm",
        "What chunking strategies work best for code files?",
    ]

    for query in queries:
        print(f"Query: {query}")

        # Hybrid retrieval
        hybrid_results = hybrid.search(query, top_k=10)

        # Re-rank top results
        reranked = reranker.rerank(query, hybrid_results, top_k=3)

        for i, (chunk, score) in enumerate(reranked):
            print(f"  [{i+1}] Score: {score:.4f} | {chunk[:120]}...")
        print()
