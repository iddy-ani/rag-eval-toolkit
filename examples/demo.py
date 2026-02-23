"""
Demo: Compare all 3 phases of RAG retrieval side-by-side.

Run from the project root:
    python examples/demo.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_PATH = str(Path(__file__).parent.parent / "data" / "sample_docs")

QUERIES = [
    # Query 1: Direct keyword match — Phase 1 should handle this OK
    "What is retrieval augmented generation?",
    # Query 2: Semantic concept — Phase 1 will struggle, Phase 2 should nail it
    "How do you make sure the AI doesn't make things up?",
    # Query 3: Specific technical term — BM25 helps here
    "BM25 keyword search algorithm",
    # Query 4: Multi-concept query — needs hybrid approach
    "What chunking strategies work best for code files?",
]


def run_phase1():
    """Phase 1: Brute force TF-IDF retrieval."""
    from retrieval.phase1_bruteforce import BruteForceRAG

    print("=" * 70)
    print("PHASE 1: Brute Force (TF-IDF + Cosine Similarity)")
    print("=" * 70)

    rag = BruteForceRAG(DATA_PATH)
    print(f"Chunks: {len(rag.chunks)} | Vocabulary: {len(rag.vocabulary)} terms\n")

    for query in QUERIES:
        print(f"  Q: {query}")
        results = rag.query(query, top_k=3)
        for i, (chunk, score) in enumerate(results):
            print(f"    [{i+1}] {score:.4f} | {chunk[:100]}...")
        print()


def run_phase2():
    """Phase 2: Hybrid retrieval with semantic search + BM25 + re-ranking."""
    from retrieval.phase2_hybrid import (
        BM25Search,
        CrossEncoderReranker,
        HybridRetriever,
        SemanticSearch,
        SentenceChunker,
        load_documents,
    )

    print("=" * 70)
    print("PHASE 2: Optimised (Semantic + BM25 + Re-ranking)")
    print("=" * 70)

    docs = load_documents(DATA_PATH)
    chunker = SentenceChunker(max_chunk_size=500, overlap=50)
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc["content"]))

    print(f"Chunks: {len(chunks)} (sentence-aware with overlap)")
    print("Building indices...")

    semantic = SemanticSearch()
    semantic.index(chunks)

    bm25 = BM25Search()
    bm25.index(chunks)

    hybrid = HybridRetriever(semantic, bm25, alpha=0.7)
    reranker = CrossEncoderReranker()
    print()

    for query in QUERIES:
        print(f"  Q: {query}")
        candidates = hybrid.search(query, top_k=10)
        results = reranker.rerank(query, candidates, top_k=3)
        for i, (chunk, score) in enumerate(results):
            print(f"    [{i+1}] {score:.4f} | {chunk[:100]}...")
        print()


def run_phase3():
    """Phase 3: Production pipeline."""
    from retrieval.phase3_production import HybridRAGPipeline, RAGConfig

    print("=" * 70)
    print("PHASE 3: Production Pipeline (Config-driven, logged, tested)")
    print("=" * 70)

    config = RAGConfig(
        max_chunk_size=500,
        chunk_overlap=50,
        semantic_weight=0.7,
        initial_candidates=20,
        final_top_k=3,
    )
    pipeline = HybridRAGPipeline(config)
    num_chunks = pipeline.add_documents([DATA_PATH])
    print(f"Chunks: {num_chunks}\n")

    for query in QUERIES:
        print(f"  Q: {query}")
        results = pipeline.query(query)
        for r in results:
            print(f"    [{r.rank}] {r.score:.4f} | {r.text[:100]}...")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RAG FROM SCRATCH — Phase Comparison Demo")
    print("  Ship It with Idriss | Episode 1")
    print("=" * 70 + "\n")

    print("Running Phase 1...")
    run_phase1()

    print("\nRunning Phase 2...")
    run_phase2()

    print("\nRunning Phase 3...")
    run_phase3()

    print("=" * 70)
    print("Done. Compare the results above to see the quality improvement")
    print("from brute force → hybrid → production pipeline.")
    print("=" * 70)
