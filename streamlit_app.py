"""
RAG From Scratch — Interactive Streamlit Demo

Part 4 of the RAG From Scratch series on Ship It with Idriss.
Upload documents, run queries, compare brute force vs hybrid retrieval.
"""

import sys
from pathlib import Path

import streamlit as st

# Add src to path so we can import our pipeline modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval.phase1_bruteforce import BruteForceRAG
from retrieval.phase2_hybrid import (
    BM25Search,
    CrossEncoderReranker,
    HybridRetriever,
    SemanticSearch,
    SentenceChunker,
    load_documents,
)

# --- Page Config ---
st.set_page_config(
    page_title="RAG From Scratch | Ship It with Idriss",
    page_icon="🔍",
    layout="wide",
)

st.title("RAG From Scratch — Interactive Demo")
st.caption(
    "Compare brute force TF-IDF vs hybrid retrieval (semantic + BM25 + re-ranking). "
    "Built live on Ship It with Idriss."
)

# --- Sidebar Configuration ---
st.sidebar.header("Pipeline Configuration")

chunk_size = st.sidebar.slider("Chunk size (chars)", 200, 1000, 500, step=50)
overlap = st.sidebar.slider("Overlap (chars)", 0, 200, 50, step=10)
alpha = st.sidebar.slider(
    "Semantic weight (alpha)",
    0.0,
    1.0,
    0.7,
    step=0.05,
    help="0.0 = pure BM25, 1.0 = pure semantic, 0.7 = recommended hybrid",
)
top_k = st.sidebar.slider("Top-K results", 1, 10, 5)
initial_candidates = st.sidebar.slider(
    "Initial candidates (before re-ranking)", 5, 50, 20
)

retrieval_mode = st.sidebar.selectbox(
    "Retrieval method",
    [
        "Hybrid (Recommended)",
        "Semantic Only",
        "BM25 Only",
        "Brute Force TF-IDF (Part 1)",
        "Compare All Methods",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Series:** [RAG From Scratch](https://github.com/ianimash/rag-eval-toolkit)  \n"
    "**Channel:** Ship It with Idriss"
)


# --- Document Loading ---
st.header("1. Load Documents")

doc_source = st.radio(
    "Choose document source:",
    ["Use sample documents", "Upload your own"],
    horizontal=True,
)

documents: list[dict[str, str]] = []

if doc_source == "Use sample documents":
    sample_path = Path(__file__).parent / "data" / "sample_docs"
    if sample_path.exists():
        documents = load_documents(str(sample_path))
        st.success(f"Loaded {len(documents)} sample documents from `data/sample_docs/`")
        with st.expander("View document names"):
            for doc in documents:
                st.write(f"- {doc['name']} ({len(doc['content'])} chars)")
    else:
        st.error(
            f"Sample documents not found at `{sample_path}`. "
            "Make sure `data/sample_docs/` exists with .txt files."
        )
else:
    uploaded_files = st.file_uploader(
        "Upload .txt files",
        type=["txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            content = uf.read().decode("utf-8")
            documents.append({"name": uf.name, "content": content})
        st.success(f"Uploaded {len(documents)} documents")

if not documents:
    st.info("Load documents to get started.")
    st.stop()


# --- Pipeline Initialisation ---
@st.cache_resource
def build_brute_force(docs: tuple, chunk_sz: int) -> BruteForceRAG:
    """Build the brute force TF-IDF pipeline."""
    # Write docs to a temp directory so BruteForceRAG can load them
    import tempfile
    import os

    tmpdir = tempfile.mkdtemp()
    for name, content in docs:
        with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
            f.write(content)
    return BruteForceRAG(tmpdir, chunk_size=chunk_sz)


@st.cache_resource
def build_hybrid_pipeline(
    docs: tuple, chunk_sz: int, ovlp: int
) -> tuple:
    """Build the hybrid retrieval pipeline components."""
    chunker = SentenceChunker(max_chunk_size=chunk_sz, overlap=ovlp)
    all_chunks = []
    for _name, content in docs:
        all_chunks.extend(chunker.chunk(content))

    semantic = SemanticSearch()
    semantic.index(all_chunks)

    bm25 = BM25Search()
    bm25.index(all_chunks)

    return all_chunks, semantic, bm25


# Convert documents to a hashable format for caching
docs_tuple = tuple((d["name"], d["content"]) for d in documents)

# Build pipelines
with st.spinner("Building pipelines... (first run downloads the embedding model)"):
    needs_hybrid = retrieval_mode != "Brute Force TF-IDF (Part 1)"
    needs_brute = retrieval_mode in [
        "Brute Force TF-IDF (Part 1)",
        "Compare All Methods",
    ]

    if needs_hybrid:
        all_chunks, semantic, bm25 = build_hybrid_pipeline(
            docs_tuple, chunk_size, overlap
        )
        hybrid = HybridRetriever(semantic, bm25, alpha=alpha)
        reranker = CrossEncoderReranker()

    if needs_brute:
        brute_force = build_brute_force(docs_tuple, chunk_size)

st.success("Pipelines ready.")

# --- Query ---
st.header("2. Search")

query = st.text_input(
    "Enter your query:",
    placeholder="e.g. How do you make sure the AI doesn't make things up?",
)

if not query:
    st.info("Enter a query above to search.")
    st.stop()


def display_results(results: list[tuple[str, float]], method_label: str):
    """Display search results with score bars."""
    if not results:
        st.warning(f"No results found for {method_label}.")
        return

    st.subheader(method_label)
    max_score = max(score for _, score in results) if results else 1.0
    min_score = min(score for _, score in results) if results else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    for rank, (text, score) in enumerate(results, 1):
        # Normalise score to 0-1 for the progress bar
        normalised = (score - min_score) / score_range if score_range > 0 else 0.5
        normalised = max(0.0, min(1.0, normalised))

        col1, col2 = st.columns([1, 4])
        with col1:
            st.metric(f"#{rank}", f"{score:.4f}")
        with col2:
            st.progress(normalised)
            with st.expander(f"Chunk text ({len(text)} chars)"):
                st.text(text)


# --- Run Queries ---
st.header("3. Results")

if retrieval_mode == "Compare All Methods":
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Part 1: Brute Force TF-IDF")
        brute_results = brute_force.query(query, top_k=top_k)
        display_results(brute_results, "TF-IDF (Part 1)")

    with col_right:
        st.markdown("### Part 2: Hybrid Retrieval")
        hybrid_results = hybrid.search(query, top_k=initial_candidates)
        reranked = reranker.rerank(query, hybrid_results, top_k=top_k)
        display_results(reranked, "Hybrid + Re-ranking (Part 2)")

elif retrieval_mode == "Brute Force TF-IDF (Part 1)":
    brute_results = brute_force.query(query, top_k=top_k)
    display_results(brute_results, "TF-IDF Brute Force")

elif retrieval_mode == "Semantic Only":
    semantic_results = semantic.search(query, top_k=top_k)
    display_results(semantic_results, "Semantic Search (all-MiniLM-L6-v2)")

elif retrieval_mode == "BM25 Only":
    bm25_results = bm25.search(query, top_k=top_k)
    display_results(bm25_results, "BM25 Keyword Search")

elif retrieval_mode == "Hybrid (Recommended)":
    hybrid_results = hybrid.search(query, top_k=initial_candidates)
    reranked = reranker.rerank(query, hybrid_results, top_k=top_k)
    display_results(reranked, "Hybrid + Cross-Encoder Re-ranking")

# --- Footer ---
st.markdown("---")
st.markdown(
    "Built live on **Ship It with Idriss** | "
    "[GitHub](https://github.com/ianimash/rag-eval-toolkit) | "
    "RAG From Scratch Series Part 4"
)
