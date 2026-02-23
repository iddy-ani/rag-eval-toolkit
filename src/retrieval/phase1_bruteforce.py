"""
Phase 1: Brute Force RAG — TF-IDF and Cosine Similarity from Scratch

No external libraries. Pure Python. Understanding the fundamentals of
text retrieval before we optimise.

This is intentionally rough — the point is to see where naive approaches break.
"""

import os
import math
from collections import Counter
from pathlib import Path


def load_documents(folder_path: str) -> list[dict[str, str]]:
    """Load all .txt files from a folder. Returns list of {name, content}."""
    docs = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append({"name": filename, "content": f.read()})
    return docs


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into fixed-size character chunks. Naive — splits mid-sentence."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    # Replace common punctuation with spaces
    for char in ".,;:!?()[]{}\"'\n\t-/":
        text = text.replace(char, " ")
    return [word for word in text.split() if len(word) > 1]


def term_frequency(term: str, document_tokens: list[str]) -> float:
    """How often does this term appear in this document?"""
    if not document_tokens:
        return 0.0
    count = document_tokens.count(term)
    return count / len(document_tokens)


def inverse_document_frequency(term: str, corpus_tokens: list[list[str]]) -> float:
    """How rare is this term across all documents?"""
    num_docs = len(corpus_tokens)
    docs_containing = sum(1 for doc in corpus_tokens if term in doc)
    if docs_containing == 0:
        return 0.0
    return math.log(num_docs / docs_containing)


def build_vocabulary(corpus_tokens: list[list[str]]) -> list[str]:
    """Build a sorted vocabulary from all tokens in the corpus."""
    vocab = set()
    for doc_tokens in corpus_tokens:
        vocab.update(doc_tokens)
    return sorted(vocab)


def tfidf_vector(
    document_tokens: list[str],
    corpus_tokens: list[list[str]],
    vocabulary: list[str],
) -> list[float]:
    """Compute the TF-IDF vector for a document given a vocabulary."""
    vector = []
    for term in vocabulary:
        tf = term_frequency(term, document_tokens)
        idf = inverse_document_frequency(term, corpus_tokens)
        vector.append(tf * idf)
    return vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors. No numpy needed."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


class BruteForceRAG:
    """Phase 1: Brute force retrieval using TF-IDF and cosine similarity."""

    def __init__(self, folder_path: str, chunk_size: int = 500):
        self.docs = load_documents(folder_path)
        self.chunks: list[str] = []
        for doc in self.docs:
            self.chunks.extend(chunk_text(doc["content"], chunk_size))

        # Tokenize all chunks
        self.corpus_tokens = [tokenize(chunk) for chunk in self.chunks]

        # Build vocabulary and compute TF-IDF vectors
        self.vocabulary = build_vocabulary(self.corpus_tokens)
        self.vectors = [
            tfidf_vector(tokens, self.corpus_tokens, self.vocabulary)
            for tokens in self.corpus_tokens
        ]

    def query(self, question: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Search for the most relevant chunks given a question."""
        question_tokens = tokenize(question)
        question_vector = tfidf_vector(
            question_tokens, self.corpus_tokens, self.vocabulary
        )

        # Score every chunk
        scores = []
        for i, chunk_vector in enumerate(self.vectors):
            score = cosine_similarity(question_vector, chunk_vector)
            scores.append((self.chunks[i], score))

        # Sort by score descending, return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# --- Demo ---
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "sample_docs"
    rag = BruteForceRAG(str(data_path))

    print(f"Loaded {len(rag.docs)} documents, {len(rag.chunks)} chunks")
    print(f"Vocabulary size: {len(rag.vocabulary)} terms")
    print()

    # Query 1: keyword match — should work OK
    query1 = "What is retrieval augmented generation?"
    print(f"Query: {query1}")
    results = rag.query(query1)
    for i, (chunk, score) in enumerate(results):
        print(f"  [{i+1}] Score: {score:.4f} | {chunk[:100]}...")
    print()

    # Query 2: semantic concept — TF-IDF will struggle
    query2 = "How do you make sure the AI doesn't make things up?"
    print(f"Query: {query2}")
    results = rag.query(query2)
    for i, (chunk, score) in enumerate(results):
        print(f"  [{i+1}] Score: {score:.4f} | {chunk[:100]}...")
    print()

    # Query 3: specific term
    query3 = "BM25 keyword search algorithm"
    print(f"Query: {query3}")
    results = rag.query(query3)
    for i, (chunk, score) in enumerate(results):
        print(f"  [{i+1}] Score: {score:.4f} | {chunk[:100]}...")
