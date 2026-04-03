import os
import json
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from app.state import AgentState

CHROMA_DIR = "./chroma_db"
BM25_DIR = "./bm25_index"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def _load_bm25_store() -> dict:
    """Load existing BM25 corpus and metadata from disk."""
    corpus_path = os.path.join(BM25_DIR, "corpus.pkl")
    meta_path = os.path.join(BM25_DIR, "metadata.json")
    if os.path.exists(corpus_path) and os.path.exists(meta_path):
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return {"corpus": corpus, "metadata": metadata}
    return {"corpus": [], "metadata": []}


def _save_bm25_store(corpus: list, metadata: list):
    """Persist BM25 corpus and metadata to disk."""
    os.makedirs(BM25_DIR, exist_ok=True)
    with open(os.path.join(BM25_DIR, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    with open(os.path.join(BM25_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def embed_documents(state: AgentState) -> dict:
    """
    Node 4 — Embedder.

    Persists chunks in two parallel indexes:

    1. ChromaDB (dense vector store):
       - Embeds each chunk with Google gemini-embedding-001 (3072 dims, full precision).
       - Stores chunk text, source, filename, file_hash, and header metadata.
       - Persisted to disk — queries work across sessions without re-ingestion.

    2. BM25 index (sparse keyword index):
       - Appends tokenized chunk text to a cumulative corpus on disk.
       - Rebuilt as BM25Okapi each time new chunks are added.
       - Enables exact keyword matching (names, codes, numbers) that dense
         search alone misses.

    Both indexes are queried in parallel during hybrid retrieval.
    """
    chunks: List[Document] = state.get("chunks", [])
    file_path = state.get("file_path", "")

    print(f"----- EMBEDDER: Indexing {len(chunks)} chunks from {file_path} -----")

    if not chunks:
        print("----- EMBEDDER: No chunks to embed -----")
        return {}

    try:
        # --- 1. Dense embeddings → ChromaDB ---
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        print(f"----- EMBEDDER: {len(chunks)} chunks saved to ChromaDB -----")

        # --- 2. Sparse index → BM25 on disk ---
        store = _load_bm25_store()
        existing_corpus: list = store["corpus"]
        existing_meta: list = store["metadata"]

        for doc in chunks:
            tokens = _tokenize(doc.page_content)
            existing_corpus.append(tokens)
            existing_meta.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "filename": doc.metadata.get("filename", ""),
                "file_hash": doc.metadata.get("file_hash", ""),
                "Header 1": doc.metadata.get("Header 1", ""),
                "Header 2": doc.metadata.get("Header 2", ""),
                "Header 3": doc.metadata.get("Header 3", ""),
            })

        _save_bm25_store(existing_corpus, existing_meta)
        print(f"----- EMBEDDER: BM25 index updated ({len(existing_corpus)} total docs) -----")

        return {}

    except Exception as e:
        print(f"----- EMBEDDER ERROR: {e} -----")
        return {}
