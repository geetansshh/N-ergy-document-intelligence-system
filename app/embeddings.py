"""
Central embeddings factory.

Set the environment variable EMBEDDING_PROVIDER to switch providers:
  - "gemini"  (default) — Google gemini-embedding-001 via API (3072 dims)
  - "bge"               — BAAI/bge-large-en-v1.5 via sentence-transformers (1024 dims)
  - "local"             — all-MiniLM-L6-v2 via sentence-transformers (384 dims, fast/light)

Each provider uses its own ChromaDB persist directory so that switching
providers never causes a dimension-mismatch error on an existing collection.
"""

import os

PROVIDER_GEMINI = "gemini"
PROVIDER_BGE    = "bge"
PROVIDER_LOCAL  = "local"

# Separate ChromaDB directories per provider to avoid dimension mismatch
CHROMA_DIRS = {
    PROVIDER_GEMINI: "./chroma_db",
    PROVIDER_BGE:    "./chroma_db_bge",
    PROVIDER_LOCAL:  "./chroma_db_local",
}


def get_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", PROVIDER_GEMINI).lower()


def get_chroma_dir() -> str:
    return CHROMA_DIRS.get(get_provider(), CHROMA_DIRS[PROVIDER_GEMINI])


def get_embeddings():
    """Return the LangChain embeddings object for the active provider."""
    provider = get_provider()

    if provider == PROVIDER_BGE:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": _best_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == PROVIDER_LOCAL:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": _best_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Default: Gemini
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _best_device() -> str:
    """Pick GPU if available, otherwise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
