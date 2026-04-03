import os
import json
import pickle
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from app.state import AgentState

CHROMA_DIR = "./chroma_db"
BM25_DIR = "./bm25_index"

# Dense retrieval relevance threshold.
# gemini-embedding-001 produces L2-normalized vectors, so ChromaDB's
# squared L2 distance is equivalent to cosine distance:
#   L2^2 = 2(1 - cos θ)
# threshold=1.0 → cos θ = 0.5 (discard chunks with < 50% cosine similarity)
RELEVANCE_THRESHOLD = 1.0

# Number of candidates to fetch per query before fusion
DENSE_K = 20
BM25_K = 20
# Final number of candidates passed to reranker
FUSION_K = 30


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _load_bm25_store() -> Tuple[List, List]:
    corpus_path = os.path.join(BM25_DIR, "corpus.pkl")
    meta_path = os.path.join(BM25_DIR, "metadata.json")
    if os.path.exists(corpus_path) and os.path.exists(meta_path):
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        return corpus, metadata
    return [], []


def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]], k: int = 60
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF) score aggregation.

    For each document across all ranked lists:
        RRF(d) = Σ  1 / (k + rank(d, list_i))

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    A higher k smooths out the influence of top-ranked documents.
    """
    scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def retrieve_chunks(state: AgentState) -> dict:
    """
    Node 6 — Hybrid Retriever (BM25 + Dense + RRF).

    For each query variant from multi_query:
        1. Dense retrieval: cosine similarity via ChromaDB (gemini-embedding-001).
           Chunks with squared L2 distance > 1.0 (cosine < 0.5) are discarded.
        2. BM25 retrieval: keyword-based scoring via BM25Okapi.
           Handles exact matches (names, numbers, codes) that dense misses.

    Results from both retrievers across all query variants are merged using
    Reciprocal Rank Fusion (RRF) — a parameter-free fusion that combines
    rankings without requiring score normalization.

    Deduplication by chunk text hash ensures no chunk appears twice even if
    retrieved by multiple query variants.

    Output: top FUSION_K unique, fused candidates passed to the reranker.
    """
    queries: List[str] = state.get("generated_queries", [])
    original_question: str = state.get("question", "")
    filenames: List[str] = state.get("filenames", [])  # optional scope filter

    if not queries:
        queries = [original_question] if original_question else []

    print(f"----- RETRIEVER: Running hybrid search for {len(queries)} queries -----")

    if not queries:
        return {"retrieved_chunks": [], "error_message": "No queries provided."}

    try:
        # --- Dense retriever setup ---
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

        # --- BM25 retriever setup ---
        corpus, bm25_meta = _load_bm25_store()
        bm25 = BM25Okapi(corpus) if corpus else None

        # chunk_id → Document (accumulated across all queries)
        all_docs: Dict[str, Document] = {}

        # ranked lists for RRF: list of doc_id lists (one per retriever per query)
        rrf_lists: List[List[str]] = []

        for query in queries:
            # --- Dense search ---
            dense_results = vectorstore.similarity_search_with_score(query, k=DENSE_K)
            dense_ids = []
            for doc, score in dense_results:
                if score > RELEVANCE_THRESHOLD:
                    continue
                # Apply filename filter if provided
                if filenames and doc.metadata.get("filename", "") not in filenames:
                    continue
                doc_id = hash(doc.page_content)
                all_docs[doc_id] = doc
                dense_ids.append(doc_id)
            if dense_ids:
                rrf_lists.append(dense_ids)

            # --- BM25 search ---
            if bm25 and bm25_meta:
                tokens = _tokenize(query)
                bm25_scores = bm25.get_scores(tokens)
                top_bm25_indices = sorted(
                    range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
                )[:BM25_K]

                bm25_ids = []
                for idx in top_bm25_indices:
                    if bm25_scores[idx] <= 0:
                        continue
                    meta = bm25_meta[idx]
                    if filenames and meta.get("filename", "") not in filenames:
                        continue
                    doc_id = hash(meta["text"])
                    if doc_id not in all_docs:
                        all_docs[doc_id] = Document(
                            page_content=meta["text"],
                            metadata={
                                "source": meta.get("source", ""),
                                "filename": meta.get("filename", ""),
                                "file_hash": meta.get("file_hash", ""),
                                "Header 1": meta.get("Header 1", ""),
                                "Header 2": meta.get("Header 2", ""),
                                "Header 3": meta.get("Header 3", ""),
                            }
                        )
                    bm25_ids.append(doc_id)
                if bm25_ids:
                    rrf_lists.append(bm25_ids)

        if not all_docs:
            print("----- RETRIEVER: No chunks passed relevance filter -----")
            return {"retrieved_chunks": [], "error_message": None}

        # --- RRF fusion ---
        fused_scores = _reciprocal_rank_fusion(rrf_lists)
        sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
        fused_docs = [all_docs[doc_id] for doc_id in sorted_ids if doc_id in all_docs][:FUSION_K]

        print(f"----- RETRIEVER: {len(fused_docs)} fused candidates after RRF -----")
        return {"retrieved_chunks": fused_docs, "error_message": None}

    except Exception as e:
        print(f"----- RETRIEVER ERROR: {e} -----")
        return {"retrieved_chunks": [], "error_message": str(e)}
