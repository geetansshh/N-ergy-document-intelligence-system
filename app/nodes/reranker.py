from typing import List, Tuple
import numpy as np
from langchain_core.documents import Document
from app.state import AgentState

# Cross-encoder loaded once at module level
_cross_encoder = None

# Final number of chunks passed to the generator
TOP_K = 10
# MMR diversity penalty weight (λ): 1.0 = pure relevance, 0.0 = pure diversity
MMR_LAMBDA = 0.6


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        # ms-marco model fine-tuned for passage re-ranking
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _cross_encoder_scores(query: str, docs: List[Document]) -> List[float]:
    """Score each (query, doc) pair with the cross-encoder."""
    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    return scores.tolist()


def _mmr_select(
    docs: List[Document],
    relevance_scores: List[float],
    k: int,
    lambda_: float,
) -> List[Document]:
    """
    Maximal Marginal Relevance selection.

    Iteratively picks documents that maximise:
        score(d) = λ * relevance(d, query)
                 - (1-λ) * max_{d' ∈ selected} similarity(d, d')

    Similarity between documents is cosine similarity on their
    all-MiniLM-L6-v2 embeddings (loaded from the chunker's model).

    This prevents the generator from receiving k near-identical chunks —
    especially important after multi-query retrieval which can surface
    many semantically overlapping results.
    """
    if len(docs) <= k:
        return docs

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(
        [d.page_content for d in docs],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    selected_indices: List[int] = []
    remaining = list(range(len(docs)))

    # Normalise relevance scores to [0, 1]
    rel = np.array(relevance_scores, dtype=float)
    rel_min, rel_max = rel.min(), rel.max()
    if rel_max > rel_min:
        rel = (rel - rel_min) / (rel_max - rel_min)
    else:
        rel = np.ones_like(rel)

    while len(selected_indices) < k and remaining:
        if not selected_indices:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: rel[i])
        else:
            selected_embs = doc_embeddings[selected_indices]
            best_score = -np.inf
            best = remaining[0]
            for i in remaining:
                redundancy = float(np.max(doc_embeddings[i] @ selected_embs.T))
                mmr_score = lambda_ * rel[i] - (1 - lambda_) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i

        selected_indices.append(best)
        remaining.remove(best)

    return [docs[i] for i in selected_indices]


def rerank_chunks(state: AgentState) -> dict:
    """
    Node 7 — Cross-Encoder Reranker + MMR.

    Two-stage refinement of the hybrid retrieval candidates:

    Stage 1 — Cross-encoder reranking:
        Each (original_question, chunk) pair is scored by a fine-tuned
        ms-marco cross-encoder. Unlike bi-encoder embeddings, the cross-
        encoder sees both texts simultaneously, enabling deep attention-based
        relevance scoring rather than approximate vector similarity.
        Result: all candidates sorted by true relevance to the query.

    Stage 2 — MMR diversity selection:
        From the reranked list, MMR iteratively selects TOP_K chunks that
        balance relevance (cross-encoder score) against redundancy (cosine
        similarity to already-selected chunks).
        λ=0.6 weights relevance slightly above diversity.

    Output: TOP_K non-redundant, highly relevant chunks for the generator.
    """
    docs: List[Document] = state.get("retrieved_chunks", [])
    question: str = state.get("question", "")

    print(f"----- RERANKER: Scoring {len(docs)} candidates -----")

    if not docs:
        return {"reranked_chunks": []}

    if not question:
        return {"reranked_chunks": docs[:TOP_K]}

    try:
        # Stage 1: cross-encoder scoring
        scores = _cross_encoder_scores(question, docs)
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        sorted_docs, sorted_scores = zip(*scored)
        sorted_docs = list(sorted_docs)
        sorted_scores = list(sorted_scores)

        # Stage 2: MMR diversity selection
        final_chunks = _mmr_select(sorted_docs, sorted_scores, k=TOP_K, lambda_=MMR_LAMBDA)

        print(f"----- RERANKER: {len(final_chunks)} chunks selected after MMR -----")
        return {"reranked_chunks": final_chunks}

    except Exception as e:
        print(f"----- RERANKER ERROR: {e} — passing through top-k -----")
        return {"reranked_chunks": docs[:TOP_K]}
