from typing import TypedDict, Optional, List
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # --- Ingestion ---
    file_path: str
    original_filename: str             # Original filename before temp-file renaming
    status: str                        # READY | SCANNED | REJECTED | ERROR | DUPLICATE
    error_message: Optional[str]
    file_hash: str                     # SHA-256 of PDF bytes (for dedup)
    raw_text: str                      # Markdown output from Docling
    chunks: List[Document]             # Final chunks with metadata

    # --- Query ---
    question: str                      # Original user question
    generated_queries: List[str]       # Multi-query variants (includes original)
    retrieved_chunks: List[Document]   # Chunks after hybrid retrieval + dedup
    reranked_chunks: List[Document]    # Chunks after cross-encoder + MMR
    retry_count: int                   # Query reformulation retry counter
    final_answer: str                  # Answer returned to user
