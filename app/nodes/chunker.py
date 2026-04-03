import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from app.state import AgentState

# Sentence-transformers is used locally — no API call, no cost.
# Loaded once at module level to avoid reloading on every request.
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _semantic_split(text: str, max_chunk_size: int = 500, similarity_threshold: float = 0.45) -> List[str]:
    """
    Splits text into semantically coherent chunks using cosine similarity
    between adjacent sentences.

    Algorithm:
    1. Split text into sentences on '. ', '! ', '? ' boundaries.
    2. Embed all sentences locally with all-MiniLM-L6-v2.
    3. Walk through sentence pairs — when cosine similarity between
       adjacent sentences drops below `similarity_threshold`, that is a
       topic boundary; start a new chunk there.
    4. If a growing chunk exceeds `max_chunk_size` characters, force-split
       regardless of similarity.

    This replaces the Groq LLM call in the inspiration system — same semantic
    quality with zero API latency or cost.
    """
    import numpy as np

    sentences = []
    for part in text.replace("\n", " ").split(". "):
        part = part.strip()
        if part:
            sentences.append(part if part.endswith(".") else part + ".")

    if len(sentences) <= 1:
        return [text]

    model = _get_embedding_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    chunks = []
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i - 1], embeddings[i]))
        sentence_len = len(sentences[i])

        topic_boundary = similarity < similarity_threshold
        size_exceeded = (current_len + sentence_len) > max_chunk_size

        if topic_boundary or size_exceeded:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_sentences = [sentences[i]]
            current_len = sentence_len
        else:
            current_sentences.append(sentences[i])
            current_len += sentence_len

    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks if chunks else [text]


def chunk_document(state: AgentState) -> dict:
    """
    Node 3 — Two-Pass Semantic Chunker.

    Pass 1 — Structural split (MarkdownHeaderTextSplitter):
        Splits the Docling Markdown at header boundaries (#, ##, ###).
        This keeps tables and their caption within the same logical section.

    Pass 2 — Semantic split (local sentence-transformers):
        For each structural section > 500 chars, applies cosine-similarity-
        based splitting to find topic boundaries within the section.
        Short sections are kept as-is.

    Each final chunk carries metadata:
        - source: original file path (for citation)
        - file_hash: SHA-256 of the source PDF (for dedup)
        - Header 1/2/3: section hierarchy from the Markdown structure
    """
    raw_markdown = state.get("raw_text", "")
    file_path = state.get("file_path", "")
    file_hash = state.get("file_hash", "")

    print(f"----- CHUNKER: Processing {file_path} -----")

    if not raw_markdown or raw_markdown.startswith("ERROR"):
        print("----- CHUNKER: No valid Markdown — skipping -----")
        return {"chunks": []}

    try:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        structural_sections = splitter.split_text(raw_markdown)

        base_metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_hash": file_hash,
        }

        final_chunks: List[Document] = []

        for section in structural_sections:
            text = section.page_content.strip()
            if not text:
                continue

            chunk_meta = {**base_metadata, **section.metadata}

            if len(text) <= 500:
                final_chunks.append(Document(page_content=text, metadata=chunk_meta))
                continue

            sub_texts = _semantic_split(text)
            for sub in sub_texts:
                if sub.strip():
                    final_chunks.append(Document(page_content=sub, metadata=chunk_meta))

        print(f"----- CHUNKER: {len(final_chunks)} chunks produced -----")
        return {"chunks": final_chunks}

    except Exception as e:
        print(f"----- CHUNKER ERROR: {e} -----")
        return {"chunks": []}
