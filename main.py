import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from app.graph import ingest_graph, query_graph, insights_graph

load_dotenv()

app = FastAPI(
    title="Document Intelligence System",
    description="""
## N-ERGY Document Intelligence System

Upload PDFs and ask natural language questions. The system uses a multi-stage
accuracy-optimised RAG pipeline:

**Ingestion:** Docling parsing → semantic chunking → Gemini embeddings → ChromaDB + BM25 index

**Query:** Multi-query expansion → Hybrid retrieval (BM25 + dense + RRF) →
Cross-encoder reranking + MMR → Gemini answer generation → Hallucination guard

**Bonus:** `/insights` — proactive key insights and next steps from all indexed documents
    """,
    version="1.0.0"
)

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    filenames: Optional[List[str]] = None  # scope retrieval to specific documents


class InsightsRequest(BaseModel):
    filename: Optional[str] = None  # scope insights to a specific document


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["Ingestion"], summary="Upload and index a PDF")
async def upload_document(file: UploadFile = File(...)):
    """
    Validates, parses, and indexes a PDF into the vector store.

    - Rejects empty, encrypted, or non-PDF files.
    - Detects duplicate uploads via SHA-256 hash — skips re-indexing.
    - Scanned (image-only) PDFs are processed via Docling's built-in OCR.
    - Returns status and chunk count on success.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    initial_state = {
        "file_path": file_path,
        "status": "",
        "error_message": None,
        "file_hash": "",
        "raw_text": "",
        "chunks": [],
    }

    result = ingest_graph.invoke(initial_state)
    status = result.get("status", "")

    if status == "DUPLICATE":
        return {
            "status": "skipped",
            "message": f"'{file.filename}' is already indexed. Skipping re-ingestion.",
            "filename": file.filename,
        }

    if status in ("REJECTED", "ERROR"):
        raise HTTPException(
            status_code=422,
            detail=result.get("error_message", "Document could not be processed.")
        )

    chunk_count = len(result.get("chunks", []))
    return {
        "status": "success",
        "message": f"'{file.filename}' indexed successfully.",
        "filename": file.filename,
        "chunks_indexed": chunk_count,
        "document_type": status,  # READY or SCANNED
    }


@app.post("/query", tags=["Query"], summary="Ask a question about uploaded documents")
async def query_documents(request: QueryRequest):
    """
    Answers a natural language question using the indexed documents.

    Pipeline:
    1. Multi-query expansion (3 query variants)
    2. Hybrid retrieval: BM25 + dense search fused with RRF
    3. Cross-encoder reranking + MMR diversity selection
    4. Gemini answer generation with inline citations (filename + section)
    5. Hallucination guard verification

    Optional `filenames` list scopes retrieval to specific documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    initial_state = {
        "question": request.question.strip(),
        "filenames": request.filenames or [],
        "generated_queries": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "retry_count": 0,
        "final_answer": "",
    }

    result = query_graph.invoke(initial_state)
    answer = result.get("final_answer", "No answer generated.")

    return {
        "question": request.question,
        "answer": answer,
        "queries_used": result.get("generated_queries", [request.question]),
    }


@app.post("/insights", tags=["Insights"], summary="Generate key insights from indexed documents")
async def get_insights(request: InsightsRequest = InsightsRequest()):
    """
    Proactively surfaces key insights and actionable next steps from all
    indexed documents without requiring a specific question.

    Uses 5 broad probe queries across findings, risks, recommendations,
    data, and context — providing better coverage than a single generic query.

    Optional `filename` scopes the analysis to a single document.
    """
    initial_state = {
        "question": request.filename or "",
    }

    result = insights_graph.invoke(initial_state)
    return {
        "insights": result.get("final_answer", "No insights generated."),
        "scoped_to": request.filename or "all documents",
    }


@app.get("/health", tags=["System"], summary="Health check")
def health():
    return {"status": "ok"}
