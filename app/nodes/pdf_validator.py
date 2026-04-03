import hashlib
import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.state import AgentState

CHROMA_DIR = "./chroma_db"


def _sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_pdf(state: AgentState) -> dict:
    """
    Node 1 — Validator + Dedup Check.

    Performs two jobs in one pass:
    1. Structural validation: rejects empty, encrypted, or image-only PDFs.
       Flags scanned PDFs (low text + images) as SCANNED — Docling's OCR handles these.
    2. Deduplication: computes SHA-256 of the file bytes and checks whether
       any previously ingested chunk carries the same hash in its metadata.
       If found, skips re-ingestion entirely to prevent vector store bloat.
    """
    file_path = state["file_path"]
    print(f"----- VALIDATOR: Checking {file_path} -----")

    try:
        doc = fitz.open(file_path)

        if doc.is_encrypted:
            doc.close()
            return {"status": "REJECTED", "error_message": "Password-protected PDF — cannot process."}

        if doc.page_count == 0:
            doc.close()
            return {"status": "REJECTED", "error_message": "PDF has zero pages."}

        total_text_chars = 0
        has_images = False
        pages_to_check = min(doc.page_count, 20)

        for i in range(pages_to_check):
            page = doc[i]
            total_text_chars += len(page.get_text().strip())
            if page.get_images():
                has_images = True

        doc.close()

        if total_text_chars == 0 and not has_images:
            return {"status": "REJECTED", "error_message": "PDF contains no extractable text or images."}

        pdf_status = "SCANNED" if (total_text_chars < 100 and has_images) else "READY"

        # --- Deduplication check ---
        file_hash = _sha256(file_path)
        print(f"----- VALIDATOR: SHA-256 = {file_hash[:12]}... -----")

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            existing = vectorstore.get(where={"file_hash": file_hash}, limit=1)
            if existing and existing.get("ids"):
                print(f"----- VALIDATOR: Duplicate detected — skipping ingestion -----")
                return {"status": "DUPLICATE", "error_message": None}
        except Exception:
            # If chroma is empty or unavailable, proceed normally
            pass

        print(f"----- VALIDATOR: Status = {pdf_status} -----")
        return {"status": pdf_status, "error_message": None, "metadata": {"file_hash": file_hash}}

    except Exception as e:
        return {"status": "ERROR", "error_message": str(e)}
