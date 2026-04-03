from docling.document_converter import DocumentConverter
from app.state import AgentState


def ingest_document(state: AgentState) -> dict:
    """
    Node 2 — Docling Parser.

    Converts the PDF to structured Markdown using Docling's two internal models:
    - DocLayNet (RT-DETR): classifies every visual element by bounding box
      (paragraphs, headers, tables, figures, equations).
    - TableFormer (vision transformer): processes detected table regions and
      emits OTSL sequences that preserve row/column spans and header hierarchy.

    Scanned PDFs (status=SCANNED) are routed through Docling's built-in
    ONNX/RapidOCR pipeline — no system-level Tesseract install required.

    Output is Markdown, which serves as a structured intermediary:
    headers become natural chunk anchors in the next stage.
    """
    file_path = state["file_path"]
    status = state.get("status", "READY")

    print(f"----- INGESTOR: Parsing {file_path} (status={status}) -----")

    if status in ("REJECTED", "ERROR", "DUPLICATE"):
        return {"raw_text": ""}

    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown = result.document.export_to_markdown()

        if not markdown.strip():
            return {"raw_text": "", "status": "ERROR", "error_message": "Docling produced empty output."}

        print(f"----- INGESTOR: Markdown generated ({len(markdown)} chars) -----")
        return {"raw_text": markdown}

    except Exception as e:
        print(f"----- INGESTOR ERROR: {e} -----")
        return {"raw_text": "", "status": "ERROR", "error_message": str(e)}
