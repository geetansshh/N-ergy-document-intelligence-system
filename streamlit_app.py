import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Must be set before importing graph nodes (they read env vars at call time)
if "embedding_provider" not in st.session_state:
    st.session_state.embedding_provider = "gemini"
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "groq"

os.environ["EMBEDDING_PROVIDER"] = st.session_state.embedding_provider
os.environ["LLM_PROVIDER"] = st.session_state.llm_provider

from app.graph import ingest_graph, query_graph, insights_graph

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Intelligence")
st.caption("Upload PDFs, ask questions, and get AI-powered insights.")

# ── Session state ─────────────────────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []   # list of filenames successfully ingested
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # list of {"question": ..., "answer": ...}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Embedding Provider")
    _embed_options = ["gemini", "bge", "local"]
    _embed_labels = {
        "gemini": "Google Gemini API (3072 dims)",
        "bge":    "BAAI/bge-large-en-v1.5 — GPU (1024 dims)",
        "local":  "all-MiniLM-L6-v2 — GPU/CPU (384 dims, fast)",
    }
    provider = st.radio(
        "Choose embedding model",
        options=_embed_options,
        format_func=lambda x: _embed_labels[x],
        index=_embed_options.index(st.session_state.embedding_provider)
              if st.session_state.embedding_provider in _embed_options else 0,
        help="Each option uses a separate index. Re-ingest PDFs after switching.",
    )
    if provider != st.session_state.embedding_provider:
        st.session_state.embedding_provider = provider
        os.environ["EMBEDDING_PROVIDER"] = provider
        st.session_state.ingested_files = []   # reset — different chroma_db dir
        st.session_state.chat_history = []
        st.rerun()

    st.caption(
        "Each provider stores embeddings in a separate index. "
        "Re-ingest your PDFs after switching."
    )

    st.header("🤖 LLM Provider")
    llm_provider = st.radio(
        "Choose generation model",
        options=["groq", "gemini"],
        format_func=lambda x: "Groq — Llama 3.3 70B" if x == "groq" else "Google Gemini 2.0 Flash",
        index=0 if st.session_state.llm_provider == "groq" else 1,
    )
    if llm_provider != st.session_state.llm_provider:
        st.session_state.llm_provider = llm_provider
        os.environ["LLM_PROVIDER"] = llm_provider
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.header("📁 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Ingest", disabled=not uploaded_files, use_container_width=True):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.ingested_files:
                st.info(f"Already ingested: **{uploaded_file.name}**")
                continue

            with st.spinner(f"Ingesting **{uploaded_file.name}** …"):
                # Write to a temp file so Docling can read it from disk
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    result = ingest_graph.invoke({
                        "file_path": tmp_path,
                        "original_filename": uploaded_file.name,
                        "status": "",
                        "error_message": None,
                        "file_hash": "",
                        "raw_text": "",
                        "chunks": [],
                    })
                finally:
                    os.unlink(tmp_path)

            status = result.get("status", "ERROR")

            if status in ("READY", "SCANNED"):
                chunk_count = len(result.get("chunks", []))
                st.success(
                    f"**{uploaded_file.name}** ingested "
                    f"({chunk_count} chunks, status: {status})"
                )
                st.session_state.ingested_files.append(uploaded_file.name)
            elif status == "DUPLICATE":
                st.info(f"**{uploaded_file.name}** is already in the index (duplicate).")
                if uploaded_file.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uploaded_file.name)
            else:
                err = result.get("error_message") or "Unknown error"
                st.error(f"Failed to ingest **{uploaded_file.name}**: {err}")

    # Show ingested files list
    if st.session_state.ingested_files:
        st.divider()
        st.subheader("Indexed documents")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- {fname}")

        if st.button("Clear list", use_container_width=True):
            st.session_state.ingested_files = []
            st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_qa, tab_insights = st.tabs(["💬 Ask a Question", "💡 Get Insights"])

# ── Q&A tab ───────────────────────────────────────────────────────────────────
with tab_qa:
    # Scope selector — auto-scope when only one doc is indexed
    scope_files = []
    if len(st.session_state.ingested_files) == 1:
        scope_files = st.session_state.ingested_files
        st.caption(f"Scoped to: **{scope_files[0]}**")
    elif st.session_state.ingested_files:
        scope_files = st.multiselect(
            "Scope to specific documents (leave empty to search all)",
            options=st.session_state.ingested_files,
            help="For questions like 'what is this document about?', select the specific document here.",
        )
        if not scope_files:
            st.info("No scope selected — searching across all indexed documents.", icon="ℹ️")

    # Chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

    # Input
    question = st.chat_input("Ask a question about your documents…")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = query_graph.invoke({
                    "question": question,
                    "filenames": scope_files,
                    "generated_queries": [],
                    "retrieved_chunks": [],
                    "reranked_chunks": [],
                    "retry_count": 0,
                    "final_answer": "",
                })

            answer = result.get("final_answer", "No answer generated.")
            queries_used = result.get("generated_queries", [])

            st.markdown(answer)

            if queries_used:
                with st.expander("Queries used for retrieval"):
                    for q in queries_used:
                        st.markdown(f"- {q}")

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
        })

    if st.session_state.chat_history:
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ── Insights tab ──────────────────────────────────────────────────────────────
with tab_insights:
    scope_insight_file = None
    if st.session_state.ingested_files:
        scope_insight_file = st.selectbox(
            "Scope insights to a specific document (optional)",
            options=["All documents"] + st.session_state.ingested_files,
        )
        if scope_insight_file == "All documents":
            scope_insight_file = None

    if st.button("Generate Insights", use_container_width=True):
        if not st.session_state.ingested_files:
            st.warning("Ingest at least one PDF first.")
        else:
            with st.spinner("Generating insights…"):
                result = insights_graph.invoke({
                    "question": scope_insight_file or "",
                })
            insights_text = result.get("final_answer", "No insights generated.")
            st.markdown(insights_text)
