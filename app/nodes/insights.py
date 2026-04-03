import os
import json
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from app.state import AgentState
from app.embeddings import get_embeddings, get_chroma_dir
from app.llm import get_llm

BM25_DIR = "./bm25_index"

# Multiple broad probe queries to get diverse coverage of the document corpus
INSIGHT_PROBES = [
    "main findings conclusions summary",
    "key risks challenges problems issues",
    "recommendations actions next steps",
    "important data statistics results",
    "background context purpose objectives",
]


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def generate_insights(state: AgentState) -> dict:
    """
    Node — Insights Generator (bonus endpoint).

    Unlike the query graph which answers a specific question, this node
    proactively surfaces what is most important across all indexed documents.

    Improvements over the inspiration system:
    - Uses 5 broad probe queries (instead of one generic query) to ensure
      diverse coverage: findings, risks, recommendations, data, and context.
    - Each probe runs hybrid retrieval (dense + BM25) so both conceptual
      summaries and specific data points are surfaced.
    - Results are deduplicated by text hash before being passed to the LLM.
    - Optional filename filter scopes the analysis to a single document.

    The LLM is prompted to produce structured output:
    - 3–5 Key Insights (specific, grounded facts from the content)
    - 3–5 Actionable Next Steps (concrete recommendations)
    """
    filename_filter: str = state.get("question", "")  # reused field for optional filter

    print("----- INSIGHTS: Generating insights from document corpus -----")

    try:
        embeddings = get_embeddings()
        vectorstore = Chroma(persist_directory=get_chroma_dir(), embedding_function=embeddings)

        # Load BM25
        corpus_path = os.path.join(BM25_DIR, "corpus.pkl")
        meta_path = os.path.join(BM25_DIR, "metadata.json")
        bm25 = None
        bm25_meta = []
        if os.path.exists(corpus_path) and os.path.exists(meta_path):
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)
            with open(meta_path, "r") as f:
                bm25_meta = json.load(f)
            bm25 = BM25Okapi(corpus)

        seen: set = set()
        all_docs: List[Document] = []

        for probe in INSIGHT_PROBES:
            # Dense retrieval
            dense_results = vectorstore.similarity_search(probe, k=5)
            for doc in dense_results:
                if filename_filter and doc.metadata.get("filename", "") != filename_filter:
                    continue
                key = hash(doc.page_content)
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

            # BM25 retrieval
            if bm25 and bm25_meta:
                scores = bm25.get_scores(_tokenize(probe))
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
                for idx in top_indices:
                    if scores[idx] <= 0:
                        continue
                    meta = bm25_meta[idx]
                    if filename_filter and meta.get("filename", "") != filename_filter:
                        continue
                    key = hash(meta["text"])
                    if key not in seen:
                        seen.add(key)
                        all_docs.append(Document(
                            page_content=meta["text"],
                            metadata={"filename": meta.get("filename", ""), "source": meta.get("source", "")}
                        ))

        if not all_docs:
            return {"final_answer": "No documents found in the database. Please upload documents first."}

        # Build context
        context_parts = []
        sources = set()
        for i, doc in enumerate(all_docs, start=1):
            fname = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            sources.add(fname)
            context_parts.append(f"[{i}] ({fname})\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        prompt = ChatPromptTemplate.from_template("""
You are an expert analyst reviewing document content.
Based ONLY on the content provided below, produce a structured analysis.

Format your response exactly as follows:

**KEY INSIGHTS**
1. [Specific insight grounded in the content]
2. ...
(3–5 insights)

**ACTIONABLE NEXT STEPS**
1. [Concrete, specific action]
2. ...
(3–5 steps)

Do not use outside knowledge. Be specific, not generic.

DOCUMENT CONTENT:
{context}
""")

        llm = get_llm(temperature=0.2)
        chain = prompt | llm
        response = chain.invoke({"context": context})

        answer_text = response.content if isinstance(response.content, str) else \
            " ".join(b.get("text", "") for b in response.content if isinstance(b, dict))

        sources_footer = "\n\n**Sources:** " + ", ".join(sorted(sources))
        print("----- INSIGHTS: Done -----")
        return {"final_answer": answer_text.strip() + sources_footer}

    except Exception as e:
        print(f"----- INSIGHTS ERROR: {e} -----")
        return {"final_answer": f"Error generating insights: {str(e)}"}
