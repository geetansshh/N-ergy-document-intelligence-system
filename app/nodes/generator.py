import os
from typing import List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.state import AgentState


def _build_context(docs: List[Document]) -> tuple[str, list[dict]]:
    """
    Formats retrieved chunks into numbered context blocks.
    Returns the context string and a list of citation metadata dicts.
    """
    context_parts = []
    citations = []

    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
        h1 = doc.metadata.get("Header 1", "")
        h2 = doc.metadata.get("Header 2", "")
        h3 = doc.metadata.get("Header 3", "")

        section_parts = [s for s in [h1, h2, h3] if s]
        section_label = " > ".join(section_parts) if section_parts else "—"

        context_parts.append(
            f"[{i}] Source: {filename} | Section: {section_label}\n{doc.page_content}"
        )
        citations.append({
            "index": i,
            "filename": filename,
            "section": section_label,
        })

    return "\n\n".join(context_parts), citations


def generate_answer(state: AgentState) -> dict:
    """
    Node 9 — Answer Generator.

    Synthesises the reranked chunks into a grounded, cited answer.

    Key design choices:
    - Uses reranked_chunks (post cross-encoder + MMR) as context, not raw
      retrieved_chunks, ensuring only high-quality, non-redundant evidence
      is provided to the LLM.
    - temperature=0: eliminates creative variance — the model must answer
      from the provided context only.
    - Strict system prompt: explicitly forbids use of outside knowledge and
      requires citing block numbers inline (e.g. [1], [2]).
    - Citations footer: lists filename + section for every source block used,
      giving users verifiable pointers back to the source material.
    - Empty context guard: if reranked_chunks is empty (both retrieval
      attempts failed), returns a clear "not found" message rather than
      hallucinating an answer.
    """
    docs: List[Document] = state.get("reranked_chunks", [])
    question: str = state.get("question", "")

    print("----- GENERATOR: Synthesising answer -----")

    if not docs:
        return {
            "final_answer": (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Please try rephrasing, or upload additional documents."
            )
        }

    context, citations = _build_context(docs)

    prompt = ChatPromptTemplate.from_template("""
You are a precise document analyst. Answer the user's question using ONLY the numbered context blocks provided.

RULES:
1. Answer strictly from the context. Do not use outside knowledge.
2. Cite your sources inline using block numbers, e.g. [1], [2].
3. If the context does not contain enough information, say: "The documents do not contain sufficient information to answer this question."
4. Be concise and factual. Do not speculate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})

        answer_text = response.content if isinstance(response.content, str) else \
            " ".join(b.get("text", "") for b in response.content if isinstance(b, dict))

        # Build citations footer
        citation_lines = [
            f"  [{c['index']}] {c['filename']} — {c['section']}"
            for c in citations
        ]
        citations_footer = "\n**Sources:**\n" + "\n".join(citation_lines)

        final_output = answer_text.strip() + "\n\n" + citations_footer

        print("----- GENERATOR: Answer produced -----")
        return {"final_answer": final_output}

    except Exception as e:
        print(f"----- GENERATOR ERROR: {e} -----")
        return {"final_answer": f"Technical error during answer generation: {str(e)}"}
