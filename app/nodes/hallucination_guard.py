import os
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.state import AgentState


def hallucination_guard(state: AgentState) -> dict:
    """
    Node 10 — Hallucination Guard.

    An independent LLM call audits the generated answer against the
    retrieved context before it is returned to the user.

    Design choices:
    - Uses reranked_chunks (the same context given to the generator) as
      ground truth — the answer should be traceable to this exact material.
    - temperature=0 + single-word verdict prompt: makes the check fast and
      deterministic. The auditor has no room for ambiguity.
    - On HALLUCINATED verdict: discards the answer and returns a safe
      refusal rather than a potentially incorrect response.
    - On error: passes the answer through unchanged to avoid blocking
      valid responses due to transient API issues.

    This is a last-line defence. Most hallucinations are already prevented
    upstream by: strict grounding prompt in generator, cross-encoder
    reranking (only highly relevant context is provided), and the
    relevance threshold filter in retrieval.
    """
    final_answer: str = state.get("final_answer", "")
    docs: List[Document] = state.get("reranked_chunks", [])

    print("----- HALLUCINATION GUARD: Verifying answer -----")

    if not final_answer or not docs:
        return {}

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checking auditor.

Verify whether the AI-generated answer below is FULLY supported by the source context provided.

RULES:
- Reply ONLY with "GROUNDED" or "HALLUCINATED". One word. No punctuation. No explanation.
- Reply "GROUNDED" if every factual claim in the answer can be directly traced to the context.
- Reply "HALLUCINATED" if the answer contains any fact, number, name, or claim NOT present in the context.

SOURCE CONTEXT:
{context}

AI-GENERATED ANSWER:
{answer}

VERDICT:
""")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        chain = prompt | llm
        response = chain.invoke({"context": context, "answer": final_answer})
        verdict = response.content.strip().upper() if isinstance(response.content, str) else "GROUNDED"

        print(f"----- HALLUCINATION GUARD: Verdict = {verdict} -----")

        if "HALLUCINATED" in verdict:
            return {
                "final_answer": (
                    "The generated response could not be fully verified against the source documents. "
                    "Please rephrase your question or upload additional supporting documents."
                )
            }
        return {}

    except Exception as e:
        print(f"----- HALLUCINATION GUARD ERROR: {e} — passing answer through -----")
        return {}
