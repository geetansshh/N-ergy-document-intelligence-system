import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.state import AgentState


def generate_queries(state: AgentState) -> dict:
    """
    Node 5 — Multi-Query Generator.

    Takes the user's original question and generates 2 additional variants
    that express the same information need from different angles.

    Why this improves retrieval:
    - A single query vector hits one region of the embedding space.
    - Differently-phrased queries cast a wider net, improving recall
      especially for generic or ambiguous questions.
    - After merging results from all 3 queries, the cross-encoder + MMR
      stage selects the truly relevant, non-redundant chunks — so the
      extra retrieval noise is cleaned up downstream.

    The original question is always included as the first query so the
    user's exact intent is never lost.
    """
    question = state.get("question", "").strip()
    print(f"----- MULTI-QUERY: Generating variants for: '{question}' -----")

    if not question:
        return {"generated_queries": []}

    try:
        prompt = ChatPromptTemplate.from_template("""
You are an expert at reformulating search queries to improve document retrieval.

Given the original question below, generate exactly 2 alternative versions that:
- Express the same information need using different vocabulary or phrasing
- Approach the topic from a slightly different angle
- Are concise and search-friendly

Return ONLY the 2 alternative questions, one per line. No numbering, no labels, no extra text.

ORIGINAL QUESTION: {question}
""")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})

        raw = response.content if isinstance(response.content, str) else ""
        variants = [line.strip() for line in raw.strip().splitlines() if line.strip()][:2]

        # Always keep original as first query
        all_queries: List[str] = [question] + variants

        print(f"----- MULTI-QUERY: Generated {len(all_queries)} queries -----")
        for i, q in enumerate(all_queries):
            print(f"  [{i}] {q}")

        return {"generated_queries": all_queries}

    except Exception as e:
        print(f"----- MULTI-QUERY ERROR: {e} — falling back to original -----")
        return {"generated_queries": [question]}
