import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.state import AgentState

MAX_RETRIES = 1  # Only one reformulation attempt to avoid infinite loops


def reformulate_query(state: AgentState) -> dict:
    """
    Node 8 — Query Reformulator (retry branch).

    Reached only when hybrid retrieval + reranking returns no usable chunks.
    The LLM is prompted to produce a broader, simpler reformulation of the
    original question — stripping domain jargon and expanding scope.

    The reformulated query replaces `generated_queries` so the retriever
    node re-runs with a fresh set of variants on the next graph iteration.

    MAX_RETRIES=1 prevents infinite loops: if the second attempt also fails,
    the graph routes to the generator which returns a safe "no information"
    message rather than looping again.
    """
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)

    print(f"----- REFORMULATOR: Reformulating (attempt {retry_count + 1}) -----")

    try:
        prompt = ChatPromptTemplate.from_template("""
The following question returned no relevant results when searched in a document database.
Reformulate it into a broader, simpler version that is more likely to find relevant passages.
Strip technical jargon. Expand abbreviations. Generalise specific terms.
Return ONLY the reformulated question. No explanation.

ORIGINAL QUESTION: {question}
""")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            api_key=os.getenv("GROQ_API_KEY")
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
        reformulated = response.content.strip() if isinstance(response.content, str) else question

        print(f"----- REFORMULATOR: New query = '{reformulated}' -----")
        return {
            "generated_queries": [reformulated],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "retry_count": retry_count + 1,
        }

    except Exception as e:
        print(f"----- REFORMULATOR ERROR: {e} -----")
        return {
            "generated_queries": [question],
            "retry_count": retry_count + 1,
        }
