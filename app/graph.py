from langgraph.graph import StateGraph, END
from app.state import AgentState

from app.nodes.pdf_validator import validate_pdf
from app.nodes.ingestor import ingest_document
from app.nodes.chunker import chunk_document
from app.nodes.embedder import embed_documents
from app.nodes.multi_query import generate_queries
from app.nodes.retriever import retrieve_chunks
from app.nodes.reranker import rerank_chunks
from app.nodes.query_reformulator import reformulate_query, MAX_RETRIES
from app.nodes.generator import generate_answer
from app.nodes.hallucination_guard import hallucination_guard
from app.nodes.insights import generate_insights


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_validation(state: AgentState) -> str:
    status = state.get("status", "")
    if status in ("REJECTED", "ERROR", "DUPLICATE"):
        return "end"
    return "ingestor"


def route_after_rerank(state: AgentState) -> str:
    """
    After reranking: if no chunks survived, attempt query reformulation
    (up to MAX_RETRIES). If retries are exhausted, go straight to generator
    which returns a safe "not found" message.
    """
    reranked = state.get("reranked_chunks", [])
    retry_count = state.get("retry_count", 0)

    if not reranked and retry_count < MAX_RETRIES:
        return "reformulator"
    return "generator"


# ---------------------------------------------------------------------------
# Ingestion Graph:  validator → ingestor → chunker → embedder
# ---------------------------------------------------------------------------

ingest_workflow = StateGraph(AgentState)

ingest_workflow.add_node("validator", validate_pdf)
ingest_workflow.add_node("ingestor", ingest_document)
ingest_workflow.add_node("chunker", chunk_document)
ingest_workflow.add_node("embedder", embed_documents)

ingest_workflow.set_entry_point("validator")

ingest_workflow.add_conditional_edges(
    "validator",
    route_after_validation,
    {"ingestor": "ingestor", "end": END}
)
ingest_workflow.add_edge("ingestor", "chunker")
ingest_workflow.add_edge("chunker", "embedder")
ingest_workflow.add_edge("embedder", END)

ingest_graph = ingest_workflow.compile()


# ---------------------------------------------------------------------------
# Query Graph:
#   multi_query → retriever → reranker
#       ↓ (no results, retries left)
#   reformulator → retriever → reranker
#       ↓ (results OR retries exhausted)
#   generator → hallucination_guard
# ---------------------------------------------------------------------------

query_workflow = StateGraph(AgentState)

query_workflow.add_node("multi_query", generate_queries)
query_workflow.add_node("retriever", retrieve_chunks)
query_workflow.add_node("reranker", rerank_chunks)
query_workflow.add_node("reformulator", reformulate_query)
query_workflow.add_node("generator", generate_answer)
query_workflow.add_node("hallucination_guard", hallucination_guard)

query_workflow.set_entry_point("multi_query")

query_workflow.add_edge("multi_query", "retriever")
query_workflow.add_edge("retriever", "reranker")

query_workflow.add_conditional_edges(
    "reranker",
    route_after_rerank,
    {"reformulator": "reformulator", "generator": "generator"}
)

# Reformulator loops back to retriever for one retry
query_workflow.add_edge("reformulator", "retriever")

query_workflow.add_edge("generator", "hallucination_guard")
query_workflow.add_edge("hallucination_guard", END)

query_graph = query_workflow.compile()


# ---------------------------------------------------------------------------
# Insights Graph:  insights_generator
# ---------------------------------------------------------------------------

insights_workflow = StateGraph(AgentState)
insights_workflow.add_node("insights_generator", generate_insights)
insights_workflow.set_entry_point("insights_generator")
insights_workflow.add_edge("insights_generator", END)

insights_graph = insights_workflow.compile()
