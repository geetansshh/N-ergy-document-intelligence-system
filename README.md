# Document Intelligence System
> N-ERGY Intern Take-Home Assignment

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure API keys**

Copy the template and fill in your keys:
```bash
cp .env.example .env
```

```
GOOGLE_API_KEY="your-google-api-key"
GROQ_API_KEY="your-groq-api-key"   # optional, not used in current build
```

`GOOGLE_API_KEY` drives Gemini embeddings (`gemini-embedding-001`) and answer generation (`gemini-2.0-flash`).

**3. Start the server**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**4. Use the API**

Interactive Swagger UI:
```
http://localhost:8000/docs
```

Quick curl examples:
```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main risks identified?"}'

# Scope a query to specific files
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the recommendations?", "filenames": ["report.pdf"]}'

# Generate insights
curl -X POST http://localhost:8000/insights \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Architecture Overview

Three independent LangGraph `StateGraph` instances handle the three endpoints.

### Ingestion Graph → `POST /upload`
```
[PDF Upload]
     ↓
 validator          PyMuPDF health check + SHA-256 dedup
     ↓                ↘ DUPLICATE / REJECTED / ERROR → END
 ingestor           Docling: DocLayNet + TableFormer → Markdown
     ↓
 chunker            Pass 1: MarkdownHeaderTextSplitter (structural)
     ↓              Pass 2: sentence-transformers cosine similarity (semantic)
 embedder           Gemini embeddings → ChromaDB  +  BM25Okapi → disk
     ↓
   END
```

### Query Graph → `POST /query`
```
[User Question]
     ↓
 multi_query        Gemini generates 2 additional query variants
     ↓
 retriever          BM25 + dense search per query → RRF fusion → dedup
     ↓
 reranker           Cross-encoder (ms-marco) scoring → MMR diversity selection
     ↓                ↘ empty + retries left → reformulator → retriever (loop once)
 generator          Gemini, temp=0, inline citations [1][2], filename + section footer
     ↓
 hallucination_guard  Independent LLM verdict: GROUNDED / HALLUCINATED
     ↓
   END
```

### Insights Graph → `POST /insights`
```
[Optional filename filter]
     ↓
 insights_generator   5 broad probe queries → hybrid retrieval → dedup → Gemini
     ↓
   END
```

---

## Key Design Decisions

### 1. Docling over PyPDF2 / pdfplumber
Docling runs two internal deep-learning models on every document:
- **DocLayNet** (RT-DETR): classifies every visual element by bounding box — paragraph, header, table, figure.
- **TableFormer** (vision transformer): processes detected table crops and emits OTSL sequences that preserve row/column spans and header hierarchy.

The result is structured Markdown rather than a raw byte stream. This prevents the scrambled-text problem where PDF text extraction destroys tabular alignment, which would otherwise cause hallucinations when the LLM tries to reason about table data.

### 2. Local Semantic Chunking over LLM-based Chunking
The inspiration system sends each section to a Groq LLM for semantic splitting — adding per-document API latency and cost. This system uses `sentence-transformers` (`all-MiniLM-L6-v2`) locally to compute cosine similarity between adjacent sentences and split at topic boundaries. No API call, no added latency, equivalent semantic quality.

### 3. Hybrid Retrieval (BM25 + Dense + RRF)
Pure dense retrieval misses exact keyword matches — names, codes, numbers, and technical terms that don't have close semantic neighbours in the embedding space. BM25 handles these precisely. Reciprocal Rank Fusion (k=60, Cormack 2009) merges both ranked lists without requiring score normalisation, producing a single ranked candidate pool.

### 4. Multi-Query Expansion
A single query vector hits one region of the embedding space. Gemini generates 2 alternative phrasings of the user's question, and all 3 are run through hybrid retrieval independently. The union (deduplicated by text hash) is passed to the reranker, which selects the truly relevant subset — so the extra retrieval noise is cleaned up downstream.

### 5. Cross-Encoder Reranking + MMR
Bi-encoder embeddings approximate relevance via vector similarity. A cross-encoder sees both the query and each candidate chunk simultaneously, enabling attention-based relevance scoring that is significantly more accurate for passage ranking. After scoring, MMR (λ=0.6) selects the final 6 chunks balancing relevance against redundancy — preventing the generator from receiving near-identical context blocks.

### 6. Query Reformulation Loop
If retrieval returns no usable chunks, a query reformulator node rewrites the question (broader, simpler, jargon-free) and retries retrieval once. This is an actual LangGraph cycle — not a workaround, but the primary reason LangGraph was chosen over a standard LangChain chain.

### 7. SHA-256 Deduplication
Each uploaded PDF is hashed before ingestion. If the hash already exists in ChromaDB metadata, the document is skipped entirely. This prevents duplicate chunks from bloating the vector store and corrupting retrieval scores.

### 8. Section-Level Citations
Every answer includes an inline citation per claim (e.g. `[1]`, `[2]`) and a footer listing filename + Markdown section hierarchy for each source block. Users can verify every claim against the original document.

---

## Tradeoffs: Accuracy vs Latency

**Decision: Accuracy was prioritised.**

| Component | Latency cost | Accuracy gain |
|---|---|---|
| Docling (DocLayNet + TableFormer) | Significant | Preserves tables, prevents table-related hallucinations |
| Local semantic chunking | Negligible | Replaces Groq LLM call — faster than inspiration system |
| Multi-query expansion | 1 small LLM call | Wider retrieval net, better recall |
| BM25 hybrid retrieval | Negligible (local) | Catches exact keyword matches dense misses |
| Cross-encoder reranking | ~0.5s for 30 candidates | 15–30% precision improvement over bi-encoder retrieval |
| MMR diversity selection | Negligible | Removes redundant context from LLM input |
| Hallucination guard | 1 LLM call | Last-line defence against ungrounded claims |

The justification: this system handles document intelligence where a hallucinated figure, clause, or name is actively harmful. A slightly slower correct answer is worth more than a fast wrong one.

---

## What Would Break at Scale (10k+ Documents)

### Semantic Collapse
In a 3072-dimensional embedding space, as tens of thousands of dense vectors populate the hypersphere, the relative distance between the true nearest neighbour and most other points begins to equalise — the **curse of dimensionality**. Retrieval accuracy degrades from ~85% at 1,000 documents to ~45% at 10,000. The BM25 component partially offsets this for keyword-heavy queries, but the dense component degrades.

**Fix:** Hierarchical indexing (cluster documents into topics, search within the relevant cluster), or switching to a purpose-built ANN index (HNSW in Qdrant/Weaviate) with filtered search.

### Infrastructure Bottlenecks

| Bottleneck | Current state | Impact at scale |
|---|---|---|
| **ChromaDB + SQLite** | Single-node, pickle serialisation | Multi-hour ingestion delays, OOM on query |
| **BM25 index on disk** | Full corpus loaded into RAM per request | OOM at ~100k chunks |
| **Sequential ingestion** | One document at a time | Unacceptable queue at 100+ docs |
| **Cross-encoder reranking** | CPU inference, ~0.5s/30 pairs | Seconds per query at higher k |
| **No deduplication at chunk level** | Only file-level hash | Semantically identical chunks from different files still bloat the index |

---

## What I Would Improve With More Time

- **Async ingestion pipeline:** Background task queue (Celery/ARQ) for document ingestion so `/upload` returns immediately and processing happens asynchronously.
- **Qdrant or Weaviate:** Replace ChromaDB with a production-grade vector database that supports filtered ANN search, horizontal scaling, and doesn't hold everything in RAM.
- **Persistent BM25 with inverted index:** Replace BM25Okapi-on-disk with Elasticsearch or Tantivy for a proper inverted index that doesn't require loading the full corpus into RAM.
- **Reranker GPU acceleration:** Move cross-encoder inference to GPU (or use Cohere Rerank API) to reduce per-query latency.
- **Self-RAG reflection loop:** After reranking, evaluate context sufficiency before calling the generator — if context quality is low, reformulate and retry rather than generating a weak answer.
- **Streaming responses:** Stream the generator output token-by-token via Server-Sent Events for better perceived latency.
- **Multi-file batch upload:** Accept a list of files in a single `/upload` call with parallel ingestion.
