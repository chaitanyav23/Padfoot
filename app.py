"""
app.py – FastAPI RAG Backend
=============================
Exposes a single POST /ask endpoint that:
  1. Embeds the user query with the same sentence-transformer model used at
     ingestion time.
  2. Retrieves the top-K most relevant chunks from ChromaDB.
  3. Injects them as context into a system prompt.
  4. Calls Gemini 2.5 Flash to generate a grounded answer.
  5. Returns the answer + source file list as JSON.

Usage:
    // start backend
    gunicorn -k uvicorn.workers.UvicornWorker app:app --workers 2 --bind 127.0.0.1:8000

    // start frontend
    cd HTML
    python3 -m http.server 5500

Required env var:
    GOOGLE_API_KEY
"""

import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "docs"
TOP_K            = 3             # Number of chunks to retrieve
MAX_TOKENS       = 1024          # Max tokens in the generated answer
TEMPERATURE      = 0.2           # Low temperature for factual, grounded answers
MAX_QUERY_LENGTH = 2000          # Character limit on incoming queries
LLM_TIMEOUT      = 20            # Seconds before Gemini call is aborted
CHAT_MODEL       = "models/gemini-2.5-flash"

GENERATION_CONFIG = genai.types.GenerationConfig(
    max_output_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)


# ── App & CORS ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Documentation RAG API", version="1.0.0")

# used during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_transformer_ef,
)


class QueryRequest(BaseModel):
    query: str
    source_page: str = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ── /ask Endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Retrieve relevant documentation chunks and generate a grounded answer.
    """
    query = request.query.strip()
    source_page = request.source_page
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"Query too long (max {MAX_QUERY_LENGTH} characters).")

    # ── Step 1: Hybrid Retrieval ───────────
    retrieved_docs = []
    retrieved_meta = []
    seen_ids = set()
    MAX_CHUNKS = 5  
    
    # 1.1: Local Boost 
    if source_page:
        local_results = collection.query(
            query_texts=[query],
            n_results=3,
            where={"source": source_page},
            include=["documents", "metadatas"],
        )
        if local_results["ids"] and local_results["ids"][0]:
            for i in range(len(local_results["ids"][0])):
                doc_id = local_results["ids"][0][i]
                retrieved_docs.append(local_results["documents"][0][i])
                retrieved_meta.append(local_results["metadatas"][0][i])
                seen_ids.add(doc_id)

    # 1.2: Global Context
    needed = MAX_CHUNKS - len(retrieved_docs)
    if needed > 0:
        # Fetch needed + 2 as a buffer against overlaps with local results
        global_results = collection.query(
            query_texts=[query],
            n_results=needed + 2,
            include=["documents", "metadatas"],
        )
        if global_results["ids"] and global_results["ids"][0]:
            for i in range(len(global_results["ids"][0])):
                if len(retrieved_docs) >= MAX_CHUNKS:
                    break
                doc_id = global_results["ids"][0][i]
                # Avoid duplicates if a top global result is also on the current page
                if doc_id not in seen_ids:
                    retrieved_docs.append(global_results["documents"][0][i])
                    retrieved_meta.append(global_results["metadatas"][0][i])
                    seen_ids.add(doc_id)

    # ── Step 2: Build the context block ───────────────────────────────────
    context_parts = []
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta)):
        source = meta.get("source", "unknown")
        context_parts.append(f"[Source {i + 1}: {source}]\n{doc}")
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Compose the system prompt ─────────────────────────────────
    page_context = ""
    if source_page:
        page_context = f"The user is currently viewing the page: {source_page}. Prioritize information from this page if relevant, but answer from the overall context if the information is elsewhere. "

    system_prompt = (
        "You are a helpful IT documentation assistant for an academic institution. "
        f"{page_context}"
        "Answer the user's question using ONLY the provided documentation context below. "
        "If the answer is not found in the context, say so clearly. "
        "Be concise, accurate, and structured. Use bullet points or numbered steps "
        "where appropriate.\n\n"
        f"## Documentation Context\n\n{context}"
    )

    chat_model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=system_prompt,
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: chat_model.generate_content(
                    query,
                    generation_config=GENERATION_CONFIG,
                )
            ),
            timeout=LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("Gemini request timed out after %ss", LLM_TIMEOUT)
        raise HTTPException(status_code=504, detail="LLM request timed out.")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled LLM error")
        raise HTTPException(status_code=500, detail="Internal server error.")

    answer = response.text

    usage = getattr(response, "usage_metadata", None)
    if usage:
        logger.info(
            "Tokens – input: %s, output: %s",
            usage.prompt_token_count,
            usage.candidates_token_count,
        )

    # Deduplicate sources while preserving order
    seen: set[str] = set()
    sources: list[str] = []
    for meta in retrieved_meta:
        src = meta.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)

    return QueryResponse(answer=answer, sources=sources)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Quick liveness check – also returns collection doc count."""
    count = collection.count()
    return {
        "status":     "ok",
        "collection": COLLECTION_NAME,
        "doc_chunks": count,
    }
