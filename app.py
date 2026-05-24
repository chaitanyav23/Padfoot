"""
app.py – Padfoot Multimodal RAG Backend (Refactored)
===================================================
A demo-focused RAG system that orchestrates text and image retrieval.
- Text Pipeline: MiniLM + Gemini 2.5 Flash
- Image Pipeline: SigLIP + Qwen2.5-VL (Local Ollama)
- Architecture: Text-Grounded Multi-Image Reasoning
"""

import os
import logging
import asyncio
import base64
import requests
import json
import io
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import torch
from transformers import AutoProcessor, SiglipModel
from PIL import Image
from dotenv import load_dotenv

# ── 1. Setup & Configuration ──────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Padfoot")

# Configuration Constants
CHROMA_DIR       = "./chroma_db"
TEXT_COLLECTION  = "docs"
IMG_COLLECTION   = "image_embeddings"

MAX_TOKENS       = 1024
TEMPERATURE      = 0.2
MAX_QUERY_LENGTH = 2000
LLM_TIMEOUT      = 180  # Seconds

# Models
CHAT_MODEL       = "models/gemini-2.5-flash"
SIGLIP_MODEL_ID  = "google/siglip-base-patch16-224"
OLLAMA_MODEL     = "qwen2.5vl:3b"
OLLAMA_URL       = "http://localhost:11434/api/generate"

# Thresholds & Limits
IMAGE_DISTANCE_THRESHOLD = 2.2  # SigLIP L2 distance (lower is better)
MAX_IMAGES_TO_VLM        = 4    # Maximum number of screenshots to send at once

# Environment Validation
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("Missing GOOGLE_API_KEY in environment.")
    raise RuntimeError("GOOGLE_API_KEY must be set to start the backend.")

# ── 2. Model Initialization ───────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load SigLIP for Multimodal Retrieval
logger.info(f"Loading SigLIP model: {SIGLIP_MODEL_ID}")
siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
siglip_model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID).to(device)

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
GENERATION_CONFIG = genai.types.GenerationConfig(
    max_output_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)

# ── 3. Database & Embeddings ──────────────────────────────────────────────────
# Text Embedding Function (all-MiniLM-L6-v2)
mini_lm_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Text Collection
text_collection = chroma_client.get_or_create_collection(
    name=TEXT_COLLECTION,
    embedding_function=mini_lm_ef,
)

# Image Collection (Manual SigLIP embeddings)
img_collection = chroma_client.get_or_create_collection(
    name=IMG_COLLECTION
)

# ── 4. Utility Functions ──────────────────────────────────────────────────────

def get_siglip_embedding(text: str) -> List[float]:
    """Generates a normalized SigLIP embedding for the query."""
    inputs = siglip_processor(text=[text], padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        features = siglip_model.get_text_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()

def encode_image(path: str) -> Optional[str]:
    """Safely encodes, resizes, and compresses an image file to base64."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        logger.error(f"Image not found: {path}")
        return None
    try:
        with Image.open(p) as img:
            img = img.convert("RGB")
            # Using thumbnail to preserve aspect ratio while staying within 384x384
            img.thumbnail((384, 384), Image.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error processing image {path}: {e}")
        return None

def rerank_and_group_images(query: str, results: dict, relevant_pages: set) -> List[dict]:
    """
    Reranks candidate images considering SigLIP distance, OCR overlap, and source page relevance.
    Returns a sorted list of image metadatas.
    """
    if not results["metadatas"] or not results["metadatas"][0]:
        return []
    
    query_words = set(query.lower().split())
    candidates = []
    
    for i, meta in enumerate(results["metadatas"][0]):
        dist = results["distances"][0][i]
        ocr_text = meta.get("ocr_text", "").lower()
        source_page = meta.get("source_page")
        
        # Calculate OCR overlap
        overlap = 0
        if ocr_text:
            overlap = sum(1 for word in query_words if word in ocr_text)
            
        # Score calculation
        score = dist - (overlap * 0.12)
        
        # Boost if the image comes from one of the retrieved text pages
        if source_page in relevant_pages:
            score -= 0.20
            
        candidates.append({"meta": meta, "score": score, "dist": dist, "page_match": source_page in relevant_pages})
        
    # Sort by composite score (lowest is best)
    candidates.sort(key=lambda x: x["score"])
    
    # Filter by threshold and extract meta
    selected_metas = []
    for c in candidates:
        if c["dist"] < IMAGE_DISTANCE_THRESHOLD:
            selected_metas.append(c["meta"])
            logger.info(f"  -> Selected: {c['meta'].get('image_src')} | Score: {c['score']:.4f} | Page Match: {c['page_match']}")
            
    return selected_metas

# ── 5. LLM Callers ────────────────────────────────────────────────────────────

async def call_gemini(query: str, system_prompt: str) -> str:
    """Invokes Gemini 2.5 Flash with timeout protection."""
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=system_prompt,
    )
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: model.generate_content(query, generation_config=GENERATION_CONFIG)
            ),
            timeout=LLM_TIMEOUT
        )
        return response.text if response.text else "The model returned an empty response."
    except asyncio.TimeoutError:
        logger.error("Gemini call timed out.")
        return "I'm sorry, the request timed out while generating a response."
    except Exception as e:
        logger.error(f"Gemini Exception: {e}")
        return "I encountered an error while processing your request with Gemini."

async def call_ollama_vision_multi(query: str, text_context: str, image_metas: List[dict]) -> Optional[str]:
    """Invokes Qwen2.5-VL via local Ollama API with multiple images and aggregated OCR."""
    images_b64 = []
    ocr_aggregations = []
    
    # Process all selected images
    for idx, meta in enumerate(image_metas):
        img_path = meta.get("full_path")
        b64 = encode_image(img_path) if img_path else None
        if b64:
            images_b64.append(b64)
            ocr = meta.get("ocr_text", "").strip()
            if ocr:
                ocr_aggregations.append(f"[Image {idx+1} OCR]: {ocr}")
                
    if not images_b64:
        return None

    ocr_context = "\n".join(ocr_aggregations) if ocr_aggregations else "No OCR text available."
    
    vision_prompt = (
        "You are an IT documentation assistant. You have been provided with relevant documentation text, "
        "multiple sequential screenshots from the documentation, and the text extracted from those screenshots (OCR).\n\n"
        "### DOCUMENTATION TEXT\n"
        f"{text_context}\n\n"
        "### EXTRACTED SCREENSHOT TEXT (OCR)\n"
        f"{ocr_context}\n\n"
        "Use ALL available text and visual information together to answer the user's question sequentially and clearly."
    )
    
    full_prompt = f"{vision_prompt}\n\nQuestion: {query}"
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "images": images_b64,
        "stream": False,
        "options": {
            "num_ctx": 1024,
            "num_predict": 128,
            "num_gpu": 25
        }
    }
    
    logger.info(f"[OLLAMA] Sending {len(images_b64)} images to {OLLAMA_MODEL}")
    try:
        response = await asyncio.to_thread(
            lambda: requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        )
        
        if response.status_code == 200:
            logger.info("[OLLAMA] Vision response received")
            res_json = response.json()
            return res_json.get("response", "")
        else:
            logger.error(f"[OLLAMA ERROR] Status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"[OLLAMA ERROR] {e}")
        return None

# ── 6. FastAPI Endpoints ──────────────────────────────────────────────────────

app = FastAPI(title="Padfoot Multimodal API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    source_page: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
async def root():
    return {"message": "Padfoot Multimodal RAG API is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": device,
        "collections": {
            "text_chunks": text_collection.count(),
            "images": img_collection.count()
        },
        "config": {
            "gemini_ready": GOOGLE_API_KEY is not None,
            "ollama_model": OLLAMA_MODEL
        }
    }

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    query = request.query.strip()
    source_page = request.source_page
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # ── STEP 1: Text Retrieval (Primary Signal) ──
    retrieved_docs = []
    retrieved_meta = []
    seen_ids = set()
    MAX_CHUNKS = 5
    relevant_pages = set()
    
    # 1.1 Local Page Boost
    if source_page:
        relevant_pages.add(source_page)
        res = text_collection.query(
            query_texts=[query], n_results=3, 
            where={"source": source_page}
        )
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                doc_id = res["ids"][0][i]
                retrieved_docs.append(res["documents"][0][i])
                retrieved_meta.append(res["metadatas"][0][i])
                seen_ids.add(doc_id)

    # 1.2 Global Retrieval
    needed = MAX_CHUNKS - len(retrieved_docs)
    if needed > 0:
        res = text_collection.query(query_texts=[query], n_results=needed + 2)
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                if len(retrieved_docs) >= MAX_CHUNKS:
                    break
                doc_id = res["ids"][0][i]
                if doc_id not in seen_ids:
                    doc = res["documents"][0][i]
                    meta = res["metadatas"][0][i]
                    retrieved_docs.append(doc)
                    retrieved_meta.append(meta)
                    seen_ids.add(doc_id)
                    if meta.get("source"):
                        relevant_pages.add(meta["source"])
                        
    logger.info(f"[TEXT RETRIEVAL] Retrieved {len(retrieved_docs)} chunks from pages: {list(relevant_pages)}")

    # ── STEP 2: Image Retrieval (Contextual Grouping) ──
    query_siglip = get_siglip_embedding(query)
    
    # We query more images to find matches across the relevant pages
    img_res = img_collection.query(
        query_embeddings=[query_siglip],
        n_results=15, 
        include=["metadatas", "distances"]
    )
    
    # Rerank prioritizing OCR overlap AND inclusion in relevant_pages
    selected_image_metas = rerank_and_group_images(query, img_res, relevant_pages)
    
    # Limit to top N images to avoid huge payloads
    selected_image_metas = selected_image_metas[:MAX_IMAGES_TO_VLM]
    
    if selected_image_metas:
        logger.info(f"[IMAGE GROUPING] Grouped {len(selected_image_metas)} associated screenshots.")
    else:
        logger.info(f"[IMAGE GROUPING] No highly relevant associated screenshots found.")

    # ── STEP 3: Prompt Construction ──
    if not retrieved_docs:
        context_str = "No specific documentation chunks were found for this query."
    else:
        formatted_chunks = []
        for doc, meta in zip(retrieved_docs, retrieved_meta):
            src_name = meta.get("source", "Documentation")
            formatted_chunks.append(f"[Source: {src_name}]\n{doc}")
        context_str = "\n\n---\n\n".join(formatted_chunks)

    page_ctx_info = f"The user is viewing the page: {source_page}. " if source_page else ""
    text_system_prompt = (
        "You are an academic IT support assistant. Answer the question using ONLY the context provided. "
        "If the context is insufficient, politely explain that. Be structured and concise.\n\n"
        f"{page_ctx_info}\n"
        f"## RETRIEVED CONTEXT\n{context_str}"
    )

    # ── STEP 4: Routing & Reasoning ──
    answer = None
    if selected_image_metas:
        logger.info("[MULTIMODAL ROUTE] Initiating text-grounded multimodal reasoning.")
        answer = await call_ollama_vision_multi(query, context_str, selected_image_metas)
        if not answer:
            logger.info("[FALLBACK] Ollama Vision multi-image reasoning failed. Falling back to Gemini.")
    
    if not answer:
        logger.info("[TEXT ROUTE] Routing to Gemini 2.5 Flash.")
        answer = await call_gemini(query, text_system_prompt)

    # ── STEP 5: Response Finalization ──
    final_sources = []
    seen_src_names = set()
    for m in retrieved_meta:
        s = m.get("source")
        if s and s not in seen_src_names:
            final_sources.append(s)
            seen_src_names.add(s)
    
    for m in selected_image_metas:
        img_src = m.get("image_src")
        if img_src and img_src not in seen_src_names:
            final_sources.append(f"Image: {img_src}")

    return QueryResponse(answer=answer, sources=final_sources)
