"""
ingest.py – Data Ingestion & Preprocessing Script
==================================================
Reads every .html, .png, .jpg, and .jpeg file in DOCS_DIR, converts each
to clean text (HTML) or a vision-generated summary (images), splits the
text into overlapping chunks, and upserts everything into a persistent
local ChromaDB collection.

Usage:
    python ingest.py

Required env var:
    GOOGLE_API_KEY  – used for the Gemini Vision API (image summarisation)
"""

import os
import re
import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_DIR        = "./HTML"          # Folder containing source files
CHROMA_DIR      = "./chroma_db"     # Persistent ChromaDB storage path
COLLECTION_NAME = "docs"
CHUNK_SIZE      = 800               # Characters per chunk
CHUNK_OVERLAP   = 150               # Overlap between consecutive chunks
VISION_MODEL    = "gemini-2.0-flash"   # Supports image input, free tier
BATCH_SIZE      = 100               # ChromaDB upsert batch size


# ── Client / Collection Setup ──────────────────────────────────────────────────
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
vision_model = genai.GenerativeModel(VISION_MODEL)

# CPU-friendly embedding model – no GPU required
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=sentence_transformer_ef,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# ── HTML Processing ────────────────────────────────────────────────────────────
def extract_text_from_html(filepath: str) -> str:
    """
    Parse an HTML file with BeautifulSoup and return clean body text.
    Boilerplate elements (navigation divs, scripts, styles) are removed first.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Strip boilerplate tags before extracting text
    for tag in soup.select("div.navigation, script, style, nav, footer, header"):
        tag.decompose()

    # get_text with a newline separator keeps paragraph structure intact
    raw = soup.get_text(separator="\n", strip=True)

    # Collapse consecutive blank lines for cleaner chunking
    lines = [line for line in raw.splitlines() if line.strip()]
    return "\n".join(lines)


# ── Image Processing ───────────────────────────────────────────────────────────
def summarize_image_with_vision(filepath: str) -> str:
    """
    Open the image with Pillow, send it to the Gemini Vision API, and return
    a concise technical summary of the UI, code snippet, or error message shown.
    Retries automatically on 429 rate-limit errors using the suggested delay.
    """
    img = Image.open(filepath)
    prompt = (
        "You are a technical documentation assistant. "
        "Summarize the technical UI, code, or error message shown in this image. "
        "Be concise and specific. Focus on actionable steps, "
        "visible settings, field names, or error details."
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = vision_model.generate_content([prompt, img])
            time.sleep(1)  # small delay to stay within per-minute quota
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                match = re.search(r'retry in (\d+)', error_str)
                wait = int(match.group(1)) + 5 if match else 60
                print(f"  → Rate limited. Waiting {wait}s …")
                time.sleep(wait)
            else:
                raise


# ── Ingestion Pipeline ─────────────────────────────────────────────────────────
def ingest_documents():
    """
    Walk DOCS_DIR, process each supported file, chunk the text, and upsert
    all chunks into ChromaDB in batches.
    """
    docs_path = Path(DOCS_DIR)
    all_files = (
        list(docs_path.glob("*.html"))
        + list(docs_path.glob("*.png"))
        + list(docs_path.glob("*.jpg"))
        + list(docs_path.glob("*.jpeg"))
    )

    print(f"Found {len(all_files)} files in '{DOCS_DIR}'.\n")

    ids: list[str]       = []
    texts: list[str]     = []
    metadatas: list[dict] = []

    for i, file_path in enumerate(all_files):
        filename = file_path.name
        ext      = file_path.suffix.lower()
        print(f"[{i+1}/{len(all_files)}] Processing: {filename}")

        try:
            if ext == ".html":
                raw_text    = extract_text_from_html(str(file_path))
                source_type = "html"
            elif ext in (".png", ".jpg", ".jpeg"):
                print("  → Skipped (images disabled).")
                continue
            else:
                continue  # skip unsupported extensions

            if not raw_text.strip():
                print("  → Skipped (empty content).")
                continue

            chunks = text_splitter.split_text(raw_text)
            for j, chunk in enumerate(chunks):
                ids.append(f"{filename}__chunk_{j}")
                texts.append(chunk)
                metadatas.append({
                    "source": filename,
                    "type":   source_type,
                    "chunk":  j,
                })

        except Exception as e:
            print(f"  → ERROR: {e}")
            continue

    if not ids:
        print("No documents were ingested.")
        return

    # Upsert in batches to avoid memory pressure
    print(f"\nUpserting {len(ids)} chunks into collection '{COLLECTION_NAME}' …")
    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Upserted chunks {start + 1}–{min(end, len(ids))}")

    print("\n✅  Ingestion complete!")
    print(f"    Total chunks stored: {len(ids)}")
    print(f"    ChromaDB path:       {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_documents()
