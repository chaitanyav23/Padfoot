"""
ingest_images_hf.py – Image Embedding Ingestion with CLIP
=========================================================
Parses HTML files to find technical image references, generates 
embeddings using CLIP (openai/clip-vit-base-patch32), and stores 
them in a separate ChromaDB collection.
"""

import os
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_DIR        = "./HTML"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "image_embeddings"
CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"

# Navigation/Boilerplate images to ignore
IGNORE_IMAGES = {"next.png", "up.png", "prev.png", "contents.png", "index.png"}

# ── Setup ──────────────────────────────────────────────────────────────────────
print(f"Loading CLIP model: {CLIP_MODEL_ID}...")
model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# We don't use a default embedding function because we're providing our own CLIP vectors
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        # Normalize and convert to list
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"  → CLIP Error for {image_path}: {e}")
        return None

def ingest_images():
    docs_path = Path(DOCS_DIR)
    html_files = list(docs_path.glob("*.html"))

    print(f"Found {len(html_files)} HTML files. Scanning for images...")

    for i, html_path in enumerate(html_files):
        print(f"[{i+1}/{len(html_files)}] Scanning: {html_path.name}")
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        # Find all <img> tags
        img_tags = soup.find_all("img")
        page_images = []
        for img in img_tags:
            src = img.get("src")
            if not src:
                continue
            
            # Simple heuristic: ignore small navigation icons and math/latex images
            if src in IGNORE_IMAGES or "img" in src.lower() and src.endswith(".png"):
                # Many latex2html images are named img1.png, img2.png etc.
                if any(char.isdigit() for char in src):
                    continue
            
            img_full_path = docs_path / src
            if img_full_path.exists() and img_full_path.is_file():
                if src not in page_images:
                    page_images.append(src)

        if not page_images:
            continue

        print(f"  → Found {len(page_images)} technical images.")
        
        for src in page_images:
            img_path = docs_path / src
            embedding = get_image_embedding(img_path)
            
            if embedding:
                doc_id = f"{html_path.name}::{src}"
                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "source_page": html_path.name,
                        "image_src": src,
                        "full_path": str(img_path)
                    }],
                    documents=[f"Image {src} from page {html_path.name}"] # Placeholder text
                )

    print("
✅ Image ingestion complete!")

if __name__ == "__main__":
    ingest_images()
