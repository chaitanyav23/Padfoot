"""
ingest_images.py – Multimodal Image Ingestion Pipeline
======================================================
1. Scans HTML files for image references.
2. Performs OCR using EasyOCR.
3. Generates SigLIP embeddings (normalized).
4. Generates image summaries using Llama 3.2 Vision (via HF API).
5. Stores everything in a separate ChromaDB 'image_embeddings' collection.
"""

import os
import base64
import time
from pathlib import Path
from bs4 import BeautifulSoup
from PIL import Image
import torch
from transformers import AutoProcessor, SiglipModel
import easyocr
import chromadb
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
DOCS_DIR          = "./HTML"
CHROMA_DIR        = "./chroma_db"
IMG_COLLECTION    = "image_embeddings"
SIGLIP_MODEL_ID   = "google/siglip-base-patch16-224"
VISION_MODEL_ID   = "meta-llama/Llama-3.2-11B-Vision-Instruct"
HF_TOKEN          = os.getenv("HF_TOKEN")

# Boilerplate icons to ignore
IGNORE_IMAGES = {"next.png", "up.png", "prev.png", "contents.png", "index.png"}

# ── Setup ──────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SigLIP model: {SIGLIP_MODEL_ID}...")
processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
model = SiglipModel.from_pretrained(SIGLIP_MODEL_ID).to(device)

print("Initializing OCR (EasyOCR)...")
reader = easyocr.Reader(['en'])

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
# Image collection - we manually manage embeddings
collection = chroma_client.get_or_create_collection(name=IMG_COLLECTION)

def get_image_embedding(image_path):
    """Generates and normalizes SigLIP embedding for an image."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # Normalization
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"  → Embedding Error: {e}")
        return None

def get_ocr_text(image_path):
    """Extracts text from image using EasyOCR."""
    try:
        results = reader.readtext(str(image_path), detail=0)
        return " ".join(results).strip()
    except Exception as e:
        print(f"  → OCR Error: {e}")
        return ""

def get_image_summary(image_path, ocr_text):
    """Generates a technical summary using Llama 3.2 Vision via HF API."""
    if not HF_TOKEN:
        return "HF_TOKEN missing, summary skipped."
    
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        API_URL = f"https://api-inference.huggingface.co/models/{VISION_MODEL_ID}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        prompt = (
            "Summarize this technical screenshot for a documentation assistant. "
            f"The OCR extracted this text: '{ocr_text}'. "
            "Describe UI elements, settings, and the technical purpose concisely."
        )
        
        payload = {
            "model": VISION_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ],
            "parameters": {"max_new_tokens": 200}
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, list): return res[0].get("generated_text", "")
            return res.get("choices", [{}])[0].get("message", {}).get("content", "")
        return f"HF API Error: {response.status_code}"
    except Exception as e:
        return f"Summary error: {e}"

def ingest_images():
    docs_path = Path(DOCS_DIR)
    html_files = list(docs_path.glob("*.html"))
    
    print(f"Scanning {len(html_files)} pages for images...")
    
    for html_path in html_files:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
            img_tags = soup.find_all("img")
            
            if not img_tags:
                continue
                
            print(f"\nProcessing {html_path.name} ({len(img_tags)} images)...")
            
            for img in img_tags:
                src = img.get("src")
                if not src or src in IGNORE_IMAGES:
                    continue
                
                # Filter latex math images
                if "img" in src.lower() and any(c.isdigit() for c in src):
                    continue
                
                img_path = docs_path / src
                if not img_path.exists():
                    continue
                
                print(f"  → Image: {src}")
                
                # 1. OCR
                ocr_text = get_ocr_text(img_path)
                
                # 2. Embedding (SigLIP)
                embedding = get_image_embedding(img_path)
                
                # 3. Summary (Llama Vision)
                summary = get_image_summary(img_path, ocr_text)
                
                if embedding:
                    # Store exact source_page for exact filtering
                    doc_id = f"{html_path.name}_{src}"
                    collection.upsert(
                        ids=[doc_id],
                        embeddings=[embedding],
                        metadatas=[{
                            "source_page": html_path.name,
                            "image_src": src,
                            "ocr_text": ocr_text,
                            "summary": summary,
                            "full_path": str(img_path),
                            "image_type": "technical_screenshot"
                        }],
                        documents=[f"OCR: {ocr_text}\nSummary: {summary}"]
                    )
                    time.sleep(0.5) # Rate limit protection

    print("\n✅ Multimodal ingestion complete!")

if __name__ == "__main__":
    ingest_images()
