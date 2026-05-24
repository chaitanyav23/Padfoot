# Padfoot: Contextual Multimodal Documentation Assistant

Padfoot is a production-grade, context-aware Retrieval-Augmented Generation (RAG) assistant explicitly engineered for technical documentation and institutional knowledge bases. Moving beyond rudimentary text-only RAG systems, Padfoot implements an advanced **Text-Grounded Multimodal Pipeline**. It does not merely retrieve paragraphs; it understands the semantic relationships between written documentation, procedural workflows, and the sequential UI screenshots and diagrams embedded within them.

Designed initially for the Indian Institute of Technology Kanpur (IITK) Computer Centre’s legacy information repository, Padfoot's architecture follows a strict "augmentation over replacement" philosophy. It injects dynamic, situational awareness directly into legacy `latex2html` static pages, delivering modern AI capabilities without sacrificing the inherent speed, security, and cacheability of a static-first web architecture.

---

## 1. System Overview & The Multimodal Imperative

Technical documentation is inherently multimodal. Procedural guides, server configurations, and software troubleshooting steps rely heavily on visual aids—screenshots of error dialogues, UI workflows, and architectural diagrams.

**Traditional RAG fails here.** A text-only retrieval system might extract a sentence like *"Click the advanced settings gear,"* but without the accompanying screenshot showing *where* that gear is, the instruction is incomplete. Conversely, single-image Vision-Language (VQA) systems lack the broader semantic context of the surrounding manual.

**Padfoot bridges this gap.** It is a workflow-aware, screenshot-aware, and UI-state-aware documentation copilot.

### Core Capabilities
*   **Text-Grounded Multimodal Retrieval:** Padfoot establishes the documentation text as the primary semantic anchor. It first identifies *where* the answer lives in the text corpus, and subsequently aggregates all visual context associated with that specific section.
*   **Sequential Workflow Awareness:** The system groups and analyzes multiple screenshots simultaneously, allowing the VLM to understand multi-step UI transitions and procedural states rather than isolated images.
*   **OCR-Augmented Semantic Reranking:** Bridges the terminology gap between user queries and visual assets by cross-referencing semantic intent with hard OCR extractions.
*   **Local, Privacy-Preserving Synthesis:** Executes multimodal reasoning entirely locally using **Qwen2.5-VL** via **Ollama**, ensuring institutional data never leaves the premises for visual analysis.
*   **Situational UI Awareness:** The frontend widget inherently understands the user's navigational state, forcing the retrieval engine to prioritize context from the currently active HTML page.

---

## 2. Multimodal Architecture & Pipeline

Padfoot utilizes a decoupled, parallel-stream architecture designed to handle the high computational demands of multimodal RAG while maintaining a responsive user experience.

### The Reasoning Pipeline Flow

```text
[User Query & State]
      │
      ├─► Captures current HTML filename via widget.js
      ▼
[Hybrid Text Retrieval] (Primary Signal)
      │
      ├─► embeds query via all-MiniLM-L6-v2
      ├─► Local Boost: queries ChromaDB specifically for current page
      ├─► Global Search: retrieves supplementary context
      ▼
[Page Context Identification]
      │
      ├─► Extracts unique source pages from retrieved text chunks
      ▼
[Image Candidate Retrieval] (Secondary Signal)
      │
      ├─► embeds query via SigLIP (google/siglip-base-patch16-224)
      ├─► retrieves top 15 candidate screenshots from image_embeddings
      ▼
[Contextual Grouping & Reranking]
      │
      ├─► Calculates OCR Overlap (query terms vs. extracted image text)
      ├─► Applies Page-Match Penalty (massive boost if image belongs to identified text pages)
      ├─► Sorts by composite score: Distance - (OCR_Overlap * 0.12) - Page_Match_Bonus
      ▼
[Multimodal Assembly]
      │
      ├─► Resizes top images to 384x384 & applies JPEG compression
      ├─► Aggregates sequential OCR text blocks
      ├─► Compiles final prompt: Text Chunks + OCR Data + Multiple Images
      ▼
[Inference Routing]
      │
      ├─► [SUCCESS] ──► Ollama (Qwen2.5-VL 3B) generates multi-image contextual answer
      │
      └─► [FAILURE/TIMEOUT/NO IMAGES] ──► Fallback to Gemini 2.5 Flash (Text Only)
```

---

## 3. Deep-Dive: Contextual Multimodal Reasoning

### Text-Grounded Retrieval Strategy
The architectural philosophy of Padfoot is: **"Retrieve workflows, not isolated screenshots."**
Image embeddings (like SigLIP) are excellent at finding pictures of "a login screen," but they lack the semantic density to answer *how to bypass the login screen's specific SSH error*. 

Therefore, Padfoot uses text retrieval as the dominant signal. By querying the `MiniLM` text index first, the system identifies the exact documentation pages relevant to the query. The image retrieval phase is then heavily biased toward fetching screenshots that live *on those specific pages*. This ensures the visual context perfectly aligns with the semantic context, preventing the VLM from hallucinating instructions based on an irrelevant UI image from a different manual.

### Multi-Image Sequential Reasoning
A single screenshot is rarely sufficient for technical support. Padfoot aggregates a contextual cluster of screenshots (up to `MAX_IMAGES_TO_VLM`) associated with the retrieved text. When the prompt is dispatched to Qwen2.5-VL, it contains an array of images. The VLM reasons over these images simultaneously, enabling it to synthesize multi-step procedural instructions by observing how the UI state changes across the screenshots.

### OCR-Augmented Reranking
To solve the classic "terminology gap" in image retrieval (e.g., the user searches for "IP Configuration," but the image is just a screenshot of a network panel), Padfoot performs a full `EasyOCR` scan during the ingestion phase. 

During retrieval, the `rerank_and_group_images` function calculates an OCR overlap score. Images containing text tokens that match the user's query receive a significant ranking boost. This hard-text extraction acts as a critical safety net against the fuzziness of dense semantic image embeddings.

---

## 4. Local VLM Deployment: Ollama & Qwen2.5-VL

Padfoot aggressively optimizes for local, private deployment using the Ollama orchestration layer. HuggingFace hosted inference was intentionally abandoned due to provider instability, latency jitter, and the necessity for on-premise execution in academic/enterprise environments.

### Hardware Optimization & Constraints
Local multimodal inference is memory-bound. Padfoot is engineered to run on consumer-grade hardware (specifically targeting the constraints of an Nvidia RTX 3050 4GB/6GB).

To achieve reliable execution without Out-Of-Memory (OOM) failures, Padfoot implements a strict payload optimization pipeline:
*   **Aggressive Resizing:** `img.thumbnail((384, 384), Image.LANCZOS)` preserves aspect ratios while drastically shrinking the tensor matrix required by the vision encoder.
*   **Lossy Compression:** JPEG compression (Quality 75) minimizes base64 payload transmission latency to the Ollama API.
*   **VRAM Capping:** The Ollama payload explicitly defines `"num_ctx": 1024` and `"num_predict": 128` to restrict the KV cache size, leaving maximum VRAM available for the model weights via `"num_gpu": 15`.

### Fallback Engineering
Local inference on shared infrastructure can queue or timeout. Padfoot wraps the Ollama invocation in an `asyncio.wait_for` block with a 180-second `LLM_TIMEOUT`. If the local VLM fails, times out, or if no relevant images are retrieved, the system gracefully degrading to a purely semantic text-RAG flow powered by Google Gemini 2.5 Flash.

---

## 5. Frontend Architecture: Non-Invasive Augmentation

Padfoot rejects the modern trend of rebuilding documentation sites into heavy Single Page Applications (SPAs). It adopts a **static-first, zero-framework philosophy**.

The IITK documentation consists of hundreds of flat HTML files. Rebuilding this in React or Next.js would break legacy URLs, require extensive CI/CD pipelines, and introduce massive JavaScript overhead for documents that are fundamentally just text and images.

### The Augmentation Strategy
*   **Idempotent Widget Injection:** A lightweight Python utility (`inject_widget.py`) iterates over the static `HTML/` directory, safely injecting `<script>` and `<link>` tags into the DOM without altering the underlying markup.
*   **Situational UI Capture:** The vanilla JavaScript widget (`widget.js`) uses browser APIs (`window.location.pathname`) to determine its exact location within the documentation tree. This acts as the `source_page` context for the backend's Local Boost retrieval.
*   **Native Multimodal Display:** When the backend returns an `images_used` array, the UI dynamically renders a horizontal gallery of rounded thumbnails, complete with a custom CSS modal for full-size viewing.

The result is an intelligent, floating AI copilot that lives transparently on top of high-performance static files.

---

## 6. Performance, Logging, & Operational Telemetry

For infrastructure engineers, Padfoot provides detailed, structured logging to monitor the health and decision-making process of the retrieval pipeline.

### Operational Logging Examples
```text
[INFO] Padfoot: [TEXT RETRIEVAL] Retrieved 4 chunks from pages: ['Abaqus2019.html', 'Abaqus.html']
[INFO] Padfoot:   -> Selected: next.png | Score: 1.1502 | Page Match: True
[INFO] Padfoot:   -> Selected: abaqus-1.png | Score: 1.4210 | Page Match: True
[INFO] Padfoot: [IMAGE GROUPING] Grouped 2 associated screenshots.
[INFO] Padfoot: [MULTIMODAL ROUTE] Initiating text-grounded multimodal reasoning.
[INFO] Padfoot: [OLLAMA] Sending 2 images to qwen2.5vl:3b
[INFO] Padfoot: [OLLAMA] Vision response received
```

These logs allow operators to instantly determine:
1. Which pages anchored the text search.
2. How the OCR and Page-Match modifiers influenced the final image selection.
3. Whether the system successfully executed the local VLM or dropped to the Gemini fallback.

---

## 7. Project Structure

```text
Padfoot/
├── HTML/                   # Modified static documentation site
│   ├── widget.js           # Vanilla JS frontend logic & API client
│   ├── widget.css          # Scoped UI styles for the floating assistant
│   ├── inject_widget.py    # Automation utility for site-wide DOM injection
│   └── *.html              # 100+ legacy documentation pages
├── chroma_db/              # SQLite-backed persistent vector store
├── app.py                  # FastAPI Backend: Retrieval & Routing Orchestrator
├── ingest.py               # Text ETL: HTML parsing, chunking, and MiniLM embedding
├── ingest_images.py        # Vision ETL: EasyOCR extraction & SigLIP embedding
├── requirements.txt        # Production Python environment dependencies
└── .env                    # Secrets (GOOGLE_API_KEY)
```

---

## 8. Why This Architecture Matters

### Traditional RAG vs. Padfoot

**Traditional Documentation RAG:**
*   Retrieves isolated text paragraphs.
*   Ignores screenshots, treating them as dead HTML `<img>` tags.
*   Cannot answer procedural questions involving visual UI states (e.g., "What does the error icon look like?").

**Padfoot Multimodal RAG:**
*   Retrieves complete procedural workflows.
*   Contextually binds text instructions to their corresponding visual screenshots.
*   Understands UI states and reasons over procedural sequences.
*   Performs multimodal synthesis, explaining *how* to perform an action using both the written manual and the visual evidence.

This architecture is essential for institutional IT support, employee onboarding, and complex software troubleshooting where text alone is inherently ambiguous.

---

## 9. Future Roadmap

Padfoot is an evolving systems engineering project. Planned infrastructural improvements include:

*   **Server-Sent Events (SSE) Streaming:** Implementing real-time token streaming from the Ollama backend to reduce perceived TTFT (Time-To-First-Token) latency in the UI.
*   **Semantic Retrieval Caching:** Integrating Redis to cache high-confidence, exact-match text embeddings and LLM responses to bypass inference entirely for common queries.
*   **Conversational State Persistence:** Moving beyond stateless requests by introducing session IDs and conversation history memory buffers.
*   **Advanced Workflow Graph Traversal:** Mapping the HTML hyperlinking structure into a directed graph, allowing the retrieval engine to automatically pull context from "Next Page" or "Previous Page" links during troubleshooting.
*   **Asynchronous Multimodal Batching:** Optimizing the Ollama payload pipeline to handle highly concurrent visual queries more efficiently on shared GPU infrastructure.
*   **Agentic Orchestration:** Allowing the VLM to execute specialized sub-routines (e.g., querying server status APIs) if the documentation context suggests an operational outage.