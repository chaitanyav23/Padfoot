✦ Padfoot: Context-Aware RAG Documentation Assistant


Padfoot is a context-aware Retrieval-Augmented Generation (RAG) assistant embedded directly into IIT Kanpur’s static documentation site.

It transforms legacy latex2html static pages into an intelligent, page-aware system without modifying the static-first architecture.

Built with:
- FastAPI + Gunicorn
- ChromaDB
- SentenceTransformers
- Gemini 2.5 Flash
- Vanilla JS frontend

The system prioritizes local page context, enforces strict hallucination safeguards, and remains fully production deployable on institutional servers.


  1. PROJECT OVERVIEW


  The original website hosted at the Indian Institute of Technology Kanpur (IITK) Computer Centre
  (https://linux.cc.iitk.ac.in/lininfo/) serves as a comprehensive information repository for servers, services, and
  software available to the academic community. Historically, this documentation was maintained in a master LaTeX file
  and compiled into static HTML pages using the latex2html utility.


  Because the generated output consisted entirely of static HTML files, it was highly performant and easily hosted on
  IITK physical servers as lightweight static content. However, this architectural simplicity introduced significant
  usability limitations:
   * Manual Navigation: Users were required to manually traverse hierarchical menus or rely on rudimentary browser text
     searches to find specific configurations, server IP addresses, or software instructions.
   * Lack of Semantic Search: Traditional keyword searches fail when terminology mismatches occur (e.g., searching for
     "email setup" when the document uses "Thunderbird configuration").
   * No Contextual Awareness: If a user was reading a page about GPU servers and had a question about access limits,
     they had no way to ask a system that understood what page they were currently viewing.
   * Absence of Intelligent Assistance: Static pages cannot synthesize information across multiple documents to provide
     unified, step-by-step troubleshooting answers.

  This project solves these limitations by injecting a Retrieval-Augmented Generation (RAG) powered documentation
  assistant directly into the static HTML pages, transforming a legacy static site into an intelligent, interactive, and
  context-aware platform without abandoning the underlying lightweight HTML architecture.

  2. THE SOLUTION


  To address the limitations of the static documentation, this project introduces a sophisticated overlay that
  transforms the user experience. The solution is an embedded, floating chat widget powered by a Retrieval-Augmented
  Generation (RAG) backend.


  Key features of the solution include:
   * RAG-Powered Documentation Assistant: Instead of answering from generalized internet training data, the assistant is
     strictly grounded in the institution's official documentation. It retrieves the exact paragraphs relevant to a
     user's query and synthesizes a direct answer.
   * Page-Aware Contextual AI: The system possesses "situational awareness." It inherently knows which specific HTML
     page the user is currently reading.
   * Hybrid Retrieval System: The retrieval engine employs a dual-pass strategy. It first performs a "Local Boost"
     search, actively seeking answers within the page the user is currently viewing. It simultaneously performs a
     "Global Search" across the entire documentation corpus to fill in any missing context.
   * Gemini-Powered Generation: The system utilizes Google's Gemini 2.5 Flash model for high-speed, high-accuracy
     natural language generation based exclusively on the retrieved vector data.

  3. SYSTEM ARCHITECTURE


  The architecture is explicitly designed to decouple the static frontend from the intelligent backend, ensuring the
  HTML files remain cacheable and highly performant while offloading heavy vector computations to a dedicated
  application server.


  Component Breakdown
   * Frontend (Static HTML + widget.js): The original HTML pages are augmented with a lightweight, vanilla JavaScript
     client (widget.js) and corresponding CSS. This client captures user input and current page state without requiring
     complex frontend frameworks like React or Angular.
   * Backend (FastAPI + Gunicorn): A high-performance asynchronous Python API handles incoming queries, orchestrates the
     vector database retrieval, and communicates with the LLM. It is served by Gunicorn utilizing Uvicorn ASGI workers
     to handle concurrent user requests efficiently.
   * Vector Database (ChromaDB): An embedded, persistent instance of ChromaDB stores the chunked documentation and their
     mathematical representations.
   * Embedding Model (SentenceTransformer all-MiniLM-L6-v2): A CPU-optimized, localized embedding model translates text
     chunks into dense vectors. Running this locally eliminates latency and API costs associated with cloud-based
     embedding services.
   * LLM (Gemini 2.5 Flash): Google's generative model is invoked to synthesize the final answer, strictly constrained
     by a highly specific system prompt and generation configuration.


  System Safeguards and Logic
   * Hybrid Retrieval Logic: Prioritizes local context (current page) while supplementing with global context.
   * Metadata-Based Filtering: Utilizes ChromaDB's where clause to specifically target documents whose source metadata
     matches the user's active page.
   * Deduplication Strategy: Implements a programmatic check using a seen_ids set to ensure that if a document chunk is
     retrieved in the local boost, it is not redundantly appended by the global search.
   * Context Cap (MAX_CHUNKS): A hard limit is enforced on the number of context chunks passed to the LLM (capped at 5)
     to prevent token bloat, reduce latency, and maintain focus.
   * Token Control & Timeout Protection: The LLM call is wrapped in an asyncio.wait_for block with a strict timeout to
     prevent thread hanging. Generation parameters dictate exact output token limits.
   * Query Length Guard: API endpoints strictly reject excessively long queries to prevent malicious payload attacks and
     system resource exhaustion.

  Architecture Diagram


    1 [User / Browser]
    2       |  (1) Views static HTML page, types query in Chat Widget
    3       v
    4 [Frontend: HTML + widget.js]
    5       |  (2) POST /ask { "query": "...", "source_page": "Abaqus2019.html" }
    6       v
    7 [Backend: FastAPI via Gunicorn]
    8       |  (3) Embed query via all-MiniLM-L6-v2
    9       v
   10 [Vector DB: ChromaDB]
   11       |  (4) Hybrid Search:
   12       |      a. n_results=3 WHERE source == source_page
   13       |      b. n_results=4 (Buffer) Global Search
   14       v
   15 [FastAPI: Context Assembler]
   16       |  (5) Deduplicate chunks, cap at MAX_CHUNKS (5)
   17       v
   18 [LLM: Gemini 2.5 Flash]
   19       |  (6) Generate answer using System Prompt + Context
   20       v
   21 [Backend -> Frontend]
   22       |  (7) Return JSON { "answer": "...", "sources": ["Abaqus2019.html"] }
   23       v
   24 [UI: widget.js]
   25          Renders text answer and generates clickable anchor tags for sources

  The data flow dictates exactly what happens when a user submits a query via the frontend widget.


   1. Capture State and Query: The user clicks the chat toggle and inputs a question. widget.js intercepts this
      submission, prevents default form behavior, and dynamically captures the current filename by parsing
      window.location.pathname.
   2. API Invocation: The frontend constructs a JSON payload containing the user's query and the source_page and issues
      a POST request to the FastAPI /ask endpoint.
   3. Validation and Embedding: The backend immediately validates the query length. Upon passing, the query is passed to
      the localized SentenceTransformer model, converting the text into a dense vector representation.
   4. Hybrid Retrieval Execution:
       * The backend queries ChromaDB for the top 3 vector matches specifically filtered by the source_page metadata.
       * It then calculates how many chunks are still needed to reach the MAX_CHUNKS limit and queries the entire
         database globally.
       * The backend iterates through the global results, adding them to the final context array only if their unique ID
         was not already captured in the local search.
   5. Prompt Composition: The retrieved text chunks are assembled into a single string. A system prompt is constructed,
      instructing the LLM to act as an IT assistant, explicitly informing it of the page the user is viewing, and
      demanding that the answer be derived solely from the provided context chunks.
   6. LLM Invocation: The prompt is sent to the Gemini 2.5 Flash API with strict TEMPERATURE and MAX_TOKENS constraints,
      wrapped in an asynchronous timeout handler.
   7. Response Delivery: The backend receives the generated text, extracts the unique source filenames from the
      retrieved metadata, and returns the compiled payload to the client.
   8. UI Rendering: The frontend parses the JSON. It appends the text answer to the chat interface. For the sources, it
      dynamically constructs standard HTML <a> tags utilizing window.location.origin, appending the filename. These
      pills are rendered as clickable links configured to open safely in a new browser tab.

  5. PROJECT STRUCTURE


  The architecture enforces a strict separation between the legacy static files and the modern application backend.


  Old Structure Paradigm
  Previously, the deployment consisted of a single directory containing hundreds of flat HTML files, images, and basic
  CSS stylesheets hosted directly via a standard web server (e.g., Apache or Nginx). There was no application logic.

  New Structure Paradigm


    1 Padfoot/
    2 ├── original_legacy/              # Untouched archive of the original latex2html output
    3 ├── HTML/                         # Modified static site served to end-users
    4 │   ├── (legacy html files)       # Restored and augmented HTML pages
    5 │   ├── widget.js                 # Vanilla JS driving the chat interface
    6 │   ├── widget.css                # Scoped CSS for the chat interface
    7 │   ├── zoom_theme.js             # Accessibility and theme toggling logic
    8 │   ├── search.css                # Placeholder to resolve legacy 404 errors
    9 │   ├── search.js                 # Placeholder to resolve legacy 404 errors
   10 │   └── icons/                    # Directory for static images and SVGs
   11 ├── chroma_db/                    # Persistent directory for the local vector database
   12 ├── app.py                        # Core FastAPI backend application
   13 ├── ingest.py                     # Script to parse HTML and populate ChromaDB
   14 ├── requirements.txt              # Python dependencies specification
   15 └── .gitignore                    # Version control exclusions


  File Responsibilities
   * `HTML/*.html`: The static documentation files. They have been injected with <script> and <link> tags pointing to
     the widget assets.
   * `app.py`: The central intelligence of the system. Handles routing, retrieval logic, context assembly, and LLM
     communication.
   * `ingest.py`: The ETL (Extract, Transform, Load) pipeline. Uses BeautifulSoup to strip boilerplate HTML, chunks the
     text, and stores the vectors and metadata in ChromaDB.
   * `widget.js` & `widget.css`: The presentation layer of the assistant. Designed to be entirely isolated from existing
     page styles to prevent CSS conflicts.

  6. FILE-BY-FILE MODIFICATIONS


  The transition from a static site to an AI-augmented platform required surgical modifications across the codebase.


  The Static HTML Transformation
  Initially, the HTML files were highly nested and prone to duplicate <div> elements due to repeated injection attempts.
   * Cleanup: Python scripts were utilized to read every HTML file, execute precise Regular Expressions to strip
     duplicated theme-selector and page-content wrappers, and restore the files to a clean state.
   * Injection: A safe, idempotent injection script was written to insert <link rel="stylesheet" href="widget.css"> in
     the <head> and <script src="widget.js"></script> immediately before the closing </body> tag.
   * Legacy Error Resolution: Ghost references to missing search.css and search.js files were causing console 404
     errors. Minimal placeholder files were generated to satisfy these browser requests gracefully.


  The Backend (app.py) Evolution
  If a basic retrieval script existed previously, it lacked production safeguards. The following critical systems were
  engineered into app.py:
   * Constants Definition: Added global configuration variables (MAX_TOKENS, TEMPERATURE, MAX_QUERY_LENGTH, LLM_TIMEOUT,
     MAX_CHUNKS) to allow immediate architectural tuning without diving into function logic.
   * Generation Configuration: Implemented genai.types.GenerationConfig to strictly enforce the low temperature required
     for factual, non-hallucinated documentation retrieval.
   * Hybrid Retrieval Implementation: Replaced a simple top-K query with a sophisticated dual-query system that parses
     source_page and executes conditional logic against ChromaDB metadata.
   * Deduplication: Added a mathematical set() check to ensure context payload efficiency.
   * Operational Logging: Integrated Python's standard logging library to track incoming requests, token usage (prompt
     vs. candidate tokens), and unhandled exceptions.


  The Frontend (widget.js) Evolution
  The client script was upgraded from a basic text-forwarder to a context-aware application interface:
   * Context Capture: Implemented window.location.pathname.split('/').pop() || 'index.html' to dynamically ascertain the
     user's location.
   * Request Payload Update: Modified the standard fetch API body to include the source_page parameter alongside the
     query text.
   * Interactive Citations: Transformed basic text spans into functional anchor tags. Implemented window.location.origin
     to ensure link resolution functions correctly regardless of the hosting environment (localhost vs. production
     domains).

  7. DESIGN DECISIONS & DEVELOPER CHOICES

  Every technical decision was made to balance accuracy, performance, and long-term maintainability.


   * Why Hybrid Retrieval: Semantic search across a large domain can surface functionally similar but contextually
     incorrect results. If a user on the "Thunderbird Setup" page asks "How do I add an account?", standard semantic
     search might return instructions for "Outlook." Hybrid retrieval forces the system to consult the current page's
     manual first, mirroring human contextual understanding.
   * Why MAX_CHUNKS = 5: LLMs suffer from "lost in the middle" syndrome when provided with massive context windows.
     Furthermore, excessively large prompts increase API latency and token costs. Five chunks (~4,000 characters)
     provides ample technical context without diluting the prompt's focus.
   * Why Temperature = 0.2: Generative models are naturally creative. In technical documentation, creativity is a
     liability. A near-zero temperature forces the model into an exploitative, deterministic state, strictly adhering to
     the provided context and reducing hallucinations.
   * Why Gemini 2.5 Flash: Chosen for its superior time-to-first-token (TTFT) metrics and generous token limits. Flash
     is optimized for high-frequency, low-latency tasks such as chat widgets.
   * Why ChromaDB: An embedded vector database removes the necessity of managing external containerized services (like
     Milvus or Qdrant) or paying for cloud vector databases (like Pinecone). It operates directly off the disk via
     SQLite, perfect for a static-first deployment.
   * Why Sentence-Transformers (`all-MiniLM-L6-v2`): This model is incredibly lightweight and runs efficiently on
     standard CPUs. It avoids the latency of API-based embeddings and prevents the backend from being bottlenecked by third-party rate limits during the high-volume ingestion phase.
   * Why Gunicorn with Uvicorn Workers: FastAPI is asynchronous by design. However, running it via a direct Python
     process is not production-viable. Gunicorn acts as a robust process manager, while Uvicorn workers handle the ASGI
     protocol, allowing the server to process multiple concurrent requests without blocking.


  Configurable Placeholders
  The system is designed to be tuned. The following constants in app.py and widget.js can be modified based on
  operational data:
   * CHAT_MODEL: To upgrade or swap the underlying LLM.
   * MAX_CHUNKS: To widen or narrow the context window.
   * TEMPERATURE: To adjust the rigidity of the assistant's tone.
   * MAX_QUERY_LENGTH: To prevent abuse of the text input field.
   * LLM_TIMEOUT: To aggressively drop hanging network requests.
   * API_URL (in widget.js): To point the frontend to the production domain.
   * --workers (in the Gunicorn execution command): To scale CPU utilization based on server hardware.

  8. HOW TO RUN LOCALLY

  To run the application in a local development environment, execute the following commands in your terminal.

   1. Create and Activate Virtual Environment:


   1    python3 -m venv venv
   2    source venv/bin/activate  # On Windows use: venv\Scripts\activate

   2. Install Dependencies:
   1    pip install -r requirements.txt

   3. Configure Environment Variables:


   1    export GOOGLE_API_KEY="your_api_key_here"  # On Windows use: set GOOGLE_API_KEY="your_api_key_here"

   4. Run Ingestion (If Vector Database is empty or files have changed):
   1    python ingest.py

   5. Start the FastAPI Backend (Using Gunicorn for production parity):


   1    gunicorn -k uvicorn.workers.UvicornWorker app:app --workers 2 --bind 127.0.0.1:8000

   6. Start the Frontend Static Server (Open a new terminal window):


   1    cd HTML
   2    python3 -m http.server 5500

   7. Access the Application:
     Open a browser and navigate to http://localhost:5500/index.html.

  9. IF STARTING FROM original_legacy ONLY


  If you are starting from a completely raw state and need to reconstruct the system using only the original_legacy
  directory, follow this disaster recovery pipeline:


   1. Reconstruct HTML Directory:
      Copy all contents from original_legacy/ into a new HTML/ directory.
   2. Asset Migration:
      Ensure widget.js, widget.css, zoom_theme.css, and zoom_theme.js are placed in the root of the new HTML/ directory.
  Create empty search.css and search.js files to prevent 404 logging.
   3. Inject Dependencies:
      You must execute a python injection script that iterates over all .html files in the HTML/ directory, safely inserting the CSS link into the head and the JS script before /body.
   4. Regenerate Vector Database:
      Delete any existing chroma_db/ folder to ensure a clean slate. Run python ingest.py. This will parse the freshly
  copied HTML files, apply BeautifulSoup transformations, generate new embeddings, and build the SQLite database from
  scratch.

  10. PRODUCTION & DEPLOYMENT (VERY DETAILED)


  Implemented Production Readiness
  The application code is already hardened for production:
   * Timeout Protection: Failsafes exist to prevent hanging LLM API calls.
   * Resource Caps: Hard limits on tokens, query lengths, and retrieval chunks prevent memory exhaustion.
   * Logging: Comprehensive error and performance logging is active.
   * Gunicorn: The application is structured to be run via an industrial WSGI/ASGI process manager.

  To prepare Padfoot for a production deployment on IITK physical servers, must transition from a development-centric directory to a clean, service-oriented structure. This involves stripping out development artifacts (backups, temporary scripts, virtual environments) and hardening the remaining files.


  1. Production Root Directory Structure
  Production root (e.g., /var/www/padfoot/) should contain only the operational core. Do not upload local venv/, __pycache__, or .git folders.


    1 /var/www/padfoot/
    2 ├── HTML/                   # The public-facing static content
    3 │   ├── icons/              # Image assets
    4 │   ├── widget.js           # Production-ready JS client
    5 │   ├── widget.css          # Production-ready CSS styles
    6 │   ├── zoom_theme.js       # Accessibility logic
    7 │   ├── search.js           # Placeholder/Stub
    8 │   ├── search.css          # Placeholder/Stub
    9 │   └── *.html              # All 100+ documentation pages
   10 ├── chroma_db/              # The SQLite/Vector database
   11 ├── app.py                  # The FastAPI backend
   12 ├── .env                    # Secrets
   13 └── requirements.txt        # Dependency list

  ---

  2. Relevant vs. Irrelevant Files



  ┌────────────────────────┬────────────────┬─────────────────────────────────────────────────────────────────┐
  │ File/Folder            │ Status         │ Reason                                                          │
  ├────────────────────────┼────────────────┼─────────────────────────────────────────────────────────────────┤
  │ `app.py`               │ Critical       │ This is the heart of the service.                               │
  │ `HTML/`                │ Critical       │ Contains all user-facing content and the AI widget.             │
  │ `chroma_db/`           │ Critical       │ The pre-built vector database.                                  │
  │ `requirements.txt`     │ Critical       │ Used to build the server-side virtual environment.              │
  │ `ingest.py`            │ Optional       │ Keep this on the server only if you plan to re-ingest files     │
  │                        │                │ frequently.                                                     │
  │ `original_legacy/`     │ Irrelevant     │ Archive only. Do not deploy to production.                      │
  │ `*.html.bak`           │ Irrelevant     │ Remove these. They bloat the server and pose a security risk if │
  │                        │                │ served.                                                         │
  │ `cleanup_html.py`      │ Irrelevant     │ Local utility script. Not needed for runtime.                   │
  │ `inject_widget.py`     │ Irrelevant     │ Local utility script.                                           │
  │ `wsl_venv/`            │ Irrelevant     │ Never upload local virtual environments.                        │
  └────────────────────────┴────────────────┴─────────────────────────────────────────────────────────────────┘

  ---

  3. Required File Modifications


  A. Backend: `app.py`
  In production, you should move the GOOGLE_API_KEY handling to use the environment variables properly (already
  implemented in your current version via os.environ), but ensure the CORS settings are restricted to the official IITK
  domain.


   * Change: Update allow_origin_regex to match your production domain.
   * Current: allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
   * Production: allow_origins=["https://linux.cc.iitk.ac.in"] (Remove the regex for stricter security).

  B. Frontend: `HTML/widget.js`
  The widget needs to know where the production API is located.


   * Change: Change API_URL to the relative or absolute production path.
   * Modification:


   1     // Change this:
   2     var API_URL = 'http://localhost:8000/ask';
   3
   4     // To this :
   5     var API_URL = window.location.origin + '/ask';


  C. The HTML Pages
  Ensure all .html files have been "cleaned." Before deploying:
   1. Run your cleanup_html.py one last time locally to ensure no div duplication exists.
   2. Verify that all script tags point to zoom_theme.js, widget.js, and widget.css without directory prefixes like
      HTML/ (since they will be in the same folder as the HTML files).

  ---

  4. Critical Deployment Checklist


  Step 1: The "Clean" Build
  Before transferring files to the IITK server, create a dist/ folder locally and copy only the Critical items listed in
  Section 1 and 2. This prevents accidental upload of several hundred megabytes of __pycache__ and venv data.


  Step 2: Environment Security
  Never hardcode the GOOGLE_API_KEY. On the server, create the .env file manually:


   1 cd /var/www/padfoot
   2 nano .env
   3 # Add: GOOGLE_API_KEY=your_actual_key
   4 chmod 600 .env # Extremely important for security


  Step 3: Gunicorn Worker Tuning
  In the systemd service file, set the --workers count based on the server's CPU cores.
   * Formula: (2 x $num_cores) + 1


  Step 4: Nginx Static Optimization
  In the Nginx config, ensure the location /lininfo/ block uses the expires directive to leverage browser caching for
  your 100+ static images:
   1 location ~* \.(?:jpg|jpeg|gif|png|ico|svg|css|js)$ {
   2     expires 7d;
   3     add_header Cache-Control "public";
   4 }


  Step 5: Health Monitoring
  Once deployed, verify the health of the production system via the endpoint:
  https://linux.cc.iitk.ac.in/health
  This will confirm ChromaDB is loaded and the backend is communicating with the vector database.

  11. Transformation
  
  A. Modified Files 
   * `*.html` (All 100+ Documentation Pages):
       * Original State: Flat HTML with navigation menus at top and bottom.
       * Modifications:
           1. Injection: Added <link rel="stylesheet" href="widget.css"> and <link rel="stylesheet"
              href="zoom_theme.css"> in the <HEAD>.
           2. Structural Wrapper: Injected <div id="theme-selector"> and wrapped the entire original page content in a
              <div id="page-content"> to enable targeted CSS scaling.
           3. Scripts: Injected <script src="zoom_theme.js"></script> and <script src="widget.js"></script> before
              </BODY>.
           4. Cleanup: Executed regex-based stripping to remove duplicated tags from failed previous injection attempts.


  B. Newly Created Supporting Assets
   * `widget.js`:
       * Function: The "brain" of the frontend. Handles the floating chat UI, captures the current filename
         (source_page), sends AJAX requests to the FastAPI backend, and renders Markdown-style responses with clickable
         source links.
   * `widget.css`:
       * Function: Provides scoped styling for the chat widget. Designed with high z-index and specific CSS prefixes
         (.padfoot-) to ensure it never conflicts with the original doc.css.
   * `zoom_theme.js`:
       * Function: Implements the "situation awareness" of accessibility. Controls the + and - zoom buttons and the
         theme swatches (Default, Dark, Blue). It uses localStorage to persist these settings across the entire site.
   * `zoom_theme.css`:
       * Function: Defines the three color themes and the layout for the floating zoom controls in the top-right corner.
   * `search.js` & `search.css`:
       * Function: These are Stubs/Placeholders. They were created to satisfy legacy 404 references found in the
         original LaTeX-to-HTML conversion, preventing console errors and improving page load performance.

  ---


  3. The Root Directory (AI Backend & Infrastructure)
  These files did not exist in the legacy system. They provide the "intelligence" layer.


   * `app.py` (FastAPI Backend):
       * Role: The core application server.
       * Modifications:
           1. Hybrid Boost Logic: Updated to prioritize the source_page passed from the widget.
           2. Deduplication: Added logic to prevent repeating the same documentation chunk in the LLM context.
           3. Safety Guards: Implemented MAX_QUERY_LENGTH, LLM_TIMEOUT, and MAX_CHUNKS.
           4. Telemetry: Added logging for token usage and error tracking.
   * `ingest.py` (Data Pipeline):
       * Role: The ETL tool.
       * Function: Iterates through the HTML/ directory, strips boilerplate (nav bars, scripts) using BeautifulSoup,
         chunks the text, and populates the ChromaDB vector store with source metadata.
   * `chroma_db/` (Vector Database):
       * Role: Persistent storage for document embeddings.
       * Status: Created during the ingestion phase. Contains the mathematical representations of the entire
         documentation corpus.
   * `requirements.txt`:
       * Role: Dependency management.
       * Content: Specifies the exact versions of FastAPI, ChromaDB, SentenceTransformers, and Google Generative AI
         required for production parity.
   * `cleanup_html.py` & `inject_widget.py`:
       * Role: Automation utilities.
       * Function: These scripts were developed to automate the transformation of the 100+ legacy HTML files, ensuring
         the injection process was consistent and error-free.



  SCALABILITY & FUTURE IMPROVEMENTS

  While the current architecture is robust and production-ready, several enhancements can be evaluated as traffic
  scales:


   * Conversation Memory: Currently, each request is stateless. Integrating a session ID and an external cache (like
     Redis) could allow the assistant to remember previous context within a single user session, enabling follow-up
     questions.
   * Response Caching: Implementing a caching layer for exact-match queries. If ten users ask "How to configure
     Thunderbird", the system should ideally bypass the LLM and serve a cached response, saving API costs and reducing
     latency.
   * Rate Limiting: Protecting the /ask endpoint against abuse by implementing IP-based rate limiting via Nginx or
     middleware within FastAPI.
   * Automated Pipeline: Creating a chron job or CI/CD hook that automatically triggers ingest.py to refresh the
     ChromaDB vectors whenever the underlying source LaTeX/HTML files are modified.
   * Cloud Vector Database: If the physical server's IOPS become a bottleneck due to extreme concurrent reads on the
     local SQLite ChromaDB file, migrating the database to a managed cloud vector store would allow the FastAPI backend
     to remain stateless and scale horizontally with ease.
