### Phase 1: Environment Setup & Core Infrastructure
Before writing any Python code, you must initialize the local vector database, SQL database, and AI inference servers using Docker.

**1. Infrastructure (Docker Compose)**
Create a `docker-compose.yml` to host Qdrant (Vector DB) and PostgreSQL (Spreadsheet DB).
*   **Command:** `docker compose up -d`
*   *Note on Qdrant Config:* Ensure the Qdrant volume is mapped to a fast NVMe SSD path, and configure the payload storage to `mmap: true` and vectors to `quantization: int8`.

**2. Environment**

Set up a clean standard Python virtual environment. Install system dependencies required for document parsing (Docling), and use standard pip for Python packages. vLLM will be run via Docker to prevent PyTorch dependency conflicts.

*   **System Dependencies (Required for Docling):**

```bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr poppler-utils libgl1 libglib2.0-0
```

*   **Python Virtual Environment:**
   
```bash
    python3.10 -m venv angol_env
    source angol_env/bin/activate
    pip install -r requirements.txt
```

*   **Start vLLM (Run in a separate terminal via Docker):**

```bash
    docker run --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --max-model-len 8192
```		
		
**3. Model Serving Engine (vLLM)**
We use vLLM to serve open-weight models as local, OpenAI-compatible APIs. Because of the 128GB RAM limit, a shell script [orchestrator.sh](orchestrator.sh) toggles between the Ingestion LLM and the Serving LLM.

### Phase 2: Ingestion & RAPTOR Pipeline (Background Process)
This phase is executed by a heavy Python script (`ingestion_pipeline.py`) run during off-hours (e.g., via a Cron job) while the Llama-3.1-8B model is loaded in vLLM.

**High-Level Structure of `ingestion_pipeline.py`:**
1.  **Parse Documents:** Use `docling` to iterate through the target directory, converting PDFs/PPTs to Markdown and mapping metadata.
2.  **SQL Routing (Spreadsheets):** If a file is a massive CSV/XLSX, use `pandas` to write it directly to the local PostgreSQL database using `psycopg2`.
3.  **Chunk & Embed:**
    *   Pass the parsed Markdown to LlamaIndex's `SemanticSplitterNodeParser`.
    *   Load the embedding model locally: `embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")`.
    *   Embed all chunks (Level 0 Leaf Nodes).
4. **Recursive RAPTOR Clustering ([`ingestion/raptor_clustering.py`](ingestion/raptor_clustering.py)):**
    Instead of a flat pass, the pipeline recursively builds a hierarchical semantic tree (up to `max_levels`).
    * **Loop Levels:** Starting at `raptor_level=0` (leaf nodes), the system fetches vectors bucket-by-bucket to respect the 128GB RAM limit.
    * **Cluster:** FAISS K-Means groups semantically similar chunks within the current bucket and level.
    * **Summarize:** Local Llama-3.1-8B (via vLLM) synthesizes each cluster into a single overarching summary node, inheriting deduplicated `child_citations`.
    * **Re-Embed:** Summaries are embedded (BGE-M3) and upserted to Qdrant tagged as `raptor_level = current_level + 1`.
    * **Recurse:** The process repeats on the newly generated summary nodes until the tree root is reached or node counts fall below the clustering threshold.    ```

---

### Phase 3: Retrieval, Reasoning & Serving
Once ingestion is complete, shut down the Llama-8B vLLM instance and start the Qwen-32B vLLM instance.

To connect Qwen-32B to Open WebUI seamlessly, we need a "Middleware API" (`serve_api.py`). This script uses FastAPI to expose LlamaIndex's Router and Qdrant integration as a standard chat endpoint.

**High-Level Structure of `serve_api.py`:**
1.  **Initialize DB Connections:** Connect LlamaIndex to Qdrant (Vector DB) and PostgreSQL (SQL DB).
2.  **Define Engines:**
    *   `vector_tool = QueryEngineTool(engine=qdrant_engine, description="Use for PDFs, text, and concepts.")`
    *   `sql_tool = QueryEngineTool(engine=nl_sql_engine, description="Use for spreadsheet math and data.")`
3.  **The Routing Agent:**
    *   Initialize the `RouterQueryEngine` pointing to the local Qwen-32B vLLM server.
4.  **FastAPI Endpoint (`/v1/chat/completions`):**
    *   Accept incoming JSON from Open WebUI.
    *   Pass the user's prompt to the `RouterQueryEngine`.
    *   LlamaIndex automatically:
        *   Embeds the query (BGE-M3).
        *   Searches Qdrant (pulling Level 0 through Level 3 nodes).
        *   Assembles the citation prompt.
        *   Streams the reasoning result from Qwen-32B.
    *   Return the synthesized, cited string back to the UI.
    *   *Command:* `uvicorn serve_api:app --host 0.0.0.0 --port 8080`

---

### Phase 4: User Interface (Frontend)
Deploy Open WebUI to provide a ChatGPT-like experience for enterprise users.

*   **Command:**
    ```bash
    docker run -d -p 3000:8080 \
      -e OPENAI_API_BASE_URL=http://localhost:8080/v1 \
      -e OPENAI_API_KEY=local-key \
      -v open-webui:/app/backend/data \
      --name open-webui ghcr.io/open-webui/open-webui:main
    ```
*   *Note:* The `OPENAI_API_BASE_URL` points directly to your custom `serve_api.py` (FastAPI) middleware, NOT directly to vLLM. This ensures Open WebUI triggers the LlamaIndex RAG/Routing pipeline rather than just talking to a "blank" LLM.

---

### Operationalizing: The Daily Bash Script
To manage the "Sequential Dual-LLM Deployment" safely within the 128GB RAM limit, create a `orchestator.sh` bash script managed by a Cron job:

```

### Directory Structure

```text
angol/
├── docker-compose.yml           # Infrastructure (Qdrant, Postgres, Open WebUI)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (ports, DB credentials)
├── orchestrator.sh              # Bash script managing the Dual-LLM daily cycle
│
├── data/                        # Local data and mapped Docker volumes
│   ├── raw_documents/           # Drop folder for PDFs, PPTs, Spreadsheets
│   ├── qdrant_storage/          # NVMe mmap storage for Vector DB
│   ├── pg_storage/              # Storage for PostgreSQL (Spreadsheets)
│   └── webui_storage/           # Open WebUI user data/history
│
├── ingestion/                   # Phase 1 & 2: Parsing and RAPTOR Pipeline
│   ├── __init__.py
│   ├── ingestion_pipeline.py    # Main entry point for nightly ingestion
│   ├── parser_docling.py        # PDF/PPT extraction to Markdown
│   ├── parser_sql.py            # Pandas script for massive spreadsheets -> Postgres
│   └── raptor_clustering.py     # FAISS K-Means and Llama-8B summarization loop
│
└── serving/                     # Phase 3 & 4: Retrieval and Generation
    ├── __init__.py
    ├── serve_api.py             # FastAPI Middleware (LlamaIndex Orchestrator)
    ├── router_tools.py          # LlamaIndex tool definitions (Vector Search vs SQL)
    └── prompt_templates.py      # The strict citation system prompt for Qwen-32B
```

---

### Core Files & Scripts Definition

#### 1. [`docker-compose.yml`](docker-compose.yml) (The Infrastructure)
This spins up the three core databases/UIs without overlapping ports.

#### 2. [`orchestrator.sh`](orchestrator.sh) (The Dual-LLM Manager)
This script executes **Option B** (Sequential Deployment) to strictly respect the 128GB RAM limit. It is designed to be run nightly via a Cron job.

#### 3. [`ingestion/ingestion_pipeline.py`](ingestion/ingestion_pipeline.py) (The RAPTOR Engine)
This script reads the raw documents, chunks them, and builds the hierarchical tree.

#### 4. [`serving/serve_api.py`](serving/serve_api.py) (The Middleware)
This translates Open WebUI's OpenAI-style requests into LlamaIndex orchestrations, routing between SQL and the RAPTOR Vector DB.

### Steps for Execution:
1.  Run `pip install -r requirements.txt`.
2.  Run `docker compose up -d` to spin up Qdrant, Postgres, and Open WebUI.
3.  Place a few test PDFs into `data/raw_documents/`.
4.  Run `chmod +x orchestrator.sh` and execute `./orchestrator.sh`.
5.  Once the script finishes, open your browser to `http://localhost:3000` to query Angol.

