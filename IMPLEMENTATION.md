# Angol: Implementation notes

## Table of Contents
- [Phase 1: Environment Setup & Core Infrastructure](#phase-1-environment-setup--core-infrastructure)
- [Phase 2: Ingestion & RAPTOR Pipeline (Background Process)](#phase-2-ingestion--raptor-pipeline-background-process)
- [Phase 3: Retrieval, Reasoning & Serving](#phase-3-retrieval-reasoning--serving)
- [Phase 4: User Interface (Frontend)](#phase-4-user-interface-frontend)
- [Directory Structure](#directory-structure)
- [Debug tools](#debug-tools)
- [TODO](#todo)
---
### Phase 1: Environment Setup & Core Infrastructure

**1. Environment**

*  **Ubuntu 22.04 LTS:**
Operating system which natively provides Python 3.10. More recent versions (e.g. Ubuntu 26.04 with Python 3.14) have compatibility and driver compilation issues with GPU-accelerated libraries (like `faiss-gpu`).

* **Environment variables:**
See the file [.env](.env).
	*  `INGESTION_MODEL` and `GENERATION_MODEL` define the Hugging Face repo IDs for the models used during RAPTOR summarization and chat serving, respectively.
	*  `HF_TOKEN` is an access token from https://huggingface.co/settings/tokens. Even though we use open-soure, open-weight models, in some cases they offer higher speed download (BGE-M3) or impose access terms (Llama-3.1-8B) via the Hugging Face token.
    *  `OPENAI_API_KEY` is purely a local dummy key required by the client library to talk to the local vLLM server. It is not used for external API calls and can be set to any arbitrary value.

*   **NVIDIA Host Drivers & Docker Toolkit (Required for GPU Access):**
Ensure the host OS can see the GPU and install the toolkit required to pass it into Docker containers.

```bash
	sudo apt-get update
	sudo apt-get install -y ubuntu-drivers-common
	sudo ubuntu-drivers autoinstall
	sudo reboot
	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
	curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	sudo apt-get update
	sudo apt-get install -y nvidia-container-toolkit
	sudo nvidia-ctk runtime configure --runtime=docker
	sudo systemctl restart docker
```

*   **System Dependencies (Required for Docling):**

```bash
	sudo apt-get update
	sudo apt-get install tesseract-ocr poppler-utils libgl1 libglib2.0-0
```

*   **Python Virtual Environment:** Load python [requirements.txt](requirements.txt) in a separate python environment to avoid conflict with existing packages.

```bash
	sudo apt install python3.10-venv
	python3.10 -m venv angol_env
	source angol_env/bin/activate
	pip install -r requirements.txt
```

**2. Infrastructure (Docker Compose)**

[`docker-compose.yml`](docker-compose.yml) spins up the three services without overlapping ports:  Qdrant (Vector DB), PostgreSQL (Spreadsheet DB) and Open WebUI.
*   **Command:** `docker compose up -d`
*   *Note on Qdrant Config:* Ensure the Qdrant volume is mapped to a fast NVMe SSD path, and configure the payload storage to `mmap: true` and vectors to `quantization: int8`.


**3. Model Serving Engine (vLLM & Orchestrator)**

We use vLLM in a separate docker container to serve open-weight models as local, OpenAI-compatible APIs (preventing PyTorch dependency conflicts in our Python environment).
Because of the 128GB RAM limit, a shell script [`orchestrator.sh`](orchestrator.sh) automatically toggles between the Ingestion LLM and the Generation LLM (defined in `.env`) in their own containers.
*   **Command:** `./orchestrator.sh`
*   *Note:* In production, this is designed to be run e.g. nightly or whenever there's new data to ingest.

---
### Phase 2: Ingestion & RAPTOR Pipeline (Background Process)

This phase is executed by a heavy Python script run by the orchestrator while the ingestion model is loaded in vLLM.
[`ingestion/ingestion_pipeline.py`](ingestion/ingestion_pipeline.py) (the RAPTOR Engine) reads the raw documents, chunks them, and builds the hierarchical tree.

**High-Level Structure:**
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
	* **Summarize:** The ingestion LLM (via vLLM) synthesizes each cluster into a summary node, inheriting deduplicated `child_citations`.
	* **Re-Embed:** Summaries are embedded (BGE-M3) and upserted to Qdrant tagged as `raptor_level = current_level + 1`.
	* **Recurse:** The process repeats on the newly generated summary nodes until the tree root is reached or node counts fall below the clustering threshold.

---

### Phase 3: Retrieval, Reasoning & Serving

Once ingestion is complete, orchestrator shuts down the ingestion vLLM instance and starts the generation vLLM instance.

To connect the generation LLM to Open WebUI seamlessly, we need  "middleware", implemented in [`serving/serve_api.py`](serving/serve_api.py). Uses FastAPI to expose LlamaIndex's Router and Qdrant integration as a standard chat endpoint.
Translates Open WebUI's OpenAI-style requests into LlamaIndex orchestrations, routing between SQL and the RAPTOR Vector DB.

**High-Level Structure:**
1.  **Initialize DB Connections:** Connect LlamaIndex to Qdrant (Vector DB) and PostgreSQL (SQL DB).
2.  **Define Engines:**
	*   `vector_tool = QueryEngineTool(engine=qdrant_engine, description="Use for PDFs, text, and concepts.")`
	*   `sql_tool = QueryEngineTool(engine=nl_sql_engine, description="Use for spreadsheet math and data.")`
3.  **The Routing Agent:**
	*   Initialize the `RouterQueryEngine` pointing to the local generation vLLM server.
4.  **FastAPI Endpoint (`/v1/chat/completions`):**
	*   Accept incoming JSON from Open WebUI.
	*   Pass the user's prompt to the `RouterQueryEngine`.
	*   LlamaIndex automatically:
		*   Embeds the query (BGE-M3).
		*   Searches Qdrant (pulling Level 0 through Level 3 nodes).
		*   Assembles the citation prompt.
		*   Streams the reasoning result from the generation LLM.
	*   Return the synthesized, cited string back to the UI.
	*   *Command:* `uvicorn serve_api:app --host 0.0.0.0 --port 8081`

---

### Phase 4: User Interface (Frontend)

Open WebUI provide a ChatGPT-like experience for enterprise users. It connects to via `OPENAI_API_BASE_URL` to our custom `serve_api.py` (FastAPI) middleware, NOT directly to vLLM. This ensures Open WebUI triggers the LlamaIndex RAG/Routing pipeline rather than just talking to a "blank" LLM. Deployment of Open WebUI is handled by docker-compose.yml.  Open your browser to `http://localhost:80` to query Angol.

---

### Directory Structure

```text
angol/
├── docker-compose.yml           # Infrastructure (Qdrant, Postgres, Open WebUI)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (ports, DB credentials, model names)
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
│   └── raptor_clustering.py     # FAISS K-Means and ingestion LLM summarization loop
│
└── serving/                     # Phase 3 & 4: Retrieval and Generation
	├── __init__.py
	├── serve_api.py             # FastAPI Middleware (LlamaIndex Orchestrator)
	├── router_tools.py          # LlamaIndex tool definitions (Vector Search vs SQL)
	└── prompt_templates.py      # The strict citation system prompt for the generation LLM
```

### Debug tools


**1. Get bucket node count**

Qdrant vector db server listens on port 6333. The **Facets API** in Qdrant acts like a `GROUP BY` statement with a `COUNT()` in SQL. Here is an example `curl` command to get the number of nodes at level 0 in each bucket.

```bash
curl -X POST 'http://localhost:6333/collections/angol/facet' \
-H 'Content-Type: application/json' \
-d '{
  "key": "bucket",
  "limit": 100,
  "filter": {
    "must": [
      {
        "key": "raptor_level",
        "match": {
          "value": 0
        }
      }
    ]
  }
}'
```

**2. Delete all Level 1 summaries**

This `curl` command uses the **Delete by Filter** endpoint.

```bash
curl -X POST 'http://localhost:6333/collections/angol/points/delete' \
-H 'Content-Type: application/json' \
-d '{
  "filter": {
    "must": [
      {
        "key": "raptor_level",
        "match": {
          "value": 1
        }
      }
    ]
  }
}'
```

### TODO

1. Instead of using a "one-shot" RouterQueryEngine, upgrade LlamaIndex orchestrator in serve_api.py to a ReAct (Reasoning and Acting) Agent. A ReAct agent works in a loop. If it tries the sql_tool and the database returns an error (e.g., "table not found"), the agent reads that error in its scratchpad, realizes it made a mistake, and autonomously decides to try the vector_tool to get whatever unstructured context it can.
1. Download script like `scp` supporting authenticated Microsoft 365 and Google Drive downloads.
