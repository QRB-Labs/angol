# Hardware Architecture & Requirements

Angol architecture is explicitly designed to operate within strict memory constraints by sequentially swapping Ingestion and Generation models.

## 1. Test & Prototype Environment (Google Cloud)
For prototyping and testing with a small to medium dataset, the architecture can run unmodified on a single modern GPU by utilizing the AWQ quantized version of the generation model (`Qwen-2.5-32B-Instruct-AWQ`).

### GCP Instance Specifications
*   **Instance Type:** `g2-standard-12` (Provision as a **Spot Instance** to minimize costs)
*   **GPU:** 1x NVIDIA L4 (24GB VRAM)
    *   *Note:* The 24GB VRAM perfectly fits the ~19GB AWQ model, leaving ~5GB for the vLLM KV cache and context window.
*   **vCPUs:** 12 vCPUs
*   **System RAM:** 48GB 
    *   *Note:* Critical for preventing Out-Of-Memory (OOM) crashes when Docker loads the model weights from disk to RAM before passing them to the GPU.
*   **Storage (Boot Disk):** 200GB Balanced Persistent Disk (`pd-balanced`)
    *   *Note:* Do not use the default 10GB HDD. 200GB is required to hold the base OS, Docker images, HuggingFace model weight caches, and local Qdrant/Postgres data.
*   **Operating System:** Ubuntu 22.04 LTS (Required for optimal NVIDIA Container Toolkit support)

## 2. Target Production Environment
The production system is targeted for enterprise-grade on-premise hardware or dedicated cloud infrastructure. The core architectural constraint driving the orchestrator pipeline is a strict **128GB System RAM limit**.

### Target Specifications
*   **Primary Platform:** NVIDIA DGX, DGX Sparx, or equivalent high-density GPU server.
*   **GPU Configuration:** 80GB+ Total VRAM required for unquantized 16-bit generation model.
    *   *Option A (Single massive GPU):* 1x NVIDIA A100 (80GB) or H100.
    *   *Option B (Multi-GPU setup):* 4x NVIDIA L4 (96GB total) or equivalent RTX/A-Series setup (requires vLLM tensor parallelism: `--tensor-parallel-size 4`).
*   **System RAM:** 128GB (Hard Architectural Limit)
*   **Storage:** 1TB+ NVMe SSD (`pd-ssd` or equivalent physical NVMe)
    *   *Note:* Extremely fast SSD storage is mandatory. The RAPTOR retrieval pipeline relies on Qdrant configured with `mmap: true`, which treats the physical disk as an extension of RAM for rapid vector similarity searches.
*   **CPU:** 32+ Cores (Required to prevent thread starvation during heavy, multi-threaded Docling PDF ingestion and OCR).

## 3. Local CPU-Only Prototyping (Fallback)
If a GPU is entirely unavailable, the pipeline logic can be built and tested locally on a standard Mini PC or laptop with the following temporary architectural swaps:
*   **Minimum Hardware:** 4-Core CPU, 16GB RAM.
*   **Code Swaps Required:**
    1.  Replace `faiss-gpu-cu12` with `faiss-cpu` in `requirements.txt`.
    2.  Replace the `vllm` Docker container in `orchestrator.sh` with `Ollama` (using `llama.cpp` under the hood for CPU-bound generation).
    3.  Limit ingestion batches to micro-corpuses (5-10 documents) to prevent CPU thermal throttling during Docling OCR passes.
