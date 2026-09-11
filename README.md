<!-- 
An enterprise "brain" using AI that knows the company's documents
(pdfs, power point slide decks, spreadsheets).

Able to summarize, synthesize, and answer questions about the
knowledge in the corpus. Answers have citations of supporting docs.

Corpus: Up to 1TB of documents

Runs locally on a NDVIDIA GDX Spark or less.
-->

# High-Level Design: Enterprise AI Knowledge Brain

## Objective
To build a highly secure, locally deployed (air-gapped) Enterprise AI system capable of ingesting large volumes of heterogeneous corporate documents (PDFs, PPTs, Spreadsheets). The system will provide accurate, reasoned answers to both granular and global queries, synthesize knowledge across multiple documents, and explicitly cite supporting sources. 

## Overview
The system utilizes an advanced Retrieval-Augmented Generation (RAG) architecture enhanced by the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) methodology. Rather than relying on simple semantic similarity, the system builds a hierarchical Knowledge Graph. It clusters related document chunks, summarizes them using a local Large Language Model (LLM), and recursively embeds the summaries. This dual-pipeline architecture (Ingestion and Retrieval) runs entirely on open-source and open-weight software optimized for an NVIDIA DGX Spark (128GB RAM limit).

## Detailed Design

### 1. Ingestion & Parsing Layer
*   **PDF/PPT Extraction:** Uses IBM's [Docling](https://github.com/DS4SD/docling) or [Unstructured](https://unstructured.io/) to parse complex layouts and slide decks via specialized Vision/OCR models.
*   **Spreadsheet Parsing:** Small sheets are converted to Markdown tables. Massive datasets bypass the vector space and are ingested into a local PostgreSQL database for Text-to-SQL agentic querying.
*   **Metadata Tagging:** Every chunk is aggressively tagged with metadata (Author, Date, Department, Page Number, Document ID) to enable accurate citation generation down the pipeline.

### 2. Chunking & Embedding Layer
*   **Orchestration:** [LlamaIndex](https://www.llamaindex.ai/) manages chunking and RAG pipelines.
*   **Chunking Strategy:** Semantic chunking targeting ~1KB of text (roughly 200 tokens) per chunk.
*   **Embedding Model:** [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3). Handles massive context windows, supports multi-linguality, and generates dense vectors at 1024 dimensions.

### 3. Clustering & Summarization (RAPTOR Pipeline)
To enable holistic reasoning across the corpus, data is grouped and summarized hierarchically:
*   **Metadata Partitioning:** Vectors are first bucketed by metadata (e.g., Department, Year) into batches of 20,000 to 40,000 chunks to prevent memory overflow.
*   **GPU Clustering:** [NVIDIA FAISS](https://github.com/facebookresearch/faiss) runs GPU-accelerated K-Means clustering on the buckets to group related chunks across different documents.
*   **Summarization:** An LLM reads the text of each cluster and generates a summary. The summary is embedded, storing the source document citations as metadata.
*   **Recursion:** Summaries are clustered and summarized iteratively until a "Root Node" executive summary is reached.

### 4. Storage Layer
*   **Database:** [Qdrant](https://qdrant.tech/) or [Milvus](https://milvus.io/), deployed locally via Docker.
*   **Memory Optimization:** Uses Scalar Quantization (Int8) to compress 32-bit floating-point vectors, coupled with memory-mapped (`mmap`) payload storage to keep raw text on the NVMe SSD and only the HNSW search index in System RAM.

### 5. Retrieval & Generation Layer
*   **Model Serving:** [vLLM](https://github.com/vllm-project/vllm) for high-throughput, memory-efficient LLM serving.
*   **Reasoning Engine:** [Qwen-2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) or [Meta Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). Selected for high reasoning capabilities within constrained VRAM.
*   **Routing Agent:** User queries are routed either to the Vector Database (for text questions) or to a Text-to-SQL Agent (for massive spreadsheet computation).

### 6. User Interface
*   **Frontend:** [Open WebUI](https://github.com/open-webui/open-webui). Provides a ChatGPT-like interface with built-in citation rendering and document snippet viewing.

## Quantitative Estimates
Tailored for the hardware constraints of an NVIDIA DGX Spark with 128GB System RAM:

*   **Max Safe RAM for Vector DB:** ~80GB (reserving 48GB for OS and FAISS clustering operations).
*   **Vector Compression:** Int8 Quantization reduces vector size from 4KB to 1KB. HNSW Index adds ~1KB overhead. Total RAM per vector: ~2KB.
*   **Maximum Vector Capacity:** ~40 Million vectors.
*   **Supported Corpus Size:** ~39.6 Million raw chunks (~40GB of raw text). 40GB equates to approximately **20 million pages** of enterprise documents.
*   **RAPTOR Scaling (100:1 ratio):**
    *   Level 0 (Raw Text): ~40,000,000 vectors
    *   Level 1 (Summaries): ~400,000 vectors
    *   Level 2 (Summaries): ~4,000 vectors
    *   Level 3 (Summaries): ~40 vectors
    *   Level 4 (Root): 1 vector
*   **Total Disk Storage Required:** ~100GB of fast NVMe SSD space for the `mmap` payload and database overhead.

## Hardware
*   **System:** NVIDIA DGX Spark or similar workstation.
*   **Memory:** 128GB CPU System RAM (Strict bottleneck; requires aggressive memory management).
*   **Storage:** Fast NVMe Gen4 SSDs for payload memory mapping.
*   **GPU:** Multi-GPU configuration capable of hosting vLLM (e.g., 2x to 4x RTX class or small A-series GPUs) serving 8B to 32B parameter models.

## Design Alternatives

*   **RAPTOR vs. Standard RAG:** Standard RAG is computationally cheaper but fails at global questions (e.g., "Summarize risks across all documents"). RAPTOR incurs high upfront compute costs during ingestion but enables complex cross-document reasoning.
*   **Text Summarization vs. Vector Averaging:** Taking the mathematical centroid of a vector cluster dilutes facts and prevents the database from returning readable text to the LLM. Using an LLM to summarize the cluster's text *before* re-embedding preserves specific facts and context.
*   **FAISS vs. UMAP/GMM:** Standard RAPTOR literature uses UMAP for dimensionality reduction followed by Gaussian Mixture Models. On a 128GB system, UMAP will trigger Out-of-Memory (OOM) errors at scale. FAISS K-Means on GPU is heavily optimized for massive batch clustering.

## Notes
*   **Citations:** To guarantee accurate citations, the chunking mechanism *must* append metadata arrays to every RAPTOR summary node. When the LLM references a Level 2 summary, the UI will parse the metadata array to show the user the 10,000 original documents that informed that node.
*   **Security:** All weights (BGE-M3, Llama/Qwen, OCR models) and software (vLLM, Qdrant) are open-source/open-weight and run entirely locally. The system requires zero external API calls, satisfying strict air-gapped compliance requirements.
