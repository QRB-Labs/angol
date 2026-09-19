import os
import argparse
import hashlib
from parser_docling import extract_markdown
from raptor_clustering import run_faiss_clustering_and_summarize
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

def main(raw_dir):
    # 1. Initialize BGE-M3 Locally
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    db_client = qdrant_client.QdrantClient(host="localhost", port=6333)

    # 2. Configure LlamaIndex to talk to Qdrant
    vector_store = QdrantVectorStore(
        client=db_client,
        collection_name="angol"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # We need an empty index to insert into iteratively
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # 3. Setup Chunking
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )

    # Clustering is done in "buckets" for memory management and semantic groups
    NODES_PER_BUCKET = 500000
    unique_buckets = set()
    total_nodes_processed = 0

    # 4. Parse and Process Documents Lazily
    # Assuming extract_markdown(raw_dir) yields one Document at a time
    for doc in extract_markdown(raw_dir):
        nodes = splitter.get_nodes_from_documents([doc])

        # Inject RAPTOR Metadata and Deterministic IDs
        for node in nodes:
            node.id_ = hashlib.md5(node.get_content().encode("utf-8")).hexdigest()
            node.metadata["raptor_level"] = 0
            bucket_name = f"bucket_{total_nodes_processed // NODES_PER_BUCKET}"
            node.metadata["bucket"] = bucket_name
            unique_buckets.add(bucket_name)
            total_nodes_processed += 1

        # Upsert this document's nodes immediately, clearing them from memory
        if nodes:
            index.insert_nodes(nodes)
            print(f"Upserted {len(nodes)} nodes from {doc.metadata.get('file_name', 'unknown')}.")

    print(f"Successfully pushed {total_nodes_processed} Level 0 nodes to Qdrant.")

    # 5. Run RAPTOR Pipeline (100:1 Compression)
    # This function uses FAISS to cluster, then calls the local Llama-8B (Port 8000)
    # to summarize, then embeds the summaries and pushes to Qdrant.
    run_faiss_clustering_and_summarize(
        db_client=db_client,
        embed_model=embed_model,
        unique_buckets=list(unique_buckets),
        collection_name="angol"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Angol RAPTOR Ingestion Pipeline.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="../data/raw_documents/",
        help="Path to the directory containing raw documents"
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir)
