import os
from parser_docling import extract_markdown
from raptor_clustering import run_faiss_clustering_and_summarize
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

def main():
    # 1. Initialize BGE-M3 Locally
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    db_client = qdrant_client.QdrantClient(host="localhost", port=6333)

    # 2. Parse New Documents
    raw_dir = "../data/raw_documents/"
    docs = extract_markdown(raw_dir)

    # 3. Chunk into Level 0 Leaf Nodes
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"Processing {len(nodes)} Level 0 Leaf Nodes...")

    # 3a. Inject RAPTOR Metadata
    # Tag all these chunks as Level 0 so the clustering algorithm knows they are raw text
    # "category" corresponds to buckets for clustering, one bucket for roughly every 1 GB of raw text (~500,000 nodes)
    NODES_PER_BUCKET = 500000
    unique_categories = set()

    for i, node in enumerate(nodes):
        node.metadata["raptor_level"] = 0
        bucket_name = f"bucket_{i // NODES_PER_BUCKET}"
        node.metadata["category"] = bucket_name
        unique_categories.add(bucket_name)

    # 3b. Configure LlamaIndex to talk to Qdrant
    vector_store = QdrantVectorStore(
        client=db_client,
        collection_name="angol"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 3c. Embed and Push
    # This single command automatically takes the text chunks, passes them to BGE-M3
    # to generate 1024-d vectors, and upserts them into Qdrant along with the metadata.
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print("Successfully pushed Level 0 nodes to Qdrant.")

    # 4. Run RAPTOR Pipeline (100:1 Compression)
    # This function uses FAISS to cluster, then calls the local Llama-8B (Port 8000)
    # to summarize, then embeds the summaries and pushes to Qdrant.
    run_faiss_clustering_and_summarize(
        db_client=db_client,
        embed_model=embed_model,
        bucket_key="category",
        unique_buckets=list(unique_categories)
    )


if __name__ == "__main__":
    main()
