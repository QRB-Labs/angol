import os
from parser_docling import extract_markdown
from raptor_clustering import run_faiss_clustering_and_summarize
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
    
    # Embed and push Level 0 nodes to Qdrant...
    # (Code omitted for brevity)
    
    # 4. Run RAPTOR Pipeline (100:1 Compression)
    # This function uses FAISS to cluster, then calls the local Llama-8B (Port 8000) 
    # to summarize, then embeds the summaries and pushes to Qdrant.
    run_faiss_clustering_and_summarize(db_client, embed_model)

if __name__ == "__main__":
    main()
