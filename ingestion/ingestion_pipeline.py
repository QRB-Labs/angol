import os
import shutil
import argparse
import hashlib
import logging
from dotenv import load_dotenv
load_dotenv()  # before other imports in case they depend on env e.g. HF_TOKEN
from sqlalchemy import create_engine
from parser_docling import extract_markdown
from parser_sql import process_tabular_files
from raptor_clustering import run_faiss_clustering_and_summarize
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Initialize BGE-M3 Locally
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", token=os.getenv("HF_TOKEN"))
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

    # 3. Setup Chunking, chunk size ~500 tokens
    splitter = SentenceSplitter(
        chunk_size=500,  
        chunk_overlap = 50
    )

    # 4. Process Tabular Data into PostgreSQL
    logger.info("Initializing PostgreSQL connection...")
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db = os.getenv("POSTGRES_DB", "angol_db")

    pg_uri = f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
    pg_engine = create_engine(pg_uri)

    logger.info("Processing tabular files into PostgreSQL...")
    last_file_path = None

    for table_name, df_chunk, is_first_chunk, file_path in process_tabular_files(raw_dir):
        # If we have moved to a new file, it's safe to move the previous successfully processed file
        if last_file_path and last_file_path != file_path:
            if os.path.exists(last_file_path):
                try:
                    shutil.move(last_file_path, os.path.join(processed_dir, os.path.basename(last_file_path)))
                    logger.info(f"Moved processed file {last_file_path}")
                except Exception as e:
                    logger.error(f"Failed to move {last_file_path}: {e}")

        if_exists_action = 'replace' if is_first_chunk else 'append'
        try:
            df_chunk.to_sql(
                table_name,
                pg_engine,
                if_exists=if_exists_action,
                index=False,
                method='multi',
                chunksize=10000
            )
            logger.info(f"Upserted chunk ({len(df_chunk)} rows) to Postgres table '{table_name}'.")
            last_file_path = file_path
        except Exception as e:
            logger.error(f"Failed to insert into Postgres table '{table_name}': {e}")
            last_file_path = None  # Clear tracking on failure so we don't move a failed file

    # Move the very last tabular file
    if last_file_path and os.path.exists(last_file_path):
        try:
            shutil.move(last_file_path, os.path.join(processed_dir, os.path.basename(last_file_path)))
            logger.info(f"Moved processed file {last_file_path}")
        except Exception as e:
            logger.error(f"Failed to move {last_file_path}: {e}")

    # 5. Parse and Process Documents into Qdrant
    # Clustering is done in "buckets" for memory management and semantic groups
    # Assuming an average chunk (node) size of ~2KB (~500 tokens),
    # 500,000 nodes equates to roughly 1GB of raw text per bucket.
    NODES_PER_BUCKET = 500000
    unique_buckets = set()
    total_nodes_processed = 0

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
            logger.info(f"Upserted {len(nodes)} nodes from {doc.metadata.get('file_name', 'unknown')}.")

            # Move the processed document immediately
            file_name = doc.metadata.get('file_name')
            file_path = doc.metadata.get('file_path')
            target_path = file_path if file_path else (os.path.join(raw_dir, file_name) if file_name else None)
            
            if target_path and os.path.exists(target_path):
                try:
                    shutil.move(target_path, os.path.join(processed_dir, os.path.basename(target_path)))
                    logger.info(f"Moved processed file {target_path}")
                except Exception as e:
                    logger.error(f"Failed to move {target_path}: {e}")

    logger.info(f"Pushed {total_nodes_processed} Level 0 nodes to Qdrant.")

    # 6. Run RAPTOR Pipeline (100:1 Compression)
    # This function uses FAISS to cluster, then calls the ingestion LLM
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
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="../data/processed_documents/",
        help="Path to move processed documents"
    )
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, processed_dir=args.processed_dir)
