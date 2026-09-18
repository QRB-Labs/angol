import os
import json
import numpy as np
import faiss
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
llm_client = OpenAI(base_url=VLLM_API_BASE, api_key="local")
MODEL_NAME = os.getenv("INGESTION_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

MAX_TOKENS_PER_CLUSTER = 6000
CHUNK_SIZE_ESTIMATE = 500

def perform_clustering(embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d=d, k=num_clusters, niter=20, verbose=False, gpu=True)
    kmeans.train(embeddings)
    _, labels = kmeans.index.search(embeddings, 1)
    return labels.flatten()

def summarize_cluster(cluster_nodes: list, level: int) -> dict:
    combined_text = "\n\n---\n\n".join([n["text"] for n in cluster_nodes])

    citations = []
    for n in cluster_nodes:
        if level == 1:
            citations.append({
                "file_name": n["metadata"].get("file_name", "Unknown"),
                "page": n["metadata"].get("page", "Unknown")
            })
        else:
            citations.extend(n["metadata"].get("child_citations", []))

    # Deduplicate citations
    unique_citations = [dict(t) for t in {tuple(d.items()) for d in citations}]

    prompt = (
        "You are an expert enterprise analyst. Synthesize and summarize the following text chunks.\n"
        "Identify the major themes, critical data points, and overarching narratives.\n\n"
        f"TEXT TO SUMMARIZE:\n{combined_text[:20000]}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=750
        )
        summary_text = response.choices[0].message.content
    except Exception as e:
        logger.error(f"vLLM Summarization failed: {e}")
        summary_text = "Summarization failed."

    return {"text": summary_text, "citations": unique_citations}

def run_faiss_clustering_and_summarize(db_client, embed_model, unique_buckets=None, collection_name="angol"):
    if unique_buckets is None:
        unique_buckets = ["default"]

    current_level = 0
    max_levels = 3

    while current_level < max_levels:
        # Bucketize Level 0. Global pass for Level 1+
        if current_level == 0:
            active_iteration = unique_buckets
        else:
            active_iteration = ["default"]

        for bucket_value in active_iteration:
            logger.info(f"Processing Level {current_level} Bucket {bucket_value}")

            # Always filter by the current RAPTOR level
            must_conditions = [FieldCondition(key="raptor_level", match=MatchValue(value=current_level))]

            # Only apply the bucket filter if not doing a global pass
            if bucket_value != "default":
                must_conditions.append(FieldCondition(key="bucket", match=MatchValue(value=bucket_value)))

            level_filter = Filter(must=must_conditions)
            # Scroll Qdrant based on the dynamic filter
            records, next_page = db_client.scroll(
                collection_name=collection_name,
                scroll_filter=level_filter,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )

            all_records = list(records)
            while next_page:
                records, next_page = db_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=level_filter,
                    limit=10000,
                    offset=next_page,
                    with_payload=True,
                    with_vectors=True
                )
                all_records.extend(records)

            if not all_records or len(all_records) < 2:
                logger.info(f"Not enough nodes to cluster. Skipping.")
                continue

            embeddings = np.array([r.vector for r in all_records], dtype=np.float32)

            def extract_text_from_payload(payload):
                if "text" in payload:
                    return payload["text"]
                if "_node_content" in payload:
                    try:
                        return json.loads(payload["_node_content"]).get("text", "")
                    except json.JSONDecodeError:
                        pass
                return ""

            nodes_data = [{"text": extract_text_from_payload(r.payload), "metadata": r.payload} for r in all_records]

            optimal_k = max(1, len(nodes_data) // (MAX_TOKENS_PER_CLUSTER // CHUNK_SIZE_ESTIMATE))
            if optimal_k >= len(nodes_data):
                continue

            labels = perform_clustering(embeddings, num_clusters=optimal_k)

            clusters = {i: [] for i in range(optimal_k)}
            for i, label in enumerate(labels):
                if label != -1:
                    clusters[label].append(nodes_data[i])

            next_level = current_level + 1
            new_summary_nodes = []

            logger.info(f"Sending {len(clusters)} clusters to Llama-8B...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_cluster = {
                    executor.submit(summarize_cluster, nodes, next_level): nodes
                    for label, nodes in clusters.items() if len(nodes) > 0
                }

                for future in as_completed(future_to_cluster):
                    result = future.result()

                    node = TextNode(
                        text=result["text"],
                        metadata={
                            "raptor_level": next_level,
                            "child_citations": result["citations"]
                        },
                        excluded_embed_metadata_keys=["child_citations"]
                    )
                    new_summary_nodes.append(node)

            if new_summary_nodes:
                logger.info(f"Pushing {len(new_summary_nodes)} Level {next_level} summaries to Qdrant...")

                vector_store = QdrantVectorStore(client=db_client, collection_name=collection_name)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                VectorStoreIndex(
                    new_summary_nodes,
                    storage_context=storage_context,
                    embed_model=embed_model,
                    show_progress=True
                )

        current_level += 1

    logger.info("RAPTOR processing complete!")
