import os
import qdrant_client
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "angol")

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "angol_db")

def get_vector_tool(llm):
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=10
    )
    
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="vector_search",
            description=(
                "Useful for answering qualitative questions about enterprise documents, "
                "reports, HR policies, IT architectures, and historical summaries. "
                "Uses a RAPTOR hierarchical clustering system to provide both high-level "
                "summaries and specific document details."
            ),
        )
    )

def get_sql_tool(llm):
    db_uri = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(db_uri)
    sql_database = SQLDatabase(engine)
    
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        llm=llm,
        context_query_kwargs={
            "context_str": "This database contains highly structured enterprise data, financial records, and metrics."
        }
    )
    
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="sql_database",
            description=(
                "Useful for translating natural language into SQL queries. "
                "Use this tool when the user asks for exact metrics, calculations, "
                "financial numbers, or structured tabular data that resides in the PostgreSQL database."
            ),
        )
    )
