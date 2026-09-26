import os
from dotenv import load_dotenv
load_dotenv()
import qdrant_client
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.base_query_engine import BaseQueryEngine
from serving.prompt_templates import VECTOR_TOOL_DESCRIPTION, SQL_TOOL_DESCRIPTION


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "angol")

PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "angol_db")


class SafeQueryEngineWrapper(CustomQueryEngine):
    query_engine: BaseQueryEngine
    fallback_message: str

    def custom_query(self, query_str: str):
        try:
            return self.query_engine.query(query_str)
        except Exception as e:
            return Response(response=f"Tool error: {str(e)}. {self.fallback_message}")

    async def acustom_query(self, query_str: str):
        try:
            return await self.query_engine.aquery(query_str)
        except Exception as e:
            return Response(response=f"Tool error: {str(e)}. {self.fallback_message}")


def get_vector_tool(llm):
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    aclient = qdrant_client.AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(client=client, aclient=aclient, collection_name=QDRANT_COLLECTION)
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", token=os.getenv("HF_TOKEN"))
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=20  # Note: slow on 24GB GPUs (VRAM limits), but fine on 128GB RAM systems.
    )
    
    safe_query_engine = SafeQueryEngineWrapper(
        query_engine=query_engine, 
        fallback_message="Vector search failed. Try rewording your query or using a different tool."
    )
    
    return QueryEngineTool(
        query_engine=safe_query_engine,
        metadata=ToolMetadata(
            name="vector_search",
            description=VECTOR_TOOL_DESCRIPTION,
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
            "context_str": "This database contains highly structured tabular data."
        }
    )
    
    safe_query_engine = SafeQueryEngineWrapper(
        query_engine=query_engine, 
        fallback_message="SQL query failed. Try a different query, table, or query tool."
    )
    
    return QueryEngineTool(
        query_engine=safe_query_engine,
        metadata=ToolMetadata(
            name="sql_database",
            description=SQL_TOOL_DESCRIPTION,
        )
    )
