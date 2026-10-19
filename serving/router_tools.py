import asyncio
import logging
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
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "angol")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", 12))

PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "angol_db")

GENERATION_MODEL_MAX_LEN = int(os.getenv("GENERATION_MODEL_MAX_LEN", "32768"))
MAX_RESULT_CHARS = GENERATION_MODEL_MAX_LEN // 4  # ~8,192 chars (approx 2,048 tokens)

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([llama_debug])
# logging.getLogger("llama_index.core").setLevel(logging.DEBUG)
# logging.getLogger("llama_index.llms").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

class SafeQueryEngineWrapper(CustomQueryEngine):
    query_engine: BaseQueryEngine
    fallback_message: str

    def custom_query(self, query_str: str):
        try:
            response = self.query_engine.query(query_str)
            if hasattr(response, "metadata") and response.metadata and "result" in response.metadata:
                if len(str(response.metadata["result"])) > MAX_RESULT_CHARS:
                    response.metadata["result"] = f"{str(response.metadata['result'])[:MAX_RESULT_CHARS]}... [TRUNCATED]"
            if not response.response or not str(response.response).strip():
                logger.error("LLM returned an empty response. Check vLLM for max-token-len limits.")
                return Response(response="[System: The model failed to generate a response, likely due to context size limits.]")
            return response
        except Exception as e:
            return Response(response=f"Tool error: {str(e)}. {self.fallback_message}")

    async def acustom_query(self, query_str: str):
        return await asyncio.to_thread(self.custom_query, query_str)


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
        similarity_top_k=VECTOR_TOP_K,
        response_mode="compact"
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


class SafeSQLDatabase(SQLDatabase):
    def run_sql(self, command: str):
        result_str, metadata = super().run_sql(command)
        if len(result_str) > MAX_RESULT_CHARS:
            result_str = f"{result_str[:MAX_RESULT_CHARS]}... [RESULT TRUNCATED]"
        return result_str, metadata


def get_sql_tool(llm):
    db_uri = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    engine = create_engine(db_uri)
    sql_database = SafeSQLDatabase(engine)
    
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
