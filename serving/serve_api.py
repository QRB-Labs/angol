from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import RouterQueryEngine
from router_tools import get_vector_tool, get_sql_tool
from prompt_templates import CITATION_SYSTEM_PROMPT

app = FastAPI()

# Connect to Qwen-32B (currently running on port 8000)
local_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="fake-key",
    model="Qwen/Qwen2.5-32B-Instruct",
    system_prompt=CITATION_SYSTEM_PROMPT
)

# Initialize Tools
vector_tool = get_vector_tool(local_llm) # Connects to Qdrant
sql_tool = get_sql_tool(local_llm)       # Connects to Postgres

# The Orchestrator / Routing Agent
router_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[vector_tool, sql_tool],
    llm=local_llm
)

class ChatRequest(BaseModel):
    messages: list

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    # Extract user prompt
    user_query = request.messages[-1]["content"]
    
    # LlamaIndex routes, searches, and generates using Qwen
    response = router_engine.query(user_query)
    
    # Return in standard OpenAI JSON format to Open WebUI
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": str(response)
            }
        }]
    }
