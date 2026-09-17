'''
This is the **Middleware**. Its job is to:
1. Pretend to be an OpenAI API server so Open WebUI can talk to it.
2. Intercept the user's question from the frontend.
3. Use the LlamaIndex library to search your Qdrant database.
4. Package the retrieved documents and the user's question into a prompt.
5. Send that packaged prompt to vLLM to get the final answer.
6. Pass the answer back up to Open WebUI.
'''
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import RouterQueryEngine
from serving.router_tools import get_vector_tool, get_sql_tool
from serving.prompt_templates import CITATION_SYSTEM_PROMPT

load_dotenv()

app = FastAPI()

# Connect to Qwen-32B (currently running on port 8000)
local_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("LLM_API_KEY"),
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
