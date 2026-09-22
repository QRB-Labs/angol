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
from llama_index.core.selectors import LLMSingleSelector
from serving.router_tools import get_vector_tool, get_sql_tool
from serving.prompt_templates import CITATION_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT

load_dotenv()

app = FastAPI()

# 1. The Standard LLM (Used for the final conversational response to the user)
local_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    system_prompt=CITATION_SYSTEM_PROMPT,
    is_chat_model=True,
    # context window + max_tokens should be < --max-model-len in
    # vllm-model for generation model
    max_tokens=1024,
    context_window=15000
)

# 2. The Router LLM (Used ONLY internally to pick the tool, strictly locked to JSON)
router_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    is_chat_model=True,
    temperature=0.0,
    max_tokens=512,
    additional_kwargs={"response_format": {"type": "json_object"}}
)

# Initialize Tools
vector_tool = get_vector_tool(local_llm) # Connects to Qdrant
sql_tool = get_sql_tool(local_llm)       # Connects to Postgres

# The Orchestrator / Routing Agent
router_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[vector_tool, sql_tool],
    llm=local_llm,
    selector=LLMSingleSelector.from_defaults(
        llm=router_llm,
        prompt_template_str=ROUTER_SYSTEM_PROMPT
    )
)

class ChatRequest(BaseModel):
    messages: list

@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "angol-orchestrator", 
                "object": "model",
                "created": 1700000000,
                "owned_by": "angol"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatRequest):
    # Extract user prompt
    user_query = request.messages[-1]["content"]
    
    # LlamaIndex routes, searches, and generates using the local model
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
