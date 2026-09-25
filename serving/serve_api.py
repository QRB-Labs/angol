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
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from serving.router_tools import get_vector_tool, get_sql_tool
from serving.prompt_templates import GENERATION_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT

app = FastAPI()

# 1. The Standard LLM (Used for the final conversational response to the user)
generation_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    system_prompt=GENERATION_SYSTEM_PROMPT,
    is_chat_model=True,
    # context window + max_tokens should be < --max-model-len in
    # vllm-model for generation model
    max_tokens=1024,
    context_window=10000
)

# 2. The Router LLM (Used by the agent for the ReAct reasoning loop)
router_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    system_prompt=ROUTER_SYSTEM_PROMPT,
    is_chat_model=True,
    temperature=0.0,
    max_tokens=512,
    context_window=4096
)

# Initialize Tools
vector_tool = get_vector_tool(generation_llm) # Connects to Qdrant
sql_tool = get_sql_tool(generation_llm)       # Connects to Postgres

# The Routing Agent
routing_agent = ReActAgent.from_tools(
    tools=[vector_tool, sql_tool],
    llm=router_llm,
    system_prompt=ROUTER_SYSTEM_PROMPT,
    verbose=True
)

class ChatRequest(BaseModel):
    messages: list

@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "angol-routing-agent",
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
    
    # Convert previous messages to chat history
    chat_history = []
    for msg in request.messages[:-1]:
        role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
        chat_history.append(ChatMessage(role=role, content=msg.get("content", "")))
    
    # LlamaIndex routes, searches, and generates using the router and generation LLMs
    response = routing_agent.chat(user_query, chat_history=chat_history)
    
    # Return in standard OpenAI JSON format to Open WebUI
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": str(response)
            }
        }]
    }
