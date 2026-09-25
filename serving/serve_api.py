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
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent.workflow import ReActAgent

# FIX FOR LLAMAINDEX WORKFLOW BUG
# Forces the Pydantic-based agent to be hashable so the internal cache doesn't crash
ReActAgent.__hash__ = object.__hash__

from serving.router_tools import get_vector_tool, get_sql_tool

# --- MINIMAL CHANGE 1: Import the new prompts ---
from serving.prompt_templates import (
    GENERATION_SYSTEM_PROMPT, 
    REACT_AGENT_SYSTEM_PROMPT,
    VECTOR_TOOL_DESCRIPTION,
    SQL_TOOL_DESCRIPTION
)

app = FastAPI()

# 1. The Standard LLM 
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

# 2. The Router LLM
router_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    # MINIMAL CHANGE 2: Remove system_prompt here so it doesn't conflict with the ReAct agent
    is_chat_model=True,
    temperature=0.0,
    max_tokens=1024, # MINIMAL CHANGE 3: Bump from 512 to 1024 because ReAct uses more tokens for its Thought/Action loop
    context_window=4096
)

# Initialize Tools
vector_tool = get_vector_tool(generation_llm) 
# MINIMAL CHANGE 4: Overwrite tool descriptions with the ReAct-specific ones
vector_tool.metadata.description = VECTOR_TOOL_DESCRIPTION

sql_tool = get_sql_tool(generation_llm)       
# MINIMAL CHANGE 4 (cont): Overwrite tool descriptions
sql_tool.metadata.description = SQL_TOOL_DESCRIPTION

# Initialize the new Workflow-based ReActAgent
routing_agent = ReActAgent(
    name="routing_agent",
    description="Routes user queries to the correct database.",
    system_prompt=REACT_AGENT_SYSTEM_PROMPT, # MINIMAL CHANGE 5: Pass the new Agent prompt here
    tools=[vector_tool, sql_tool],
    llm=router_llm,
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
    
    # Run the workflow async, passing both the query and the chat history
    response = await routing_agent.run(
        user_msg=user_query, 
        chat_history=chat_history
    )
    
    # Return in standard OpenAI JSON format to Open WebUI
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": str(response)
            }
        }]
    }
