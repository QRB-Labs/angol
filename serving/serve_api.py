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
import json
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent.workflow import ReActAgent

# FIX FOR LLAMAINDEX WORKFLOW BUG
# Forces the Pydantic-based agent to be hashable so the internal cache doesn't crash
ReActAgent.__hash__ = object.__hash__

from serving.router_tools import get_vector_tool, get_sql_tool
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
    # context_window + max_tokens should be < --max-model-len in
    # vllm-model for generation model
    max_tokens=1024,
    context_window=10000
)

# 2. The Router LLM
router_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    is_chat_model=True,
    temperature=0.0,
    max_tokens=1024,
    context_window=10000
)

# Initialize Tools
vector_tool = get_vector_tool(generation_llm)
vector_tool.metadata.description = VECTOR_TOOL_DESCRIPTION

sql_tool = get_sql_tool(generation_llm)
sql_tool.metadata.description = SQL_TOOL_DESCRIPTION

# Initialize the new Workflow-based ReActAgent
routing_agent = ReActAgent(
    name="routing_agent",
    description="Routes user queries to the correct database.",
    system_prompt=REACT_AGENT_SYSTEM_PROMPT,
    tools=[vector_tool, sql_tool],
    llm=router_llm,
    verbose=True,
)

class ChatRequest(BaseModel):
    messages: list
    stream: bool = False

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
    user_query = request.messages[-1]["content"]

    chat_history = []
    for msg in request.messages[:-1]:
        role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
        chat_history.append(ChatMessage(role=role, content=msg.get("content", "")))

    if not request.stream:
        response = await routing_agent.run(
            user_msg=user_query,
            chat_history=chat_history,
            # default max 20 thought-action-observation iterations is too much
            max_iterations=4,
            early_stopping_method="generate"
        )
        return {"choices": [{"message": {"role": "assistant", "content": str(response)}}]}

    def make_chunk(text):
        return f'data: {json.dumps({"choices": [{"delta": {"content": text}}]})}\n\n'

    async def event_generator():
        yield make_chunk("⚙️ *ReAct Agent Analyzing Query...*\n\n")

        try:
            # 1. Start the workflow (returns a background handler immediately)
            handler = routing_agent.run(user_msg=user_query, chat_history=chat_history,
                                        # default max 20 thought-action-observation iterations is too much
                                        max_iterations=4, early_stopping_method="generate")

            last_thought = ""

            # 2. Iterate over internal events as they happen
            async for event in handler.stream_events():
                event_name = type(event).__name__
                event_str = str(event)

                # --- Catch Structured ThinkingBlocks (for <think> models) ---
                if event_name == "AgentOutput":
                    if hasattr(event, "response") and hasattr(event.response, "blocks"):
                        for block in event.response.blocks:
                            if getattr(block, "block_type", "") == "thinking":
                                thought = getattr(block, "content", "").strip()
                                if thought and thought != last_thought:
                                    yield make_chunk(f"🧠 *Thinking: {thought}*\n\n")
                                    last_thought = thought

                if "ToolCall" in event_name or "ToolCall" in event_str:
                    # --- If it's a Tool ACTION ---
                    if "Result" not in event_name and "Result" not in event_str:
                        if "sql" in event_str.lower():
                            yield make_chunk("🔀 *Agent Action: Querying PostgreSQL Database...*\n\n")
                        else:
                            yield make_chunk("🔀 *Agent Action: Searching Qdrant Vector Data...*\n\n")

                    # --- If it's a Tool RESULT ---
                    else:
                        # Extract the output payload from the event safely
                        tool_output = getattr(event, "tool_output", event_str)
                        # ToolOutput objects usually have a content attribute, otherwise stringify it
                        result_text = getattr(tool_output, "content", str(tool_output))

                        # Yield it wrapped in a Markdown code block so it looks clean in the UI
                        yield make_chunk(f"**📄 Database Results:**\n```text\n{result_text}\n```\n\n⚙️ *Agent Synthesizing Answer...*\n\n")

            # 3. Once the workflow is done, await the final answer
            response = await handler
            yield make_chunk(f"**✅ Final Answer:**\n\n{str(response)}")

        except Exception as e:
            yield make_chunk(f"\n\n**Error:** {str(e)}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
