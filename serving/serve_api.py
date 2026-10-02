'''
This is the **Middleware**. Its job is to:
1. Pretend to be an OpenAI API server so Open WebUI can talk to it.
2. Intercept the user's question from the frontend.
3. Use the LlamaIndex library to search your Qdrant database.
4. Route the query using a Reasoning Agent to select the right tool.
5. Use the Generation LLM to summarize tool results into an observation.
6. Pass the Agent's final synthesized answer back up to Open WebUI.
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

# default max 20 thought-action-observation iterations is too much
MAX_ITERATIONS = 8
GENERATION_MODEL_MAX_LEN = int(os.getenv("GENERATION_MODEL_MAX_LEN", "16384"))

app = FastAPI()

# 1. The Generation LLM
generation_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    system_prompt=GENERATION_SYSTEM_PROMPT,
    is_chat_model=True,
    # context_window + max_tokens should be < --max-model-len in
    # vllm-model for generation model
    max_tokens=1024,
    context_window=15000
)

# 2. The Router LLM
router_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    is_chat_model=True,
    temperature=0.0,
    max_tokens=1024,
    context_window=15000
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
            max_iterations=MAX_ITERATIONS,
            early_stopping_method="generate"
        )
        return {"choices": [{"message": {"role": "assistant", "content": str(response)}}]}

    def make_chunk(text):
        return f'data: {json.dumps({"choices": [{"delta": {"content": text}}]})}\n\n'

    async def event_generator():
        yield make_chunk("⚙️ *Working...*\n\n")

        try:
            # 1. Start the workflow (returns a background handler immediately)
            handler = routing_agent.run(user_msg=user_query, chat_history=chat_history,
                                        max_iterations=MAX_ITERATIONS, early_stopping_method="generate")

            last_thought = ""

            # 2. Iterate over internal events as they happen
            async for event in handler.stream_events():
                event_name = type(event).__name__

                # --- Catch Structured ThinkingBlocks (for <think> models) ---
                if event_name == "AgentOutput" and hasattr(event, "response") and hasattr(event.response, "blocks"):
                    for block in event.response.blocks:
                        if getattr(block, "block_type", "") == "thinking":
                            thought = getattr(block, "content", "").strip()
                            if thought and thought != last_thought:
                                collapsible_thought = (
                                    "<details>\n"
                                    "<summary>🧠 Thought...</summary>\n\n"
                                    f"{thought}\n"
                                    "</details>\n\n"
                                )
                                yield make_chunk(collapsible_thought)
                                last_thought = thought
                # --- If it's a Tool ACTION ---
                elif event_name == "ToolCall":
                    tool_name = getattr(event, "tool_name", "tool")
                    tool_kwargs = getattr(event, "tool_kwargs", {})
                    kwargs_str = json.dumps(tool_kwargs, ensure_ascii=False) if tool_kwargs else "()"
                    collapsible_action = ( "<details>\n"
                                           f"<summary>🔀 Action: {tool_name}</summary>\n\n"
                                           f"```json\n{kwargs_str}```\n"
                                           "</details>\n\n")
                    yield make_chunk(collapsible_action)
                # --- If it's a Tool RESULT ---
                elif event_name == "ToolCallResult":
                    tool_output = getattr(event, "tool_output", None)
                    if not tool_output:
                        continue

                    raw_output = getattr(tool_output, "raw_output", None)

                    if raw_output and hasattr(raw_output, "source_nodes"):
                        nodes = raw_output.source_nodes
                        if nodes:
                            snippet_html = f"<details>\n<summary>🔍 {len(nodes)} nodes retrieved</summary>\n\n"
                            nodes_size = 0
                            for i, node in enumerate(nodes):
                                text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
                                nodes_size += len(text)
                                clean_text = text.replace('\n', ' ').strip()
                                snippet = clean_text[:80] + ("..." if len(clean_text) > 80 else "")
                                snippet_html += f"- `{snippet}`\n"
                            snippet_html += f"- {nodes_size} bytes\n"
                            approx_tokens = nodes_size // 4
                            if approx_tokens > generation_llm.context_window:
                                snippet_html +=  f"- ~{approx_tokens} tokens > configured context size: {generation_llm.context_window}\n"
                            if approx_tokens + generation_llm.max_tokens > GENERATION_MODEL_MAX_LEN:
                                snippet_html += f"- {approx_tokens} + {generation_llm.max_tokens} > --max-model-len {GENERATION_MODEL_MAX_LEN}, generation LLM may return an empty response!"
                            snippet_html += "\n</details>\n\n"
                            yield make_chunk(snippet_html)

                    if raw_output and hasattr(raw_output, "metadata") and raw_output.metadata:
                        metadata = raw_output.metadata
                        yield make_chunk(
                            "<details>\n"
                            "<summary>🏷️  Metadata</summary>\n\n"
                            f"```json\n{metadata}\n```\n"
                            "</details>\n\n"
                        )

                    observation_text = getattr(tool_output, "content", str(tool_output))
                    collapsible_tool_result = (
                        "<details>\n"
                        "<summary>📄 Observation</summary>\n\n"
                        f"```text\n{observation_text}\n```\n"
                        "</details>\n\n"
                    )
                    yield make_chunk(collapsible_tool_result)

            # 3. Once the workflow is done, await the final answer
            response = await handler
            yield make_chunk(f"**✅ Answer:**\n\n{str(response)}")

        except Exception as e:
            yield make_chunk(f"\n\n**Error:** {str(e)}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
