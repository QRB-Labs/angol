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
import logging
import time
import uuid
import asyncio
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.callbacks.schema import EventPayload

# FIX FOR LLAMAINDEX WORKFLOW BUG
# Forces Pydantic-based agent to be hashable so internal cache doesn't crash
ReActAgent.__hash__ = object.__hash__

from serving.router_tools import get_vector_tool, get_sql_tool
from serving.prompt_templates import (
    GENERATION_SYSTEM_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    VECTOR_TOOL_DESCRIPTION,
    SQL_TOOL_DESCRIPTION
)

# default max 20 thought-action-observation iterations is too much
MAX_ITERATIONS = 15
GENERATION_MODEL_MAX_LEN = int(os.getenv("GENERATION_MODEL_MAX_LEN", "32768"))

app = FastAPI()

# Uncomment the following lines to get details from llama
#logging.basicConfig(level=logging.INFO)
#httpx_logger = logging.getLogger("httpx")
#httpx_logger.setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# 1. The Generation LLM
generation_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    system_prompt=GENERATION_SYSTEM_PROMPT,
    is_chat_model=True,
    timeout=600.0,
    # context_window + max_tokens should be < max-model-len in vllm-model
    max_tokens=1024,
    context_window=GENERATION_MODEL_MAX_LEN - 2048
)

# 2. The Router LLM
router_llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY", "fake-key"),
    model=os.getenv("GENERATION_MODEL"),
    is_chat_model=True,
    temperature=0.0,
    timeout=600.0,
    max_tokens=1024,
    context_window=GENERATION_MODEL_MAX_LEN - 1024
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
        content=msg.get("content", "")
        if role == MessageRole.ASSISTANT and "**✅ Answer:**" in content:
            # Keep ONLY the final answer in conversation history, not intermediate observations, etc.
            content = content.split("**✅ Answer:**")[-1].strip()
        chat_history.append(ChatMessage(role=role, content=content))

    req_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if not request.stream:
        response = await routing_agent.run(
            user_msg=user_query,
            chat_history=chat_history,
            max_iterations=MAX_ITERATIONS,
            early_stopping_method="generate"
        )
        
        final_text = str(response)
        if not final_text.strip():
            final_text = "[System: The model returned an empty string. Check context limits.]"
            
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created_time,
            "model": "angol-routing-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    def make_chunk(text):
        return f'data: {json.dumps({"id": req_id, "object": "chat.completion.chunk", "created": created_time, "model": "angol-routing-agent", "choices": [{"index": 0, "delta": {"content": text}}]})}\n\n'

    async def event_generator():
        try:
            start_time = time.time()
            # 1. Start the workflow (returns a background handler immediately)
            handler = routing_agent.run(user_msg=user_query, chat_history=chat_history,
                                        max_iterations=MAX_ITERATIONS, early_stopping_method="generate")

            last_thought = ""

            # --- HEARTBEAT LOGIC ADDED HERE ---
            # Set up a queue to decouple the event stream from the timed heartbeat loop
            queue = asyncio.Queue()

            async def fetch_events():
                """Background task to fetch events and push them to the queue."""
                try:
                    async for ev in handler.stream_events():
                        await queue.put(("event", ev))
                    await queue.put(("done", None))
                except Exception as ex:
                    await queue.put(("error", ex))

            # Start fetching events in the background
            fetch_task = asyncio.create_task(fetch_events())

            try:
                # 2. Iterate over internal events as they arrive in the queue
                while True:
                    try:
                        # Wait for an event, but time out every 15 seconds to send a heartbeat
                        msg_type, event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield make_chunk("\n")
                        continue

                    # Handle the queue message type
                    if msg_type == "done":
                        break
                    elif msg_type == "error":
                        raise event  # Raise it to be caught by the outer try-except handler

                    # Process the actual event (EXISTING LOGIC)
                    event_name = type(event).__name__

                    if event_name == "StopEvent":
                        elapsed_time = time.time() - start_time
                        hours, rem = divmod(elapsed_time, 3600)
                        minutes, seconds = divmod(rem, 60)
                        if hours > 0:
                            duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                        elif minutes > 0:
                            duration_str = f"{int(minutes)}m {int(seconds)}s"
                        else:
                            duration_str = f"{seconds:.1f}s"
                        yield make_chunk(f"🏁 *Finished in {duration_str}...*\n\n")
                    elif event_name == "AgentInput":
                        yield make_chunk(f"➡️  *Input received, working...*\n\n")
                    # --- Catch Structured ThinkingBlocks (for <think> models) ---
                    elif event_name == "AgentOutput" and hasattr(event, "response") and hasattr(event.response, "blocks"):
                        for block in event.response.blocks:
                            if getattr(block, "block_type", "") == "thinking":
                                thought = getattr(block, "content", "").strip()
                                if thought and thought != last_thought:
                                    collapsible_thought = (
                                        "<details>\n"
                                        "<summary>🧠 Thought</summary>\n\n"
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
                                               f"<summary>🔀 Action: {tool_name}...</summary>\n\n"
                                               f"```json\n{kwargs_str}```\n"
                                               "</details>\n\n")
                        yield make_chunk(collapsible_action)
                    # --- If it's a Tool RESULT ---
                    elif event_name == "ToolCallResult":
                        tool_output = getattr(event, "tool_output", None)
                        if not tool_output:
                            continue

                        if getattr(tool_output, "is_error", False):
                            error_msg = getattr(tool_output, "content", "Unknown tool execution error")
                            yield make_chunk(f"⚠️  **Tool Error:** {error_msg}\n\n")
                            # Continue, ReAct agent will see the error observation and try to fix it
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
                                    clean_text = text.replace('\n', ' ').replace('`', "'").replace('<', '&lt;').replace('>', '&gt;').strip()
                                    snippet = clean_text[:80] + ("..." if len(clean_text) > 80 else "")
                                    snippet_html += f"- `{snippet}`\n"
                                snippet_html += f"{nodes_size} chars\n"
                                approx_tokens = nodes_size // 4
                                if approx_tokens > generation_llm.context_window:
                                    snippet_html +=  f"~{approx_tokens} tokens > configured context size: {generation_llm.context_window}\n"
                                if approx_tokens + generation_llm.max_tokens > GENERATION_MODEL_MAX_LEN:
                                    snippet_html += f"~{approx_tokens} + {generation_llm.max_tokens} > max-model-len {GENERATION_MODEL_MAX_LEN}, generation LLM may return an empty response!"
                                snippet_html += "\n</details>\n\n"
                                yield make_chunk(snippet_html)

                        if raw_output and hasattr(raw_output, "metadata") and raw_output.metadata:
                            metadata = raw_output.metadata
                            yield make_chunk(
                                "<details>\n"
                                "<summary>🏷️  Metadata</summary>\n\n"
                                f"```json\n{json.dumps(metadata, indent=2)}\n```\n"
                                "</details>\n\n"
                            )

                        observation_text = getattr(tool_output, "content", None)
                        if observation_text is None or str(observation_text).strip() in ["", "None"]:
                            # No Observation, dump debug info
                            debug_data = {
                                "tool_name": getattr(tool_output, "tool_name", "Unknown"),
                                "tool_inputs": getattr(tool_output, "raw_input", {}), 
                                "raw_output_type": str(type(raw_output)) if raw_output else "None",
                            }
                            
                            # Inspect the underlying Response object
                            if raw_output:
                                # LlamaIndex Response objects store the string answer in .response
                                debug_data["raw_output_response_attr"] = getattr(raw_output, "response", "Not found")
                                # Check if the query engine suppressed an exception
                                if hasattr(raw_output, "exception") and raw_output.exception:
                                    debug_data["raw_output_exception"] = str(raw_output.exception)
                                    
                            # Inspect the LlamaIndex Event Payload for hidden exceptions
                            if hasattr(event, "payload") and event.payload:
                                debug_data["payload_keys"] = list(event.payload.keys())
                                # Try to extract the literal exception from the payload if it exists
                                if EventPayload.EXCEPTION in event.payload:
                                    debug_data["payload_exception"] = str(event.payload[EventPayload.EXCEPTION])
                            logger.error(f"{json.dumps(debug_data)}")
                            yield make_chunk((
                                "<details>\n"
                                "<summary>⚠️  No observation</summary>\n\n"
                                f"```json\n{json.dumps(debug_data, indent=2, default=str)}\n```\n"
                                "</details>\n\n"
                            ))
                        else:
                            collapsible_tool_result = (
                                "<details>\n"
                                "<summary>📄 Observation</summary>\n\n"
                                f"{observation_text}\n\n"
                                "</details>\n\n"
                            )
                            yield make_chunk(collapsible_tool_result)
            finally:
                # Ensure the background task cleans up properly if the client disconnects
                fetch_task.cancel()

            # 3. Once the workflow is done, await the final answer
            response = await handler
            final_answer = str(response)

            if not final_answer.strip():
                final_answer = "[System: The model returned an empty string. Check context limits.]"
            yield make_chunk(f"**✅ Answer:**\n\n{final_answer}")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield make_chunk(f"\n\n**Error:** {str(e)}")

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
