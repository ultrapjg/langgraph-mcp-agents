import asyncio
import json
import os
import platform
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from utils import astream_graph, random_uuid

load_dotenv(override=True)

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

CONFIG_FILE_PATH = "config.json"

DEFAULT_CONFIG = {
    "get_current_time": {
        "command": "python",
        "args": ["./mcp_server_time.py"],
        "transport": "stdio",
    }
}

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools.
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question.
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
}

# Simple in-memory state
state: Dict[str, Optional[object]] = {
    "agent": None,
    "mcp_client": None,
    "thread_id": random_uuid(),
    "recursion_limit": 100,
    "tool_count": 0,
}


async def cleanup_mcp_client() -> None:
    if state.get("mcp_client") is not None:
        try:
            await state["mcp_client"].__aexit__(None, None, None)
        except Exception:
            pass
        finally:
            state["mcp_client"] = None


def load_config_from_json() -> dict:
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            return DEFAULT_CONFIG
    except Exception:
        return DEFAULT_CONFIG


def save_config_to_json(config: dict) -> bool:
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


async def initialize_session(selected_model: str, mcp_config=None) -> bool:
    await cleanup_mcp_client()

    if mcp_config is None:
        mcp_config = load_config_from_json()
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    state["tool_count"] = len(tools)
    state["mcp_client"] = client

    if selected_model in [
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ]:
        model = ChatAnthropic(
            model=selected_model,
            temperature=0.1,
            max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
        )
    else:
        model = ChatOpenAI(
            model=selected_model,
            temperature=0.1,
            max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
        )
    agent = create_react_agent(
        model,
        tools,
        checkpointer=MemorySaver(),
        prompt=SYSTEM_PROMPT,
    )
    state["agent"] = agent
    if state.get("thread_id") is None:
        state["thread_id"] = random_uuid()
    return True


async def process_query(query: str, timeout_seconds: int = 60) -> Dict[str, str]:
    if state.get("agent") is None:
        return {"error": "Agent not initialized."}

    text_chunks: List[str] = []
    tool_chunks: List[str] = []

    def streaming_callback(message: dict):
        message_content = message.get("content", None)
        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                if message_chunk["type"] == "text":
                    text_chunks.append(message_chunk["text"])
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        tool_chunks.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunk = message_content.tool_call_chunks[0]
                        tool_chunks.append(str(tool_call_chunk))
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_chunks.append(str(message_content.tool_calls[0]))
            elif isinstance(content, str):
                text_chunks.append(content)
        elif isinstance(message_content, ToolMessage):
            tool_chunks.append(str(message_content.content))
        return None

    try:
        await asyncio.wait_for(
            astream_graph(
                state["agent"],
                {"messages": [HumanMessage(content=query)]},
                callback=streaming_callback,
                config=RunnableConfig(
                    recursion_limit=state["recursion_limit"],
                    thread_id=state["thread_id"],
                ),
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return {"error": f"Request time exceeded {timeout_seconds} seconds."}

    final_text = "".join(text_chunks)
    final_tool = "".join(tool_chunks)
    return {"text": final_text, "tool": final_tool}


# FastAPI application
app = FastAPI()


class InitRequest(BaseModel):
    selected_model: str
    mcp_config: Optional[Dict[str, object]] = None


class InitResponse(BaseModel):
    status: str
    tool_count: int


class QueryRequest(BaseModel):
    query: str
    timeout_seconds: int = 60


class QueryResponse(BaseModel):
    text: str
    tool: str


@app.post("/initialize", response_model=InitResponse)
async def initialize_endpoint(req: InitRequest):
    if req.mcp_config is not None:
        save_config_to_json(req.mcp_config)
    success = await initialize_session(req.selected_model, req.mcp_config)
    if not success:
        raise HTTPException(status_code=500, detail="Initialization failed")
    return {"status": "ok", "tool_count": state["tool_count"]}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    result = await process_query(req.query, req.timeout_seconds)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await cleanup_mcp_client()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend_service:app", host="0.0.0.0", port=8000)
