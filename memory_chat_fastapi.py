

import os
from typing import TypedDict, Annotated, Any
import operator
from dotenv import load_dotenv
import json
import logging
from logging.handlers import RotatingFileHandler
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.nodes import EpisodeType

from pydantic import BaseModel
import google.generativeai as genai
import numpy as np
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel as PydanticBaseModel
import uvicorn


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in .env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

LOG_DIR = os.getenv("LOG_DIR", ".")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("memory_chat")
logger.setLevel(logging.DEBUG)

# console
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# rotating file
fh = RotatingFileHandler(
    os.path.join(LOG_DIR, "memory_chat.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=5,
)
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)

logger.info("Starting memory_chat application")

logger.info("Initializing Graphiti with official Gemini clients")

llm_config = LLMConfig(
    api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash",
    temperature=0.4,
)

embedder_config = GeminiEmbedderConfig(
    api_key=GOOGLE_API_KEY,
    embedding_model="models/text-embedding-004",
)

reranker_config = LLMConfig(
    api_key=GOOGLE_API_KEY,
    model="gemini-2.5-flash",
)

gemini_llm = GeminiClient(config=llm_config)
gemini_embedder = GeminiEmbedder(config=embedder_config)
gemini_cross_encoder = GeminiRerankerClient(config=reranker_config)

graphiti = Graphiti(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    llm_client=gemini_llm,
    embedder=gemini_embedder,
    cross_encoder=gemini_cross_encoder,
)


@tool
async def retrieve_memory_tool(query: str, user_id: str) -> str:
    """Retrieve relevant memories for the user based on the query."""
    logger.info(f"[retrieve_memory] query='{query}', user={user_id}")

    try:
        results = await graphiti.search(
            query=query,
            group_ids=[user_id],
            num_results=5,
        )

        if not results:
            return "No relevant memories found."

        facts = [edge.fact for edge in results]

        return "Relevant memories:\n" + "\n".join(f"- {f}" for f in facts)

    except Exception as e:
        logger.exception("Retrieve memory error")
        return f"Error retrieving memories: {str(e)}"


@tool
async def store_memory_tool(conversation: str, user_id: str) -> str:
    """Store the conversation as a memory for the user."""
    logger.info(f"[store_memory] user={user_id}")

    try:
        await graphiti.add_episode(
            name=f"conversation_{user_id}_{datetime.now().timestamp()}",
            episode_body=conversation,
            source_description=f"Chat conversation with {user_id}",
            reference_time=datetime.now(),
            source=EpisodeType.message,
            group_id=user_id,
        )
        return "Memory stored successfully."

    except Exception as e:
        logger.exception("Store memory error")
        return f"Error storing memory: {str(e)}"


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str


async def call_model(state: AgentState):
    logger.info("[Agent] calling Gemini LLM")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
    )

    llm_with_tools = llm.bind_tools([retrieve_memory_tool, store_memory_tool])

    system_msg = SystemMessage(content=(
        "You are a helpful AI assistant with access to a memory system.\n"
        "- ALWAYS call retrieve_memory_tool at conversation start.\n"
        "- ALWAYS call store_memory_tool after responding.\n"
        "- Store conversations as: 'User: ...\\nAssistant: ...'\n"
    ))

    messages = [system_msg] + state["messages"]

    try:
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    except Exception as e:
        logger.exception("LLM error")
        return {"messages": [AIMessage(content="An internal error occurred.")]}


async def call_tools(state: AgentState):
    last = state["messages"][-1]

    if not getattr(last, "tool_calls", None):
        return {"messages": []}

    tool_messages = []

    for call in last.tool_calls:
        name = call["name"]
        args = call["args"]
        args["user_id"] = state["user_id"]

        try:
            if name == "retrieve_memory_tool":
                result = await retrieve_memory_tool.ainvoke(args)
            elif name == "store_memory_tool":
                result = await store_memory_tool.ainvoke(args)
            else:
                result = f"Unknown tool: {name}"

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
        except Exception as e:
            logger.exception("Tool error")
            tool_messages.append(
                ToolMessage(content=f"Error executing tool: {str(e)}", tool_call_id=call["id"])
            )

    return {"messages": tool_messages}


def should_continue(state: AgentState):
    last = state["messages"][-1]

    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()



async def chat(user_id: str, message: str):
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
    })

    # return the last assistant response
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg.content

    return result["messages"][-1].content

app = FastAPI(title="Memory Chat API")

class ChatRequest(PydanticBaseModel):
    user_id: str
    message: str

class ChatResponse(PydanticBaseModel):
    user_id: str
    response: str




@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        response = await chat(payload.user_id, payload.message)
        return ChatResponse(
            user_id=payload.user_id,
            response=response,
        )
    except Exception as e:
        logger.exception("HTTP error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Running FastAPI on {host}:{port}")
    uvicorn.run("memory_chat_fastapi:app", host=host, port=port, log_level="info")
