"""Ollama agent loop that can call MCP tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ollama import AsyncClient

from .config import MAX_TOOL_ITERATIONS, OLLAMA_HOST, OLLAMA_MODEL
from .mcp_client import McpHub
from .memory import MemoryStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are PrivateAI, a personal Linux administration assistant.

You may use MCP tools when they help answer the user accurately.
Prefer inspection and read-only tools before any write or destructive action.
If a tool fails, explain the failure briefly and suggest a safer next step.
Keep answers concise and practical.
When no tool is needed, answer directly.
"""


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is not None:
            data["tool_calls"] = tool_calls
    return data


def _log_background_failure(task: asyncio.Task[bool]) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("Background memory summarization failed")


async def _remember_response(
    memory: MemoryStore | None,
    session_id: str | int | None,
    user_prompt: str,
    response: str,
    model: str,
) -> str:
    if memory is None or session_id is None:
        return response

    await memory.remember_turn(session_id, user_prompt, response)
    task = asyncio.create_task(
        memory.summarize_if_needed(
            session_id,
            client=AsyncClient(host=OLLAMA_HOST),
            model=model,
        )
    )
    task.add_done_callback(_log_background_failure)
    return response


async def run_agent(
    user_prompt: str,
    hub: McpHub | None,
    model: str = OLLAMA_MODEL,
    memory: MemoryStore | None = None,
    session_id: str | int | None = None,
) -> str:
    """Run one user turn, optionally using MCP tools via Ollama tool calling."""
    client = AsyncClient(host=OLLAMA_HOST)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory is not None and session_id is not None:
        messages.extend(await memory.context_messages(session_id))
    messages.append({"role": "user", "content": user_prompt})

    tools = hub.as_ollama_tools() if hub and hub.tool_names else None

    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_TOOL_ITERATIONS)
        response = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
        )
        message = response.message
        messages.append(_message_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            final_response = (message.content or "").strip() or "(empty model response)"
            return await _remember_response(
                memory, session_id, user_prompt, final_response, model
            )

        if hub is None:
            final_response = "Model requested tools, but no MCP servers are configured."
            return await _remember_response(
                memory, session_id, user_prompt, final_response, model
            )

        for call in tool_calls:
            function = call.function
            name = function.name
            arguments = function.arguments or {}
            if not isinstance(arguments, dict):
                arguments = dict(arguments)
            tool_result = await hub.call_tool(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                }
            )

    final_response = (
        "Stopped after too many tool iterations. "
        "Narrow the request or raise MAX_TOOL_ITERATIONS."
    )
    return await _remember_response(
        memory, session_id, user_prompt, final_response, model
    )
