"""Ollama agent loop that can call MCP tools."""

from __future__ import annotations

import logging
from typing import Any

from ollama import AsyncClient

from .config import MAX_TOOL_ITERATIONS, OLLAMA_HOST, OLLAMA_MODEL
from .mcp_client import McpHub

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


async def run_agent(user_prompt: str, hub: McpHub | None, model: str = OLLAMA_MODEL) -> str:
    """Run one user turn, optionally using MCP tools via Ollama tool calling."""
    client = AsyncClient(host=OLLAMA_HOST)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

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
            return (message.content or "").strip() or "(empty model response)"

        if hub is None:
            return "Model requested tools, but no MCP servers are configured."

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

    return (
        "Stopped after too many tool iterations. "
        "Narrow the request or raise MAX_TOOL_ITERATIONS."
    )
