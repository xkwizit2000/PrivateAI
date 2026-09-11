"""Ollama agent loop that can call MCP tools."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import monotonic
from typing import Any

from ollama import AsyncClient

from .config import (
    MAX_TOOL_ITERATIONS,
    MEMORY_RECENT_TURNS,
    OLLAMA_HOST,
    OLLAMA_MAX_CONTINUATIONS,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_STUCK_WARN_SECONDS,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_SECONDS,
)
from .mcp_client import McpHub
from .memory import MemoryStore
from .rag import RagStore

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, str | None], Awaitable[None]]

SYSTEM_PROMPT = """You are PrivateAI, a personal Systems, Security, AI, Automation, Options trading, and strategy development assistant.

You may use MCP tools when they help answer the user accurately.
Prefer inspection and read-only tools before any write or destructive action.
If a tool fails, explain the failure briefly and suggest a safer next step.
Keep answers concise and practical.
When no tool is needed, answer directly.
"""


CONTINUE_PROMPT = (
    "Continue your previous reply exactly where you left off. "
    "Do not restart, apologize, or repeat text already written."
)


def _ollama_options() -> dict[str, Any] | None:
    """Build optional Ollama generation options from config."""
    options: dict[str, Any] = {}
    if OLLAMA_NUM_PREDICT is not None and OLLAMA_NUM_PREDICT != -1:
        options["num_predict"] = OLLAMA_NUM_PREDICT
    elif OLLAMA_NUM_PREDICT == -1:
        # Explicit uncapped predict avoids low Modelfile caps truncating replies.
        options["num_predict"] = -1
    if OLLAMA_NUM_CTX:
        options["num_ctx"] = OLLAMA_NUM_CTX
    return options or None


def _prefer_longer_text(assembled: str, final: str | None) -> str:
    """Prefer assembled stream deltas unless the final payload is longer."""
    if not final:
        return assembled
    return final if len(final) > len(assembled) else assembled


def _done_reason(response: Any) -> str | None:
    reason = getattr(response, "done_reason", None)
    return str(reason) if reason is not None else None



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


def _ns_to_s(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None


def _rate(count: Any, duration_ns: Any) -> float | None:
    seconds = _ns_to_s(duration_ns)
    if seconds is None or seconds <= 0 or count is None:
        return None
    try:
        return float(count) / seconds
    except (TypeError, ValueError):
        return None


def _log_ollama_stats(response: Any, *, model: str, wall_s: float) -> None:
    """Log Ollama timing/token stats from a chat response."""
    total_s = _ns_to_s(getattr(response, "total_duration", None))
    load_s = _ns_to_s(getattr(response, "load_duration", None))
    prompt_s = _ns_to_s(getattr(response, "prompt_eval_duration", None))
    eval_s = _ns_to_s(getattr(response, "eval_duration", None))
    prompt_tokens = getattr(response, "prompt_eval_count", None)
    eval_tokens = getattr(response, "eval_count", None)
    prompt_tok_s = _rate(prompt_tokens, getattr(response, "prompt_eval_duration", None))
    eval_tok_s = _rate(eval_tokens, getattr(response, "eval_duration", None))
    done_reason = getattr(response, "done_reason", None)

    logger.info(
        "Ollama stats model=%s wall=%.2fs total=%s load=%s "
        "prompt_tokens=%s prompt_s=%s prompt_tok/s=%s "
        "gen_tokens=%s gen_s=%s gen_tok/s=%s done_reason=%s",
        model,
        wall_s,
        f"{total_s:.2f}s" if total_s is not None else "n/a",
        f"{load_s:.2f}s" if load_s is not None else "n/a",
        prompt_tokens if prompt_tokens is not None else "n/a",
        f"{prompt_s:.2f}s" if prompt_s is not None else "n/a",
        f"{prompt_tok_s:.1f}" if prompt_tok_s is not None else "n/a",
        eval_tokens if eval_tokens is not None else "n/a",
        f"{eval_s:.2f}s" if eval_s is not None else "n/a",
        f"{eval_tok_s:.1f}" if eval_tok_s is not None else "n/a",
        done_reason if done_reason is not None else "n/a",
    )


async def _watch_for_stall(model: str, started: float, timeout_seconds: float) -> None:
    """Periodically warn while an Ollama chat call has not returned."""
    try:
        while True:
            await asyncio.sleep(OLLAMA_STUCK_WARN_SECONDS)
            waited = monotonic() - started
            logger.warning(
                "Ollama model appears stuck: model=%s waited=%.0fs "
                "(still waiting for chat response; will time out at %.0fs)",
                model,
                waited,
                timeout_seconds,
            )
    except asyncio.CancelledError:
        raise


async def _chat_with_monitoring(
    client: AsyncClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    think: bool,
    timeout_seconds: float,
    progress: ProgressCallback | None = None,
    verbose: bool = False,
) -> Any:
    """Call Ollama chat with stall warnings, timeout, and performance logging."""
    started = monotonic()
    watcher = asyncio.create_task(_watch_for_stall(model, started, timeout_seconds))
    stream_reasoning = bool(think and verbose and progress is not None)

    options = _ollama_options()

    async def _non_stream() -> Any:
        return await client.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=think,
            options=options,
        )

    async def _stream() -> Any:
        assert progress is not None
        stream = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=think,
            stream=True,
            options=options,
        )
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: Any = None
        last_chunk: Any = None
        last_progress_at = 0.0

        async for chunk in stream:
            last_chunk = chunk
            message = chunk.message
            thinking_piece = getattr(message, "thinking", None)
            if thinking_piece:
                thinking_parts.append(str(thinking_piece))
                now = monotonic()
                if now - last_progress_at >= 2.0:
                    excerpt = "".join(thinking_parts)[-700:]
                    await progress("reasoning", excerpt)
                    last_progress_at = now
            content_piece = getattr(message, "content", None)
            if content_piece:
                content_parts.append(str(content_piece))
            calls = getattr(message, "tool_calls", None)
            if calls:
                tool_calls = calls

        if thinking_parts:
            await progress("reasoning", "".join(thinking_parts)[-700:])

        if last_chunk is None:
            raise RuntimeError("Ollama stream returned no chunks")

        # Assemble deltas; only prefer a final payload when it is longer (full text).
        # Overwriting with the last delta alone truncates the reply.
        final_message = getattr(last_chunk, "message", None)
        content = "".join(content_parts)
        thinking = "".join(thinking_parts)
        if final_message is not None:
            final_content = getattr(final_message, "content", None)
            if final_content:
                content = _prefer_longer_text(content, str(final_content))
            final_thinking = getattr(final_message, "thinking", None)
            if final_thinking:
                thinking = _prefer_longer_text(thinking, str(final_thinking))
            if getattr(final_message, "tool_calls", None):
                tool_calls = final_message.tool_calls

        from types import SimpleNamespace

        assembled_message = SimpleNamespace(
            role="assistant",
            content=content or None,
            thinking=thinking or None,
            tool_calls=tool_calls,
        )
        return SimpleNamespace(
            model=getattr(last_chunk, "model", model),
            created_at=getattr(last_chunk, "created_at", None),
            done=getattr(last_chunk, "done", True),
            done_reason=getattr(last_chunk, "done_reason", None),
            total_duration=getattr(last_chunk, "total_duration", None),
            load_duration=getattr(last_chunk, "load_duration", None),
            prompt_eval_count=getattr(last_chunk, "prompt_eval_count", None),
            prompt_eval_duration=getattr(last_chunk, "prompt_eval_duration", None),
            eval_count=getattr(last_chunk, "eval_count", None),
            eval_duration=getattr(last_chunk, "eval_duration", None),
            message=assembled_message,
        )

    try:
        response = await asyncio.wait_for(
            _stream() if stream_reasoning else _non_stream(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        waited = monotonic() - started
        logger.error(
            "Ollama chat timed out: model=%s waited=%.0fs timeout=%.0fs "
            "(model may be stuck; check ollama on %s)",
            model,
            waited,
            timeout_seconds,
            OLLAMA_HOST,
        )
        raise TimeoutError(
            f"Ollama model '{model}' timed out after {int(waited)}s"
        ) from exc
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher

    _log_ollama_stats(response, model=model, wall_s=monotonic() - started)
    return response


def _log_background_failure(task: asyncio.Task[Any]) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("Background memory task failed")


async def _remember_response(
    memory: MemoryStore | None,
    rag: RagStore | None,
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

    if rag is not None:
        index_task = asyncio.create_task(
            rag.index_turn(session_id, user_prompt, response)
        )
        index_task.add_done_callback(_log_background_failure)

    return response


async def _rag_context_message(
    rag: RagStore,
    memory: MemoryStore | None,
    session_id: str | int,
    user_prompt: str,
) -> dict[str, str] | None:
    exclude_texts: list[str] = []
    if memory is not None:
        records = await memory.load_transcript(session_id)
        exclude_texts = rag.recent_exclude_texts(
            records, recent_turns=MEMORY_RECENT_TURNS
        )

    snippets = await rag.retrieve(
        session_id,
        user_prompt,
        exclude_texts=exclude_texts,
    )
    formatted = rag.format_context(snippets)
    if not formatted:
        return None
    return {"role": "system", "content": formatted}


async def run_agent(
    user_prompt: str,
    hub: McpHub | None,
    model: str = OLLAMA_MODEL,
    memory: MemoryStore | None = None,
    rag: RagStore | None = None,
    session_id: str | int | None = None,
    progress: ProgressCallback | None = None,
    think: bool | None = None,
    timeout_seconds: float | None = None,
    verbose: bool = False,
) -> str:
    """Run one user turn, optionally using MCP tools via Ollama tool calling."""
    think_enabled = OLLAMA_THINK if think is None else think
    chat_timeout = (
        OLLAMA_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    )
    client = AsyncClient(host=OLLAMA_HOST, timeout=chat_timeout)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory is not None and session_id is not None:
        messages.extend(await memory.context_messages(session_id))
    if rag is not None and session_id is not None:
        if progress is not None:
            await progress("thinking", None)
        rag_message = await _rag_context_message(rag, memory, session_id, user_prompt)
        if rag_message is not None:
            # Insert after system prompt / summary, before recent turns when possible.
            # context_messages returns [summary?, ...recent]. Put RAG after summary.
            insert_at = 1
            if (
                len(messages) > 1
                and messages[1].get("role") == "system"
                and str(messages[1].get("content", "")).startswith(
                    "Durable memory from earlier conversations:"
                )
            ):
                insert_at = 2
            messages.insert(insert_at, rag_message)
    messages.append({"role": "user", "content": user_prompt})

    tools = hub.as_ollama_tools() if hub and hub.tool_names else None
    logger.info(
        "Agent think mode=%s verbose=%s timeout=%.0fs",
        think_enabled,
        verbose,
        chat_timeout,
    )

    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_TOOL_ITERATIONS)
        if progress is not None:
            await progress("thinking", None)
        try:
            response = await _chat_with_monitoring(
                client,
                model=model,
                messages=messages,
                tools=tools,
                think=think_enabled,
                timeout_seconds=chat_timeout,
                progress=progress,
                verbose=verbose,
            )
        except TimeoutError:
            final_response = (
                f"System Error: the model timed out after "
                f"{int(chat_timeout)}s and may be stuck. "
                f"Check Ollama on {OLLAMA_HOST}."
            )
            if progress is not None:
                await progress("responding", None)
            return await _remember_response(
                memory, rag, session_id, user_prompt, final_response, model
            )
        message = response.message
        messages.append(_message_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            pieces = [(message.content or "").strip()]
            continuations = 0
            while (
                _done_reason(response) == "length"
                and continuations < OLLAMA_MAX_CONTINUATIONS
            ):
                continuations += 1
                logger.warning(
                    "Ollama stopped with done_reason=length; continuing %d/%d",
                    continuations,
                    OLLAMA_MAX_CONTINUATIONS,
                )
                if progress is not None:
                    await progress("thinking", None)
                messages.append({"role": "user", "content": CONTINUE_PROMPT})
                try:
                    response = await _chat_with_monitoring(
                        client,
                        model=model,
                        messages=messages,
                        tools=None,
                        think=think_enabled,
                        timeout_seconds=chat_timeout,
                        progress=progress,
                        verbose=verbose,
                    )
                except TimeoutError:
                    break
                message = response.message
                messages.append(_message_to_dict(message))
                more = (message.content or "").strip()
                if more:
                    pieces.append(more)
                if getattr(message, "tool_calls", None):
                    # Unexpected tool call during continuation; stop extending.
                    break

            final_response = "\n".join(p for p in pieces if p) or "(empty model response)"
            if _done_reason(response) == "length":
                final_response += (
                    "\n\n_(Reply truncated by the model length limit. "
                    "Raise OLLAMA_NUM_PREDICT / OLLAMA_NUM_CTX or "
                    "OLLAMA_MAX_CONTINUATIONS.)_"
                )
            if progress is not None:
                await progress("responding", None)
            return await _remember_response(
                memory, rag, session_id, user_prompt, final_response, model
            )

        if hub is None:
            final_response = "Model requested tools, but no MCP servers are configured."
            if progress is not None:
                await progress("responding", None)
            return await _remember_response(
                memory, rag, session_id, user_prompt, final_response, model
            )

        for call in tool_calls:
            function = call.function
            name = function.name
            arguments = function.arguments or {}
            if not isinstance(arguments, dict):
                arguments = dict(arguments)
            if progress is not None:
                await progress("processing", name)
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
    if progress is not None:
        await progress("responding", None)
    return await _remember_response(
        memory, rag, session_id, user_prompt, final_response, model
    )
