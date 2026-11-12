"""Ollama agent loop that can call MCP tools."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import monotonic
from typing import Any

import httpx
from ollama import AsyncClient, ResponseError

from .config import (
    MAX_TOOL_ITERATIONS,
    MEMORY_RECENT_TURNS,
    MEMORY_TOOLS,
    OLLAMA_HOST,
    OLLAMA_MAX_CONTINUATIONS,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_RETRY_POLL_SECONDS,
    OLLAMA_RETRY_WAIT_SECONDS,
    OLLAMA_STUCK_WARN_SECONDS,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_SECONDS,
    TOOL_ROUTING,
)
from .mcp_client import McpHub
from .memory import MemoryStore
from .memory_service_client import ServiceMemory
from .rag import RagStore
from .tool_routing import DEFAULT_ROUTER, ToolRouter

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


OLLAMA_UNAVAILABLE_MESSAGE = (
    "System Error: Ollama was unavailable or restarted while answering. "
    "Please resend your message in a minute."
)

# Connection-type failures that typically happen while Ollama is restarting.
_RETRYABLE_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    ConnectionError,  # the ollama client re-raises httpx.ConnectError as this
)


class OllamaUnavailableError(Exception):
    """Ollama stayed unreachable (or failed again) after the one allowed retry."""


def _is_retryable_ollama_error(exc: BaseException) -> bool:
    """True for disconnect/restart-type failures and Ollama HTTP 5xx responses."""
    if isinstance(exc, ResponseError):
        return exc.status_code >= 500
    return isinstance(exc, _RETRYABLE_CONNECTION_ERRORS)


async def _wait_for_ollama(host: str, max_wait: float, poll: float) -> bool:
    """Poll GET {host}/api/version until Ollama answers or max_wait elapses."""
    url = f"{host.rstrip('/')}/api/version"
    deadline = monotonic() + max_wait
    async with httpx.AsyncClient(timeout=5.0) as http:
        while True:
            try:
                response = await http.get(url)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            if monotonic() + poll > deadline:
                return False
            await asyncio.sleep(poll)


async def _chat_with_retry(
    client: AsyncClient,
    *,
    retry_state: dict[str, bool],
    **kwargs: Any,
) -> Any:
    """Run one model call, retrying it once if Ollama dropped/restarted.

    Only the failed model request is repeated (never tool calls). The retry
    budget lives in ``retry_state`` so a single user request retries at most
    once. Timeouts, 4xx errors and other exceptions propagate unchanged.
    """
    try:
        return await _chat_with_monitoring(client, **kwargs)
    except Exception as exc:
        if not _is_retryable_ollama_error(exc):
            raise
        if retry_state.get("used"):
            raise OllamaUnavailableError(str(exc)) from exc
        retry_state["used"] = True
        logger.warning(
            "Ollama chat failed (%s: %s); waiting up to %.0fs for %s then retrying once",
            type(exc).__name__,
            exc,
            OLLAMA_RETRY_WAIT_SECONDS,
            OLLAMA_HOST,
        )
        if not await _wait_for_ollama(
            OLLAMA_HOST, OLLAMA_RETRY_WAIT_SECONDS, OLLAMA_RETRY_POLL_SECONDS
        ):
            raise OllamaUnavailableError(
                f"Ollama did not come back within {int(OLLAMA_RETRY_WAIT_SECONDS)}s"
            ) from exc
        logger.info("Ollama is reachable again; retrying the chat call")
        try:
            return await _chat_with_monitoring(client, **kwargs)
        except Exception as retry_exc:
            if _is_retryable_ollama_error(retry_exc):
                raise OllamaUnavailableError(str(retry_exc)) from retry_exc
            raise


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
    """Return Ollama's done_reason string, or None if absent."""
    reason = getattr(response, "done_reason", None)
    return str(reason) if reason is not None else None


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Normalize an Ollama message object into a plain dict for the chat history."""
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
    """Convert a nanosecond duration to seconds, or None if invalid."""
    if value is None:
        return None
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None


def _rate(count: Any, duration_ns: Any) -> float | None:
    """Compute items-per-second from a count and nanosecond duration."""
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
        """Run a single non-streaming Ollama chat call."""
        return await client.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=think,
            options=options,
        )

    async def _stream() -> Any:
        """Stream an Ollama chat call, assemble deltas, and surface reasoning progress."""
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
    """Log exceptions from fire-and-forget memory/RAG background tasks."""
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
    memory_service: ServiceMemory | None = None,
) -> str:
    """Persist the completed turn and kick off summary/RAG indexing in the background.

    With a memory service configured the same completed turn is also sent to it
    (fire-and-forget dual-write; the reply never waits and failures are swallowed).
    Only turns that reach this function are saved, so failed/retried turns that
    return an error message earlier are stored nowhere, exactly as before.
    """
    if memory is None or session_id is None:
        return response

    timestamp = await memory.remember_turn(session_id, user_prompt, response)
    service_backend = memory_service is not None and memory_service.backend == "service"
    if memory_service is not None and (memory_service.dual_write or service_backend):
        memory_service.dual_write_turn(
            session_id,
            user_prompt,
            response,
            timestamp if isinstance(timestamp, str) else None,
        )
    if service_backend:
        return response  # service backend: no rolling summary and no RAG indexing

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
    """Build an optional system message with retrieved past conversation snippets."""
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
    tool_mode: str | None = None,
    router: ToolRouter | None = None,
    memory_service: ServiceMemory | None = None,
    memory_tools: bool | None = None,
) -> str:
    """Run one user turn, optionally using MCP tools via Ollama tool calling.

    ``tool_mode`` (auto|always|never; default TOOL_ROUTING) controls whether the
    tool schemas are sent to the model for this turn.

    ``memory_service`` (optional) receives a copy of each completed turn and answers
    the per-turn recall query according to its ``recall_mode``: ``shadow`` runs the
    query in the background and only logs metadata; ``on`` injects the gated block as
    a system message. ``memory_tools`` (default MEMORY_TOOLS=on) enables routing of
    the memory MCP tool group.
    """
    think_enabled = OLLAMA_THINK if think is None else think
    chat_timeout = (
        OLLAMA_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    )
    memory_tools_on = (MEMORY_TOOLS == "on") if memory_tools is None else memory_tools
    client = AsyncClient(host=OLLAMA_HOST, timeout=chat_timeout)

    # Per-turn recall against the memory service. shadow: scheduled and forgotten (zero
    # added latency, metadata-only log line). on: runs concurrently with the legacy
    # context assembly below and is awaited just before the model call (bounded timeout).
    recall_task: asyncio.Task[str] | None = None
    if (
        memory_service is not None
        and session_id is not None
        and user_prompt.strip()
        and memory_service.recall_mode in ("shadow", "on")
    ):
        if memory_service.recall_mode == "shadow":
            memory_service.start_shadow_recall(user_prompt, session_id)
        else:
            recall_task = memory_service.spawn(
                memory_service.recall_block(user_prompt, session_id)
            )

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
    if recall_task is not None:
        recall_block = ""
        try:
            recall_block = await recall_task
        except Exception:  # noqa: BLE001 - recall must never break a turn
            logger.warning("memory recall failed; continuing without it")
        if recall_block:
            insert_at = 1
            while (
                insert_at < len(messages) and messages[insert_at].get("role") == "system"
            ):
                insert_at += 1
            messages.insert(insert_at, {"role": "system", "content": recall_block})
    messages.append({"role": "user", "content": user_prompt})

    router = router or DEFAULT_ROUTER
    routing_mode = tool_mode or TOOL_ROUTING
    offered_names: set[str] | None = None
    if hub is not None and hub.tool_names:
        if not memory_tools_on:
            decision = router.decide(routing_mode, session_id, user_prompt)
            tools = hub.as_ollama_tools() if decision.tools_on else None
        else:
            decision = router.decide(
                routing_mode, session_id, user_prompt, memory_tools=True
            )
            groups = {
                group
                for group, on in (("files", decision.tools_on), ("memory", decision.memory_on))
                if on
            }
            tools = (hub.as_ollama_tools(groups=groups) or None) if groups else None
            if decision.memory_on:
                hub.kick_reconnect()  # background; heals a memory server that was down
            offered_names = {t["function"]["name"] for t in tools or []}
        logger.info(
            "tool routing: tools=%s reason=%s",
            "on" if decision.tools_on else "off",
            decision.reason,
        )
        if memory_tools_on:
            logger.info(
                "tool routing: memory=%s reason=%s",
                "on" if decision.memory_on else "off",
                decision.memory_reason,
            )
    else:
        tools = None
        logger.info("tool routing: tools=off reason=no MCP tools registered")
    logger.info(
        "Agent think mode=%s verbose=%s timeout=%.0fs",
        think_enabled,
        verbose,
        chat_timeout,
    )

    retry_state: dict[str, bool] = {}

    for iteration in range(MAX_TOOL_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_TOOL_ITERATIONS)
        if progress is not None:
            await progress("thinking", None)
        try:
            response = await _chat_with_retry(
                client,
                retry_state=retry_state,
                model=model,
                messages=messages,
                tools=tools,
                think=think_enabled,
                timeout_seconds=chat_timeout,
                progress=progress,
                verbose=verbose,
            )
        except OllamaUnavailableError:
            logger.exception("Ollama unavailable after retry; replying with an error")
            if progress is not None:
                await progress("responding", None)
            # Not stored in memory/RAG: the user is asked to resend the message.
            return OLLAMA_UNAVAILABLE_MESSAGE
        except TimeoutError:
            final_response = (
                f"System Error: the model timed out after "
                f"{int(chat_timeout)}s and may be stuck. "
                f"Check Ollama on {OLLAMA_HOST}."
            )
            if progress is not None:
                await progress("responding", None)
            return await _remember_response(
                memory, rag, session_id, user_prompt, final_response, model,
                memory_service,
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
                    response = await _chat_with_retry(
                        client,
                        retry_state=retry_state,
                        model=model,
                        messages=messages,
                        tools=None,
                        think=think_enabled,
                        timeout_seconds=chat_timeout,
                        progress=progress,
                        verbose=verbose,
                    )
                except OllamaUnavailableError:
                    logger.exception(
                        "Ollama unavailable during continuation; returning partial reply"
                    )
                    pieces.append(
                        "_(Reply may be incomplete: Ollama became unavailable "
                        "while continuing.)_"
                    )
                    break
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
                memory, rag, session_id, user_prompt, final_response, model,
                memory_service,
            )

        if hub is None:
            final_response = "Model requested tools, but no MCP servers are configured."
            if progress is not None:
                await progress("responding", None)
            return await _remember_response(
                memory, rag, session_id, user_prompt, final_response, model,
                memory_service,
            )

        for call in tool_calls:
            function = call.function
            name = function.name
            arguments = function.arguments or {}
            if not isinstance(arguments, dict):
                arguments = dict(arguments)
            if progress is not None:
                await progress("processing", name)
            if tools is None:
                # The model was not given any tool schemas this turn; don't run
                # a hallucinated call.
                logger.warning("Model called tool %s although tools were off", name)
                tool_result = (
                    f"Error: tool '{name}' is not available for this message. "
                    "Answer without tools."
                )
            elif offered_names is not None and name not in offered_names:
                # Memory routing is on and this tool's group was not offered this turn.
                logger.warning("Model called tool %s which was not offered this turn", name)
                tool_result = (
                    f"Error: tool '{name}' is not available for this message. "
                    "Answer without it."
                )
            else:
                if offered_names is None:
                    router.mark_active(session_id)
                else:
                    router.mark_active(session_id, hub.tool_group(name) or "files")
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
        memory, rag, session_id, user_prompt, final_response, model, memory_service
    )
