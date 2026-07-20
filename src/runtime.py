"""Shared PrivateAI runtime used by Telegram and the Session bridge."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .agent import ProgressCallback, run_agent
from .commands import handle_command, parse_slash_command
from .config import (
    MCP_CONFIG_PATH,
    MEMORY_DIR,
    MEMORY_MAX_CONTEXT_CHARS,
    MEMORY_RECENT_TURNS,
    MEMORY_SUMMARIZE_EVERY,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_SECONDS,
    RAG_DIR,
    RAG_EMBED_MODEL,
    RAG_ENABLED,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MIN_SCORE,
    RAG_TOP_K,
    SESSION_SETTINGS_PATH,
)
from .mcp_client import McpHub
from .memory import MemoryStore
from .rag import RagStore
from .session_settings import SessionSettings

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    settings: SessionSettings
    memory: MemoryStore
    rag: RagStore | None
    hub: McpHub

    @classmethod
    async def create(cls) -> AppRuntime:
        settings = SessionSettings(
            SESSION_SETTINGS_PATH,
            default_think=OLLAMA_THINK,
            default_timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        logger.info("Session settings enabled at %s", SESSION_SETTINGS_PATH)

        memory = MemoryStore(
            MEMORY_DIR,
            recent_turns=MEMORY_RECENT_TURNS,
            summarize_every=MEMORY_SUMMARIZE_EVERY,
            max_context_chars=MEMORY_MAX_CONTEXT_CHARS,
        )
        logger.info("Durable conversation memory enabled at %s", MEMORY_DIR)

        rag: RagStore | None = None
        if RAG_ENABLED:
            rag = RagStore(
                RAG_DIR,
                embed_model=RAG_EMBED_MODEL,
                top_k=RAG_TOP_K,
                max_context_chars=RAG_MAX_CONTEXT_CHARS,
                min_score=RAG_MIN_SCORE,
                ollama_host=OLLAMA_HOST,
                recent_turns=MEMORY_RECENT_TURNS,
            )
            logger.info(
                "Conversation RAG enabled at %s (embed model=%s)",
                RAG_DIR,
                RAG_EMBED_MODEL,
            )
            total = 0
            for session_id in memory.list_session_ids():
                records = await memory.load_transcript(session_id)
                total += await rag.backfill_from_transcript(session_id, records)
            if total:
                logger.info(
                    "Backfilled %d RAG chunk(s) from existing transcripts", total
                )
        else:
            logger.info("Conversation RAG disabled")

        hub = McpHub(MCP_CONFIG_PATH)
        await hub.start()
        if hub.tool_names:
            logger.info("Tools available: %s", ", ".join(hub.tool_names))
        else:
            logger.info("Running in chat-only mode (no MCP tools loaded)")

        return cls(settings=settings, memory=memory, rag=rag, hub=hub)

    async def close(self) -> None:
        await self.hub.close()

    async def handle_text(
        self,
        session_id: str | int,
        text: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> str:
        command = parse_slash_command(text)
        if command is not None:
            reply = await handle_command(self.settings, session_id, command)
            if reply is not None:
                return reply
            return f"Unknown command: /{command.name}"

        think = self.settings.get_think(session_id)
        timeout_seconds = self.settings.get_timeout(session_id)
        verbose = self.settings.get_verbose(session_id)
        return await run_agent(
            text,
            hub=self.hub,
            model=OLLAMA_MODEL,
            memory=self.memory,
            rag=self.rag,
            session_id=session_id,
            progress=progress,
            think=think,
            timeout_seconds=timeout_seconds,
            verbose=verbose,
        )
