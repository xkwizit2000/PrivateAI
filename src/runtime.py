"""Shared PrivateAI runtime used by Telegram and the Session bridge."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .agent import ProgressCallback, run_agent
from .commands import handle_command, parse_slash_command
from .config import (
    MCP_CONFIG_PATH,
    MEMORY_BACKEND,
    MEMORY_DIR,
    MEMORY_DUAL_WRITE,
    MEMORY_MAX_CONTEXT_CHARS,
    MEMORY_RECALL,
    MEMORY_RECENT_TURNS,
    MEMORY_SERVICE_URL,
    MEMORY_SUMMARIZE_EVERY,
    MEMORY_TOKEN,
    MEMORY_TOOLS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_SECONDS,
    RAG_DIR,
    RAG_EMBED_MODEL,
    RAG_EMBED_NUM_GPU,
    RAG_ENABLED,
    RAG_MAX_CONTEXT_CHARS,
    RAG_MIN_SCORE,
    RAG_TOP_K,
    SESSION_SETTINGS_PATH,
    TOOL_ROUTING,
)
from .mcp_client import McpHub
from .memory import MemoryStore
from .memory_service_client import CoreMemoryStore, ServiceMemory
from .rag import RagStore
from .session_settings import SessionSettings

logger = logging.getLogger(__name__)


@dataclass
class AppRuntime:
    """Shared settings, memory, RAG, and MCP hub used by both chat front ends."""

    settings: SessionSettings
    memory: MemoryStore
    rag: RagStore | None
    hub: McpHub
    memory_service: ServiceMemory | None = None

    @classmethod
    async def create(cls) -> AppRuntime:
        """Construct runtime services, optionally backfill RAG, and start MCP."""
        settings = SessionSettings(
            SESSION_SETTINGS_PATH,
            default_think=OLLAMA_THINK,
            default_timeout=OLLAMA_TIMEOUT_SECONDS,
            default_tools=TOOL_ROUTING,
        )
        logger.info("Session settings enabled at %s", SESSION_SETTINGS_PATH)

        memory_service: ServiceMemory | None = None
        if MEMORY_SERVICE_URL:
            memory_service = ServiceMemory(
                MEMORY_SERVICE_URL,
                MEMORY_TOKEN,
                recall_mode=MEMORY_RECALL,
                dual_write=MEMORY_DUAL_WRITE,
                backend=MEMORY_BACKEND,
            )
        service_backend = MEMORY_BACKEND == "service" and memory_service is not None
        if MEMORY_BACKEND == "service" and memory_service is None:
            logger.warning(
                "MEMORY_BACKEND=service but MEMORY_SERVICE_URL is empty; using the legacy backend"
            )
        logger.info(
            "Memory config: MEMORY_BACKEND=%s MEMORY_RECALL=%s MEMORY_TOOLS=%s "
            "MEMORY_DUAL_WRITE=%s service=%s",
            "service" if service_backend else "legacy",
            MEMORY_RECALL if memory_service is not None else "off",
            MEMORY_TOOLS,
            "on" if MEMORY_DUAL_WRITE or service_backend else "off",
            MEMORY_SERVICE_URL or "not configured",
        )

        memory_args = dict(
            recent_turns=MEMORY_RECENT_TURNS,
            summarize_every=MEMORY_SUMMARIZE_EVERY,
            max_context_chars=MEMORY_MAX_CONTEXT_CHARS,
        )
        memory: MemoryStore
        if service_backend:
            memory = CoreMemoryStore(MEMORY_DIR, service=memory_service, **memory_args)
            logger.info("Core-only memory (service backend) at %s", MEMORY_DIR)
        else:
            memory = MemoryStore(MEMORY_DIR, **memory_args)
            logger.info("Durable conversation memory enabled at %s", MEMORY_DIR)

        rag: RagStore | None = None
        if service_backend:
            logger.info("Conversation RAG disabled (service backend)")
        elif RAG_ENABLED:
            rag = RagStore(
                RAG_DIR,
                embed_model=RAG_EMBED_MODEL,
                embed_num_gpu=RAG_EMBED_NUM_GPU,
                top_k=RAG_TOP_K,
                max_context_chars=RAG_MAX_CONTEXT_CHARS,
                min_score=RAG_MIN_SCORE,
                ollama_host=OLLAMA_HOST,
                recent_turns=MEMORY_RECENT_TURNS,
            )
            logger.info(
                "Conversation RAG enabled at %s (embed model=%s, num_gpu=%s)",
                RAG_DIR,
                RAG_EMBED_MODEL,
                RAG_EMBED_NUM_GPU,
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

        return cls(
            settings=settings,
            memory=memory,
            rag=rag,
            hub=hub,
            memory_service=memory_service,
        )

    async def close(self) -> None:
        """Shut down MCP servers and the memory-service client owned by this runtime."""
        await self.hub.close()
        if self.memory_service is not None:
            await self.memory_service.close()

    async def handle_text(
        self,
        session_id: str | int,
        text: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> str:
        """Route slash commands or run the agent for a normal chat turn."""
        command = parse_slash_command(text)
        if command is not None:
            reply = await handle_command(
                self.settings,
                session_id,
                command,
                memory_service=self.memory_service,
                runtime=self,
            )
            if reply is not None:
                return reply
            return f"Unknown command: /{command.name}"

        think = self.settings.get_think(session_id)
        timeout_seconds = self.settings.get_timeout(session_id)
        verbose = self.settings.get_verbose(session_id)
        tool_mode = self.settings.get_tool_mode(session_id)
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
            tool_mode=tool_mode,
            memory_service=self.memory_service,
        )
