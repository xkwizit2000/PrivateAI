import asyncio
import logging
from contextlib import suppress
from time import monotonic

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import run_agent
from .config import (
    ALLOWED_USER_ID,
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
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_TOKEN,
)
from .mcp_client import McpHub
from .memory import MemoryStore
from .ollama_status import fetch_ollama_status
from .rag import RagStore
from .session_settings import MAX_TIMEOUT_SECONDS, MIN_TIMEOUT_SECONDS, SessionSettings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# httpx logs full request URLs; Telegram embeds the bot token in the path.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Telegram drops the typing indicator after ~5s; refresh before that.
TYPING_REFRESH_SECONDS = 4.0
# Edit the temporary progress message often enough to show the bot is alive.
STATUS_REFRESH_SECONDS = 5.0


def chunk_text(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into Telegram-safe message chunks."""
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _session_id(update: Update) -> int:
    if update.effective_chat is not None:
        return update.effective_chat.id
    assert update.effective_user is not None
    return update.effective_user.id


def _authorized(update: Update) -> bool:
    return (
        update.effective_user is not None
        and update.effective_user.id == ALLOWED_USER_ID
    )


async def keep_typing(message: Message) -> None:
    """Resend ChatAction.TYPING until cancelled."""
    try:
        while True:
            await message.reply_chat_action(action=ChatAction.TYPING)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Typing indicator refresh failed", exc_info=True)


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show per-chat settings plus Ollama running-model info."""
    if update.message is None or update.effective_user is None:
        return
    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    settings: SessionSettings = context.application.bot_data["session_settings"]
    session_text = settings.format_status(_session_id(update))
    ollama_text = await fetch_ollama_status()
    await update.message.reply_text(f"{session_text}\n\n{ollama_text}")


async def handle_think(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle or show per-chat Ollama thinking mode: /think [on|off|status]."""
    if update.message is None or update.effective_user is None:
        return
    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    settings: SessionSettings = context.application.bot_data["session_settings"]
    session_id = _session_id(update)
    args = [arg.lower() for arg in (context.args or [])]
    current = settings.get_think(session_id)

    if not args or args[0] in {"status", "show"}:
        await update.message.reply_text(
            f"Thinking mode is {'on' if current else 'off'} for this chat.\n"
            f"(Default from OLLAMA_THINK is {'on' if OLLAMA_THINK else 'off'}.)\n"
            "Use /think on or /think off to change it.\n"
            "Also: /verbose status, /timeout status, /status"
        )
        return

    if args[0] in {"on", "true", "1", "enable"}:
        settings.set_think(session_id, True)
        await update.message.reply_text(
            "Thinking mode enabled for this chat. Replies may be slower but deeper.\n"
            "Consider raising the timeout, e.g. /timeout 600\n"
            "Status: think=on"
        )
        return

    if args[0] in {"off", "false", "0", "disable"}:
        settings.set_think(session_id, False)
        await update.message.reply_text(
            "Thinking mode disabled for this chat. Replies should be faster.\n"
            "Status: think=off"
        )
        return

    await update.message.reply_text("Usage: /think [on|off|status]")


async def handle_timeout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show or set per-chat Ollama timeout: /timeout [seconds|reset|status]."""
    if update.message is None or update.effective_user is None:
        return
    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    settings: SessionSettings = context.application.bot_data["session_settings"]
    session_id = _session_id(update)
    args = [arg.lower() for arg in (context.args or [])]
    current = settings.get_timeout(session_id)

    if not args or args[0] in {"status", "show"}:
        await update.message.reply_text(
            f"Timeout is {int(current)}s for this chat.\n"
            f"(Default from OLLAMA_TIMEOUT_SECONDS is {int(OLLAMA_TIMEOUT_SECONDS)}s.)\n"
            f"Allowed range: {int(MIN_TIMEOUT_SECONDS)}–{int(MAX_TIMEOUT_SECONDS)}s.\n"
            "Use /timeout <seconds> or /timeout reset to change it.\n"
            "Also: /think status, /verbose status, /status"
        )
        return

    if args[0] in {"reset", "default", "clear"}:
        restored = settings.clear_timeout(session_id)
        await update.message.reply_text(
            f"Timeout reset to default ({int(restored)}s) for this chat.\n"
            f"Status: timeout={int(restored)}s"
        )
        return

    try:
        seconds = float(args[0])
        applied = settings.set_timeout(session_id, seconds)
    except ValueError as exc:
        await update.message.reply_text(
            f"{exc}\nUsage: /timeout [status|<seconds>|reset]"
        )
        return

    await update.message.reply_text(
        f"Timeout set to {int(applied)}s for this chat.\n"
        f"Status: timeout={int(applied)}s"
    )


async def handle_verbose(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle streaming of model reasoning into status: /verbose [on|off|status]."""
    if update.message is None or update.effective_user is None:
        return
    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    settings: SessionSettings = context.application.bot_data["session_settings"]
    session_id = _session_id(update)
    args = [arg.lower() for arg in (context.args or [])]
    current = settings.get_verbose(session_id)
    think = settings.get_think(session_id)

    if not args or args[0] in {"status", "show"}:
        note = ""
        if current and not think:
            note = "\nNote: /think is off, so no reasoning text will stream yet."
        await update.message.reply_text(
            f"Verbose mode is {'on' if current else 'off'} for this chat.\n"
            "Use /verbose on or /verbose off to change it.\n"
            "With /verbose on and /think on, reasoning streams into the status message."
            f"{note}\n"
            "Also: /think status, /timeout status, /status"
        )
        return

    if args[0] in {"on", "true", "1", "enable"}:
        settings.set_verbose(session_id, True)
        extra = ""
        if not think:
            extra = "\n/think is currently off — enable it with /think on to stream reasoning."
        await update.message.reply_text(
            "Verbose mode enabled. Reasoning will stream into the status when thinking is on."
            f"{extra}\n"
            "Status: verbose=on"
        )
        return

    if args[0] in {"off", "false", "0", "disable"}:
        settings.set_verbose(session_id, False)
        await update.message.reply_text(
            "Verbose mode disabled. Status will show the simple progress indicator.\n"
            "Status: verbose=off"
        )
        return

    await update.message.reply_text("Usage: /verbose [on|off|status]")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    user_prompt = update.message.text or ""
    hub: McpHub | None = context.application.bot_data.get("mcp_hub")
    memory: MemoryStore | None = context.application.bot_data.get("memory")
    rag: RagStore | None = context.application.bot_data.get("rag")
    settings: SessionSettings = context.application.bot_data["session_settings"]
    session_id = _session_id(update)
    think = settings.get_think(session_id)
    timeout_seconds = settings.get_timeout(session_id)
    verbose = settings.get_verbose(session_id)

    status_started = monotonic()
    progress_stage = "thinking"
    progress_detail: str | None = None
    displayed_status = "🤔 Thinking…\nStill thinking — 0s elapsed"
    status_message = await update.message.reply_text(displayed_status)
    status_lock = asyncio.Lock()
    typing_task: asyncio.Task[None] | None = None

    def render_status() -> str:
        elapsed = int(monotonic() - status_started)
        if progress_stage == "processing":
            label = f"⚙️ Processing: {progress_detail or 'tool'}…"
            return f"{label}\nStill thinking — {elapsed}s elapsed"
        if progress_stage == "responding":
            return f"✍️ Preparing response…\nStill thinking — {elapsed}s elapsed"
        if progress_stage == "reasoning" and progress_detail:
            excerpt = progress_detail.strip()
            if len(excerpt) > 700:
                excerpt = "…" + excerpt[-700:]
            return (
                f"🧠 Reasoning… ({elapsed}s)\n"
                f"{excerpt}"
            )
        return f"🤔 Thinking…\nStill thinking — {elapsed}s elapsed"

    async def update_status() -> None:
        nonlocal displayed_status
        async with status_lock:
            next_status = render_status()
            if next_status == displayed_status:
                return
            try:
                await status_message.edit_text(next_status)
                displayed_status = next_status
            except Exception:
                logger.debug("Progress status update failed", exc_info=True)

    async def refresh_status() -> None:
        while True:
            await asyncio.sleep(STATUS_REFRESH_SECONDS)
            await update_status()

    async def report_progress(stage: str, detail: str | None) -> None:
        nonlocal progress_stage, progress_detail, typing_task
        progress_stage = stage
        progress_detail = detail
        if stage == "responding" and typing_task is None:
            typing_task = asyncio.create_task(keep_typing(update.message))
        await update_status()

    status_task = asyncio.create_task(refresh_status())
    try:
        response = await run_agent(
            user_prompt,
            hub=hub,
            model=OLLAMA_MODEL,
            memory=memory,
            rag=rag,
            session_id=session_id,
            progress=report_progress,
            think=think,
            timeout_seconds=timeout_seconds,
            verbose=verbose,
        )
        for part in chunk_text(response):
            await update.message.reply_text(part)
    except Exception:
        logger.exception("Agent request failed")
        await update.message.reply_text(
            "System Error: the agent request failed. Check server logs for details."
        )
    finally:
        status_task.cancel()
        with suppress(asyncio.CancelledError):
            await status_task
        if typing_task is not None:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task
        try:
            await status_message.delete()
        except Exception:
            logger.debug("Progress status cleanup failed", exc_info=True)


async def post_init(app: Application) -> None:
    app.bot_data["session_settings"] = SessionSettings(
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
    app.bot_data["memory"] = memory
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
        app.bot_data["rag"] = rag
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
            logger.info("Backfilled %d RAG chunk(s) from existing transcripts", total)
    else:
        app.bot_data["rag"] = None
        logger.info("Conversation RAG disabled")

    hub = McpHub(MCP_CONFIG_PATH)
    await hub.start()
    app.bot_data["mcp_hub"] = hub
    if hub.tool_names:
        logger.info("Tools available: %s", ", ".join(hub.tool_names))
    else:
        logger.info("Running in chat-only mode (no MCP tools loaded)")


async def post_shutdown(app: Application) -> None:
    hub: McpHub | None = app.bot_data.get("mcp_hub")
    if hub is not None:
        await hub.close()


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise ValueError("Missing TELEGRAM_TOKEN environment variable.")
    if not ALLOWED_USER_ID:
        raise ValueError("Missing or invalid ALLOWED_USER_ID environment variable.")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("think", handle_think))
    app.add_handler(CommandHandler("timeout", handle_timeout))
    app.add_handler(CommandHandler("verbose", handle_verbose))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Headless Bot Server Online via Telegram Pipeline...")
    app.run_polling()


if __name__ == "__main__":
    main()
