import asyncio
import logging
from contextlib import suppress
from time import monotonic

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from .agent import run_agent
from .config import (
    ALLOWED_USER_ID,
    MCP_CONFIG_PATH,
    MEMORY_DIR,
    MEMORY_MAX_CONTEXT_CHARS,
    MEMORY_RECENT_TURNS,
    MEMORY_SUMMARIZE_EVERY,
    OLLAMA_MODEL,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_TOKEN,
)
from .mcp_client import McpHub
from .memory import MemoryStore

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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    if update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    user_prompt = update.message.text or ""
    hub: McpHub | None = context.application.bot_data.get("mcp_hub")
    memory: MemoryStore | None = context.application.bot_data.get("memory")
    session_id = (
        update.effective_chat.id
        if update.effective_chat is not None
        else update.effective_user.id
    )

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
        elif progress_stage == "responding":
            label = "✍️ Preparing response…"
        else:
            label = "🤔 Thinking…"
        return f"{label}\nStill thinking — {elapsed}s elapsed"

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
            session_id=session_id,
            progress=report_progress,
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
    app.bot_data["memory"] = MemoryStore(
        MEMORY_DIR,
        recent_turns=MEMORY_RECENT_TURNS,
        summarize_every=MEMORY_SUMMARIZE_EVERY,
        max_context_chars=MEMORY_MAX_CONTEXT_CHARS,
    )
    logger.info("Durable conversation memory enabled at %s", MEMORY_DIR)

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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Headless Bot Server Online via Telegram Pipeline...")
    app.run_polling()


if __name__ == "__main__":
    main()
