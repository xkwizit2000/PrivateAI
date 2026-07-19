import logging

from telegram import Update
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
logger = logging.getLogger(__name__)


def chunk_text(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into Telegram-safe message chunks."""
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    if update.effective_user.id != ALLOWED_USER_ID:
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    await update.message.reply_chat_action(action=ChatAction.TYPING)
    user_prompt = update.message.text or ""
    hub: McpHub | None = context.application.bot_data.get("mcp_hub")
    memory: MemoryStore | None = context.application.bot_data.get("memory")
    session_id = (
        update.effective_chat.id
        if update.effective_chat is not None
        else update.effective_user.id
    )

    try:
        response = await run_agent(
            user_prompt,
            hub=hub,
            model=OLLAMA_MODEL,
            memory=memory,
            session_id=session_id,
        )
        for part in chunk_text(response):
            await update.message.reply_text(part)
    except Exception:
        logger.exception("Agent request failed")
        await update.message.reply_text(
            "System Error: the agent request failed. Check server logs for details."
        )


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
