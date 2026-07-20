"""Telegram front end for PrivateAI."""

from __future__ import annotations

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

from .commands import handle_command, parse_slash_command
from .config import (
    ALLOWED_USER_ID,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_TOKEN,
)
from .runtime import AppRuntime

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


async def _reply_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_user is None:
        return
    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    runtime: AppRuntime = context.application.bot_data["runtime"]
    text = update.message.text or ""
    command = parse_slash_command(text)
    if command is None:
        await update.message.reply_text("Unrecognized command.")
        return
    reply = await handle_command(runtime.settings, _session_id(update), command)
    if reply is None:
        await update.message.reply_text(f"Unknown command: /{command.name}")
        return
    await update.message.reply_text(reply)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.message is None:
        return

    if not _authorized(update):
        await update.message.reply_text("Access Denied: Unauthorized Linux Admin.")
        return

    runtime: AppRuntime = context.application.bot_data["runtime"]
    user_prompt = update.message.text or ""
    session_id = _session_id(update)

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
            return f"🧠 Reasoning… ({elapsed}s)\n{excerpt}"
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
        response = await runtime.handle_text(
            session_id,
            user_prompt,
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
    app.bot_data["runtime"] = await AppRuntime.create()


async def post_shutdown(app: Application) -> None:
    runtime: AppRuntime | None = app.bot_data.get("runtime")
    if runtime is not None:
        await runtime.close()


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
    app.add_handler(CommandHandler("think", _reply_command))
    app.add_handler(CommandHandler("timeout", _reply_command))
    app.add_handler(CommandHandler("verbose", _reply_command))
    app.add_handler(CommandHandler("status", _reply_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Headless Bot Server Online via Telegram Pipeline...")
    app.run_polling()


if __name__ == "__main__":
    main()
