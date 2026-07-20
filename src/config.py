import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
# Front-end channel: telegram (python-telegram-bot) or session (Session.js adapter).
CHAT_BACKEND = os.getenv("CHAT_BACKEND", "telegram").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# Hard limit for a single Ollama chat call (seconds).
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
# Log a "model still waiting" warning this often while a chat call is in flight.
OLLAMA_STUCK_WARN_SECONDS = float(os.getenv("OLLAMA_STUCK_WARN_SECONDS", "30"))
# Gemma/Qwen thinking/reasoning mode. Set 0/false to disable (faster replies).
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "0").lower() in {"1", "true", "yes"}
MCP_CONFIG_PATH = Path(os.getenv("MCP_CONFIG_PATH", PROJECT_ROOT / "mcp.json"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))
# When true, expose only inspect/read MCP tools (recommended default).
MCP_READONLY = os.getenv("MCP_READONLY", "1").lower() not in {"0", "false", "no"}
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", PROJECT_ROOT / "data" / "sessions"))
SESSION_SETTINGS_PATH = Path(
    os.getenv("SESSION_SETTINGS_PATH", PROJECT_ROOT / "data" / "session_settings.json")
)
MEMORY_RECENT_TURNS = int(os.getenv("MEMORY_RECENT_TURNS", "4"))
MEMORY_SUMMARIZE_EVERY = int(os.getenv("MEMORY_SUMMARIZE_EVERY", "6"))
MEMORY_MAX_CONTEXT_CHARS = int(os.getenv("MEMORY_MAX_CONTEXT_CHARS", "12000"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "1").lower() not in {"0", "false", "no"}
RAG_DIR = Path(os.getenv("RAG_DIR", PROJECT_ROOT / "data" / "rag"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Session.js adapter → Python agent bridge (used when CHAT_BACKEND=session).
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8787"))
BRIDGE_TOKEN = os.getenv("BRIDGE_TOKEN", "").strip()
# Comma-separated Session IDs allowed to talk to the bot (adapter + bridge).
SESSION_ALLOWED_IDS = [
    item.strip()
    for item in os.getenv("SESSION_ALLOWED_ID", "").split(",")
    if item.strip()
]
# Used by the Session.js adapter process (not read by the Python bridge).
SESSION_MNEMONIC = os.getenv("SESSION_MNEMONIC", "").strip()
SESSION_DISPLAY_NAME = os.getenv("SESSION_DISPLAY_NAME", "PrivateAI").strip() or "PrivateAI"
AGENT_BRIDGE_URL = os.getenv(
    "AGENT_BRIDGE_URL", f"http://{BRIDGE_HOST}:{BRIDGE_PORT}"
).rstrip("/")
SESSION_STORAGE_PATH = os.getenv(
    "SESSION_STORAGE_PATH",
    str(PROJECT_ROOT / "data" / "session-adapter" / "storage.db"),
)
SESSION_MESSAGE_CHUNK = int(os.getenv("SESSION_MESSAGE_CHUNK", "2000"))