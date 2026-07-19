import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MCP_CONFIG_PATH = Path(os.getenv("MCP_CONFIG_PATH", PROJECT_ROOT / "mcp.json"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))
# When true, expose only inspect/read MCP tools (recommended default).
MCP_READONLY = os.getenv("MCP_READONLY", "1").lower() not in {"0", "false", "no"}
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", PROJECT_ROOT / "data" / "sessions"))
MEMORY_RECENT_TURNS = int(os.getenv("MEMORY_RECENT_TURNS", "4"))
MEMORY_SUMMARIZE_EVERY = int(os.getenv("MEMORY_SUMMARIZE_EVERY", "6"))
MEMORY_MAX_CONTEXT_CHARS = int(os.getenv("MEMORY_MAX_CONTEXT_CHARS", "12000"))
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
