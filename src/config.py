"""Environment-backed configuration for PrivateAI.

This module only defines constants loaded from the environment / .env file;
there are no runtime functions to document here.
"""

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
# Max tokens to generate per chat call (-1 = Ollama default / uncapped).
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "-1"))
# Context window size in tokens. Empty/0 = leave Ollama/model default.
_num_ctx_raw = os.getenv("OLLAMA_NUM_CTX", "").strip()
OLLAMA_NUM_CTX = int(_num_ctx_raw) if _num_ctx_raw else None
# When generation stops with done_reason=length, continue this many times.
OLLAMA_MAX_CONTINUATIONS = int(os.getenv("OLLAMA_MAX_CONTINUATIONS", "2"))
# After a connection-type failure, wait up to this long for Ollama to come back
# (polling /api/version) and retry the chat call once.
OLLAMA_RETRY_WAIT_SECONDS = float(os.getenv("OLLAMA_RETRY_WAIT_SECONDS", "90"))
OLLAMA_RETRY_POLL_SECONDS = float(os.getenv("OLLAMA_RETRY_POLL_SECONDS", "2"))
MCP_CONFIG_PATH = Path(os.getenv("MCP_CONFIG_PATH", PROJECT_ROOT / "mcp.json"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "8"))
# When true, expose only inspect/read MCP tools (recommended default).
MCP_READONLY = os.getenv("MCP_READONLY", "1").lower() not in {"0", "false", "no"}
# Comma-separated MCP tools to expose to the model. Names may be the registered
# name (filesystem_read_text_file) or the server-local name (read_text_file),
# case-insensitive. "*" or "all" exposes everything; empty = built-in default
# list (see src/mcp_client.py DEFAULT_TOOL_ALLOWLIST). Applied on top of MCP_READONLY.
MCP_TOOL_ALLOWLIST = os.getenv("MCP_TOOL_ALLOWLIST", "").strip()
# Cap (characters) for each tool's top-level description sent to Ollama (0 = unlimited).
MCP_TOOL_DESC_MAX_CHARS = int(os.getenv("MCP_TOOL_DESC_MAX_CHARS", "200") or "200")
# Cap (characters) for each parameter description inside tool schemas (0 = unlimited).
MCP_PARAM_DESC_MAX_CHARS = int(os.getenv("MCP_PARAM_DESC_MAX_CHARS", "80") or "80")
# Tool routing: auto = send tools only when the message looks like it needs file
# access; always = always send tools; never = never send tools.
_tool_routing_raw = os.getenv("TOOL_ROUTING", "auto").strip().lower() or "auto"
TOOL_ROUTING = _tool_routing_raw if _tool_routing_raw in {"auto", "always", "never"} else "auto"
# In auto mode, keep tools on for follow-up messages this long (seconds) after a
# turn that matched the file-access check or actually called a tool.
TOOL_ROUTING_FOLLOWUP_SECONDS = float(os.getenv("TOOL_ROUTING_FOLLOWUP_SECONDS", "600") or "600")
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", PROJECT_ROOT / "data" / "sessions"))
SESSION_SETTINGS_PATH = Path(
    os.getenv("SESSION_SETTINGS_PATH", PROJECT_ROOT / "data" / "session_settings.json")
)
MEMORY_RECENT_TURNS = int(os.getenv("MEMORY_RECENT_TURNS", "4"))
MEMORY_SUMMARIZE_EVERY = int(os.getenv("MEMORY_SUMMARIZE_EVERY", "6"))
MEMORY_MAX_CONTEXT_CHARS = int(os.getenv("MEMORY_MAX_CONTEXT_CHARS", "12000"))


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    """Read a lower-cased enum-like env var; unknown/empty values fall back to default."""
    value = os.getenv(name, default).strip().lower() or default
    return value if value in allowed else default


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var (0/false/no/off = False)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# --- External memory service (see docs/memory-service-design.md) -------------------
# legacy  = rolling summary + RAG (today's behaviour); service = tiny core context
#           (fact profile + last N turns) and no summary/RAG. Keep `legacy` until the
#           shadow-mode logs look good.
MEMORY_BACKEND = _env_choice("MEMORY_BACKEND", "legacy", {"legacy", "service"})
# Base URL of the memory service on the compose network. Empty = integration disabled.
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "").strip().rstrip("/")
# Shared bearer token (also used for ${MEMORY_TOKEN} expansion in mcp.json headers).
MEMORY_TOKEN = os.getenv("MEMORY_TOKEN", "").strip()
# off    = never ask the service for recall
# shadow = ask in the background and only LOG metadata (gate/hits/score/chars/ms)
# on     = inject the service's gated recall block as a system message
MEMORY_RECALL = _env_choice("MEMORY_RECALL", "shadow", {"off", "shadow", "on"})
# on = expose the memory MCP tools (mcp.json server named MEMORY_MCP_SERVER) to the model.
MEMORY_TOOLS = _env_choice("MEMORY_TOOLS", "off", {"off", "on"})
# Name of the mcp.json server whose tools form the "memory" tool group.
MEMORY_MCP_SERVER = os.getenv("MEMORY_MCP_SERVER", "memory").strip() or "memory"
# 1 = also send every completed turn to the service (in addition to the JSONL).
MEMORY_DUAL_WRITE = _env_flag("MEMORY_DUAL_WRITE", True)
# Short timeouts (seconds): a slow/down service must never hold up a chat turn.
MEMORY_RECALL_TIMEOUT = float(os.getenv("MEMORY_RECALL_TIMEOUT", "2.5") or "2.5")
MEMORY_APPEND_TIMEOUT = float(os.getenv("MEMORY_APPEND_TIMEOUT", "3") or "3")
MEMORY_TOOL_TIMEOUT = float(os.getenv("MEMORY_TOOL_TIMEOUT", "5") or "5")
# Connecting an SSE MCP server (startup / reconnect) gives up after this long.
MEMORY_SSE_CONNECT_TIMEOUT = float(os.getenv("MEMORY_SSE_CONNECT_TIMEOUT", "8") or "8")
# Circuit breaker: after this many consecutive failures stop calling the service for a
# cooldown (doubling up to MEMORY_BREAKER_MAX_COOLDOWN seconds) so a dead service costs nothing.
MEMORY_BREAKER_FAILURES = int(os.getenv("MEMORY_BREAKER_FAILURES", "3") or "3")
MEMORY_BREAKER_COOLDOWN = float(os.getenv("MEMORY_BREAKER_COOLDOWN", "30") or "30")
MEMORY_BREAKER_MAX_COOLDOWN = float(os.getenv("MEMORY_BREAKER_MAX_COOLDOWN", "300") or "300")
# Hard cap (chars) of the recall block injected in `on` mode.
MEMORY_RECALL_MAX_CHARS = int(os.getenv("MEMORY_RECALL_MAX_CHARS", "700") or "700")
RAG_ENABLED = os.getenv("RAG_ENABLED", "1").lower() not in {"0", "false", "no"}
RAG_DIR = Path(os.getenv("RAG_DIR", PROJECT_ROOT / "data" / "rag"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-embed-text")
# GPU layers for the embedding model. 0 = CPU only (default, keeps the iGPU free
# for the chat model); -1 or empty = let Ollama decide; N>0 = offload N layers.
_embed_num_gpu_raw = os.getenv("RAG_EMBED_NUM_GPU", "0").strip()
RAG_EMBED_NUM_GPU: int | None = (
    int(_embed_num_gpu_raw) if _embed_num_gpu_raw not in {"", "-1"} else None
)
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
SESSION_MESSAGE_CHUNK = int(os.getenv("SESSION_MESSAGE_CHUNK", "1800"))