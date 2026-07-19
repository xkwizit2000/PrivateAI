# PrivateAI

Headless personal AI admin assistant for Ubuntu Server. Telegram is the remote interface; a local Ollama model reasons; MCP servers expose scoped tools.

## Architecture

```text
Telegram (authorized user)
        ↓
src/bot.py          authorization + messaging
        ↓
src/agent.py        Ollama tool-calling loop
        ↓
src/mcp_client.py   MCP hub (stdio servers from mcp.json)
        ↓
MCP tools (e.g. filesystem) → results back to the model → Telegram reply

src/memory.py persists each session's transcript and rolling summary under
`data/sessions/`, then adds the summary and recent turns to future prompts.
```

## Prerequisite Configurations
* Ubuntu Server 24.04+ LTS (Headless via SSH)
* Ollama background runtime daemon active (`ollama serve`)
* Node.js / NPM / `npx` for common MCP servers

## Quick Installation & Launch
1. Clone the project locally:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd PrivateAI
   ```
2. Set up the local runtime environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Establish a local system environment runtime configuration:
   ```bash
   echo "TELEGRAM_TOKEN=your_token_here" > .env
   echo "ALLOWED_USER_ID=your_id_here" >> .env
   ```
4. Configure MCP tools (local file, gitignored):
   ```bash
   cp mcp.json.example mcp.json
   # Edit the allowed directory path in mcp.json before enabling filesystem tools.
   ```
5. Run under `tmux` or `systemd`:
   ```bash
   ollama serve
   python -m src
   ```

## MCP notes
* `mcp.json` is intentionally gitignored — treat it like a secret/capability boundary.
* Start with a **scoped filesystem** root, not `/`.
* `MCP_READONLY=1` (default) hides write/edit/move/create tools from the model.
* Set `MCP_READONLY=0` only when you intentionally want write tools enabled.
* Do not add unrestricted shell MCP servers until you have an approval/allowlist policy.
* If `mcp.json` is missing or empty, the bot still runs in chat-only mode.

## Durable memory

The model itself has no memory between requests. The agent manages memory in
two local, gitignored files per Telegram chat:

* `<chat-id>.jsonl` — append-only user/assistant transcript
* `<chat-id>.summary.json` — compact long-term summary

Each prompt receives the rolling summary plus a limited number of recent turns.
After a configurable number of completed turns, a background Ollama request
updates the summary.

Optional `.env` settings:

```env
MEMORY_DIR=/path/to/privateai/data/sessions
MEMORY_RECENT_TURNS=4
MEMORY_SUMMARIZE_EVERY=6
MEMORY_MAX_CONTEXT_CHARS=12000
```

Memory files may contain private conversation content. Keep `MEMORY_DIR` on
encrypted or appropriately permissioned storage and include it in backups if
the memory must survive server loss.
