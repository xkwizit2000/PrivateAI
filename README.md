# PrivateAI

PrivateAI is a headless personal assistant for Ubuntu Server. Telegram or
[Session](https://getsession.org/) is the remote interface, Ollama provides
local or remote model inference, and Model Context Protocol (MCP) servers
expose scoped tools.

The bot currently supports:

- Configurable chat backend (`CHAT_BACKEND=telegram` or `session`)
- Telegram user-ID or Session ID authorization
- Local or network-accessible Ollama inference
- Ollama tool calling through local stdio MCP servers (and MCP-over-SSE servers such as the memory service)
- Read-only MCP mode by default
- Durable per-chat transcripts and rolling summaries
- Safe response chunking and sanitized error messages

## Architecture

```text
Telegram app              Session app (getsession.org)
      │                            │
      ▼                            ▼
Telegram Bot API          Session.js adapter (Bun)
      │                            │ HTTP POST /v1/message
      └────────────┬───────────────┘
                   ▼
PrivateAI bot / agent host
  ├── src/bot.py / bridge.py   front-end adapters
  ├── src/agent.py             Ollama reasoning and tool-call loop
  ├── src/mcp_client.py        MCP server lifecycle (stdio + SSE) and tool routing
  ├── src/memory.py            transcript and rolling-summary storage
  └── src/memory_service_client.py  dual-write / recall client for memory-service/
        │                            │
        │ HTTP                       │ stdio
        ▼                            ▼
Ollama model host               Scoped MCP servers
(local or remote GPU)           (filesystem, future tools)
```

The bot and MCP servers can run on a smaller tool host while Ollama runs on a
dedicated GPU server. Your normal interaction with the whole system remains
through Telegram or Session.

### Reference development host

The primary lab box used while building PrivateAI is a
[Beelink SER5 Max](https://www.amazon.com/dp/B0CGRDSMDN) mini PC configured as
a single-machine GPU host:

| Item | Detail |
| --- | --- |
| Hardware | Beelink SER5 Max, 24 GB system RAM |
| GPU | AMD Radeon 680M (iGPU) |
| BIOS | 16 GB allocated to video RAM (UMA / shared graphics) |
| OS | Ubuntu 26.04 |
| Inference | Ollama installed and serving models on this host |
| Chat model | `gemma4:e4b` — smaller than `gemma4:12b` so the Radeon 680M keeps headroom for the RAG embedding model (`nomic-embed-text`) alongside chat |
| PrivateAI | Agent, Telegram bot, and MCP tools run on the same machine |

On this layout, keep `OLLAMA_HOST=http://127.0.0.1:11434` (or the Docker
`host.docker.internal` / LAN equivalent if the bot runs in Compose). Splitting
the bot onto a separate host is optional; co-locating everything on the SER5
is the simplest development path. Larger chat tags can starve VRAM when chat
and embeddings are both loaded.

## Requirements

- Python 3.10+
- Ubuntu Server 24.04+ recommended
- Ollama with a tool-capable model, such as `gemma4:e4b` or `gemma4:12b`
- Node.js, npm, and `npx` for the example filesystem MCP server
- For Telegram: a bot token and your numeric Telegram user ID
- For Session: [Bun](https://bun.sh/) 1.x, a Session mnemonic for the bot identity,
  and your personal Session ID
## Installation

```bash
git clone https://github.com/xkwizit2000/PrivateAI.git
cd PrivateAI

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Telegram setup

1. Open the official Telegram bot `@BotFather`.
2. Send `/newbot` and follow its prompts.
3. Copy the generated HTTP API token.
4. Open `@userinfobot`, send `/start`, and copy your numeric user ID.

Do not use your Telegram username for `ALLOWED_USER_ID`; the application
compares numeric IDs.

Set `CHAT_BACKEND=telegram` (the default) in `.env`.

## Session setup

[Session](https://getsession.org/) has no official Bot API. PrivateAI uses a
low-risk [Session.js](https://sessionjs.github.io/docs/) adapter that speaks
normal Session DMs and calls the Python agent over a local HTTP bridge.

1. Install the Session app and note **your** Session ID (Account ID).
2. Create a dedicated Session identity for the bot (or reuse a spare mnemonic).
   Put that recovery phrase in `SESSION_MNEMONIC` — treat it like a password.
3. Set `SESSION_ALLOWED_ID` to your personal Session ID (comma-separated if
   you need more than one).
4. Switch the Python process to the bridge:

```env
CHAT_BACKEND=session
BRIDGE_HOST=127.0.0.1
BRIDGE_PORT=8787
AGENT_BRIDGE_URL=http://127.0.0.1:8787
# Optional shared secret checked by the bridge:
# BRIDGE_TOKEN=some-long-random-string
```

5. Run both processes:

```bash
# terminal 1 — agent + MCP + memory/RAG bridge
source .venv/bin/activate
CHAT_BACKEND=session python -m src

# terminal 2 — Session.js DM adapter (requires Bun)
cd session-adapter
bun install
bun start
```

6. In Session, message the bot’s Session ID (printed at adapter startup).

Slash commands (`/status`, `/think`, `/verbose`, `/timeout`) work the same as
on Telegram. The adapter stores Sync/poll state under
`data/session-adapter/storage.db` so restarts do not re-process old DMs.

## Environment configuration

Copy the complete example and replace its placeholder values:

```bash
cp .env.example .env
```

Required variables:

```env
TELEGRAM_TOKEN=your_botfather_token
ALLOWED_USER_ID=your_numeric_telegram_id
```

Available variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHAT_BACKEND` | `telegram` | Front end: `telegram` or `session` |
| `TELEGRAM_TOKEN` | none | Telegram Bot API token; required for Telegram |
| `ALLOWED_USER_ID` | `0` | Only Telegram user permitted to use the bot |
| `SESSION_MNEMONIC` | none | Bot Session recovery phrase; required for Session adapter |
| `SESSION_DISPLAY_NAME` | `PrivateAI` | Display name for the Session bot identity |
| `SESSION_ALLOWED_ID` | none | Comma-separated Session IDs allowed to chat |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `127.0.0.1` / `8787` | Python HTTP bridge bind address |
| `BRIDGE_TOKEN` | empty | Optional Bearer token for bridge requests |
| `AGENT_BRIDGE_URL` | `http://127.0.0.1:8787` | URL the Session.js adapter calls |
| `SESSION_MESSAGE_CHUNK` | `1800` | Max characters per Session outbound message (under 2000 limit) |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Ollama model tag |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local or remote Ollama API |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Hard timeout per Ollama chat call |
| `OLLAMA_STUCK_WARN_SECONDS` | `30` | Log a stuck warning while still waiting |
| `OLLAMA_THINK` | `0` | Default model thinking/reasoning (`1`/`0`; overridable with `/think`) |
| `OLLAMA_NUM_PREDICT` | `-1` | Max tokens to generate per chat call (`-1` = uncapped) |
| `OLLAMA_NUM_CTX` | empty | Optional context window override (tokens); empty keeps model default |
| `OLLAMA_MAX_CONTINUATIONS` | `2` | Auto-continue when Ollama stops with `done_reason=length` |
| `OLLAMA_RETRY_WAIT_SECONDS` | `90` | After a connection-type failure (e.g. Ollama restarting), wait up to this long for `/api/version` before retrying the chat call once |
| `OLLAMA_RETRY_POLL_SECONDS` | `2` | Poll interval while waiting for Ollama to come back |
| `MCP_CONFIG_PATH` | `<project>/mcp.json` | Local MCP server configuration |
| `MAX_TOOL_ITERATIONS` | `8` | Maximum model/tool loops per request |
| `MCP_READONLY` | `1` | Hide tools whose names appear to modify data |
| `MCP_TOOL_ALLOWLIST` | empty (= built-in 9-tool default) | Comma-separated tools exposed to the model; registered (`filesystem_read_text_file`) or server-local (`read_text_file`) names, case-insensitive; `all`/`*` = everything. Applied on top of `MCP_READONLY` |
| `MCP_TOOL_DESC_MAX_CHARS` | `200` | Max characters of each tool description sent to Ollama (`0` = unlimited) |
| `MCP_PARAM_DESC_MAX_CHARS` | `80` | Max characters of each parameter description in tool schemas (`0` = unlimited) |
| `TOOL_ROUTING` | `auto` | `auto`: send tool schemas only when the message looks like it needs file access; `always`: every request; `never`: no tools. Overridable per chat with `/tools` |
| `TOOL_ROUTING_FOLLOWUP_SECONDS` | `600` | In `auto` mode, follow-up messages keep tools on for this long after a turn that needed files or called a tool |
| `MEMORY_DIR` | `<project>/data/sessions` | Durable per-chat memory directory |
| `MEMORY_RECENT_TURNS` | `4` | Recent turns inserted into each model prompt |
| `MEMORY_SUMMARIZE_EVERY` | `6` | Completed turns between summary updates |
| `MEMORY_MAX_CONTEXT_CHARS` | `12000` | Soft memory-text budget per prompt |
| `MEMORY_BACKEND` | `legacy` | `legacy` = rolling summary + RAG (today); `service` = core-only context from the memory service (fact profile + last N turns, no summary/RAG) |
| `MEMORY_SERVICE_URL` | empty (compose: `http://memory-service:8765`) | Memory service base URL; empty disables the whole integration |
| `MEMORY_TOKEN` | empty | Shared bearer token with the memory-service container (also expanded for `${MEMORY_TOKEN}` in `mcp.json` headers) |
| `MEMORY_RECALL` | `shadow` | `off`; `shadow` = per-turn recall query in the background, metadata-only log line; `on` = inject the gated recall block as a system message (≤700 chars) |
| `MEMORY_TOOLS` | `off` | `on` = expose the memory MCP tools (server named `memory` in `mcp.json`) to the model; hidden when `off` |
| `MEMORY_DUAL_WRITE` | `1` | Send a copy of every completed turn to the service (always on with `MEMORY_BACKEND=service`) |
| `MEMORY_RECALL_TIMEOUT` / `MEMORY_APPEND_TIMEOUT` / `MEMORY_TOOL_TIMEOUT` | `2.5` / `3` / `5` | Seconds before a recall, append or memory tool call gives up |
| `MEMORY_BREAKER_FAILURES` / `MEMORY_BREAKER_COOLDOWN` | `3` / `30` | Circuit breaker: after this many consecutive failures, stop calling the service for the cooldown (doubles up to `MEMORY_BREAKER_MAX_COOLDOWN`=300 s) |
| `MEMORY_RECALL_MAX_CHARS` | `700` | Hard cap of the recall block injected in `on` mode |
| `RAG_ENABLED` | `1` | Retrieve older relevant turns via embeddings |
| `RAG_DIR` | `<project>/data/rag` | Per-session SQLite vector store directory |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | Embedding model on `OLLAMA_HOST` |
| `RAG_EMBED_NUM_GPU` | `0` | GPU layers for the embedding model (`0` = CPU only, `-1`/empty = Ollama decides) |
| `RAG_TOP_K` | `4` | Max retrieved chunks per prompt |
| `RAG_MAX_CONTEXT_CHARS` | `4000` | Soft character budget for retrieved snippets |
| `RAG_MIN_SCORE` | `0.25` | Drop weak cosine similarity matches |

The `.env` file is gitignored and must never be committed.

## Ollama setup

### Ollama on the same machine

```bash
ollama pull gemma4:12b
ollama pull nomic-embed-text
ollama serve
```

Keep the default:

```env
OLLAMA_HOST=http://127.0.0.1:11434
```

Each successful chat call logs performance stats (`prompt_tok/s`, `gen_tok/s`,
durations). While a call is in flight, the bot logs a stuck warning every
`OLLAMA_STUCK_WARN_SECONDS` (default 30s). If the call exceeds
`OLLAMA_TIMEOUT_SECONDS` (default 180s), it fails with a clear timeout error
instead of hanging indefinitely.

Gemma 4 and similar models may "think" before answering, which is slower.
PrivateAI defaults to `OLLAMA_THINK=0` so reasoning mode is off. Set
`OLLAMA_THINK=1` only when you want that behavior as the default.

To avoid truncated replies, PrivateAI sends `num_predict=-1` by default and
auto-continues when Ollama stops with `done_reason=length` (see
`OLLAMA_MAX_CONTINUATIONS`). On Session, long replies are split under the
2000-character client limit via `SESSION_MESSAGE_CHUNK` (default 1800).

Per Telegram chat, override it without restarting:

```text
/status
/think status
/think on
/think off
/verbose status
/verbose on
/verbose off
/timeout status
/timeout 600
/timeout reset
/tools status
/tools auto
/tools on
/tools off
/tools reset
```

`/tools` sets tool routing for the chat: `auto` (default, from `TOOL_ROUTING`)
sends file tools only when the message looks like it needs them, `on` always
sends them, `off` never does.

`/status` lists every per-chat setting (think, verbose, timeout, tools, and any
future ones), plus Ollama host/chat model and currently loaded models from
`GET /api/ps`. Each of `/think`, `/verbose`, and `/timeout` also accepts
`status` (or no args) for that setting alone.

With `/think on` and `/verbose on`, model reasoning streams into the temporary
status message. With `/verbose off`, status stays as the simple progress
indicator.

When enabling `/think on`, raise `/timeout` for that chat so deep reasoning
has enough time. Per-chat choices are stored in `data/session_settings.json`.

### Ollama on a dedicated GPU server

On the GPU server, expose Ollama only on a trusted private network:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull gemma4:12b
ollama pull nomic-embed-text
```

On the PrivateAI bot/tool host:

```env
OLLAMA_HOST=http://gpu-server.internal:11434
```

Use a firewall, WireGuard, Tailscale, or an SSH tunnel. Ollama does not provide
an authentication boundary suitable for direct public Internet exposure.

## MCP tool configuration

Copy the example configuration:

```bash
cp mcp.json.example mcp.json
```

Edit its allowed filesystem root:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/absolute/path/the/agent/may/access"
      ]
    }
  }
}
```

At startup, PrivateAI launches each configured MCP server over stdio, discovers
its tools, converts their schemas for Ollama, and routes model tool calls back
to the correct server.

Security notes:

- `mcp.json` is gitignored because it defines the agent's capabilities.
- Use a narrowly scoped directory, never `/`, as the filesystem root.
- `MCP_READONLY=1` is the default and hides tools with write-like names.
- The read-only filter is a safety layer, not a security sandbox.
- Every exposed tool schema is sent to the model on every request, which costs
  prompt-processing time on small GPUs. `MCP_TOOL_ALLOWLIST` limits which tools
  the model sees (default: `read_text_file`, `read_multiple_files`, `write_file`,
  `edit_file`, `create_directory`, `list_directory`, `directory_tree`,
  `move_file`, `search_files`; use `all` for every tool) and
  `MCP_TOOL_DESC_MAX_CHARS` / `MCP_PARAM_DESC_MAX_CHARS` shorten the descriptions
  (names, types, `required` and `enum` are never changed). A call to a hidden
  tool is rejected with an error message. The exposed tools are logged at startup.
- With `TOOL_ROUTING=auto` (default) the schemas are only sent when the user's
  message matches a cheap keyword/path/file-extension check (or follows such a
  turn within `TOOL_ROUTING_FOLLOWUP_SECONDS`); every request logs
  `tool routing: tools=on|off reason=...`. Plain chat then carries no tool
  schemas at all. Use `/tools on` (or `TOOL_ROUTING=always`) if a file request
  is ever missed.
- Do not add unrestricted shell tools without command allowlists and explicit
  user approval.
- An empty or missing `mcp.json` leaves the bot in chat-only mode.
- The MCP client supports local stdio servers and MCP-over-SSE servers
  (`{"url": "...", "headers": {...}}`, header values may use `${ENV_VAR}`). SSE
  calls have a short timeout and one reconnect, return errors as text, and an
  unreachable SSE server only logs a warning at startup. Streamable-HTTP is not
  available in `mcp==1.3.0`.

## Durable memory

Ollama models do not remember previous API requests. PrivateAI therefore keeps
memory in the application layer.

For every completed turn, the agent immediately appends:

```text
data/sessions/<telegram-chat-id>.jsonl
```

After `MEMORY_SUMMARIZE_EVERY` completed turns, a background Ollama request
updates:

```text
data/sessions/<telegram-chat-id>.summary.json
```

Each new prompt receives:

1. The system instructions
2. The compact rolling summary
3. Relevant older turns retrieved by conversation RAG (when enabled)
4. A limited number of recent user/assistant turns
5. The current Telegram message

This gives a small-context model durable continuity without loading the entire
conversation into VRAM. The character budget is an approximation rather than
an exact tokenizer-based context limit.

## Conversation RAG

PrivateAI embeds past conversation turns with an Ollama embedding model (default
`nomic-embed-text` on the same `OLLAMA_HOST` as the chat model) and stores
vectors in per-session SQLite files:

```text
data/rag/<telegram-chat-id>.sqlite3
```

On each request the agent embeds the user message, ranks earlier chunks by
cosine similarity, and injects the top matches (excluding turns already present
in the recent window). After each reply, new turns are indexed in the
background. On startup, existing `data/sessions/*.jsonl` transcripts are
backfilled into the RAG store.

Pull the embedding model on the GPU/Ollama host:

```bash
ollama pull nomic-embed-text
```

Optional `.env` settings:

```env
RAG_ENABLED=1
RAG_DIR=/path/to/privateai/data/rag
RAG_EMBED_MODEL=nomic-embed-text
RAG_EMBED_NUM_GPU=0
RAG_TOP_K=4
RAG_MAX_CONTEXT_CHARS=4000
RAG_MIN_SCORE=0.25
```

Embedding calls pass the Ollama option `num_gpu: 0` by default
(`RAG_EMBED_NUM_GPU=0`), so `nomic-embed-text` runs on the CPU and never loads a
second runner on the (integrated) GPU next to the chat model. Set
`RAG_EMBED_NUM_GPU=-1` (or leave it empty) to let Ollama choose. Note that a
model already resident with different options is reloaded on the next request,
so run `ollama stop nomic-embed-text` once if it stays on the GPU after
changing this setting.

Set `RAG_ENABLED=0` to disable retrieval and indexing. Document/file corpus
indexing is not implemented yet — RAG currently covers conversation history
only.

Memory and RAG files are gitignored, permissioned for the local user, and may
contain private conversation content. Store them on protected storage and back
up `MEMORY_DIR` and `RAG_DIR` if they must survive host failure.

### Ollama restarts and automatic retry

If a chat call fails because Ollama dropped the connection or returned an HTTP
5xx (typically while Ollama is being restarted), the agent waits for
`GET /api/version` to answer again (every `OLLAMA_RETRY_POLL_SECONDS`, up to
`OLLAMA_RETRY_WAIT_SECONDS`) and retries that one model call **once** per
request. Timeouts, 4xx errors and tool errors are not retried, and tools that
already ran are never re-executed. If the retry fails too, the bot replies
`System Error: Ollama was unavailable or restarted while answering. Please
resend your message in a minute.` (the failed turn is not stored in memory).

## Memory service (shadow mode)

`memory-service/` is an external long-term memory (SQLite + FTS5 + CPU embeddings,
no Ollama calls; design and measurements in
[`docs/memory-service-design.md`](docs/memory-service-design.md)). The bot integrates
with it in stages, all behind flags; with the defaults below **replies, prompts and
tools are exactly what they were without it**:

| Setting | Shadow rollout | Effect |
| --- | --- | --- |
| `MEMORY_BACKEND` | `legacy` | rolling summary + RAG unchanged |
| `MEMORY_DUAL_WRITE` | `1` | every completed turn is also POSTed to the service (fire-and-forget; failed/retried turns are saved nowhere, same as the JSONL) |
| `MEMORY_RECALL` | `shadow` | one `/v1/recall` per turn in the background (never awaited before the model call); result is only logged |
| `MEMORY_TOOLS` | `off` | memory tools are not visible to the model |

A down or slow service costs nothing: calls are bounded by 2.5–3 s timeouts, run
outside the reply path, and a circuit breaker stops calling it after 3 failures
(retrying after 30 s, doubling to 5 min). The bot does not depend on the service
container to start. The service gets no published port; it shares the compose network
and the `MEMORY_TOKEN` secret with the bot.

### Reading the shadow stats

Every turn logs one metadata-only line (never message text):

```text
memory shadow: gate=fired|skipped hits=N top=0.xx chars=N ms=N why=cue|strong|none|empty
memory dual-write: turns_new=2 chunks_new=2 ms=9
```

```bash
docker logs privateai 2>&1 | grep 'memory shadow' | tail            # recent decisions
docker logs privateai 2>&1 | grep -c 'memory shadow: gate=fired'     # how often it would inject
docker logs privateai 2>&1 | grep -E 'memory service (call failed|unavailable|recovered)'
docker exec privateai-memory python -m memory_service stats           # archive size
```

In chat, `/memory` shows reachability, archive counts, breaker state and the
recall/dual-write counters since the last bot start, and `/recall <query>` runs one
explicit search of the archive (capped text).

### Deploy / backfill

```bash
echo "MEMORY_TOKEN=$(openssl rand -hex 24)" >> .env      # never print or commit it
sudo install -d -o 10002 -g 10002 -m 700 memory-data       # DB directory (uid of the image)
docker compose build
docker compose up -d memory-service                         # wait until `docker ps` says healthy
# backfill the archive from the existing transcripts (read-only mount, over HTTP so the
# running service indexes them); output is counts only:
docker run --rm --network privateai_default --user 10001 --entrypoint "" \
  -v "$PWD/data/sessions:/in:ro" -v "$PWD/memory-service/integration:/it:ro" \
  -e MEMORY_TOKEN -e MEM_URL=http://memory-service:8765 \
  privateai:latest python /it/backfill_http.py /in
docker compose up -d                                         # recreates the bot (brief restart)
```

(`MEMORY_TOKEN` must be exported in the shell for the `docker run` line, e.g.
`set -a; . ./.env; set +a`, or read from `.env` without echoing it.)

### Enabling later

* **Inject recalled snippets** (`on`): `MEMORY_RECALL=on` (keep `MEMORY_BACKEND=legacy`
  to add the block next to the legacy RAG, or `MEMORY_BACKEND=service` to replace
  summary + RAG by the core-only context). Restart: `docker compose up -d privateai`.
* **Memory tools for the model**: add the `memory` server from `mcp.json.example` to
  `mcp.json` (URL `http://memory-service:8765/sse`, header
  `Authorization: Bearer ${MEMORY_TOKEN}`) and set `MEMORY_TOOLS=on`. They are then
  offered only for messages with memory cues ("remember", "earlier", "last time",
  "we discussed", "yesterday", …) and for a follow-up window, independently of the
  file tools; `/tools off` still disables all tools for a chat.

Put these in `.env` (compose passes them through with defaults).

### Rollback

Nothing in the legacy files (`data/sessions/*.jsonl`, `*.summary.json`, `data/rag`) is
ever modified by the service, and the bot keeps writing its JSONL, so rollback is safe at
any time:

* **Soft (no code change)**: put `MEMORY_RECALL=off` and `MEMORY_DUAL_WRITE=0` in `.env`,
  then `docker compose up -d privateai` (and optionally `docker compose stop memory-service`).
* **Full (back to `main`)**:

  ```bash
  cd ~/PrivateAI
  docker compose rm -sf memory-service            # stop + remove the container (compose file of this branch)
  sudo mv memory-data ~/backups/memory-data-$(date -u +%Y%m%dT%H%M%SZ)   # REQUIRED, see note
  git checkout main
  docker compose up -d --build --remove-orphans   # rebuilds the bot from main; drops any leftover orphan
  ```

  **Why the `mv` is required:** `memory-data/` is owned by uid 10002 (mode 700). `main`'s
  `.dockerignore` does not exclude it, so with the directory still present
  `docker compose up --build` fails with `checking context: can't stat '.../memory-data'`
  (verified on pai01 with a scratch context). Moving it aside also keeps the archive and
  keeps `git status` on `main` clean.

  Optional clean-up: `docker image rm privateai-memory:latest`, deleting the moved archive
  (`sudo rm -rf ~/backups/memory-data-*`; the JSONL stays the system of record), and
  restoring the previous `.env` from the backup you made before the rollout
  (`cp /home/<user>/backups/env-<timestamp>.bak .env`) or just deleting the
  `MEMORY_TOKEN=` line (it is ignored by `main`).

## Running PrivateAI

Activate the environment and start the selected front end:

```bash
source .venv/bin/activate
# CHAT_BACKEND=telegram (default) → Telegram polling
# CHAT_BACKEND=session → local HTTP bridge for the Session.js adapter
python -m src
```

If `CHAT_BACKEND=session`, also start `session-adapter` (see **Session setup**).

If Ollama is local, ensure `ollama serve` is already running. For persistent
operation, run PrivateAI under `systemd`, a container supervisor, or `tmux`.

On startup, logs report:

- Whether durable memory is enabled
- Whether conversation RAG is enabled
- Which MCP tools were loaded
- Whether the bot is running in chat-only mode
- Telegram polling status, or Session bridge bind address

## Docker

The image bundles Python and Node.js (for `npx`-based MCP servers). Secrets and
tool configuration are provided at runtime rather than baked into the image.

### Docker Compose (recommended)

Telegram (default):

```bash
cp .env.example .env      # fill in TELEGRAM_TOKEN and ALLOWED_USER_ID
cp mcp.json.example mcp.json
docker compose up -d --build
docker compose logs -f
```

Session (Python bridge + Session.js adapter):

```bash
# In .env: CHAT_BACKEND=session, SESSION_MNEMONIC, SESSION_ALLOWED_ID, …
docker compose --profile session up -d --build
docker compose logs -f
```

Compose mounts `mcp.json` read-only, persists `./data` for memory/RAG/settings,
and defaults `OLLAMA_HOST` for a LAN/GPU host. Override as needed:

```bash
OLLAMA_HOST=http://gpu-server.internal:11434 docker compose up -d
```

### Plain Docker

```bash
docker build -t privateai:latest .
docker run -d --name privateai \
  --env-file .env \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  -v "$PWD/mcp.json:/app/mcp.json:ro" \
  -v privateai-data:/data \
  privateai:latest
```

### Container notes

- `MCP_CONFIG_PATH` defaults to `/app/mcp.json`, `MEMORY_DIR` to
  `/data/sessions`, and `RAG_DIR` to `/data/rag` inside the image.
- A filesystem MCP root must be a path that exists **inside** the container.
  Mount the host directory (e.g. `-v /srv/work:/srv/work`) and use that path in
  `mcp.json`.
- The container runs as a non-root user; the `/data` volume keeps memory and
  RAG indexes across restarts.

## Testing

Run the current unit tests:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The tests cover transcript persistence, recent-turn loading, summary thresholds,
session-path validation, conversation RAG, tool routing, the MCP SSE transport
(against an in-process fake SSE server), and the memory-service client (timeouts,
circuit breaker, dual-write, shadow/`on` recall, per-group tool routing). The memory
service has its own suite: `docker build --target test -t memsvc-test memory-service`.

## Project layout

```text
PrivateAI/
├── .env.example          complete environment template
├── mcp.json.example      scoped MCP server template
├── Dockerfile            container image (Python + Node for MCP)
├── docker-compose.yml    compose services (Telegram or Session profile)
├── requirements.txt
├── memory-service/       external memory service (own Dockerfile, tests, README)
├── session-adapter/      Bun + Session.js DM front end
├── src/
│   ├── __main__.py       `python -m src` entry (telegram|session)
│   ├── agent.py          Ollama agent/tool loop
│   ├── bot.py            Telegram application
│   ├── bridge.py         HTTP bridge for Session.js
│   ├── commands.py       shared slash commands
│   ├── config.py         environment configuration
│   ├── mcp_client.py     MCP client hub
│   ├── memory.py         durable session memory
│   ├── memory_service_client.py  memory-service client (dual-write, shadow/on recall)
│   ├── rag.py            conversation RAG store
│   └── runtime.py        shared agent/MCP/memory bootstrap
└── tests/
```

## Current limitations

- Telegram authorizes a single user ID; Session authorizes an allowlist of IDs.
- Session support uses community Session.js (not an official Bot API).
- MCP connections use local stdio or MCP-over-SSE (no streamable-HTTP with `mcp==1.3.0`).
- There is no confirmation workflow for write-capable tools.
- RAG indexes conversation turns only, not arbitrary workspace documents.
- There is no packaged `systemd` unit yet.