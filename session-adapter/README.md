# Session.js adapter

Bun + [Session.js](https://sessionjs.github.io/docs/) front end for PrivateAI.
It polls Session DMs, forwards text to the Python agent bridge
(`POST /v1/message`), and sends the reply back.

Requires `CHAT_BACKEND=session` on the Python process and Bun 1.x.

```bash
# terminal 1 — Python bridge
CHAT_BACKEND=session python -m src

# terminal 2 — Session adapter
cd session-adapter
bun install
bun start
```

See the root README for `.env` settings (`SESSION_MNEMONIC`,
`SESSION_ALLOWED_ID`, `AGENT_BRIDGE_URL`, etc.).
