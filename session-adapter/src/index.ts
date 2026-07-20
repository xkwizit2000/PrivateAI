import { mkdirSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { config as loadEnv } from "dotenv"
import { FileKeyvalStorage } from "@session.js/file-keyval-storage"
import { Poller, Session, ready } from "@session.js/client"

loadEnv({ path: resolve(import.meta.dir, "../../.env") })
await ready

type IncomingMessage = {
  type: "private" | "group" | string
  from: string
  text?: string
  getReplyToMessage: () =>
    | {
        timestamp: number
        author: string
        text?: string
      }
    | undefined
}

const mnemonic = (process.env.SESSION_MNEMONIC || "").trim()
const displayName =
  (process.env.SESSION_DISPLAY_NAME || "PrivateAI").trim() || "PrivateAI"
const bridgeUrl = (
  process.env.AGENT_BRIDGE_URL || "http://127.0.0.1:8787"
).replace(/\/$/, "")
const bridgeToken = (process.env.BRIDGE_TOKEN || "").trim()
const allowedIds = (process.env.SESSION_ALLOWED_ID || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean)
const storagePath = resolve(
  process.env.SESSION_STORAGE_PATH ||
    resolve(import.meta.dir, "../../data/session-adapter/storage.db"),
)
const chunkSize = Number(process.env.SESSION_MESSAGE_CHUNK || "2000")

if (!mnemonic) {
  throw new Error("Missing SESSION_MNEMONIC environment variable.")
}
if (allowedIds.length === 0) {
  console.warn(
    "SESSION_ALLOWED_ID is empty; ignoring all inbound messages until configured.",
  )
}

mkdirSync(dirname(storagePath), { recursive: true })

const session = new Session({
  storage: new FileKeyvalStorage({ filePath: storagePath }),
})
session.setMnemonic(mnemonic, displayName)
console.log(`Session bot online as ${session.getSessionID()} (${displayName})`)
console.log(`Agent bridge: ${bridgeUrl}`)

const inflight = new Set<string>()

function chunkText(text: string, limit: number): string[] {
  if (text.length <= limit) return [text]
  const parts: string[] = []
  for (let i = 0; i < text.length; i += limit) {
    parts.push(text.slice(i, i + limit))
  }
  return parts
}

async function callBridge(sessionId: string, text: string): Promise<string> {
  const headers: Record<string, string> = {
    "content-type": "application/json",
  }
  if (bridgeToken) {
    headers.authorization = `Bearer ${bridgeToken}`
  }

  const response = await fetch(`${bridgeUrl}/v1/message`, {
    method: "POST",
    headers,
    body: JSON.stringify({ session_id: sessionId, text }),
  })

  const body = (await response.json().catch(() => ({}))) as {
    reply?: string
    error?: string
  }

  if (!response.ok) {
    throw new Error(body.error || `bridge HTTP ${response.status}`)
  }
  if (!body.reply) {
    throw new Error("bridge returned an empty reply")
  }
  return body.reply
}

async function replyAll(
  to: string,
  text: string,
  source?: IncomingMessage,
): Promise<void> {
  const limit = Number.isFinite(chunkSize) ? chunkSize : 2000
  const parts = chunkText(text, limit)
  for (const [index, part] of parts.entries()) {
    await session.sendMessage({
      to,
      text: part,
      replyToMessage:
        index === 0 && source ? source.getReplyToMessage() : undefined,
    })
  }
}

session.on("message", async (msg: IncomingMessage) => {
  if (msg.type !== "private") {
    return
  }
  const from = msg.from
  const text = (msg.text || "").trim()
  if (!text) {
    return
  }
  if (allowedIds.length === 0 || !allowedIds.includes(from)) {
    console.warn(`Ignoring unauthorized Session ID ${from}`)
    await session.sendMessage({
      to: from,
      text: "Access Denied: Unauthorized.",
      replyToMessage: msg.getReplyToMessage(),
    })
    return
  }
  if (inflight.has(from)) {
    await session.sendMessage({
      to: from,
      text: "Still working on your previous message — please wait.",
      replyToMessage: msg.getReplyToMessage(),
    })
    return
  }

  inflight.add(from)
  try {
    await session.sendMessage({
      to: from,
      text: "Thinking…",
      replyToMessage: msg.getReplyToMessage(),
    })
    const reply = await callBridge(from, text)
    await replyAll(from, reply, msg)
  } catch (error) {
    console.error("Failed to handle Session message", error)
    await session.sendMessage({
      to: from,
      text: "System Error: the agent request failed. Check server logs for details.",
      replyToMessage: msg.getReplyToMessage(),
    })
  } finally {
    inflight.delete(from)
  }
})

session.addPoller(new Poller())
console.log("Polling Session DMs…")
