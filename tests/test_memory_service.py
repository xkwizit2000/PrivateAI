"""ServiceMemory client, dual-write, shadow/on recall, core-only backend and commands."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from src.agent import OllamaUnavailableError, _remember_response, run_agent
from src.commands import ParsedCommand, handle_command, parse_slash_command
from src.memory import MemoryStore
from src.memory_service_client import (
    CoreMemoryStore,
    RecallResult,
    ServiceMemory,
    format_recall_block,
)

ROOT = Path(__file__).resolve().parent.parent
SECRET_USER = "PRIVATE-USER-WORDS do you remember what we discussed earlier?"
SECRET_HIT = "PRIVATE-HIT-TEXT the earlier answer about the car"


class FakeService:
    """In-process stand-in for the memory service reachable through httpx.MockTransport."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.appended: list[dict] = []
        self.recall_payloads: list[dict] = []
        self.fail = False
        self.delay = 0.0
        self.recall_body: dict = {
            "inject": True,
            "reason": "past-reference cue + cos=0.61",
            "text": f"[t5 2026-08-04 user] {SECRET_HIT}\n[t6 2026-08-04 assistant] more",
            "top_cos": 0.61,
            "hits": [{"turn_id": 5}, {"turn_id": 6}],
            "ms": 4.0,
        }
        self.core_body: dict = {"profile": "- likes concise answers", "turns": []}
        self.transport = httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            return httpx.Response(500, json={"error": "boom"})
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path == "/v1/turns":
            self.appended.append(body)
            return httpx.Response(200, json={"turns_new": 2, "chunks_new": 2})
        if path == "/v1/recall":
            self.recall_payloads.append(body)
            return httpx.Response(200, json=self.recall_body)
        if path == "/v1/search":
            return httpx.Response(200, json={"text": "[t1 2026-01-01 user] found it", "hits": []})
        if path == "/v1/stats":
            return httpx.Response(
                200, json={"turns": 228, "chunks": 351, "sessions": 2, "days": 17, "facts_active": 0}
            )
        if path == "/v1/core":
            return httpx.Response(200, json=self.core_body)
        return httpx.Response(404, json={})

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _client(fake: FakeService, **kwargs) -> ServiceMemory:
    kwargs.setdefault("token", "tok")
    token = kwargs.pop("token")
    return ServiceMemory("http://memory-service:8765", token, transport=fake.transport, **kwargs)


def _reply(text="ok", tool_calls=None):
    return SimpleNamespace(
        done_reason="stop", message=SimpleNamespace(content=text, tool_calls=tool_calls)
    )


def _chat(*replies):
    """AsyncMock model call that snapshots the messages it was sent (the agent mutates them later)."""
    queue = list(replies) or [_reply("the answer")]
    snapshots: list[list[dict]] = []

    async def fake(*_args, **kwargs):
        snapshots.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return queue.pop(0)

    mock = AsyncMock(side_effect=fake)
    mock.snapshots = snapshots
    return mock


async def _drain(svc: ServiceMemory) -> None:
    if svc._tasks:
        await asyncio.gather(*list(svc._tasks), return_exceptions=True)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeService()

    async def test_append_sends_auth_header_and_records(self):
        svc = _client(self.fake)
        ok = await svc.append_turn("42", "hi", "hello", "2026-11-12T00:00:00+00:00")
        self.assertTrue(ok)
        request = self.fake.requests[0]
        self.assertEqual(request.headers["authorization"], "Bearer tok")
        self.assertEqual(request.url.host, "memory-service")
        self.assertEqual(
            self.fake.appended[0],
            {
                "session_id": "42",
                "turns": [
                    {"role": "user", "content": "hi", "timestamp": "2026-11-12T00:00:00+00:00"},
                    {"role": "assistant", "content": "hello", "timestamp": "2026-11-12T00:00:00+00:00"},
                ],
            },
        )
        self.assertEqual(svc.counters["append_ok"], 1)
        await svc.close()

    async def test_append_timeout_is_bounded_and_swallowed(self):
        self.fake.delay = 3.0
        svc = _client(self.fake, append_timeout=0.1)
        started = time.monotonic()
        with self.assertLogs("src.memory_service_client", level="WARNING") as logs:
            ok = await svc.append_turn("42", "hi", "hello")
        self.assertFalse(ok)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(svc.counters["append_fail"], 1)
        self.assertTrue(any("op=append" in line for line in logs.output))
        await svc.close()

    async def test_http_error_counts_as_failure(self):
        self.fake.fail = True
        svc = _client(self.fake)
        with self.assertLogs("src.memory_service_client", level="WARNING") as logs:
            self.assertIsNone(await svc.recall("q", "42"))
        self.assertTrue(any("HTTP 500" in line for line in logs.output))
        await svc.close()

    async def test_breaker_opens_skips_calls_then_probes_and_recovers(self):
        clock = Clock()
        self.fake.fail = True
        svc = _client(self.fake, breaker_failures=3, breaker_cooldown=30, clock=clock)
        with self.assertLogs("src.memory_service_client", level="INFO") as logs:
            for _ in range(3):
                self.assertFalse(await svc.append_turn("42", "a", "b"))
            self.assertEqual(svc.breaker_state, "open")
            calls = len(self.fake.requests)
            for _ in range(5):  # breaker open: no network at all
                self.assertFalse(await svc.append_turn("42", "a", "b"))
                self.assertIsNone(await svc.recall("q", "42"))
            self.assertEqual(len(self.fake.requests), calls)
            self.assertGreaterEqual(svc.counters["skipped_open"], 10)

            clock.now += 31  # half-open: exactly one probe goes out; it fails -> longer cooldown
            self.assertFalse(await svc.append_turn("42", "a", "b"))
            self.assertEqual(len(self.fake.requests), calls + 1)
            self.assertEqual(svc.breaker_state, "open")
            self.assertAlmostEqual(svc.breaker_retry_in(), 60, delta=1)  # doubled

            clock.now += 61
            self.fake.fail = False
            self.assertTrue(await svc.append_turn("42", "a", "b"))
            self.assertEqual(svc.breaker_state, "closed")
        self.assertTrue(any("pausing calls" in line and "WARNING" in line for line in logs.output))
        self.assertTrue(any("recovered" in line for line in logs.output))
        self.assertFalse(any(line.startswith("ERROR") for line in logs.output))
        await svc.close()

    async def test_only_one_probe_while_half_open(self):
        clock = Clock()
        self.fake.fail = True
        svc = _client(self.fake, breaker_failures=1, breaker_cooldown=10, clock=clock)
        with self.assertLogs("src.memory_service_client", level="WARNING"):
            await svc.append_turn("42", "a", "b")
            clock.now += 11
            self.fake.fail = False
            self.fake.delay = 0.2
            before = len(self.fake.requests)
            results = await asyncio.gather(*(svc.append_turn("42", "a", "b") for _ in range(4)))
        self.assertEqual(results.count(True), 1)
        self.assertEqual(len(self.fake.requests), before + 1)
        await svc.close()

    async def test_recall_result_metadata(self):
        svc = _client(self.fake)
        result = await svc.recall("q", "42")
        assert result is not None
        self.assertTrue(result.fired)
        self.assertEqual((result.hits, result.top, result.why), (2, 0.61, "cue"))
        self.assertEqual(result.chars, len(self.fake.recall_body["text"]))
        self.assertEqual(self.fake.recall_payloads[0]["session"], "42")
        self.assertEqual(self.fake.recall_payloads[0]["max_hits"], 2)
        await svc.close()

    async def test_recall_skipped_gate(self):
        self.fake.recall_body = {
            "inject": False, "reason": "no gate (cue=False, cos=0.58)", "text": "",
            "top_cos": 0.58, "hits": [{"turn_id": 1}],
        }
        svc = _client(self.fake)
        result = await svc.recall("q", "42")
        assert result is not None
        self.assertFalse(result.fired)
        self.assertEqual((result.chars, result.why), (0, "none"))
        self.assertEqual(svc.counters["recall_skipped"], 1)
        await svc.close()

    async def test_stats_and_search_bypass_open_breaker(self):
        clock = Clock()
        self.fake.fail = True
        svc = _client(self.fake, breaker_failures=1, clock=clock)
        with self.assertLogs("src.memory_service_client", level="WARNING"):
            await svc.append_turn("42", "a", "b")
        self.assertEqual(svc.breaker_state, "open")
        self.fake.fail = False
        stats = await svc.stats()  # explicit status checks may heal the breaker
        assert stats is not None
        self.assertEqual(stats["turns"], 228)
        self.assertEqual(svc.breaker_state, "closed")
        self.assertEqual(await svc.search("q"), "[t1 2026-01-01 user] found it")
        await svc.close()

    def test_format_recall_block_is_capped(self):
        block = format_recall_block("x" * 5000, max_chars=300)
        self.assertLessEqual(len(block), 300)
        self.assertTrue(block.startswith("Possibly relevant earlier conversation"))
        self.assertEqual(format_recall_block("   "), "")


class ShadowAndOnModeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeService()

    async def _run(self, svc, text=SECRET_USER, memory=None, rag=None, chat=None):
        chat = chat or _chat()
        with patch("src.agent._chat_with_monitoring", chat):
            reply = await run_agent(
                text, hub=None, session_id="42", memory=memory, rag=rag, memory_service=svc
            )
        return reply, chat

    async def test_shadow_logs_metadata_only_and_never_injects(self):
        svc = _client(self.fake, recall_mode="shadow")
        with self.assertLogs(level="INFO") as logs:
            reply, chat = await self._run(svc)
            await _drain(svc)
        joined = "\n".join(logs.output)
        self.assertEqual(reply, "the answer")
        self.assertRegex(
            joined,
            r"memory shadow: gate=fired hits=2 top=0\.61 chars=\d+ ms=\d+ why=cue",
        )
        self.assertNotIn("PRIVATE", joined)
        sent = json.dumps(chat.snapshots[0])
        self.assertNotIn(SECRET_HIT, sent)  # reply path unchanged: nothing injected
        self.assertEqual([m["role"] for m in chat.snapshots[0]], ["system", "user"])
        self.assertEqual(svc.counters["recall_fired"], 1)
        await svc.close()

    async def test_shadow_skipped_line(self):
        self.fake.recall_body = {
            "inject": False, "reason": "no gate (cue=False, cos=0.58)", "text": "",
            "top_cos": 0.58, "hits": [{"turn_id": 1}, {"turn_id": 2}],
        }
        svc = _client(self.fake, recall_mode="shadow")
        with self.assertLogs("src.memory_service_client", level="INFO") as logs:
            await self._run(svc, "hello")
            await _drain(svc)
        self.assertTrue(
            any("memory shadow: gate=skipped hits=2 top=0.58 chars=0" in line for line in logs.output)
        )
        await svc.close()

    async def test_shadow_does_not_add_latency_even_if_service_hangs(self):
        self.fake.delay = 5.0
        svc = _client(self.fake, recall_mode="shadow", recall_timeout=10, append_timeout=10)
        started = time.monotonic()
        reply, _ = await self._run(svc, memory=MemoryStore(Path(tempfile.mkdtemp())))
        elapsed = time.monotonic() - started
        self.assertEqual(reply, "the answer")
        self.assertLess(elapsed, 0.5)  # recall + dual-write are still in flight
        self.assertTrue(svc._tasks)
        await svc.close()  # cancels the hung background calls

    async def test_shadow_with_dead_service_is_silent_for_the_reply(self):
        self.fake.fail = True
        svc = _client(self.fake, recall_mode="shadow")
        with self.assertLogs("src.memory_service_client", level="INFO") as logs:
            reply, _ = await self._run(svc, memory=MemoryStore(Path(tempfile.mkdtemp())))
            await _drain(svc)
        self.assertEqual(reply, "the answer")
        self.assertFalse(any(line.startswith("ERROR") for line in logs.output))
        await svc.close()

    async def test_recall_off_makes_no_recall_request(self):
        svc = _client(self.fake, recall_mode="off")
        await self._run(svc)
        await _drain(svc)
        self.assertEqual(self.fake.paths(), [])
        await svc.close()

    async def test_on_mode_injects_capped_system_message_before_recent_turns(self):
        memory = MemoryStore(Path(tempfile.mkdtemp()))
        await memory.remember_turn("42", "earlier question", "earlier answer")
        svc = _client(self.fake, recall_mode="on", dual_write=False)
        with self.assertLogs("src.memory_service_client", level="INFO") as logs:
            reply, chat = await self._run(svc, memory=memory)
        messages = chat.snapshots[0]
        self.assertEqual(reply, "the answer")
        self.assertEqual([m["role"] for m in messages], ["system", "system", "user", "assistant", "user"])
        injected = messages[1]["content"]
        self.assertIn(SECRET_HIT, injected)
        self.assertLessEqual(len(injected), 700)
        self.assertTrue(injected.startswith("Possibly relevant earlier conversation"))
        self.assertEqual(messages[-1]["content"], SECRET_USER)
        joined = "\n".join(logs.output)
        self.assertIn("memory recall: gate=fired", joined)
        self.assertNotIn("PRIVATE", joined)
        self.assertEqual(svc.counters["recall_injected"], 1)
        await svc.close()

    async def test_on_mode_skipped_gate_injects_nothing(self):
        self.fake.recall_body = {"inject": False, "reason": "no gate", "text": "", "top_cos": 0.4, "hits": []}
        svc = _client(self.fake, recall_mode="on")
        _, chat = await self._run(svc, "hello")
        self.assertEqual([m["role"] for m in chat.snapshots[0]], ["system", "user"])
        await svc.close()

    async def test_on_mode_service_failure_falls_back_without_delaying_much(self):
        self.fake.delay = 5.0
        svc = _client(self.fake, recall_mode="on", recall_timeout=0.2)
        started = time.monotonic()
        with self.assertLogs("src.memory_service_client", level="WARNING"):
            reply, chat = await self._run(svc)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(reply, "the answer")
        self.assertEqual([m["role"] for m in chat.snapshots[0]], ["system", "user"])
        await svc.close()

    async def test_on_mode_runs_alongside_legacy_rag_message(self):
        rag_message = {"role": "system", "content": "RAG BLOCK"}
        svc = _client(self.fake, recall_mode="on")
        with patch("src.agent._rag_context_message", AsyncMock(return_value=rag_message)):
            _, chat = await self._run(
                svc,
                rag=SimpleNamespace(index_turn=AsyncMock()),
                memory=MemoryStore(Path(tempfile.mkdtemp())),
            )
        roles_contents = [(m["role"], m["content"][:12]) for m in chat.snapshots[0]]
        self.assertEqual(roles_contents[1], ("system", "RAG BLOCK"))
        self.assertEqual(roles_contents[2][0], "system")
        self.assertTrue(roles_contents[2][1].startswith("Possibly rel"))


class DualWriteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = FakeService()
        self.dir = Path(tempfile.mkdtemp())
        self.memory = MemoryStore(self.dir)

    async def test_completed_turn_goes_to_jsonl_and_service_with_same_timestamp(self):
        svc = _client(self.fake, recall_mode="off")
        with patch.object(self.memory, "summarize_if_needed", AsyncMock(return_value=False)):
            reply = await _remember_response(
                self.memory, None, "42", "question", "answer", "m", svc
            )
        await _drain(svc)
        self.assertEqual(reply, "answer")
        lines = [json.loads(l) for l in (self.dir / "42.jsonl").read_text().splitlines()]
        sent = self.fake.appended[0]
        self.assertEqual(sent["session_id"], "42")
        self.assertEqual(
            [(t["role"], t["content"], t["timestamp"]) for t in sent["turns"]],
            [(r["role"], r["content"], r["timestamp"]) for r in lines],
        )
        await svc.close()

    async def test_dual_write_disabled_sends_nothing(self):
        svc = _client(self.fake, dual_write=False, recall_mode="off")
        with patch.object(self.memory, "summarize_if_needed", AsyncMock(return_value=False)):
            await _remember_response(self.memory, None, "42", "q", "a", "m", svc)
        await _drain(svc)
        self.assertEqual(self.fake.paths(), [])
        self.assertTrue((self.dir / "42.jsonl").exists())
        await svc.close()

    async def test_legacy_summary_and_rag_still_run_in_legacy_backend(self):
        svc = _client(self.fake, backend="legacy")
        rag = SimpleNamespace(index_turn=AsyncMock())
        summarize = AsyncMock(return_value=False)
        with patch.object(self.memory, "summarize_if_needed", summarize):
            await _remember_response(self.memory, rag, "42", "q", "a", "m", svc)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        summarize.assert_awaited_once()
        rag.index_turn.assert_awaited_once()
        await svc.close()

    async def test_service_backend_skips_summary_and_rag_but_still_writes(self):
        svc = _client(self.fake, backend="service", dual_write=False)  # forced on by the backend
        rag = SimpleNamespace(index_turn=AsyncMock())
        summarize = AsyncMock(return_value=False)
        with patch.object(self.memory, "summarize_if_needed", summarize):
            await _remember_response(self.memory, rag, "42", "q", "a", "m", svc)
            await _drain(svc)
        summarize.assert_not_awaited()
        rag.index_turn.assert_not_awaited()
        self.assertEqual(self.fake.paths(), ["/v1/turns"])
        await svc.close()

    async def test_dual_write_failure_never_breaks_the_turn(self):
        self.fake.fail = True
        svc = _client(self.fake, recall_mode="off")
        with patch.object(self.memory, "summarize_if_needed", AsyncMock(return_value=False)):
            with self.assertLogs("src.memory_service_client", level="WARNING"):
                reply = await _remember_response(self.memory, None, "42", "q", "a", "m", svc)
                await _drain(svc)
        self.assertEqual(reply, "a")
        self.assertEqual(len((self.dir / "42.jsonl").read_text().splitlines()), 2)
        await svc.close()

    async def test_failed_turn_is_saved_nowhere(self):
        svc = _client(self.fake, recall_mode="off")
        chat = AsyncMock(side_effect=OllamaUnavailableError("down"))
        with patch("src.agent._chat_with_retry", chat):
            reply = await run_agent(
                "question", hub=None, session_id="42", memory=self.memory, memory_service=svc
            )
        await _drain(svc)
        self.assertIn("Ollama was unavailable", reply)
        self.assertFalse((self.dir / "42.jsonl").exists())
        self.assertEqual(self.fake.appended, [])
        await svc.close()

    async def test_retried_turn_is_written_once(self):
        import httpx as _httpx

        svc = _client(self.fake, recall_mode="off")
        chat = AsyncMock(side_effect=[_httpx.RemoteProtocolError("drop"), _reply("recovered")])
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", AsyncMock(return_value=True)),
            patch.object(self.memory, "summarize_if_needed", AsyncMock(return_value=False)),
        ):
            reply = await run_agent(
                "question", hub=None, session_id="42", memory=self.memory, memory_service=svc
            )
        await _drain(svc)
        self.assertEqual(reply, "recovered")
        self.assertEqual(len(self.fake.appended), 1)
        self.assertEqual(len((self.dir / "42.jsonl").read_text().splitlines()), 2)
        await svc.close()


class CoreMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_core_context_is_profile_plus_recent_turns_without_summary(self):
        fake = FakeService()
        svc = _client(fake)
        directory = Path(tempfile.mkdtemp())
        store = CoreMemoryStore(directory, recent_turns=1, service=svc)
        for i in range(3):
            await store.remember_turn("42", f"question {i}", f"answer {i}")
        (directory / "42.summary.json").write_text(
            json.dumps({"summary": "LEGACY SUMMARY", "summarized_messages": 2})
        )
        messages = await store.context_messages("42")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Known facts about the user:\n- likes concise answers"},
                {"role": "user", "content": "question 2"},
                {"role": "assistant", "content": "answer 2"},
            ],
        )
        self.assertFalse(await store.summarize_if_needed("42", None, "m"))
        await svc.close()

    async def test_core_context_survives_service_outage(self):
        fake = FakeService()
        fake.fail = True
        svc = _client(fake)
        directory = Path(tempfile.mkdtemp())
        store = CoreMemoryStore(directory, recent_turns=1, service=svc)
        await store.remember_turn("42", "q", "a")
        with self.assertLogs("src.memory_service_client", level="WARNING"):
            messages = await store.context_messages("42")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        await svc.close()

    async def test_service_backend_prompt_has_no_rag_and_is_compact(self):
        fake = FakeService()
        svc = _client(fake, backend="service", recall_mode="on")
        directory = Path(tempfile.mkdtemp())
        store = CoreMemoryStore(directory, recent_turns=2, service=svc)
        await store.remember_turn("42", "older q", "older a")
        chat = _chat(_reply("fine"))
        with patch("src.agent._chat_with_monitoring", chat):
            await run_agent(
                SECRET_USER, hub=None, session_id="42", memory=store, rag=None, memory_service=svc
            )
        roles = [m["role"] for m in chat.snapshots[0]]
        self.assertEqual(roles, ["system", "system", "system", "user", "assistant", "user"])
        await _drain(svc)
        await svc.close()


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse(self):
        self.assertEqual(parse_slash_command("/recall what about cars").args[:1], ["what"])

    async def test_memory_command_reports_stats_and_counters(self):
        fake = FakeService()
        svc = _client(fake, recall_mode="shadow")
        await svc.recall("q", "42")
        reply = await handle_command(None, "42", ParsedCommand("memory", []), memory_service=svc)
        self.assertIn("reachable", reply)
        self.assertIn("turns=228 chunks=351", reply)
        self.assertIn("backend=legacy recall=shadow dual_write=on", reply)
        self.assertIn("recalls=1 fired=1", reply)
        await svc.close()

    async def test_memory_command_unreachable_and_unconfigured(self):
        fake = FakeService()
        fake.fail = True
        svc = _client(fake)
        with self.assertLogs("src.memory_service_client", level="WARNING"):
            reply = await handle_command(None, "42", ParsedCommand("memory", []), memory_service=svc)
        self.assertIn("UNREACHABLE", reply)
        reply = await handle_command(None, "42", ParsedCommand("memory", []))
        self.assertIn("not configured", reply)
        await svc.close()

    async def test_recall_command(self):
        fake = FakeService()
        svc = _client(fake)
        reply = await handle_command(
            None, "42", ParsedCommand("recall", ["the", "car"]), memory_service=svc
        )
        self.assertEqual(reply, "[t1 2026-01-01 user] found it")
        usage = await handle_command(None, "42", ParsedCommand("recall", []), memory_service=svc)
        self.assertIn("Usage", usage)
        none = await handle_command(None, "42", ParsedCommand("recall", ["x"]))
        self.assertIn("not configured", none)
        await svc.close()


class ConfigDefaultTests(unittest.TestCase):
    def _config(self, **env):
        clean = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
        clean.update(env)
        code = (
            "import json, src.config as c;"
            "print(json.dumps({k: getattr(c, k) for k in ("
            "'MEMORY_BACKEND','MEMORY_SERVICE_URL','MEMORY_RECALL','MEMORY_TOOLS',"
            "'MEMORY_DUAL_WRITE','MEMORY_RECALL_TIMEOUT','MEMORY_APPEND_TIMEOUT',"
            "'MEMORY_TOOL_TIMEOUT')}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], env=clean, cwd=tempfile.mkdtemp(),
            capture_output=True, text=True, check=True,
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_defaults_preserve_todays_behaviour(self):
        cfg = self._config()
        self.assertEqual(cfg["MEMORY_BACKEND"], "legacy")
        self.assertEqual(cfg["MEMORY_SERVICE_URL"], "")  # integration inert without a URL
        self.assertEqual(cfg["MEMORY_RECALL"], "shadow")
        self.assertEqual(cfg["MEMORY_TOOLS"], "off")
        self.assertTrue(cfg["MEMORY_DUAL_WRITE"])
        self.assertEqual(
            (cfg["MEMORY_RECALL_TIMEOUT"], cfg["MEMORY_APPEND_TIMEOUT"], cfg["MEMORY_TOOL_TIMEOUT"]),
            (2.5, 3.0, 5.0),
        )

    def test_overrides_and_garbage_fall_back(self):
        cfg = self._config(
            MEMORY_BACKEND="service", MEMORY_RECALL="on", MEMORY_TOOLS="on", MEMORY_DUAL_WRITE="0"
        )
        self.assertEqual(
            (cfg["MEMORY_BACKEND"], cfg["MEMORY_RECALL"], cfg["MEMORY_TOOLS"], cfg["MEMORY_DUAL_WRITE"]),
            ("service", "on", "on", False),
        )
        bad = self._config(MEMORY_BACKEND="banana", MEMORY_RECALL="", MEMORY_TOOLS="maybe")
        self.assertEqual((bad["MEMORY_BACKEND"], bad["MEMORY_RECALL"], bad["MEMORY_TOOLS"]), ("legacy", "shadow", "off"))


class NoServiceConfiguredTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_without_service_is_unchanged(self):
        chat = _chat(_reply("plain"))
        directory = Path(tempfile.mkdtemp())
        memory = MemoryStore(directory)
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch.object(memory, "summarize_if_needed", AsyncMock(return_value=False)),
        ):
            reply = await run_agent("hi", hub=None, session_id="42", memory=memory)
        self.assertEqual(reply, "plain")
        self.assertEqual(len((directory / "42.jsonl").read_text().splitlines()), 2)
        self.assertEqual([m["role"] for m in chat.snapshots[0]], ["system", "user"])


class RecallResultTests(unittest.TestCase):
    def test_chars_property(self):
        self.assertEqual(RecallResult(True, 1, 0.5, "abc", 1.0, "cue").chars, 3)
        self.assertEqual(RecallResult(False, 0, None, "", 1.0, "empty").chars, 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
