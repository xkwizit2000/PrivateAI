"""Tests for the one-shot Ollama retry and the friendlier failure message."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from ollama import ResponseError

from src.agent import (
    OLLAMA_UNAVAILABLE_MESSAGE,
    _is_retryable_ollama_error,
    _wait_for_ollama,
    run_agent,
)


def _ok(text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        done_reason="stop",
        message=SimpleNamespace(content=text, tool_calls=None),
    )


def _disconnect() -> httpx.RemoteProtocolError:
    return httpx.RemoteProtocolError("Server disconnected without sending a response.")


class RetryableClassificationTests(unittest.TestCase):
    def test_connection_type_errors_are_retryable(self):
        for exc in (
            _disconnect(),
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("slow"),
            httpx.ReadError("reset"),
            ConnectionError("Failed to connect to Ollama"),
            ResponseError("boom", 500),
            ResponseError("unavailable", 503),
        ):
            self.assertTrue(_is_retryable_ollama_error(exc), repr(exc))

    def test_other_errors_are_not_retryable(self):
        for exc in (
            ResponseError("bad request", 400),
            ResponseError("not found", 404),
            httpx.ReadTimeout("slow read"),
            TimeoutError("bot timeout"),
            ValueError("nope"),
        ):
            self.assertFalse(_is_retryable_ollama_error(exc), repr(exc))


class RunAgentRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_succeeds_after_one_remote_protocol_error(self):
        chat = AsyncMock(side_effect=[_disconnect(), _ok("recovered")])
        wait = AsyncMock(return_value=True)
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", wait),
        ):
            reply = await run_agent("hi", hub=None)

        self.assertEqual(reply, "recovered")
        self.assertEqual(chat.await_count, 2)
        wait.assert_awaited_once()

    async def test_no_retry_for_4xx(self):
        chat = AsyncMock(side_effect=ResponseError("bad request", 400))
        wait = AsyncMock(return_value=True)
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", wait),
        ):
            with self.assertRaises(ResponseError):
                await run_agent("hi", hub=None)

        self.assertEqual(chat.await_count, 1)
        wait.assert_not_awaited()

    async def test_no_retry_on_bot_timeout(self):
        chat = AsyncMock(side_effect=TimeoutError("timed out"))
        wait = AsyncMock(return_value=True)
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", wait),
        ):
            reply = await run_agent("hi", hub=None)

        self.assertIn("timed out", reply)
        self.assertEqual(chat.await_count, 1)
        wait.assert_not_awaited()

    async def test_friendly_message_after_double_failure(self):
        chat = AsyncMock(side_effect=[_disconnect(), _disconnect()])
        wait = AsyncMock(return_value=True)
        progress = AsyncMock()
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", wait),
        ):
            reply = await run_agent("hi", hub=None, progress=progress)

        self.assertEqual(reply, OLLAMA_UNAVAILABLE_MESSAGE)
        self.assertTrue(reply.startswith("System Error:"))
        self.assertEqual(chat.await_count, 2)  # original + exactly one retry
        progress.assert_any_await("responding", None)

    async def test_friendly_message_when_ollama_never_returns(self):
        chat = AsyncMock(side_effect=[ResponseError("down", 502)])
        wait = AsyncMock(return_value=False)
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", wait),
        ):
            reply = await run_agent("hi", hub=None)

        self.assertEqual(reply, OLLAMA_UNAVAILABLE_MESSAGE)
        self.assertEqual(chat.await_count, 1)

    async def test_failed_turn_is_not_remembered(self):
        chat = AsyncMock(side_effect=[_disconnect(), _disconnect()])
        memory = MagicMock()
        memory.context_messages = AsyncMock(return_value=[])
        memory.remember_turn = AsyncMock()
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", AsyncMock(return_value=True)),
        ):
            reply = await run_agent("hi", hub=None, memory=memory, session_id="1")

        self.assertEqual(reply, OLLAMA_UNAVAILABLE_MESSAGE)
        memory.remember_turn.assert_not_awaited()

    async def test_retry_does_not_rerun_tools_and_is_used_once_per_request(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="read_thing", arguments={"x": 1})
        )
        tool_turn = SimpleNamespace(
            done_reason="stop",
            message=SimpleNamespace(content="", tool_calls=[tool_call]),
        )
        # call 1 -> tool request; call 2 -> disconnect; retry -> final answer
        chat = AsyncMock(side_effect=[tool_turn, _disconnect(), _ok("done")])
        hub = MagicMock()
        hub.tool_names = ["read_thing"]
        hub.as_ollama_tools.return_value = []
        hub.call_tool = AsyncMock(return_value="tool output")
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", AsyncMock(return_value=True)),
        ):
            reply = await run_agent("hi", hub=hub)

        self.assertEqual(reply, "done")
        self.assertEqual(chat.await_count, 3)
        hub.call_tool.assert_awaited_once_with("read_thing", {"x": 1})

    async def test_second_disconnect_in_same_request_is_not_retried_again(self):
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name="read_thing", arguments={})
        )
        tool_turn = SimpleNamespace(
            done_reason="stop",
            message=SimpleNamespace(content="", tool_calls=[tool_call]),
        )
        # disconnect (retried ok -> tool call), then another disconnect: budget spent
        chat = AsyncMock(side_effect=[_disconnect(), tool_turn, _disconnect()])
        hub = MagicMock()
        hub.tool_names = ["read_thing"]
        hub.as_ollama_tools.return_value = []
        hub.call_tool = AsyncMock(return_value="out")
        wait = AsyncMock(return_value=True)
        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent._wait_for_ollama", wait),
        ):
            reply = await run_agent("hi", hub=hub)

        self.assertEqual(reply, OLLAMA_UNAVAILABLE_MESSAGE)
        self.assertEqual(chat.await_count, 3)
        wait.assert_awaited_once()
        hub.call_tool.assert_awaited_once()


class WaitForOllamaTests(unittest.IsolatedAsyncioTestCase):
    def _patched_client(self, handler):
        real = httpx.AsyncClient
        return patch(
            "src.agent.httpx.AsyncClient",
            side_effect=lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
        )

    async def test_returns_true_once_version_endpoint_answers(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/version")
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json={"version": "0.0.0"})

        with self._patched_client(handler):
            ok = await _wait_for_ollama("http://ollama:11434/", max_wait=5, poll=0.01)

        self.assertTrue(ok)
        self.assertEqual(calls["n"], 3)

    async def test_returns_false_when_ollama_never_answers(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        with self._patched_client(handler):
            ok = await _wait_for_ollama("http://ollama:11434", max_wait=0.05, poll=0.01)

        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
