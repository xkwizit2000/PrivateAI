"""Tests for tool routing (auto/always/never) and the /tools command."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent import run_agent
from src.commands import handle_command, parse_slash_command
from src.session_settings import SessionSettings
from src.tool_routing import ToolRouter, match_file_access

OFF_MESSAGES = [
    "hello",
    "what is 2+2",
    "summarize our chat",
    "Tell me a joke about options trading",
    "explain how TCP works",
    "is 24/7 monitoring worth it, and/or 50/50?",
    "see https://example.com for details",
    "thanks!",
]
ON_MESSAGES = [
    "list the files in /workingdir",
    "read notes.txt",
    "what's in config.yaml",
    "show me the directory tree",
    "Please SAVE this to disk",
    "create a folder called x",
    "delete old logs",
    "rename it",
    "cat report",
    "ls",
    "open ~/docs",
    "look at C:\\Users\\me",
    "fix main.py",
    "edit it please",
    "move that over",
    "search for TODO",
    "find the config",
    "what is in the working dir?",
    "listing files",
]


class MatchTests(unittest.TestCase):
    def test_misses(self):
        for text in OFF_MESSAGES:
            self.assertIsNone(match_file_access(text), text)

    def test_hits(self):
        for text in ON_MESSAGES:
            self.assertIsNotNone(match_file_access(text), text)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class RouterTests(unittest.TestCase):
    def test_auto_keyword_hits_and_misses(self):
        router = ToolRouter(600, _Clock())
        for text in OFF_MESSAGES:
            decision = router.decide("auto", f"miss-{text}", text)
            self.assertFalse(decision.tools_on, text)
        for text in ON_MESSAGES:
            decision = router.decide("auto", f"hit-{text}", text)
            self.assertTrue(decision.tools_on, text)
            self.assertTrue(decision.matched)

    def test_followup_after_match_then_expires(self):
        clock = _Clock()
        router = ToolRouter(600, clock)
        self.assertTrue(router.decide("auto", "1", "list the files in /workingdir").tools_on)
        follow = router.decide("auto", "1", "yes do it")
        self.assertTrue(follow.tools_on)
        self.assertIn("follow-up", follow.reason)
        clock.now += 601
        self.assertFalse(router.decide("auto", "1", "yes do it").tools_on)

    def test_followup_does_not_renew_itself(self):
        clock = _Clock()
        router = ToolRouter(600, clock)
        router.decide("auto", "1", "read notes.txt")
        clock.now += 400
        self.assertTrue(router.decide("auto", "1", "ok").tools_on)  # follow-up
        clock.now += 400  # 800s after the matching turn
        self.assertFalse(router.decide("auto", "1", "ok").tools_on)

    def test_tool_use_marks_active(self):
        router = ToolRouter(600, _Clock())
        self.assertFalse(router.decide("auto", "1", "hello").tools_on)
        router.mark_active("1")
        self.assertTrue(router.decide("auto", "1", "now do that again").tools_on)

    def test_followup_is_per_session(self):
        router = ToolRouter(600, _Clock())
        router.decide("auto", "1", "read notes.txt")
        self.assertFalse(router.decide("auto", "2", "hello").tools_on)

    def test_always_and_never(self):
        router = ToolRouter(600, _Clock())
        always = router.decide("always", "1", "hello")
        self.assertTrue(always.tools_on)
        self.assertEqual(always.reason, "mode=always")
        never = router.decide("never", "1", "list the files in /workingdir")
        self.assertFalse(never.tools_on)
        self.assertEqual(never.reason, "mode=never")
        # never must not arm the follow-up window
        self.assertFalse(router.decide("auto", "1", "hello").tools_on)

    def test_no_session_id(self):
        router = ToolRouter(600, _Clock())
        self.assertFalse(router.decide("auto", None, "hello").tools_on)
        self.assertTrue(router.decide("auto", None, "read a.txt").tools_on)
        self.assertFalse(router.decide("auto", None, "ok").tools_on)

    def test_followup_disabled_with_zero_ttl(self):
        router = ToolRouter(0, _Clock())
        router.decide("auto", "1", "read notes.txt")
        self.assertFalse(router.decide("auto", "1", "ok").tools_on)


def _reply(text="ok", tool_calls=None):
    return SimpleNamespace(
        done_reason="stop",
        message=SimpleNamespace(content=text, tool_calls=tool_calls),
    )


def _fake_hub(result="a.txt\nb.txt"):
    hub = MagicMock()
    hub.tool_names = ["filesystem_list_directory"]
    hub.as_ollama_tools.return_value = [
        {"type": "function", "function": {"name": "filesystem_list_directory"}}
    ]
    hub.call_tool = AsyncMock(return_value=result)
    return hub


class AgentRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, text, hub, router, mode="auto", chat=None):
        chat = chat or AsyncMock(side_effect=[_reply("done")])
        with patch("src.agent._chat_with_monitoring", chat):
            reply = await run_agent(
                text, hub=hub, session_id="42", tool_mode=mode, router=router
            )
        return reply, chat

    async def test_hello_sends_no_tools_and_logs_off(self):
        hub = _fake_hub()
        with self.assertLogs("src.agent", level="INFO") as logs:
            reply, chat = await self._run("hello", hub, ToolRouter(600, _Clock()))
        self.assertEqual(reply, "done")
        self.assertIsNone(chat.await_args.kwargs["tools"])
        hub.as_ollama_tools.assert_not_called()
        self.assertTrue(any("tool routing: tools=off reason=" in l for l in logs.output))

    async def test_file_request_sends_tools_and_logs_on(self):
        hub = _fake_hub()
        with self.assertLogs("src.agent", level="INFO") as logs:
            _, chat = await self._run(
                "list the files in /workingdir", hub, ToolRouter(600, _Clock())
            )
        self.assertEqual(
            chat.await_args.kwargs["tools"][0]["function"]["name"],
            "filesystem_list_directory",
        )
        self.assertTrue(any("tool routing: tools=on reason=" in l for l in logs.output))

    async def test_followup_after_tool_turn_keeps_tools(self):
        hub = _fake_hub()
        router = ToolRouter(600, _Clock())
        call = SimpleNamespace(
            function=SimpleNamespace(name="filesystem_list_directory", arguments={"path": "/workingdir"})
        )
        chat = AsyncMock(side_effect=[_reply("", [call]), _reply("two files")])
        reply, chat = await self._run("list the files in /workingdir", hub, router, chat=chat)
        self.assertEqual(reply, "two files")
        hub.call_tool.assert_awaited_once_with(
            "filesystem_list_directory", {"path": "/workingdir"}
        )
        reply, chat2 = await self._run("how many were there?", hub, router)
        self.assertIsNotNone(chat2.await_args.kwargs["tools"])

    async def test_tool_use_arms_followup_even_without_keyword(self):
        hub = _fake_hub()
        router = ToolRouter(600, _Clock())
        call = SimpleNamespace(
            function=SimpleNamespace(name="filesystem_list_directory", arguments={})
        )
        # Force tools on via 'always' for turn 1; follow-up in auto mode stays off
        # because 'always' neither matched nor called a tool here.
        chat = AsyncMock(side_effect=[_reply("", [call]), _reply("x")])
        await self._run("hello", hub, router, mode="always", chat=chat)
        _, chat2 = await self._run("and then?", hub, router)
        self.assertIsNotNone(chat2.await_args.kwargs["tools"])

    async def test_always_and_never_modes(self):
        hub = _fake_hub()
        _, chat = await self._run("hello", hub, ToolRouter(600, _Clock()), mode="always")
        self.assertIsNotNone(chat.await_args.kwargs["tools"])
        _, chat = await self._run(
            "list the files in /workingdir", hub, ToolRouter(600, _Clock()), mode="never"
        )
        self.assertIsNone(chat.await_args.kwargs["tools"])

    async def test_hallucinated_call_while_tools_off_is_not_executed(self):
        hub = _fake_hub()
        call = SimpleNamespace(
            function=SimpleNamespace(name="filesystem_list_directory", arguments={})
        )
        chat = AsyncMock(side_effect=[_reply("", [call]), _reply("sorry")])
        reply, chat = await self._run("hello", hub, ToolRouter(600, _Clock()), chat=chat)
        self.assertEqual(reply, "sorry")
        hub.call_tool.assert_not_awaited()
        tool_msg = chat.await_args_list[1].kwargs["messages"][-2]  # list is mutated after the call
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("not available", tool_msg["content"])

    async def test_no_hub_logs_off(self):
        with self.assertLogs("src.agent", level="INFO") as logs:
            reply, chat = await self._run("read notes.txt", None, ToolRouter(600, _Clock()))
        self.assertIsNone(chat.await_args.kwargs["tools"])
        self.assertTrue(any("tool routing: tools=off" in l for l in logs.output))


class OllamaClientAcceptsNoToolsTests(unittest.TestCase):
    def test_chat_signature_accepts_tools_none(self):
        import inspect

        from ollama import AsyncClient

        params = inspect.signature(AsyncClient.chat).parameters
        self.assertIn("tools", params)
        self.assertIsNone(params["tools"].default)


class ToolsCommandTests(unittest.IsolatedAsyncioTestCase):
    async def _cmd(self, settings, text):
        parsed = parse_slash_command(text)
        assert parsed is not None
        return await handle_command(settings, "123", parsed)

    async def test_set_status_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = SessionSettings(path, default_tools="auto")
            self.assertEqual(settings.get_tool_mode("123"), "auto")

            reply = await self._cmd(settings, "/tools off")
            self.assertIn("tools=off", reply)
            self.assertEqual(settings.get_tool_mode("123"), "never")
            self.assertEqual(SessionSettings(path).get_tool_mode("123"), "never")

            reply = await self._cmd(settings, "/tools on")
            self.assertEqual(settings.get_tool_mode("123"), "always")
            self.assertIn("tools=on", reply)

            reply = await self._cmd(settings, "/tools")
            self.assertIn("Tools are on", reply)

            await self._cmd(settings, "/tools auto")
            self.assertEqual(settings.get_tool_mode("123"), "auto")

            await self._cmd(settings, "/tools off")
            await self._cmd(settings, "/tools reset")
            self.assertEqual(settings.get_tool_mode("123"), "auto")

    async def test_bad_arg(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(Path(directory) / "s.json")
            reply = await self._cmd(settings, "/tools banana")
            self.assertIn("Usage: /tools", reply)

    def test_invalid_stored_value_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.json"
            path.write_text('{"123": {"tools": "bogus"}}')
            self.assertEqual(SessionSettings(path, default_tools="never").get_tool_mode("123"), "never")

    def test_status_lists_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(Path(directory) / "s.json", default_tools="auto")
            settings.set_tool_mode("123", "never")
            self.assertEqual(dict(settings.status_items("123"))["tools"], "off (default auto)")


if __name__ == "__main__":
    unittest.main()
