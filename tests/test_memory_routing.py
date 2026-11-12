"""Per-tool-group routing (files vs memory) in ToolRouter and run_agent."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent import run_agent
from src.tool_routing import ToolRouter, match_file_access, match_memory_cue


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class MemoryCueTests(unittest.TestCase):
    def test_cues_match(self):
        for text in (
            "do you remember what we discussed earlier about the car?",
            "Remember when I told you about my guitar?",
            "what did we decide last time",
            "you said something yesterday about backups",
            "we talked about this a few days ago",
            "remind me what I asked 3 weeks ago",
            "what did I tell you about my job",
            "our earlier conversation about VRAM",
            "check our chat history",
        ):
            self.assertIsNotNone(match_memory_cue(text), text)

    def test_plain_chat_and_file_requests_do_not_match(self):
        for text in (
            "hello",
            "what is 2+2",
            "tell me a joke",
            "list the files in /workingdir",
            "read notes.txt",
            "how do I write a for loop in python",
        ):
            self.assertIsNone(match_memory_cue(text), text)

    def test_memory_cue_is_not_a_file_cue(self):
        self.assertIsNone(match_file_access("do you remember what we discussed earlier?"))


class RouterGroupTests(unittest.TestCase):
    def test_memory_group_off_unless_enabled(self):
        router = ToolRouter(600, _Clock())
        decision = router.decide("auto", "1", "do you remember what we discussed earlier?")
        self.assertFalse(decision.memory_on)
        self.assertFalse(decision.tools_on)
        self.assertEqual(decision.memory_reason, "memory tools disabled")

    def test_memory_cue_turns_memory_on_without_file_tools(self):
        router = ToolRouter(600, _Clock())
        decision = router.decide(
            "auto", "1", "do you remember what we discussed earlier?", memory_tools=True
        )
        self.assertTrue(decision.memory_on)
        self.assertFalse(decision.tools_on)  # no file keyword
        self.assertIn("memory cue", decision.memory_reason)

    def test_file_cue_turns_files_on_without_memory_tools(self):
        router = ToolRouter(600, _Clock())
        decision = router.decide("auto", "1", "list the files in /workingdir", memory_tools=True)
        self.assertTrue(decision.tools_on)
        self.assertFalse(decision.memory_on)

    def test_hello_is_off_for_both(self):
        router = ToolRouter(600, _Clock())
        decision = router.decide("auto", "1", "hello", memory_tools=True)
        self.assertFalse(decision.tools_on)
        self.assertFalse(decision.memory_on)

    def test_both_groups_can_match(self):
        router = ToolRouter(600, _Clock())
        decision = router.decide(
            "auto", "1", "what did we decide earlier about reading the config file?", memory_tools=True
        )
        self.assertTrue(decision.tools_on)
        self.assertTrue(decision.memory_on)

    def test_memory_followup_window_and_expiry_are_per_group(self):
        clock = _Clock()
        router = ToolRouter(600, clock)
        router.decide("auto", "1", "remember what we discussed earlier?", memory_tools=True)
        clock.now = 100
        follow = router.decide("auto", "1", "and what else?", memory_tools=True)
        self.assertTrue(follow.memory_on)
        self.assertFalse(follow.tools_on)  # the memory window never arms the file group
        self.assertIn("follow-up", follow.memory_reason)
        clock.now = 700
        self.assertFalse(router.decide("auto", "1", "and what else?", memory_tools=True).memory_on)

    def test_file_window_does_not_arm_memory(self):
        clock = _Clock()
        router = ToolRouter(600, clock)
        router.decide("auto", "1", "list the files", memory_tools=True)
        follow = router.decide("auto", "1", "how many?", memory_tools=True)
        self.assertTrue(follow.tools_on)
        self.assertFalse(follow.memory_on)

    def test_mark_active_group(self):
        router = ToolRouter(600, _Clock())
        router.mark_active("1", "memory")
        self.assertTrue(router.followup_active("1", "memory"))
        self.assertFalse(router.followup_active("1"))  # files untouched

    def test_always_and_never(self):
        router = ToolRouter(600, _Clock())
        always = router.decide("always", "1", "hello", memory_tools=True)
        self.assertTrue(always.tools_on and always.memory_on)
        never = router.decide("never", "1", "remember earlier", memory_tools=True)
        self.assertFalse(never.tools_on or never.memory_on)
        self.assertFalse(router.decide("always", "1", "hello").memory_on)  # disabled stays off


def _reply(text="ok", tool_calls=None):
    return SimpleNamespace(
        done_reason="stop", message=SimpleNamespace(content=text, tool_calls=tool_calls)
    )


def _tool(name):
    return {"type": "function", "function": {"name": name}}


def _hub():
    hub = MagicMock()
    hub.tool_names = ["filesystem_list_directory", "memory_search_memory"]
    catalog = {
        "files": [_tool("filesystem_list_directory")],
        "memory": [_tool("memory_search_memory")],
    }

    def as_tools(groups=None, **_):
        groups = catalog.keys() if groups is None else groups
        return [t for g in catalog if g in groups for t in catalog[g]]

    hub.as_ollama_tools.side_effect = as_tools
    hub.tool_group.side_effect = lambda name: "memory" if name.startswith("memory_") else "files"
    hub.call_tool = AsyncMock(return_value="[t1 2026-01-01 user] a hit")
    return hub


class AgentGroupRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, text, hub, router=None, chat=None, memory_tools=True):
        chat = chat or AsyncMock(side_effect=[_reply("done")])
        with patch("src.agent._chat_with_monitoring", chat):
            reply = await run_agent(
                text,
                hub=hub,
                session_id="42",
                router=router or ToolRouter(600, _Clock()),
                memory_tools=memory_tools,
            )
        return reply, chat

    def _names(self, chat):
        tools = chat.await_args_list[0].kwargs["tools"]
        return None if tools is None else sorted(t["function"]["name"] for t in tools)

    async def test_memory_cue_offers_only_memory_tools(self):
        _, chat = await self._run("do you remember what we discussed earlier?", _hub())
        self.assertEqual(self._names(chat), ["memory_search_memory"])

    async def test_file_request_offers_only_file_tools(self):
        _, chat = await self._run("list the files in /workingdir", _hub())
        self.assertEqual(self._names(chat), ["filesystem_list_directory"])

    async def test_hello_offers_nothing(self):
        _, chat = await self._run("hello", _hub())
        self.assertIsNone(self._names(chat))

    async def test_memory_tools_disabled_never_offers_memory_group(self):
        hub = _hub()
        _, chat = await self._run(
            "do you remember what we discussed earlier?", hub, memory_tools=False
        )
        self.assertIsNone(self._names(chat))
        hub.kick_reconnect.assert_not_called()

    async def test_memory_tool_call_runs_and_arms_memory_followup(self):
        hub = _hub()
        router = ToolRouter(600, _Clock())
        call = SimpleNamespace(
            function=SimpleNamespace(name="memory_search_memory", arguments={"query": "x"})
        )
        chat = AsyncMock(side_effect=[_reply("", [call]), _reply("found it")])
        reply, _ = await self._run(
            "do you remember what we discussed earlier?", hub, router, chat=chat
        )
        self.assertEqual(reply, "found it")
        hub.call_tool.assert_awaited_once_with("memory_search_memory", {"query": "x"})
        self.assertTrue(router.followup_active("42", "memory"))
        self.assertFalse(router.followup_active("42"))

    async def test_tool_of_a_group_not_offered_is_not_executed(self):
        hub = _hub()
        call = SimpleNamespace(
            function=SimpleNamespace(name="memory_search_memory", arguments={"query": "x"})
        )
        chat = AsyncMock(side_effect=[_reply("", [call]), _reply("sorry")])
        reply, chat = await self._run("list the files in /workingdir", hub, chat=chat)
        self.assertEqual(reply, "sorry")
        hub.call_tool.assert_not_awaited()
        tool_msg = chat.await_args_list[1].kwargs["messages"][-2]
        self.assertIn("not available", tool_msg["content"])

    async def test_logs_memory_routing_line(self):
        with self.assertLogs("src.agent", level="INFO") as logs:
            await self._run("hello", _hub())
        self.assertTrue(any("tool routing: memory=off" in line for line in logs.output))
        self.assertTrue(any("tool routing: tools=off" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
