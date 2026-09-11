"""Tests for agent reply truncation safeguards."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.agent import (
    _done_reason,
    _ollama_options,
    _prefer_longer_text,
    run_agent,
)


class PreferLongerTextTests(unittest.TestCase):
    def test_keeps_assembled_when_final_is_shorter_delta(self):
        assembled = "Hello, this is the full streamed reply."
        final_delta = "reply."
        self.assertEqual(_prefer_longer_text(assembled, final_delta), assembled)

    def test_uses_final_when_it_contains_full_text(self):
        assembled = ""
        final_full = "Complete answer from non-delta final payload."
        self.assertEqual(_prefer_longer_text(assembled, final_full), final_full)


class OllamaOptionsTests(unittest.TestCase):
    def test_default_options_set_uncapped_predict(self):
        options = _ollama_options()
        self.assertIsNotNone(options)
        assert options is not None
        self.assertEqual(options.get("num_predict"), -1)

    def test_done_reason_helper(self):
        self.assertEqual(_done_reason(SimpleNamespace(done_reason="length")), "length")
        self.assertIsNone(_done_reason(SimpleNamespace()))


class LengthContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_continues_when_done_reason_is_length(self):
        first = SimpleNamespace(
            done_reason="length",
            message=SimpleNamespace(content="Part one ", tool_calls=None),
        )
        second = SimpleNamespace(
            done_reason="stop",
            message=SimpleNamespace(content="part two.", tool_calls=None),
        )
        chat = AsyncMock(side_effect=[first, second])

        with patch("src.agent._chat_with_monitoring", chat):
            reply = await run_agent("Say more", hub=None)

        self.assertIn("Part one", reply)
        self.assertIn("part two.", reply)
        self.assertEqual(chat.await_count, 2)

    async def test_appends_notice_when_still_truncated(self):
        truncated = SimpleNamespace(
            done_reason="length",
            message=SimpleNamespace(content="Still going", tool_calls=None),
        )
        chat = AsyncMock(return_value=truncated)

        with (
            patch("src.agent._chat_with_monitoring", chat),
            patch("src.agent.OLLAMA_MAX_CONTINUATIONS", 1),
        ):
            reply = await run_agent("Say more", hub=None)

        self.assertIn("Still going", reply)
        self.assertIn("truncated by the model length limit", reply)
        # initial + 1 continuation
        self.assertEqual(chat.await_count, 2)


if __name__ == "__main__":
    unittest.main()
