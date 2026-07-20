import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.memory import MemoryStore


class FakeOllamaClient:
    async def chat(self, **kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(content="User prefers concise answers.")
        )


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_persists_and_loads_recent_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory), recent_turns=1)
            await store.remember_turn("123", "first question", "first answer")
            await store.remember_turn("123", "second question", "second answer")

            messages = await store.context_messages("123")

            self.assertEqual(
                messages,
                [
                    {"role": "user", "content": "second question"},
                    {"role": "assistant", "content": "second answer"},
                ],
            )

    async def test_summarizes_after_configured_number_of_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(
                Path(directory),
                recent_turns=1,
                summarize_every=2,
            )
            await store.remember_turn("123", "question one", "answer one")
            self.assertFalse(
                await store.summarize_if_needed("123", FakeOllamaClient(), "test")
            )

            await store.remember_turn("123", "question two", "answer two")
            self.assertTrue(
                await store.summarize_if_needed("123", FakeOllamaClient(), "test")
            )

            summary_path = Path(directory) / "123.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["summarized_messages"], 4)
            self.assertEqual(summary["summary"], "User prefers concise answers.")

            messages = await store.context_messages("123")
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("User prefers concise answers.", messages[0]["content"])

    async def test_rejects_unsafe_session_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory))
            with self.assertRaises(ValueError):
                await store.context_messages("../escape")

    async def test_accepts_session_hex_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory), recent_turns=1)
            session_id = (
                "05abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            )
            await store.remember_turn(session_id, "hello", "world")
            messages = await store.context_messages(session_id)
            self.assertEqual(
                messages,
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
