import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.commands import handle_command, parse_slash_command
from src.ids import safe_session_id
from src.session_settings import SessionSettings


class IdsTests(unittest.TestCase):
    def test_allows_telegram_and_session_ids(self):
        self.assertEqual(safe_session_id(12345), "12345")
        self.assertEqual(
            safe_session_id(
                "05abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
            ),
            "05abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        )

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            safe_session_id("../escape")
        with self.assertRaises(ValueError):
            safe_session_id("bad/id")


class CommandsTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_slash_command(self):
        parsed = parse_slash_command("/think on")
        assert parsed is not None
        self.assertEqual(parsed.name, "think")
        self.assertEqual(parsed.args, ["on"])

        parsed = parse_slash_command("/status@PrivateAIBot")
        assert parsed is not None
        self.assertEqual(parsed.name, "status")
        self.assertEqual(parsed.args, [])

    async def test_think_command(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(
                Path(directory) / "settings.json",
                default_think=False,
            )
            parsed = parse_slash_command("/think on")
            assert parsed is not None
            reply = await handle_command(settings, "123", parsed)
            self.assertIn("think=on", reply or "")
            self.assertTrue(settings.get_think("123"))

    async def test_status_includes_ollama_block(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(Path(directory) / "settings.json")
            parsed = parse_slash_command("/status")
            assert parsed is not None
            with patch(
                "src.commands.fetch_ollama_status",
                new=AsyncMock(return_value="Ollama (http://x):\n• running: none loaded"),
            ):
                reply = await handle_command(settings, "123", parsed)
            self.assertIn("Session settings:", reply or "")
            self.assertIn("Ollama (http://x):", reply or "")


if __name__ == "__main__":
    unittest.main()
