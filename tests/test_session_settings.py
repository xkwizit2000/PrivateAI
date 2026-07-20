import tempfile
import unittest
from pathlib import Path

from src.session_settings import SessionSettings


class SessionSettingsTests(unittest.TestCase):
    def test_defaults_and_persists_think(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = SessionSettings(path, default_think=False)
            self.assertFalse(settings.get_think("123"))

            settings.set_think("123", True)
            reloaded = SessionSettings(path, default_think=False)
            self.assertTrue(reloaded.get_think("123"))
            self.assertFalse(reloaded.get_think("999"))

    def test_timeout_override_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = SessionSettings(path, default_timeout=180)
            self.assertEqual(settings.get_timeout("123"), 180.0)

            settings.set_timeout("123", 600)
            reloaded = SessionSettings(path, default_timeout=180)
            self.assertEqual(reloaded.get_timeout("123"), 600.0)

            restored = reloaded.clear_timeout("123")
            self.assertEqual(restored, 180.0)
            self.assertEqual(reloaded.get_timeout("123"), 180.0)

    def test_timeout_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(Path(directory) / "settings.json")
            with self.assertRaises(ValueError):
                settings.set_timeout("123", 5)
            with self.assertRaises(ValueError):
                settings.set_timeout("123", 99999)

    def test_verbose_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = SessionSettings(path)
            self.assertFalse(settings.get_verbose("123"))
            settings.set_verbose("123", True)
            reloaded = SessionSettings(path)
            self.assertTrue(reloaded.get_verbose("123"))

    def test_rejects_bad_session_id(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(Path(directory) / "settings.json")
            with self.assertRaises(ValueError):
                settings.get_think("../escape")

    def test_format_status_includes_all_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SessionSettings(
                Path(directory) / "settings.json",
                default_think=False,
                default_timeout=180,
            )
            settings.set_think("123", True)
            settings.set_verbose("123", True)
            settings.set_timeout("123", 600)

            items = dict(settings.status_items("123"))
            self.assertEqual(items["think"], "on (default off)")
            self.assertEqual(items["verbose"], "on")
            self.assertEqual(items["timeout"], "600s (default 180s)")

            text = settings.format_status("123")
            self.assertIn("• think: on (default off)", text)
            self.assertIn("• verbose: on", text)
            self.assertIn("• timeout: 600s (default 180s)", text)
            self.assertIn("Change with: /think, /verbose, /timeout", text)


if __name__ == "__main__":
    unittest.main()
