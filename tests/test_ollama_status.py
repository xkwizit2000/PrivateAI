import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.ollama_status import format_ollama_status


class OllamaStatusTests(unittest.TestCase):
    def test_format_empty_running_models(self):
        text = format_ollama_status(
            [],
            host="http://10.0.0.37:11434",
            chat_model="gemma4:e4b-jt",
        )
        self.assertIn("Ollama (http://10.0.0.37:11434):", text)
        self.assertIn("• chat model: gemma4:e4b-jt", text)
        self.assertIn("• running: none loaded", text)

    def test_format_running_models(self):
        now = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
        model = SimpleNamespace(
            name="gemma4:e4b-jt",
            model="gemma4:e4b-jt",
            size=6_591_830_464,
            size_vram=5_333_539_264,
            context_length=8192,
            expires_at=now + timedelta(minutes=4, seconds=20),
            details=SimpleNamespace(
                parameter_size="8.0B",
                quantization_level="Q4_K_M",
            ),
        )
        text = format_ollama_status(
            [model],
            host="http://10.0.0.37:11434",
            chat_model="gemma4:e4b-jt",
            now=now,
        )
        self.assertIn("• running:", text)
        self.assertIn("gemma4:e4b-jt", text)
        self.assertIn("6.1 GiB", text)
        self.assertIn("5.0 GiB VRAM", text)
        self.assertIn("ctx 8192", text)
        self.assertIn("8.0B", text)
        self.assertIn("Q4_K_M", text)
        self.assertIn("expires in 4m", text)


if __name__ == "__main__":
    unittest.main()
