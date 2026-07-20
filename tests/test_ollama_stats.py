import unittest
from types import SimpleNamespace

from src.agent import _log_ollama_stats, _ns_to_s, _rate


class OllamaStatsHelpersTests(unittest.TestCase):
    def test_ns_to_s(self):
        self.assertAlmostEqual(_ns_to_s(2_000_000_000), 2.0)
        self.assertIsNone(_ns_to_s(None))

    def test_rate(self):
        self.assertAlmostEqual(_rate(100, 2_000_000_000), 50.0)
        self.assertIsNone(_rate(100, 0))
        self.assertIsNone(_rate(None, 1_000_000_000))

    def test_log_ollama_stats_handles_partial_fields(self):
        response = SimpleNamespace(
            total_duration=3_000_000_000,
            load_duration=None,
            prompt_eval_count=20,
            prompt_eval_duration=1_000_000_000,
            eval_count=40,
            eval_duration=2_000_000_000,
            done_reason="stop",
        )
        # Should not raise.
        _log_ollama_stats(response, model="test-model", wall_s=3.1)


if __name__ == "__main__":
    unittest.main()
