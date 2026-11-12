"""Tests for the host-side exporter (ops/pai-status/pai-status-export)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "ops" / "pai-status" / "pai-status-export"


def load_module():
    """Import the extension-less script as a module (without running main)."""
    loader = importlib.machinery.SourceFileLoader("pai_status_export", str(SCRIPT))
    spec = importlib.util.spec_from_loader("pai_status_export", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class ExporterRunTests(unittest.TestCase):
    def run_exporter(self, tmp: Path, **extra_env):
        env = dict(os.environ, PAI_STATUS_DIR=str(tmp / "out"), PAI_STATUS_STATE=str(tmp / "state" / "cpu.json"),
                   PAI_DATA_PATH=str(tmp), PAI_CONTAINERS="pai-test-no-such-container", **extra_env)
        return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)

    def test_writes_one_valid_json_file_atomically_with_safe_perms(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            result = self.run_exporter(tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            files = sorted(p.name for p in (tmp / "out").iterdir())
            self.assertEqual(files, ["host.json"])  # no temp file left behind
            mode = stat.S_IMODE((tmp / "out" / "host.json").stat().st_mode)
            self.assertEqual(mode, 0o644)
            self.assertEqual(stat.S_IMODE((tmp / "out").stat().st_mode), 0o755)
            data = json.loads((tmp / "out" / "host.json").read_text())
            self.assertEqual(data["schema"], 1)
            for key in ("generated_at", "generated_epoch", "cpu", "mem", "disks", "units", "errors", "exporter"):
                self.assertIn(key, data)
            self.assertGreater(data["mem"]["total"], 0)
            self.assertEqual(len(data["cpu"]["load"]), 3)
            self.assertEqual({d["label"] for d in data["disks"]}, {"root", "data"})

    def test_failing_collectors_are_isolated_and_second_run_uses_state(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self.run_exporter(tmp, PATH="/nonexistent")  # no docker/systemctl/journalctl at all
            first = json.loads((tmp / "out" / "host.json").read_text())
            self.assertIn("units", first["errors"])
            self.assertIn("watchdog", first["errors"])
            self.assertIsNotNone(first["cpu"]["count"])  # healthy collectors unaffected
            self.assertTrue((tmp / "state" / "cpu.json").exists())
            self.run_exporter(tmp, PATH="/nonexistent")
            second = json.loads((tmp / "out" / "host.json").read_text())
            self.assertGreaterEqual(second["generated_epoch"], first["generated_epoch"])

    def test_no_secret_looking_content(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self.run_exporter(tmp, TELEGRAM_TOKEN="123456789:SECRETSECRETSECRETSECRETSECRET1234",
                              MEMORY_TOKEN="topsecretmemorytoken")
            text = (tmp / "out" / "host.json").read_text()
            self.assertNotIn("SECRET", text)
            self.assertNotIn("topsecretmemorytoken", text)


class ExporterUnitTests(unittest.TestCase):
    def test_watchdog_event_regex(self):
        module = load_module()
        for line, kind in (
            ("2026-11-12T07:07:18+00:00 pai01 x[1]: ollama-watchdog: HUNG confirmed: probe failed", "HUNG confirmed"),
            ("t h x[1]: ollama-watchdog: restarting ollama: systemctl restart ollama", "restarting ollama"),
            ("t h x[1]: ollama-watchdog: RECOVERED: ok", "RECOVERED"),
            ("t h x[1]: ollama-watchdog: COOLDOWN: last 5s ago", "COOLDOWN"),
        ):
            self.assertEqual(module._EVENT_RE.search(line).group(1), kind)
        self.assertIsNone(module._EVENT_RE.search("t h x[1]: ollama-watchdog: ok: idle (cpu=0%)"))

    def test_systemd_unit_files_are_read_only_and_low_priority(self):
        unit = (SCRIPT.parent / "pai-status-export.service").read_text()
        for needle in ("Type=oneshot", "Nice=19", "IOSchedulingClass=idle", "TimeoutStartSec=20",
                       "ProtectSystem=strict", "NoNewPrivileges=yes"):
            self.assertIn(needle, unit)
        timer = (SCRIPT.parent / "pai-status-export.timer").read_text()
        self.assertIn("OnUnitActiveSec=30s", timer)


if __name__ == "__main__":
    unittest.main()
