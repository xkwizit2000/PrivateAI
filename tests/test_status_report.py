"""Tests for the /status report: verdict, collectors, formatting, secrets, isolation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import time
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from src import status_report as sr
from src.mcp_client import McpHub
from src.ollama_status import (
    LoadedModel,
    OllamaProbe,
    format_keep_alive,
    model_matches,
    parse_ps,
    probe_ollama,
)
from src.runtime_stats import ModelCallStats
from src.session_settings import SessionSettings

NOW = 1_800_000_000.0


def iso(epoch: float) -> str:
    """UTC ISO timestamp for an epoch."""
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def host_json(**overrides) -> dict:
    """A healthy host.json as the exporter writes it (generated 'now')."""
    data = {
        "schema": 1,
        "generated_at": iso(NOW - 10),
        "generated_epoch": NOW - 10,
        "hostname": "pai01",
        "uptime_s": 90061.0,
        "reboot_required": False,
        "cpu": {"count": 16, "usage_pct": 4.5, "load": [0.4, 0.5, 0.6]},
        "mem": {"total": 14 * 2**30, "available": 8 * 2**30, "swap_total": 4 * 2**30, "swap_free": 4 * 2**30},
        "disks": [
            {"label": "root", "path": "/", "mount": "/", "total": 98 * 2**30, "used": 47 * 2**30, "pct": 48.0},
            {"label": "data", "path": "/d", "mount": "/", "total": 98 * 2**30, "used": 47 * 2**30, "pct": 48.0},
        ],
        "gpu": {"busy_pct": 0, "vram_used": 4 * 2**30, "vram_total": 8 * 2**30},
        "temps": {"cpu_c": 45.0, "gpu_c": 41.0, "nvme_c": 38.0},
        "units": {"states": {"ollama.service": "active", "docker.service": "active", "ollama-watchdog.timer": "active"},
                  "failed": [], "failed_count": 0},
        "containers": [
            {"name": "privateai", "state": "running", "health": "none"},
            {"name": "privateai-memory", "state": "running", "health": "healthy"},
        ],
        "watchdog": {"restarts_24h": 0, "hung_24h": 0, "cooldown_24h": 0, "last_restart_at": None,
                     "last_line": {"ts": iso(NOW - 30), "text": "ok: idle (cpu=0% gpu=0%)"}, "events": []},
        "errors": {},
        "exporter": {"duration_ms": 90},
    }
    data.update(overrides)
    return data


def cfg_for(tmp: Path, **overrides) -> sr.StatusConfig:
    """Config pointing at temp paths; everything else uses defaults."""
    values = dict(
        host_file=tmp / "host.json",
        rag_dir=tmp / "rag",
        version_file=tmp / "VERSION",
        cgroup_dir=tmp / "cgroup",
        cpu_sample_seconds=0.0,
        ollama_host="http://10.0.0.181:11434",
        ollama_model="gemma4:e4b-jt-8k",
    )
    values.update(overrides)
    return sr.StatusConfig(**values)


def model(name="gemma4:e4b-jt-8k", size=3_400_000_000, vram=3_400_000_000, ctx=8192):
    """A LoadedModel with far-future keep-alive."""
    return LoadedModel(name, size, vram, ctx, datetime(2319, 2, 22, tzinfo=timezone.utc), "8.0B", "Q4_K_M")


def ctx_for(tmp: Path, **kw) -> sr.StatusContext:
    """StatusContext with temp-path config."""
    cfg = kw.pop("cfg", None) or cfg_for(tmp)
    return sr.StatusContext(cfg=cfg, now=NOW, **kw)


class FakeService:
    """Stands in for ServiceMemory (probe, breaker, counters)."""

    def __init__(self, healthy=True, stats=None, breaker="closed", error="ConnectError"):
        self.backend = "legacy"
        self.breaker_state = breaker
        self._failures = 0
        self.counters = {"append_ok": 3, "append_fail": 0, "recall_calls": 5, "recall_fired": 1,
                         "recall_skipped": 4, "recall_errors": 0, "recall_injected": 0, "skipped_open": 0}
        self.last_append_epoch = NOW - 120
        self._recall_ms = [100.0, 200.0]
        self.pending_tasks = 0
        self._probe = {"healthy": healthy, "ms": 12.0, "error": None if healthy else error,
                       "stats": stats if healthy else None}

    def breaker_retry_in(self):
        return 30.0

    async def probe(self, timeout=2.0):
        return self._probe

    def shadow_summary(self):
        return "recalls=5"


SERVICE_STATS = {"turns": 100, "chunks": 150, "sessions": 2, "days": 4, "facts_active": 7, "db_bytes": 5 * 2**20,
                 "last_ts": iso(NOW - 90), "summaries": 0, "embed_model": "m", "embed_dim": 384,
                 "index_mb": 0.5, "redactions": 0}


def write_legacy(tmp: Path, records: int, summarized: int = 0, newest: float = NOW - 60, session="123",
                 summary_updated: float | None = None) -> Path:
    """Write a legacy transcript (+ optional summary) under tmp/sessions."""
    directory = tmp / "sessions"
    directory.mkdir(exist_ok=True)
    lines = [
        json.dumps({"timestamp": iso(newest - (records - i)), "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"line {i}"})
        for i in range(records)
    ]
    (directory / f"{session}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summarized or summary_updated:
        (directory / f"{session}.summary.json").write_text(
            json.dumps({"updated_at": iso(summary_updated or newest), "summarized_messages": summarized,
                        "summary": "x" * 100}), encoding="utf-8")
    return directory


# ----------------------------------------------------------------------- verdict logic
class VerdictTests(unittest.TestCase):
    def test_ok_when_nothing_is_wrong(self):
        sections = [sr.Section("A", items=[sr.Item("fine", sr.OK), sr.Item("note", sr.INFO), sr.Item("-", sr.NA)])]
        self.assertEqual(sr.compute_verdict(sections), ("OK", []))

    def test_warn_is_degraded_and_fail_sorts_first(self):
        sections = [sr.Section("A", items=[sr.Item("w", sr.WARN, issue="warn issue"),
                                           sr.Item("f", sr.FAIL, issue="fail issue")])]
        verdict, issues = sr.compute_verdict(sections)
        self.assertEqual(verdict, "DEGRADED")
        self.assertEqual(issues, ["fail issue", "warn issue"])

    def test_critical_fail_is_down(self):
        sections = [sr.Section("A", items=[sr.Item("x", sr.FAIL, critical=True, issue="Ollama unreachable")])]
        self.assertEqual(sr.compute_verdict(sections)[0], "DOWN")

    def test_header_line_lists_issues_with_cap(self):
        items = [sr.Item(f"t{i}", sr.WARN, issue=f"issue {i}") for i in range(6)]
        text = sr.render_report([sr.Section("A", items=items)], sr.StatusConfig(), full=False, now=NOW)
        first = text.splitlines()[0]
        self.assertTrue(first.startswith("Overall: DEGRADED - issue 0; issue 1"))
        self.assertIn("+2 more", first)

    def test_text_markers(self):
        cfg = sr.StatusConfig(markers="text")
        text = sr.render_report([sr.Section("A", items=[sr.Item("w", sr.WARN)])], cfg, full=False, now=NOW)
        self.assertIn("WARN A", text)
        self.assertIn("WARN w", text)
        self.assertNotIn("⚠", text)

    def test_full_only_items_hidden_in_compact(self):
        sections = [sr.Section("A", items=[sr.Item("always"), sr.Item("only full", full=True)])]
        compact = sr.render_report(sections, sr.StatusConfig(), full=False, now=NOW)
        full = sr.render_report(sections, sr.StatusConfig(), full=True, now=NOW)
        self.assertNotIn("only full", compact)
        self.assertIn("only full", full)
        self.assertIn("THRESHOLDS", full)


class HostThresholdTests(unittest.TestCase):
    def levels(self, host, **cfg):
        _, items = sr._host_items(host, sr.StatusConfig(**cfg), NOW)
        return {t for i in items if i.level in (sr.WARN, sr.FAIL)
                for t in ([i.issue] if isinstance(i.issue, str) else i.issue)}

    def test_healthy_host_has_no_issues(self):
        self.assertEqual(self.levels(host_json()), set())

    def test_load_threshold(self):
        host = host_json(cpu={"count": 16, "usage_pct": 5, "load": [20, 17.0, 10]})
        self.assertEqual(self.levels(host), {"load5 17.0 on 16 cpus"})
        self.assertEqual(self.levels(host, load_warn_ratio=2.0), set())  # env-overridable

    def test_cpu_memory_swap(self):
        host = host_json(
            cpu={"count": 16, "usage_pct": 97, "load": [1, 1, 1]},
            mem={"total": 100, "available": 5, "swap_total": 100, "swap_free": 10},
        )
        issues = self.levels(host)
        self.assertIn("host cpu 97%", issues)
        self.assertIn("swap 90% used", issues)  # swap check wins the shared memory line

    def test_disk_warn_and_fail(self):
        def disks(pct):
            return [{"label": "root", "mount": "/", "total": 100, "used": pct, "pct": float(pct)}]

        self.assertEqual(self.levels(host_json(disks=disks(86))), {"disk root 86% full"})
        _, items = sr._host_items(host_json(disks=disks(96)), sr.StatusConfig(), NOW)
        disk = next(i for i in items if i.text.startswith("disk"))
        self.assertEqual(disk.level, sr.FAIL)

    def test_temps_vram_units_containers(self):
        host = host_json(
            temps={"cpu_c": 95.0, "gpu_c": 50, "nvme_c": 40},
            gpu={"busy_pct": 3, "vram_used": 99, "vram_total": 100},
            units={"states": {"ollama.service": "failed", "docker.service": "active",
                              "ollama-watchdog.timer": "inactive"}, "failed": ["x.service"]},
            containers=[{"name": "privateai-memory", "state": "running", "health": "unhealthy"}],
        )
        issues = self.levels(host)
        self.assertIn("cpu temperature 95C", issues)
        self.assertIn("VRAM 99% used", issues)
        self.assertTrue(any("ollama failed" in i or "watchdog-timer inactive" in i for i in issues))
        self.assertIn("container privateai-memory running/unhealthy", issues)

    def test_watchdog_restarts_threshold(self):
        wd = host_json()["watchdog"] | {"restarts_24h": 3, "last_restart_at": iso(NOW - 3600)}
        self.assertEqual(self.levels(host_json(watchdog=wd)), {"Ollama restarted 3x in 24h (watchdog)"})
        self.assertEqual(self.levels(host_json(watchdog=wd), watchdog_restarts_warn=5), set())

    def test_failed_units_are_informational_only(self):
        host = host_json()
        host["units"]["failed"] = ["foo.service"]
        self.assertEqual(self.levels(host), set())

    def test_same_volume_is_deduplicated(self):
        _, items = sr._host_items(host_json(), sr.StatusConfig(), NOW)
        disk = next(i for i in items if i.text.startswith("disk"))
        self.assertIn("data: same volume as root", disk.text)

    def test_garbage_fields_do_not_raise(self):
        host = {"cpu": "x", "mem": [], "disks": [1, None], "gpu": 5, "temps": None, "units": "u",
                "containers": [3], "watchdog": {"restarts_24h": "NaN"}, "errors": []}
        sr._host_items(host, sr.StatusConfig(), NOW)


# ------------------------------------------------------------------------- collectors
class HostFileTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, content: str | None, **cfg):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            if content is not None:
                (tmp / "host.json").write_text(content)
            return await sr.collect_host(ctx_for(tmp, cfg=cfg_for(tmp, **cfg)))

    async def test_fresh(self):
        section = await self.collect(json.dumps(host_json()))
        self.assertEqual(section.level, sr.OK)
        self.assertIn("up 1d1h", section.head)
        self.assertTrue(any("watchdog" in i.text for i in section.items))

    async def test_missing_is_warn_not_error(self):
        section = await self.collect(None)
        self.assertEqual(section.level, sr.WARN)
        self.assertIn("n/a (exporter not running", section.items[0].text)
        self.assertEqual(sr.compute_verdict([section])[0], "DEGRADED")

    async def test_stale(self):
        stale = host_json(generated_epoch=NOW - 3 * 3600, generated_at=iso(NOW - 3 * 3600))
        section = await self.collect(json.dumps(stale))
        self.assertEqual(section.level, sr.WARN)
        self.assertIn("exporter not running", section.items[0].text)
        self.assertIn("3h00m old", section.items[0].text)

    async def test_stale_threshold_is_configurable(self):
        slightly_old = host_json(generated_epoch=NOW - 200, generated_at=iso(NOW - 200))
        self.assertEqual((await self.collect(json.dumps(slightly_old))).level, sr.WARN)
        ok = await self.collect(json.dumps(slightly_old), host_stale_seconds=600)
        self.assertEqual(ok.level, sr.OK)

    async def test_corrupt_and_wrong_shape(self):
        for content, reason in (("{not json", "corrupt JSON"), ("[1,2]", "unexpected format"),
                                (json.dumps({"hostname": "x"}), "no timestamp")):
            section = await self.collect(content)
            self.assertEqual(section.level, sr.WARN, content)
            self.assertIn(reason, section.items[0].text)


class CgroupTests(unittest.TestCase):
    def write(self, tmp: Path, **files):
        tmp.mkdir(exist_ok=True)
        for name, content in files.items():
            (tmp / name.replace("_", ".")).write_text(content)

    def test_parses_v2_files(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(Path(d), memory_current="209715200\n", memory_max="536870912\n",
                       cpu_stat="usage_usec 5000000\nuser_usec 4000000\n", pids_current="12\n",
                       memory_events="low 0\noom_kill 2\n", cpu_max="200000 100000\n")
            sample = sr.read_cgroup(Path(d))
            assert sample is not None
            self.assertEqual(sample.mem_current, 209715200)
            self.assertEqual(sample.mem_max, 536870912)
            self.assertEqual(sample.cpu_usage_usec, 5_000_000)
            self.assertEqual(sample.pids, 12)
            self.assertEqual(sample.oom_kills, 2)
            self.assertEqual(sample.cpu_quota_cores, 2.0)

    def test_max_means_unlimited(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(Path(d), memory_current="100\n", memory_max="max\n", cpu_max="max 100000\n",
                       cpu_stat="usage_usec 1\n")
            sample = sr.read_cgroup(Path(d))
            assert sample is not None
            self.assertIsNone(sample.mem_max)
            self.assertIsNone(sample.cpu_quota_cores)

    def test_missing_dir_gives_none(self):
        self.assertIsNone(sr.read_cgroup(Path("/nonexistent/cgroup")))


class AgentCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_limit_warning_and_version(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "cgroup").mkdir()
            (tmp / "cgroup/memory.current").write_text(str(480 * 2**20))
            (tmp / "cgroup/memory.max").write_text(str(512 * 2**20))
            (tmp / "cgroup/cpu.stat").write_text("usage_usec 1000000\n")
            (tmp / "VERSION").write_text("commit=abcdef123456789\nbranch=feature/x\n")
            section = await sr.collect_agent(ctx_for(tmp))
            self.assertIn("v abcdef123456 (feature/x)", section.head)
            self.assertEqual(section.level, sr.WARN)
            self.assertIn("bot memory 94% of limit", sr.compute_verdict([section])[1][0])

    async def test_missing_cgroup_and_version_degrade_to_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            section = await sr.collect_agent(ctx_for(Path(d)))
            self.assertIn("v unknown", section.head)
            self.assertIn("cgroup v2 files not visible", section.items[0].text)
            self.assertEqual(section.level, sr.OK)

    def test_read_version_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "VERSION"
            self.assertEqual(sr.read_version(path), ("unknown", "unknown"))
            path.write_text("garbage\n")
            self.assertEqual(sr.read_version(path), ("unknown", "unknown"))
            path.write_text("commit=a77c974\nbranch=main\n")
            self.assertEqual(sr.read_version(path), ("a77c974", "main"))


PS_JSON = {
    "models": [
        {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest", "size": 376449269,
         "details": {"parameter_size": "137M", "quantization_level": "F16"},
         "expires_at": "2319-02-22T10:54:08.456299023Z", "size_vram": 0, "context_length": 2048},
        {"name": "gemma4:e4b-jt-8k", "model": "gemma4:e4b-jt-8k", "size": 3423223151,
         "details": {"parameter_size": "8.0B", "quantization_level": "Q4_K_M"},
         "expires_at": "2319-02-22T10:54:07.939684623Z", "size_vram": 3423223151, "context_length": 8192},
        "junk", {"name": ""},
    ]
}


class OllamaParsingTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_ps(self):
        models = parse_ps(PS_JSON)
        self.assertEqual([m.name for m in models], ["nomic-embed-text:latest", "gemma4:e4b-jt-8k"])
        embed, chat = models
        self.assertEqual(embed.placement, "CPU")
        self.assertEqual(chat.placement, "100% GPU")
        self.assertEqual(chat.context_length, 8192)
        self.assertEqual(chat.parameter_size, "8.0B")
        self.assertEqual(format_keep_alive(chat.expires_at, now=datetime(2026, 11, 12, tzinfo=timezone.utc)), "forever")

    def test_parse_ps_garbage(self):
        for payload in (None, [], {"models": None}, {"models": "x"}):
            self.assertEqual(parse_ps(payload), [])

    def test_split_placement_and_name_matching(self):
        self.assertEqual(model(size=100, vram=60).placement, "60% GPU/40% CPU")
        self.assertTrue(model_matches("nomic-embed-text:latest", "nomic-embed-text"))
        self.assertFalse(model_matches("a:1", "a:2"))

    def test_keep_alive_relative(self):
        now = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)
        soon = datetime(2026, 7, 20, 12, 4, 20, tzinfo=timezone.utc)
        self.assertEqual(format_keep_alive(soon, now=now), "expires in 4m")

    async def test_probe_uses_only_get_version_and_ps(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, request.url.path))
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.32.1"})
            return httpx.Response(200, json=PS_JSON)

        real = httpx.AsyncClient
        with patch("src.ollama_status.httpx.AsyncClient",
                   lambda **kw: real(transport=httpx.MockTransport(handler), **kw)):
            probe = await probe_ollama("http://x:11434", 1.0)
        self.assertEqual(sorted(seen), [("GET", "/api/ps"), ("GET", "/api/version")])
        self.assertEqual(probe.version, "0.32.1")
        self.assertEqual(len(probe.models), 2)
        self.assertTrue(probe.reachable)

    async def test_probe_unreachable_reports_category_only(self):
        def handler(request):
            raise httpx.ConnectError("boom http://user:pw@host/")

        real = httpx.AsyncClient
        with patch("src.ollama_status.httpx.AsyncClient",
                   lambda **kw: real(transport=httpx.MockTransport(handler), **kw)):
            probe = await probe_ollama("http://x:11434", 1.0)
        self.assertFalse(probe.reachable)
        self.assertEqual(probe.error, "connection refused/unreachable")
        self.assertNotIn("pw", probe.error)


class OllamaItemsTests(unittest.TestCase):
    def items(self, probe, stats=None, now=NOW, **cfg):
        with tempfile.TemporaryDirectory() as d:
            ctx = ctx_for(Path(d), cfg=cfg_for(Path(d), **cfg))
            ctx.now = now
            with patch("src.status_report.MODEL_STATS", stats or ModelCallStats()):
                return sr._ollama_items(ctx, probe)

    def probe(self, models, **kw):
        return OllamaProbe("0.32.1", models, 12.0, **kw)

    def test_healthy_gpu_model(self):
        items = self.items(self.probe([model(), model("nomic-embed-text:latest", 376_000_000, 0, 2048)]))
        chat = items[0]
        self.assertEqual(chat.level, sr.OK)
        self.assertIn("100% GPU", chat.text)
        self.assertIn("ctx 8192", chat.text)
        self.assertIn("keep-alive forever", chat.text)
        embed = items[1]
        self.assertIn("CPU", embed.text)
        self.assertEqual(embed.level, sr.OK)

    def test_unreachable_is_critical_fail(self):
        items = self.items(OllamaProbe(None, [], 5.0, error="timeout", ps_error="timeout"))
        self.assertEqual(items[0].level, sr.FAIL)
        self.assertTrue(items[0].critical)

    def test_not_loaded_is_informational(self):
        items = self.items(self.probe([]))
        self.assertEqual(items[0].level, sr.INFO)
        self.assertIn("not loaded", items[0].text)

    def test_partially_on_cpu_warns(self):
        items = self.items(self.probe([model(size=100, vram=60)]))
        self.assertEqual(items[0].level, sr.WARN)
        self.assertIn("not fully on GPU", items[0].issue)

    def test_context_mismatch_warns(self):
        items = self.items(self.probe([model(ctx=4096)]), num_ctx=8192)
        self.assertEqual(items[0].level, sr.WARN)
        self.assertIn("4096", items[0].issue)

    def test_embed_on_gpu_when_cpu_requested_warns(self):
        items = self.items(self.probe([model(), model("nomic-embed-text:latest", 100, 100)]),
                           rag_embed_num_gpu=0)
        self.assertEqual(items[1].level, sr.WARN)

    def test_ps_failure_warns(self):
        items = self.items(OllamaProbe("0.32.1", [], 5.0, ps_error="timeout"))
        self.assertEqual(items[0].level, sr.WARN)

    def test_call_stats_error_and_stuck(self):
        stats = ModelCallStats()
        ok = stats.begin()
        stats.end(ok, None)
        bad = stats.begin()
        stats.end(bad, "timeout")
        stats.record_retry()
        pending = stats.begin()
        stats._in_flight[pending] -= 120  # running for 2 minutes
        items = self.items(self.probe([model()]), stats=stats, now=time.time())
        texts = " | ".join(i.text for i in items)
        self.assertIn("errors 1/2", texts)
        self.assertIn("retries 1", texts)
        error = next(i for i in items if i.text.startswith("last model error"))
        self.assertEqual(error.level, sr.WARN)  # inside the 15 min window
        stuck = next(i for i in items if "in flight" in i.text)
        self.assertEqual(stuck.level, sr.WARN)
        self.assertIn("may be stuck", stuck.text)


class ModelStatsTests(unittest.TestCase):
    def test_ring_buffer_and_outcomes(self):
        stats = ModelCallStats(maxlen=3)
        for outcome in (None, "timeout", None, None):
            stats.end(stats.begin(), outcome)
        self.assertEqual(stats.total_calls, 4)
        self.assertEqual(stats.total_errors, 1)
        self.assertEqual(len(stats.recent_durations()), 2)  # maxlen 3 dropped the oldest
        self.assertEqual(stats.last_error.outcome, "timeout")
        self.assertIsNone(stats.last_call.outcome)
        self.assertEqual(stats.in_flight, 0)
        self.assertIsNone(stats.oldest_in_flight_seconds())

    def test_end_of_unknown_call_is_ignored(self):
        stats = ModelCallStats()
        stats.end(999, None)
        self.assertEqual(stats.total_calls, 0)


class ToolsCollectorTests(unittest.IsolatedAsyncioTestCase):
    def tool(self, name):
        return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}})

    async def test_snapshot_and_section(self):
        hub = McpHub(Path("/nonexistent"), readonly=True, allowlist="", memory_tools=False)
        hub._server_info["filesystem"] = {"transport": "stdio", "state": "connected", "error": None}
        hub._server_info["memory"] = {"transport": "sse", "state": "disabled", "error": None}
        for name in ("read_text_file", "list_directory", "write_file", "get_file_info"):
            hub._register_tool("filesystem", MagicMock(), self.tool(name))
        snap = hub.status_snapshot()
        fs = snap["servers"][0]
        self.assertEqual(fs["exposed"], ["filesystem_list_directory", "filesystem_read_text_file"])
        self.assertEqual(fs["hidden"], ["filesystem_get_file_info", "filesystem_write_file"])
        self.assertEqual(snap["allowlist"], "default list")
        with tempfile.TemporaryDirectory() as d:
            for full in (False, True):
                ctx = ctx_for(Path(d), hub=hub, full=full)
                section = await sr.collect_tools(ctx)
                self.assertEqual(section.level, sr.OK)
                text = sr.render_report([section], ctx.cfg, full=full, now=NOW)
                self.assertIn("filesystem (stdio): connected, 2 exposed, 2 hidden", text)
                self.assertIn("memory (sse): disabled", text)
                self.assertEqual("filesystem_read_text_file" in text, full)  # names only in full

    async def test_disconnected_server_warns(self):
        hub = McpHub(Path("/nonexistent"), allowlist="*")
        hub._server_info["memory"] = {"transport": "sse", "state": "failed", "error": "ConnectError"}
        with tempfile.TemporaryDirectory() as d:
            section = await sr.collect_tools(ctx_for(Path(d), hub=hub))
        self.assertEqual(section.level, sr.WARN)
        self.assertIn("'memory' disconnected", sr.compute_verdict([section])[1][0])

    async def test_no_hub_is_na(self):
        with tempfile.TemporaryDirectory() as d:
            section = await sr.collect_tools(ctx_for(Path(d)))
        self.assertEqual(section.items[0].level, sr.NA)

    async def test_start_records_stdio_and_failed_sse(self):
        path = Path(tempfile.mkdtemp()) / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {"memory": {"url": "http://127.0.0.1:1/sse"}}}))
        hub = McpHub(path, allowlist="", memory_tools=True, connect_timeout=1.0)
        await hub.start()
        self.addAsyncCleanup(hub.close)
        info = hub.status_snapshot()["servers"][0]
        self.assertEqual((info["transport"], info["state"]), ("sse", "disconnected"))
        self.assertTrue(info["error"])


class MemoryCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def run_memory(self, tmp: Path, service, records=10, summarized=10, rag=None, **cfg):
        write_legacy(tmp, records, summarized)
        memory = SimpleNamespace(data_dir=tmp / "sessions", recent_turns=4)
        ctx = ctx_for(tmp, cfg=cfg_for(tmp, **cfg), memory=memory, memory_service=service, rag=rag)
        ctx.legacy_task = asyncio.ensure_future(asyncio.to_thread(sr.scan_legacy_memory, memory.data_dir))
        return await sr.collect_memory(ctx)

    async def test_healthy_in_sync(self):
        with tempfile.TemporaryDirectory() as d:
            section = await self.run_memory(Path(d), FakeService(stats=SERVICE_STATS | {"turns": 10}))
            self.assertEqual(section.level, sr.OK)
            text = " | ".join(i.text for i in section.items)
            self.assertIn("reachable (12 ms)", text)
            self.assertIn("dual-write current", text)
            self.assertIn("breaker: closed", text)
            self.assertIn("DB 5.0 MiB", text)

    async def test_service_down_is_fail_but_not_critical_on_legacy_backend(self):
        with tempfile.TemporaryDirectory() as d:
            section = await self.run_memory(Path(d), FakeService(healthy=False, error="timeout"))
            self.assertEqual(section.level, sr.FAIL)
            self.assertEqual(sr.compute_verdict([section])[0], "DEGRADED")
            self.assertIn("UNREACHABLE (timeout)", section.items[0].text)

    async def test_service_down_on_service_backend_is_down(self):
        with tempfile.TemporaryDirectory() as d:
            service = FakeService(healthy=False)
            section = await self.run_memory(Path(d), service, memory_backend="service")
            self.assertEqual(sr.compute_verdict([section])[0], "DOWN")

    async def test_breaker_open_warns(self):
        with tempfile.TemporaryDirectory() as d:
            section = await self.run_memory(Path(d), FakeService(stats=SERVICE_STATS | {"turns": 10}, breaker="open"))
            self.assertIn("memory breaker open", sr.compute_verdict([section])[1])

    async def test_dual_write_drift_by_count_and_by_lag(self):
        with tempfile.TemporaryDirectory() as d:
            section = await self.run_memory(Path(d), FakeService(stats=SERVICE_STATS | {"turns": 10}), records=40,
                                            summarized=40)
            self.assertIn("memory dual-write drift", sr.compute_verdict([section])[1])
        with tempfile.TemporaryDirectory() as d:  # same count, service 20 min behind
            stats = SERVICE_STATS | {"turns": 10, "last_ts": iso(NOW - 1260)}
            section = await self.run_memory(Path(d), FakeService(stats=stats))
            self.assertIn("memory dual-write drift", sr.compute_verdict([section])[1])

    async def test_small_dedup_difference_is_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            section = await self.run_memory(Path(d), FakeService(stats=SERVICE_STATS | {"turns": 8}))
            self.assertEqual(section.level, sr.OK)

    async def test_backlog_and_stale_summary_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            section = await self.run_memory(Path(d), None, records=60, summarized=10)
            self.assertIn("summary backlog 50 > 24", sr.compute_verdict([section])[1])
        with tempfile.TemporaryDirectory() as d:  # backlog 20 >= trigger 12 and summary 10 days old
            tmp = Path(d)
            write_legacy(tmp, 30, 10, summary_updated=NOW - 10 * 86400)
            memory = SimpleNamespace(data_dir=tmp / "sessions", recent_turns=4, summarize_every=6)
            ctx = ctx_for(tmp, memory=memory)
            ctx.legacy_task = asyncio.ensure_future(asyncio.to_thread(sr.scan_legacy_memory, memory.data_dir))
            section = await sr.collect_memory(ctx)
            self.assertIn("summary older than 7d with backlog >= 12", sr.compute_verdict([section])[1])
        with tempfile.TemporaryDirectory() as d:  # quiet chat: old summary + small leftover backlog is normal
            tmp = Path(d)
            write_legacy(tmp, 14, 10, summary_updated=NOW - 100 * 86400)
            memory = SimpleNamespace(data_dir=tmp / "sessions", recent_turns=4, summarize_every=6)
            ctx = ctx_for(tmp, memory=memory)
            ctx.legacy_task = asyncio.ensure_future(asyncio.to_thread(sr.scan_legacy_memory, memory.data_dir))
            self.assertEqual((await sr.collect_memory(ctx)).level, sr.OK)
        with tempfile.TemporaryDirectory() as d:  # old summary but no backlog is fine
            section = await self.run_memory(Path(d), None, records=10, summarized=10)
            self.assertEqual(section.level, sr.OK)

    async def test_rag_scan_and_service_backend_note(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "rag").mkdir()
            conn = sqlite3.connect(tmp / "rag" / "123.sqlite3")
            conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)")
            conn.executemany("INSERT INTO chunks(content) VALUES (?)", [("a",), ("b",), ("c",)])
            conn.commit()
            conn.close()
            before = (tmp / "rag" / "123.sqlite3").stat().st_mtime_ns
            self.assertEqual(sr.scan_rag(tmp / "rag")[:2], (1, 3))
            self.assertEqual((tmp / "rag" / "123.sqlite3").stat().st_mtime_ns, before)  # read-only
            section = await self.run_memory(tmp, None, rag=object())
            self.assertTrue(any("RAG: on" in i.text and "3 chunks" in i.text for i in section.items))
            self.assertTrue(any("num_gpu 0 = CPU" in i.text for i in section.items))


class LegacyScanTests(unittest.TestCase):
    def test_scan_counts_without_text(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            directory = write_legacy(tmp, 6, 4, newest=NOW - 5, session="05abc")
            write_legacy(tmp, 2, session="9")
            scan = sr.scan_legacy_memory(directory)
            self.assertEqual(scan.records, 8)
            self.assertEqual(scan.summarized, 4)
            self.assertEqual(scan.backlog_max, 2)
            self.assertEqual(len(scan.sessions), 2)
            self.assertAlmostEqual(scan.newest_record_epoch, NOW - 5 - 0, delta=7)

    def test_missing_dir_and_corrupt_lines(self):
        self.assertEqual(sr.scan_legacy_memory(Path("/nonexistent/x")).records, 0)
        self.assertEqual(sr.scan_legacy_memory(None).error, "no memory store")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "1.jsonl").write_text("not json\n{}\n")
            (Path(d) / "1.summary.json").write_text("{broken")
            scan = sr.scan_legacy_memory(Path(d))
            self.assertEqual(scan.records, 2)
            self.assertEqual(scan.summarized, 0)


class SessionCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_section_has_settings_and_history_but_no_id(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            settings = SessionSettings(tmp / "s.json")
            settings.set_tool_mode("424242424", "never")
            write_legacy(tmp, 30, 10, session="424242424")
            memory = SimpleNamespace(data_dir=tmp / "sessions", recent_turns=4)
            ctx = ctx_for(tmp, settings=settings, session_id="424242424", memory=memory)
            ctx.legacy_task = asyncio.ensure_future(asyncio.to_thread(sr.scan_legacy_memory, memory.data_dir))
            section = await sr.collect_session(ctx)
            text = "\n".join(i.text for i in section.items)
            self.assertIn("tools off (default auto)", text)
            self.assertIn("history: 30 records", text)
            self.assertIn("last 8 records + summary 100 chars", text)
            self.assertNotIn("424242424", text)


# --------------------------------------------------------------- whole-report behaviour
class ReportShapeTests(unittest.IsolatedAsyncioTestCase):
    async def full_report(self, tmp: Path, full: bool, host=None, service=None, probe=None) -> str:
        write_legacy(tmp, 120, 100)
        (tmp / "host.json").write_text(json.dumps(host or host_json()))
        hub = McpHub(Path("/nonexistent"), allowlist="", memory_tools=False)
        hub._server_info["filesystem"] = {"transport": "stdio", "state": "connected", "error": None}
        for name in ("read_text_file", "list_directory", "write_file", "get_file_info", "directory_tree"):
            hub._register_tool("filesystem", MagicMock(), SimpleNamespace(
                name=name, description="d", inputSchema={"type": "object", "properties": {}}))
        memory = SimpleNamespace(data_dir=tmp / "sessions", recent_turns=4)
        settings = SessionSettings(tmp / "s.json")
        probe = probe or OllamaProbe("0.32.1", [model(), model("nomic-embed-text:latest", 376_000_000, 0, 2048)], 20.0)

        async def fake_probe(host, timeout):
            return probe

        with patch("src.status_report.probe_ollama", fake_probe):
            return await sr.build_status_report(
                settings=settings, session_id="777", memory=memory, hub=hub, rag=None,
                memory_service=service or FakeService(stats=SERVICE_STATS | {"turns": 118}),
                full=full, cfg=cfg_for(tmp))

    async def test_compact_fits_budget_and_full_chunks_for_telegram(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            host = host_json(watchdog=host_json()["watchdog"] | {
                "restarts_24h": 6, "hung_24h": 6, "last_restart_at": iso(NOW - 9000),
                "events": [{"ts": iso(NOW - 9000 + i), "kind": "RECOVERED", "text": "RECOVERED: ollama restarted " * 3}
                           for i in range(8)]}, reboot_required=True)
            compact = await self.full_report(tmp, False, host=host)
            full = await self.full_report(tmp, True, host=host)
        self.assertLess(len(compact), sr.COMPACT_TARGET_CHARS, compact)
        self.assertGreater(len(full), len(compact))
        for limit in (4096 - 96, 1800):  # Telegram / Session chunk sizes
            parts = sr.split_message(full, limit)
            self.assertTrue(all(len(p) <= limit for p in parts))
            self.assertEqual("\n\n".join(parts).replace("\n\n", "\n"), full.replace("\n\n", "\n"))
        self.assertTrue(compact.startswith("Overall: DEGRADED - Ollama restarted 6x in 24h (watchdog)"))
        for section in ("AGENT", "OLLAMA", "TOOLS", "MEMORY", "HOST", "THIS CHAT"):
            self.assertIn(section, compact)

    async def test_overall_ok_on_healthy_system(self):
        with tempfile.TemporaryDirectory() as d:
            report = await self.full_report(Path(d), False)
        # agent has no cgroup in the test config -> info only; everything else healthy
        self.assertTrue(report.startswith("Overall: OK"), report)

    async def test_split_message_edge_cases(self):
        self.assertEqual(sr.split_message("short", 100), ["short"])
        parts = sr.split_message("a" * 250, 100)
        self.assertEqual([len(p) for p in parts], [100, 100, 50])
        paragraphs = "\n\n".join(["x" * 60] * 5)
        self.assertTrue(all(len(p) <= 130 for p in sr.split_message(paragraphs, 130)))


class SecretSafetyTests(unittest.IsolatedAsyncioTestCase):
    TOKEN = "123456789:AAH" + "x" * 32
    MNEMONIC = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike"
    MEM = "m3m0ry-t0k3n-" + "9" * 20
    SESSION_ID = "05" + "ab12" * 16

    def test_redacts_secret_values_and_token_shapes(self):
        with patch.multiple("src.status_report.app_config", TELEGRAM_TOKEN=self.TOKEN,
                            SESSION_MNEMONIC=self.MNEMONIC, MEMORY_TOKEN=self.MEM, BRIDGE_TOKEN="bridge-secret-1",
                            ALLOWED_USER_ID=987654321, SESSION_ALLOWED_IDS=[self.SESSION_ID]):
            leaky = (f"a {self.TOKEN} b {self.MNEMONIC} c {self.MEM} d bridge-secret-1 e 987654321 "
                     f"f {self.SESSION_ID} g http://user:pw@host/ h Bearer abcdefgh12345 "
                     f"i {'0f' * 25} j {'Zz9_' * 14}")
            clean = sr.redact_secrets(leaky)
        for needle in (self.TOKEN, self.MNEMONIC, self.MEM, "bridge-secret-1", "987654321", self.SESSION_ID,
                       "user:pw@", "abcdefgh12345", "0f" * 25, "Zz9_" * 14):
            self.assertNotIn(needle, clean)
        self.assertIn("[redacted]", clean)

    def test_normal_text_is_untouched(self):
        text = "Overall: OK - gemma4:e4b-jt-8k 100% GPU 3.2 GiB ctx 8192 host 10.0.0.181:11434 a77c974"
        self.assertEqual(sr.redact_secrets(text), text)

    def test_safe_host_drops_credentials_path_and_query(self):
        self.assertEqual(sr.safe_host("http://user:pw@10.0.0.181:11434/x?token=1"), "10.0.0.181:11434")
        self.assertEqual(sr.safe_host("http://memory-service:8765"), "memory-service:8765")
        self.assertEqual(sr.safe_host(""), "n/a")

    async def test_rendered_report_contains_no_secret_values(self):
        with tempfile.TemporaryDirectory() as d, patch.multiple(
            "src.status_report.app_config", TELEGRAM_TOKEN=self.TOKEN, SESSION_MNEMONIC=self.MNEMONIC,
            MEMORY_TOKEN=self.MEM, ALLOWED_USER_ID=987654321, SESSION_ALLOWED_IDS=[self.SESSION_ID]):
            tmp = Path(d)
            # Hostile inputs: credentials in URLs, secrets inside host.json text.
            host = host_json(watchdog=host_json()["watchdog"] | {
                "last_line": {"ts": iso(NOW), "text": f"leak {self.TOKEN} {self.MEM}"}})
            (tmp / "host.json").write_text(json.dumps(host))
            cfg = cfg_for(tmp, ollama_host=f"http://bot:{self.MEM}@10.0.0.181:11434/api",
                          memory_service_url=f"http://u:{self.MEM}@memory-service:8765")

            async def fake_probe(host, timeout):
                return OllamaProbe("0.32.1", [model()], 5.0)

            with patch("src.status_report.probe_ollama", fake_probe):
                for full in (False, True):
                    text = await sr.build_status_report(
                        settings=SessionSettings(tmp / "s.json"), session_id=self.SESSION_ID,
                        memory_service=FakeService(stats=SERVICE_STATS), full=full, cfg=cfg)
                    for needle in (self.TOKEN, self.MNEMONIC, self.MEM, "987654321", self.SESSION_ID):
                        self.assertNotIn(needle, text)
                    self.assertIn("10.0.0.181:11434", text)


class FailureIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_raising_and_hanging_collectors_still_yield_a_full_report(self):
        async def boom(ctx):
            raise RuntimeError("secret detail that must not leak")

        async def hang(ctx):
            await asyncio.sleep(30)

        async def fine(ctx):
            return sr.Section("FINE", "ok", [sr.Item("works", sr.OK)])

        with tempfile.TemporaryDirectory() as d:
            cfg = cfg_for(Path(d), collector_timeout=0.2)
            started = time.monotonic()
            text = await sr.build_status_report(
                cfg=cfg, collectors=(("BOOM", boom), ("HANG", hang), ("FINE", fine)))
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.0)
        self.assertIn("n/a (RuntimeError)", text)
        self.assertIn("n/a (timed out after 0.2s)", text)
        self.assertNotIn("secret detail", text)
        self.assertIn("works", text)
        self.assertTrue(text.startswith("Overall: DEGRADED"))
        self.assertIn("boom check failed", text.splitlines()[0].lower())

    async def test_real_collectors_survive_missing_everything(self):
        with tempfile.TemporaryDirectory() as d:
            async def dead_probe(host, timeout):
                raise OSError("network down")

            with patch("src.status_report.probe_ollama", dead_probe):
                text = await sr.build_status_report(cfg=cfg_for(Path(d)))
        self.assertIn("OLLAMA", text)
        self.assertIn("n/a (OSError)", text)
        self.assertIn("HOST", text)
        self.assertIn("n/a (exporter not running", text)
        self.assertIn("MEMORY", text)

    async def test_handler_is_fast_with_concurrent_probes(self):
        async def slow_probe(host, timeout):
            await asyncio.sleep(0.5)
            return OllamaProbe("1", [], 500.0)

        with tempfile.TemporaryDirectory() as d, patch("src.status_report.probe_ollama", slow_probe):
            service = FakeService(stats=SERVICE_STATS)

            async def slow(timeout=2.0):
                await asyncio.sleep(0.5)
                return service._probe

            service.probe = slow
            started = time.monotonic()
            await sr.build_status_report(cfg=cfg_for(Path(d)), memory_service=service,
                                         memory=SimpleNamespace(data_dir=Path(d), recent_turns=4))
            self.assertLess(time.monotonic() - started, 1.5)  # probes overlap, not add up


class GenVersionTests(unittest.TestCase):
    def setUp(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gen_version", Path(__file__).resolve().parent.parent / "scripts" / "gen_version.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    SHA = "a77c974" + "0" * 33

    def test_loose_ref(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d)
            (meta / "HEAD").write_text("ref: refs/heads/feature/memory-service\n")
            (meta / "refs/heads/feature").mkdir(parents=True)
            (meta / "refs/heads/feature/memory-service").write_text(self.SHA + "\n")
            self.assertEqual(self.mod.read_git_meta(meta), ("a77c97400000", "feature/memory-service"))

    def test_packed_ref_detached_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d)
            (meta / "HEAD").write_text("ref: refs/heads/main\n")
            (meta / "packed-refs").write_text(f"# pack-refs\n{self.SHA} refs/heads/main\n")
            self.assertEqual(self.mod.read_git_meta(meta), ("a77c97400000", "main"))
            (meta / "HEAD").write_text(self.SHA + "\n")
            self.assertEqual(self.mod.read_git_meta(meta), ("a77c97400000", "detached"))
        self.assertEqual(self.mod.read_git_meta(Path("/nonexistent")), ("unknown", "unknown"))

    def test_main_writes_file_and_env_overrides(self):
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {"GIT_COMMIT": "deadbeefcafe1234",
                                                                           "GIT_BRANCH": "custom"}):
            out = Path(d) / "VERSION"
            self.assertEqual(self.mod.main(["x", d, str(out)]), 0)
            self.assertEqual(out.read_text(), "commit=deadbeefcafe\nbranch=custom\n")


class TelegramChunkingTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_command_reply_is_sent_in_telegram_sized_parts(self):
        from src import bot

        text = "\n\n".join(f"section {i}\n" + "line of status text\n" * 40 for i in range(12))
        self.assertGreater(len(text), 4096)
        message = SimpleNamespace(text="/status full", reply_text=unittest.mock.AsyncMock())
        update = SimpleNamespace(message=message, effective_user=SimpleNamespace(id=5),
                                 effective_chat=SimpleNamespace(id=5))
        context = SimpleNamespace(application=SimpleNamespace(bot_data={"runtime": SimpleNamespace(
            settings=None, memory_service=None)}))
        with patch.object(bot, "ALLOWED_USER_ID", 5), patch.object(
                bot, "handle_command", unittest.mock.AsyncMock(return_value=text)) as handler:
            await bot._reply_command(update, context)
        sent = [call.args[0] for call in message.reply_text.await_args_list]
        self.assertGreater(len(sent), 1)
        self.assertTrue(all(len(part) <= 4096 for part in sent))
        self.assertIn("runtime", handler.await_args.kwargs)


if __name__ == "__main__":
    unittest.main()
