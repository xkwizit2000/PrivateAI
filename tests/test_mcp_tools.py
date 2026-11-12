"""Tests for MCP tool allowlist filtering and description trimming."""

from __future__ import annotations

import copy
import json
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.mcp_client import (
    DEFAULT_TOOL_ALLOWLIST,
    McpHub,
    parse_tool_allowlist,
    trim_description,
    trim_schema_descriptions,
)

ALL_TOOLS = [
    "read_file",
    "read_text_file",
    "read_media_file",
    "read_multiple_files",
    "write_file",
    "append_file",
    "edit_file",
    "create_directory",
    "list_directory",
    "list_directory_with_sizes",
    "directory_tree",
    "move_file",
    "copy_file",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
]

LONG_DESC = (
    "Read the complete contents of a file as text. DEPRECATED: Use read_text_file "
    "instead. Handles various text encodings and provides detailed error messages "
    "if the file cannot be read. Only works within allowed directories."
)


def _tool(name: str, description: str = LONG_DESC) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file. " * 8},
                "mode": {"type": "string", "enum": ["a", "b"], "description": "Short."},
            },
            "required": ["path"],
        },
    )


def _hub(allowlist: str | None = "", readonly: bool = False, **kwargs) -> McpHub:
    hub = McpHub(Path("/nonexistent/mcp.json"), readonly=readonly, allowlist=allowlist, **kwargs)
    for name in ALL_TOOLS:
        hub._register_tool("filesystem", MagicMock(), _tool(name))
    return hub


class ParseAllowlistTests(unittest.TestCase):
    def test_empty_uses_default(self):
        for raw in ("", "   ", None, " , ,"):
            self.assertEqual(
                parse_tool_allowlist(raw),
                frozenset(DEFAULT_TOOL_ALLOWLIST),
            )

    def test_all_values(self):
        self.assertIsNone(parse_tool_allowlist("*"))
        self.assertIsNone(parse_tool_allowlist("all"))
        self.assertIsNone(parse_tool_allowlist(" ALL "))
        self.assertIsNone(parse_tool_allowlist("read_text_file, *"))

    def test_normalizes_case_and_whitespace(self):
        self.assertEqual(
            parse_tool_allowlist(" Read_Text_File ,LIST_directory,"),
            frozenset({"read_text_file", "list_directory"}),
        )


class AllowlistFilteringTests(unittest.TestCase):
    def test_default_exposes_nine_tools(self):
        hub = _hub("")
        self.assertEqual(
            hub.tool_names,
            sorted(f"filesystem_{n}" for n in DEFAULT_TOOL_ALLOWLIST),
        )
        self.assertEqual(len(hub.tool_names), 9)
        for hidden in (
            "read_file",
            "read_media_file",
            "append_file",
            "copy_file",
            "list_directory_with_sizes",
            "get_file_info",
            "list_allowed_directories",
        ):
            self.assertNotIn(f"filesystem_{hidden}", hub.tool_names)

    def test_local_name_matches(self):
        hub = _hub("read_text_file,list_directory")
        self.assertEqual(hub.tool_names, ["filesystem_list_directory", "filesystem_read_text_file"])

    def test_registered_name_matches_case_insensitively(self):
        hub = _hub("Filesystem_Read_Text_File, LIST_DIRECTORY")
        self.assertEqual(hub.tool_names, ["filesystem_list_directory", "filesystem_read_text_file"])

    def test_server_dot_tool_form_matches(self):
        hub = _hub("filesystem.directory_tree")
        self.assertEqual(hub.tool_names, ["filesystem_directory_tree"])

    def test_all_and_star_expose_everything(self):
        self.assertEqual(len(_hub("all").tool_names), 16)
        self.assertEqual(len(_hub("*").tool_names), 16)

    def test_unknown_names_warn_not_crash(self):
        hub = _hub("read_text_file,does_not_exist")
        with self.assertLogs("src.mcp_client", level=logging.WARNING) as logs:
            unknown = hub.warn_unknown_allowlist_entries()
        self.assertEqual(unknown, ["does_not_exist"])
        self.assertTrue(any("does_not_exist" in line for line in logs.output))
        self.assertEqual(hub.tool_names, ["filesystem_read_text_file"])

    def test_known_names_do_not_warn(self):
        hub = _hub("read_file,filesystem_copy_file")
        self.assertEqual(hub.warn_unknown_allowlist_entries(), [])

    def test_allowlist_applies_on_top_of_readonly(self):
        hub = _hub("all", readonly=True)
        names = hub.tool_names
        self.assertNotIn("filesystem_write_file", names)
        self.assertIn("filesystem_read_text_file", names)
        # allowlisting a write tool does not override readonly
        hub = _hub("write_file,read_text_file", readonly=True)
        self.assertEqual(hub.tool_names, ["filesystem_read_text_file"])

    def test_readonly_with_default_allowlist(self):
        hub = _hub("", readonly=True)
        for name in ("write_file", "edit_file", "create_directory", "move_file"):
            self.assertNotIn(f"filesystem_{name}", hub.tool_names)
        self.assertIn("filesystem_list_directory", hub.tool_names)

    def test_ollama_tools_only_contain_exposed(self):
        hub = _hub("")
        names = [t["function"]["name"] for t in hub.as_ollama_tools()]
        self.assertEqual(sorted(names), hub.tool_names)


class HiddenToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_tool_call_rejected_gracefully(self):
        session = MagicMock()
        session.call_tool = AsyncMock()
        hub = McpHub(Path("/x"), readonly=False, allowlist="")
        hub._register_tool("filesystem", session, _tool("read_text_file"))
        hub._register_tool("filesystem", session, _tool("copy_file"))
        with self.assertLogs("src.mcp_client", level=logging.WARNING):
            result = await hub.call_tool("filesystem_copy_file", {"a": 1})
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("not available", result)
        self.assertIn("filesystem_read_text_file", result)
        session.call_tool.assert_not_awaited()

    async def test_unknown_tool_call_rejected(self):
        hub = McpHub(Path("/x"), readonly=False, allowlist="")
        result = await hub.call_tool("totally_made_up", {})
        self.assertEqual(result, "Error: unknown tool 'totally_made_up'")

    async def test_exposed_tool_call_goes_through(self):
        session = MagicMock()
        session.call_tool = AsyncMock(
            return_value=SimpleNamespace(
                content=[SimpleNamespace(text="hello")], isError=False
            )
        )
        hub = McpHub(Path("/x"), readonly=False, allowlist="")
        hub._register_tool("filesystem", session, _tool("read_text_file"))
        self.assertEqual(await hub.call_tool("filesystem_read_text_file", {}), "hello")
        session.call_tool.assert_awaited_once_with("read_text_file", arguments={})


class TrimDescriptionTests(unittest.TestCase):
    def test_unlimited(self):
        self.assertEqual(trim_description(LONG_DESC, 0), LONG_DESC)

    def test_short_text_unchanged_apart_from_whitespace(self):
        self.assertEqual(trim_description("Short one.", 200), "Short one.")
        self.assertEqual(trim_description("a\n  b", 200), "a b")

    def test_cuts_at_sentence_boundary(self):
        out = trim_description(LONG_DESC, 100)
        self.assertEqual(
            out,
            "Read the complete contents of a file as text. "
            "DEPRECATED: Use read_text_file instead.",
        )
        self.assertLessEqual(len(out), 100)

    def test_keeps_multiple_sentences_up_to_cap(self):
        out = trim_description(LONG_DESC, 200)
        self.assertLessEqual(len(out), 200)
        self.assertTrue(out.endswith("."))
        self.assertTrue(LONG_DESC.startswith(out))

    def test_cuts_at_word_boundary_when_first_sentence_too_long(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        out = trim_description(text, 25)
        self.assertTrue(out.endswith("..."))
        body = out[:-3]
        self.assertLessEqual(len(out), 25)
        self.assertIn(body, text)
        self.assertIn(body.split()[-1], text.split())  # no half word
        self.assertEqual(body, "alpha beta gamma delta")

    def test_no_mid_word_cut_when_boundary_falls_on_space(self):
        out = trim_description("aaaa bbbb cccc dddd eeee", 13)
        self.assertEqual(out, "aaaa bbbb...")

    def test_param_cap(self):
        out = trim_description("Absolute path to the file. " * 8, 80)
        self.assertLessEqual(len(out), 80)
        self.assertTrue(out.endswith("file."))


class TrimSchemaTests(unittest.TestCase):
    def test_structure_unchanged_only_descriptions_shorter(self):
        schema = {
            "type": "object",
            "description": "Root. " * 40,
            "properties": {
                "path": {"type": "string", "description": "Path to file. " * 20},
                "description": {"type": "string", "description": "A property named description. " * 10},
                "mode": {"type": "string", "enum": ["a", "b"]},
                "nested": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"x": {"type": "integer", "description": "x. " * 50}}},
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }
        original = copy.deepcopy(schema)
        trimmed = trim_schema_descriptions(schema, 60)
        self.assertEqual(schema, original, "input must not be mutated")

        def strip(node):
            if isinstance(node, dict):
                return {
                    k: strip(v)
                    for k, v in node.items()
                    if not (k == "description" and isinstance(v, str))
                }
            if isinstance(node, list):
                return [strip(v) for v in node]
            return node

        self.assertEqual(strip(trimmed), strip(original))
        self.assertIn("description", trimmed["properties"])  # property named "description" kept
        self.assertEqual(trimmed["properties"]["mode"]["enum"], ["a", "b"])
        self.assertEqual(trimmed["required"], ["path"])
        for node in (
            trimmed,
            trimmed["properties"]["path"],
            trimmed["properties"]["description"],
            trimmed["properties"]["nested"]["items"]["properties"]["x"],
        ):
            self.assertLessEqual(len(node["description"]), 60)

    def test_unlimited_returns_equal_copy(self):
        schema = {"type": "object", "properties": {"p": {"type": "string", "description": "x" * 500}}}
        out = trim_schema_descriptions(schema, 0)
        self.assertEqual(out, schema)
        self.assertIsNot(out, schema)


class HubPayloadTests(unittest.TestCase):
    def test_payload_names_params_unchanged_and_smaller(self):
        hub = _hub("all", desc_max_chars=120, param_desc_max_chars=40)
        full = hub.as_ollama_tools(trimmed=False)
        short = hub.as_ollama_tools()
        self.assertEqual(len(full), len(short))
        for a, b in zip(full, short):
            self.assertEqual(a["type"], b["type"])
            self.assertEqual(a["function"]["name"], b["function"]["name"])
            pa, pb = a["function"]["parameters"], b["function"]["parameters"]
            self.assertEqual(pa["required"], pb["required"])
            self.assertEqual(pa["type"], pb["type"])
            self.assertEqual(list(pa["properties"]), list(pb["properties"]))
            self.assertEqual(pa["properties"]["mode"]["enum"], pb["properties"]["mode"]["enum"])
            self.assertLessEqual(len(b["function"]["description"]), 120)
        self.assertLess(len(json.dumps(short)), len(json.dumps(full)))

    def test_originals_kept_untrimmed(self):
        hub = _hub("all", desc_max_chars=50)
        self.assertEqual(hub._tools["filesystem_read_file"]["description"], LONG_DESC)
        self.assertEqual(hub.as_ollama_tools(trimmed=False)[0]["function"]["description"], LONG_DESC)

    def test_zero_means_unlimited(self):
        hub = _hub("all", desc_max_chars=0, param_desc_max_chars=0)
        self.assertEqual(hub.as_ollama_tools(), hub.as_ollama_tools(trimmed=False))


if __name__ == "__main__":
    unittest.main()
