#!/usr/bin/env python3
"""Write the build's VERSION file (commit + branch) for the /status report.

Used by the Dockerfile's ``version`` stage: it reads the few git metadata files that
.dockerignore lets into the build context (HEAD, refs/, packed-refs), so a plain
``docker compose build`` records the exact commit without git or build args. Explicit
GIT_COMMIT / GIT_BRANCH build args (env here) win when set. Never fails the build:
anything it cannot determine is written as ``unknown``.

Usage: gen_version.py <gitmeta-dir> <output-file>
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def read_git_meta(meta: Path) -> tuple[str, str]:
    """Return (commit, branch) from a copy of the .git metadata, 'unknown' if absent."""
    try:
        head = (meta / "HEAD").read_text().strip()
    except OSError:
        return "unknown", "unknown"
    if _SHA_RE.match(head):  # detached HEAD
        return head[:12], "detached"
    if not head.startswith("ref: "):
        return "unknown", "unknown"
    ref = head[5:].strip()
    branch = ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
    sha = ""
    try:
        sha = (meta / ref).read_text().strip()
    except OSError:
        try:
            for line in (meta / "packed-refs").read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    sha = parts[0]
        except OSError:
            pass
    return (sha[:12] if _SHA_RE.match(sha) else "unknown"), branch


def main(argv: list[str]) -> int:
    """Write the VERSION file; always exits 0."""
    meta, out = Path(argv[1]), Path(argv[2])
    commit, branch = read_git_meta(meta)
    commit = os.environ.get("GIT_COMMIT", "").strip()[:12] or commit
    branch = os.environ.get("GIT_BRANCH", "").strip() or branch
    out.write_text(f"commit={commit}\nbranch={branch}\n")
    print(f"VERSION: commit={commit} branch={branch}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
