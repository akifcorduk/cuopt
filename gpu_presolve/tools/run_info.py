# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity of a run: source tree, tool binary, host, start time and load.

The runners write it into every run record and summary, and reuse a stored
result (--skip-done) only when the identity of the run that made it is the
identity of the current run.

The source tree is src, include, tools, check and CMakeLists.txt of the
project. Its commit is the last commit that changed one of them, so a commit
of results does not make earlier results stale; "commit" is the checked-out
commit. The checksum of the files tells two trees apart where git is not
available or the tree has changes that are not committed.

Run as a script, it prints the source identity on one line (for the step
markers of the shell scripts).
"""

import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATHS = ("src", "include", "tools", "check", "CMakeLists.txt")


def git(*args):
    """Return the output of a git command in the project, or None."""
    try:
        done = subprocess.run(
            ("git", "-C", str(PROJECT_DIR)) + args,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def file_checksum(path):
    """Return the SHA-256 of a file, or None when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def source_files():
    """Return the files of the source tree, sorted by relative path."""
    files = []
    for name in SOURCE_PATHS:
        path = PROJECT_DIR / name
        if path.is_file():
            files.append(path)
        else:
            files += [
                p
                for p in path.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            ]
    return sorted(files, key=lambda p: p.relative_to(PROJECT_DIR).as_posix())


def source_identity():
    """Return commit, source commit, uncommitted files and source checksum."""
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(path.relative_to(PROJECT_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update((file_checksum(path) or "unreadable").encode())
        digest.update(b"\0")
    dirty = git("status", "--porcelain", "--", *SOURCE_PATHS)
    return {
        "commit": git("rev-parse", "HEAD"),
        "source_commit": git("log", "-1", "--format=%H", "--", *SOURCE_PATHS),
        "source_dirty": None if dirty is None else dirty.splitlines(),
        "source_checksum": digest.hexdigest(),
    }


def run_identity(tool):
    """Return the source identity with the tool binary and its checksum."""
    identity = source_identity()
    identity["tool"] = str(tool)
    identity["tool_checksum"] = file_checksum(tool)
    return identity


def same_identity(a, b):
    """Return whether two runs have the same source tree and tool binary."""
    if not a or not b:
        return False
    for key in ("source_checksum", "tool_checksum"):
        if a.get(key) is None or a.get(key) != b.get(key):
            return False
    return a.get("source_commit") == b.get("source_commit")


def host_state():
    """Return host name, the time and the load average of this moment."""
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {
        "host": socket.gethostname(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_average": load,
    }


def main():
    """Print the source identity on one line."""
    identity = source_identity()
    dirty = identity["source_dirty"]
    print(
        "%s %s %s"
        % (
            identity["source_commit"] or "no-commit",
            identity["source_checksum"],
            "no-git" if dirty is None else "dirty" if dirty else "clean",
        )
    )


if __name__ == "__main__":
    sys.exit(main())
