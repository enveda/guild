"""
Helper for writing per-invocation subprocess logs.

guild's docking helpers (``deploy_boltz``, ``deploy_diffdock``,
``deploy_gnina``) invoke external CLIs via ``subprocess.run``. When one of
them fails (or even succeeds in a degraded way — e.g. Boltz returning an
empty manifest with exit code 0), the orchestrator surfaces a FAILED line
in the progress log but the actual stderr/stdout is left mixed inside the
batch-wide ``output.log``. That makes debugging from outside the container
painful for the user.

This helper writes a focused, predictable subprocess transcript to
``batches/<batch>/<method>/<run_id>.subprocess.log`` (or
``batches/<batch>/<method>/_batch.subprocess.log`` for batch-level methods
like DiffDock). Always written, success or failure — the file is small
(typical subprocess stdout under ~100 KB) and "always available" is the
contract external consumers actually want when something looks wrong.
"""

from __future__ import annotations

import os
import shlex
from typing import Sequence


def write_subprocess_log(
    log_path: str | os.PathLike,
    argv: Sequence[str],
    returncode: int,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    extra_header: str | None = None,
) -> None:
    """
    Write a full subprocess transcript to ``log_path``.

    :param log_path: Destination file. Parent dirs are created if missing.
    :param argv: The argv passed to subprocess.run (recorded for reproducibility).
    :param returncode: Exit code (``0`` = success).
    :param stdout: Captured stdout. ``None`` is treated as empty.
    :param stderr: Captured stderr. ``None`` is treated as empty.
    :param extra_header: Optional extra context line above the [STDOUT] block —
        useful for non-zero-exit-but-logically-failed cases (e.g. Boltz empty
        manifest).
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    status = "SUCCESS" if returncode == 0 else f"FAILED (exit {returncode})"
    quoted_argv = " ".join(shlex.quote(str(a)) for a in argv)

    def _decode(blob):
        if blob is None:
            return ""
        if isinstance(blob, bytes):
            return blob.decode("utf-8", errors="replace")
        return blob

    with open(log_path, "w") as f:
        f.write(f"[INVOCATION {status}]\n")
        f.write(f"argv: {quoted_argv}\n")
        if extra_header:
            f.write(f"note: {extra_header}\n")
        f.write("\n[STDOUT]\n")
        f.write(_decode(stdout))
        f.write("\n\n[STDERR]\n")
        f.write(_decode(stderr))
        f.write("\n")
