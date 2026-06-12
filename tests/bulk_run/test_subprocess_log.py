"""
Tests for ``guild.tools.subprocess_log.write_subprocess_log``.

That helper backs the per-combination ``*.subprocess.log`` files our docking
helpers emit so external consumers can read a focused, predictable
transcript when a method fails — rather than grepping the batch-wide
``output.log``. The helper's contract is:

- Always writes a complete transcript (success or failure).
- Creates parent dirs if missing.
- Decodes bytes stdout/stderr; tolerates None.
- Quotes argv for reproducibility.
"""

from pathlib import Path

from guild.tools.subprocess_log import write_subprocess_log


def test_success_transcript(tmp_path: Path):
    log = tmp_path / "method" / "combo.subprocess.log"
    write_subprocess_log(
        log,
        argv=["/usr/bin/echo", "hello world"],
        returncode=0,
        stdout="hello world\n",
        stderr="",
    )

    text = log.read_text()
    assert "[INVOCATION SUCCESS]" in text
    assert "argv: /usr/bin/echo 'hello world'" in text
    assert "[STDOUT]\nhello world" in text
    assert "[STDERR]\n" in text


def test_failure_transcript_with_header(tmp_path: Path):
    log = tmp_path / "boltz.subprocess.log"
    write_subprocess_log(
        log,
        argv=["boltz", "predict", "in.yaml"],
        returncode=1,
        stdout="some progress...\n",
        stderr="ImportError: foo\n",
        extra_header="exited 1 — Boltz crashed before writing manifest",
    )

    text = log.read_text()
    assert "[INVOCATION FAILED (exit 1)]" in text
    assert "note: exited 1" in text
    assert "ImportError: foo" in text


def test_bytes_decoded(tmp_path: Path):
    log = tmp_path / "b.log"
    write_subprocess_log(
        log,
        argv=["true"],
        returncode=0,
        stdout=b"binary out\n",
        stderr=b"\xc3\xa9rror\n",  # 'érror' UTF-8
    )
    text = log.read_text()
    assert "binary out" in text
    assert "érror" in text


def test_none_blobs_tolerated(tmp_path: Path):
    log = tmp_path / "n.log"
    write_subprocess_log(log, argv=["x"], returncode=0, stdout=None, stderr=None)
    # Should not raise; file should still exist with the header block.
    assert "[INVOCATION SUCCESS]" in log.read_text()


def test_parent_dirs_created(tmp_path: Path):
    log = tmp_path / "deep" / "nested" / "path" / "combo.log"
    assert not log.parent.exists()
    write_subprocess_log(log, argv=["x"], returncode=0, stdout="", stderr="")
    assert log.exists()
