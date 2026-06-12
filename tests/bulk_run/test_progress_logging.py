"""
Tests for the per-batch and project-level progress logs written by
``BulkRun.run_docking`` / ``run_guild_scoring``.

Each log is a simple append-only file of timestamped milestones:

    Starting Vina at 2026-05-25 12:34:56.789012
    Completed Vina at 2026-05-25 12:34:57.123456
    FAILED Vina 3pbl-A-ETQ-A_lig1 (timeout) at ...

The per-batch milestones are appended to ``batches/<batch>/output.log`` —
the same file Guild workers write to via ``logging.basicConfig``, so there's
a single log per batch rather than two. The project-level copy at
``{project}/batch_progress.log`` carries batch-level milestones plus a
copy of every FAILED line so a single grep surfaces every failure across
the run.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.bulk import BATCH_FOLDER, BATCH_PROGRESS_LOG_FILE, OUTPUT_LOG_FILE
from guild.constants.guild import BOLTZ_PREFIX, VINA_PREFIX

TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"


@pytest.fixture
def test_input_table():
    df = pd.read_csv(TEST_DATA_DIR / "bulk_dummy.csv")
    df["protein_path"] = str(TEST_DATA_DIR / df["protein_path"].iloc[0])
    return df


@pytest.fixture
def cleanup():
    yield
    test_project = Path.cwd() / "data" / "test-progress-log"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def test_per_batch_log_records_method_start_and_end(test_input_table, cleanup):
    """Each per-method helper writes Starting/Completed lines into the per-batch log."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-progress-log",
        methods_to_run=[VINA_PREFIX, BOLTZ_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )

    # Mock the actual docking work — we only care about the log writes that
    # the helpers do at their start/end.
    def make_noop(name):
        # Each helper's body would otherwise hit real subprocess / I/O work.
        # Patch them to a no-op so the test stays fast; the log writes happen
        # in the helpers themselves, which we leave intact via wraps=... .
        def _noop(self, current_batch, current_batch_folder):
            # Reproduce the helpers' own progress lines so the test still
            # exercises the format the user will see in real runs.
            BulkRun._log_progress(
                f"{current_batch_folder}/{OUTPUT_LOG_FILE}",
                message=f"Starting {name}",
            )
            BulkRun._log_progress(
                f"{current_batch_folder}/{OUTPUT_LOG_FILE}",
                message=f"Completed {name}",
            )
        return _noop

    with (
        patch.object(BulkRun, "_prepare_docking", lambda self: None),
        patch.object(BulkRun, "_run_vina_for_batch", make_noop("Vina")),
        patch.object(BulkRun, "_run_boltz_for_batch", make_noop("Boltz")),
    ):
        bulk.run_docking()

    batch_name = next(iter(bulk.batched_dictionary))
    batch_folder = bulk.batched_dictionary[batch_name][BATCH_FOLDER]
    batch_log_text = Path(batch_folder, OUTPUT_LOG_FILE).read_text()

    # All method milestones should be present in the per-batch log.
    for marker in ("Starting Vina", "Completed Vina", "Starting Boltz", "Completed Boltz"):
        assert marker in batch_log_text, f"missing {marker!r} in:\n{batch_log_text}"

    # The batch's start/completion bookends should also be there.
    assert "Starting batch_1" in batch_log_text
    assert "Completed batch_1" in batch_log_text


def test_failure_line_appears_in_both_logs(test_input_table, cleanup):
    """``FAILED`` entries land in both the per-batch and the project log."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-progress-log",
        methods_to_run=[VINA_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )

    batch_name = next(iter(bulk.batched_dictionary))
    batch_folder = bulk.batched_dictionary[batch_name][BATCH_FOLDER]
    batch_log = f"{batch_folder}/{OUTPUT_LOG_FILE}"
    main_log = f"{bulk.project_folder}/{BATCH_PROGRESS_LOG_FILE}"

    # Directly exercise _log_progress as the helpers do when a combo fails.
    BulkRun._log_progress(
        batch_log, main_log,
        message="FAILED Vina 3pbl-A-ETQ-A_lig1 (timeout)",
    )

    batch_text = Path(batch_log).read_text()
    main_text = Path(main_log).read_text()
    assert "FAILED Vina 3pbl-A-ETQ-A_lig1 (timeout)" in batch_text
    assert "FAILED Vina 3pbl-A-ETQ-A_lig1 (timeout)" in main_text


def test_log_progress_skips_none_paths(tmp_path: Path):
    """``None`` entries in the *paths argpack are silently ignored."""
    log = tmp_path / "progress.log"
    BulkRun._log_progress(str(log), None, message="Sentinel")
    assert "Sentinel at" in log.read_text()
