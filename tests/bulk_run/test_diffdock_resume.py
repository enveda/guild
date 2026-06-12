"""
Tests for the hardened DiffDock resume-skip check in
``BulkRun.run_docking()``. The previous check skipped DiffDock when the
top-level results directory was non-empty, which silently masked
interrupted runs whose per-combination subfolders were empty. The check now
requires every combination to have at least one ``*_confidence*.sdf`` on
disk before declaring the batch done.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.bulk import BATCH_FOLDER
from guild.constants.diffdock import DIFFDOCK_RESULTS_FOLDER
from guild.constants.guild import (
    DIFFDOCK_FOLDER,
    DIFFDOCK_PREFIX,
    LIGAND_ID,
    PROTEIN_CONF_ID,
)

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
    test_project = Path.cwd() / "data" / "test-diffdock-resume"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def _build_bulk(test_input_table):
    """Construct a minimal BulkRun configured for DiffDock only."""
    return BulkRun(
        input_table=test_input_table,
        project_name="test-diffdock-resume",
        methods_to_run=[DIFFDOCK_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )


def _results_dir(bulk):
    """Resolve the DiffDock results dir for the first (and only) batch."""
    batch_name = next(iter(bulk.batched_dictionary))
    batch_folder = bulk.batched_dictionary[batch_name][BATCH_FOLDER]
    return Path(batch_folder) / DIFFDOCK_FOLDER / DIFFDOCK_RESULTS_FOLDER, batch_name


def _populate_full_results(bulk):
    """Drop one valid *_confidence*.sdf into every per-combo subfolder."""
    results_dir, batch_name = _results_dir(bulk)
    combinations_df = bulk.batched_dictionary[batch_name]["combinations_table"]
    for _, row in combinations_df.iterrows():
        combo_dir = results_dir / f"{row[PROTEIN_CONF_ID]}_{row[LIGAND_ID]}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        (combo_dir / "rank1_confidence-1.23.sdf").write_text("MOCK\n")


def _populate_empty_subfolders(bulk):
    """Create per-combo subfolders but leave them empty (the bug scenario)."""
    results_dir, batch_name = _results_dir(bulk)
    combinations_df = bulk.batched_dictionary[batch_name]["combinations_table"]
    for _, row in combinations_df.iterrows():
        combo_dir = results_dir / f"{row[PROTEIN_CONF_ID]}_{row[LIGAND_ID]}"
        combo_dir.mkdir(parents=True, exist_ok=True)


def test_skip_when_all_combos_have_confidence_sdf(test_input_table, cleanup):
    bulk = _build_bulk(test_input_table)
    _populate_full_results(bulk)

    with (
        patch.object(BulkRun, "_prepare_docking", lambda self: None),
        patch("guild.bulk.deploy_diffdock") as mock_deploy,
        patch("guild.bulk.generate_diffdock_complex_pdbs"),
    ):
        bulk.run_docking()

    mock_deploy.assert_not_called()


def test_rerun_when_combo_folder_empty(test_input_table, cleanup):
    bulk = _build_bulk(test_input_table)
    _populate_empty_subfolders(bulk)

    with (
        patch.object(BulkRun, "_prepare_docking", lambda self: None),
        patch("guild.bulk.deploy_diffdock") as mock_deploy,
        patch("guild.bulk.generate_diffdock_complex_pdbs"),
    ):
        bulk.run_docking()

    mock_deploy.assert_called_once()


def test_rerun_when_combo_folder_missing(test_input_table, cleanup):
    bulk = _build_bulk(test_input_table)
    # Intentionally do NOT create any per-combo subfolders.

    with (
        patch.object(BulkRun, "_prepare_docking", lambda self: None),
        patch("guild.bulk.deploy_diffdock") as mock_deploy,
        patch("guild.bulk.generate_diffdock_complex_pdbs"),
    ):
        bulk.run_docking()

    mock_deploy.assert_called_once()
