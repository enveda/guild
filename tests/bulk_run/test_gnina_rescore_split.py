"""
Tests for the gnina re-scoring tracks: ``gnina_rescore_boltz`` and
``gnina_rescore_diffdock`` are additive alongside the existing Vina rescore
tracks — requesting ``boltz``/``diffdock`` auto-enables both the Vina and the
gnina rescore for that upstream method, never one in place of the other.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import (
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
    GNINA_RESCORE_BOLTZ_PREFIX,
    GNINA_RESCORE_BOLTZ_SCORE,
    GNINA_RESCORE_DIFFDOCK_PREFIX,
    GNINA_RESCORE_DIFFDOCK_SCORE,
    SCORES_DICTIONARY,
    VINA_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX,
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
    test_project = Path.cwd() / "data" / "test-gnina-rescore-split"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def test_scores_dictionary_distinguishes_gnina_rescore_methods():
    """Both gnina rescore prefixes have their own distinct score columns."""
    assert SCORES_DICTIONARY[GNINA_RESCORE_BOLTZ_PREFIX] == GNINA_RESCORE_BOLTZ_SCORE
    assert SCORES_DICTIONARY[GNINA_RESCORE_DIFFDOCK_PREFIX] == GNINA_RESCORE_DIFFDOCK_SCORE
    assert GNINA_RESCORE_BOLTZ_SCORE != GNINA_RESCORE_DIFFDOCK_SCORE


def test_boltz_auto_enables_both_vina_and_gnina_rescore(test_input_table, cleanup):
    """Requesting boltz auto-adds vina_rescore_boltz AND gnina_rescore_boltz,
    but neither diffdock rescore track."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-gnina-rescore-split",
        methods_to_run=[BOLTZ_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        database_update=False,
        use_gpu=False,
        n_workers=1,
    )
    assert BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert GNINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run
    assert GNINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run


def test_diffdock_auto_enables_both_vina_and_gnina_rescore(test_input_table, cleanup):
    """Requesting diffdock auto-adds vina_rescore_diffdock AND
    gnina_rescore_diffdock, but neither boltz rescore track."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-gnina-rescore-split",
        methods_to_run=[DIFFDOCK_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        database_update=False,
        use_gpu=False,
        n_workers=1,
    )
    assert DIFFDOCK_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX in bulk.methods_to_run
    assert GNINA_RESCORE_DIFFDOCK_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run
    assert GNINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run


def test_both_methods_enable_all_four_rescore_tracks(test_input_table, cleanup):
    """Requesting both Boltz and DiffDock auto-enables all four rescore tracks."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-gnina-rescore-split",
        methods_to_run=[BOLTZ_PREFIX, DIFFDOCK_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        database_update=False,
        use_gpu=False,
        n_workers=1,
    )
    assert VINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX in bulk.methods_to_run
    assert GNINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert GNINA_RESCORE_DIFFDOCK_PREFIX in bulk.methods_to_run


def test_vina_alone_does_not_enable_any_rescore(test_input_table, cleanup):
    """Plain Vina docking does not auto-enable any rescore track."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-gnina-rescore-split",
        methods_to_run=[VINA_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        database_update=False,
        use_gpu=False,
        n_workers=1,
    )
    assert VINA_PREFIX in bulk.methods_to_run
    assert GNINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run
    assert GNINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run


def test_rescore_scoring_functions_emit_distinct_columns():
    """The two gnina guild_scoring entry points return distinct score columns,
    and don't collide with the Vina rescore columns."""
    from guild.constants.bulk import BATCH_FOLDER, COMBINATIONS_TO_RUN_KEY
    from guild.docking.boltz import gnina_rescore_boltz_guild_scoring
    from guild.docking.diffdock import gnina_rescore_diffdock_guild_scoring

    # Empty batch (no outputs on disk) → both functions return frames with
    # their respective score columns, no overlap.
    empty_batch = {BATCH_FOLDER: "/nonexistent", COMBINATIONS_TO_RUN_KEY: []}

    boltz_df = gnina_rescore_boltz_guild_scoring(empty_batch)
    diffdock_df = gnina_rescore_diffdock_guild_scoring(empty_batch)

    assert GNINA_RESCORE_BOLTZ_SCORE in boltz_df.columns
    assert GNINA_RESCORE_DIFFDOCK_SCORE not in boltz_df.columns
    assert GNINA_RESCORE_DIFFDOCK_SCORE in diffdock_df.columns
    assert GNINA_RESCORE_BOLTZ_SCORE not in diffdock_df.columns
