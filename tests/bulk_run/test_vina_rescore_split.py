"""
Tests for the split Vina re-scoring tracks: ``vina_rescore_boltz`` and
``vina_rescore_diffdock`` are independent and must coexist when both upstream
methods are present in ``methods_to_run``.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import (
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
    SCORES_DICTIONARY,
    VINA_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_BOLTZ_SCORE,
    VINA_RESCORE_DIFFDOCK_PREFIX,
    VINA_RESCORE_DIFFDOCK_SCORE,
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
    test_project = Path.cwd() / "data" / "test-rescore-split"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def test_scores_dictionary_distinguishes_rescore_methods():
    """Both rescore prefixes have their own distinct score columns."""
    assert SCORES_DICTIONARY[VINA_RESCORE_BOLTZ_PREFIX] == VINA_RESCORE_BOLTZ_SCORE
    assert SCORES_DICTIONARY[VINA_RESCORE_DIFFDOCK_PREFIX] == VINA_RESCORE_DIFFDOCK_SCORE
    assert VINA_RESCORE_BOLTZ_SCORE != VINA_RESCORE_DIFFDOCK_SCORE


def test_boltz_auto_enables_only_boltz_rescore(test_input_table, cleanup):
    """Requesting boltz auto-adds vina_rescore_boltz but NOT vina_rescore_diffdock."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-rescore-split",
        methods_to_run=[BOLTZ_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run


def test_diffdock_auto_enables_only_diffdock_rescore(test_input_table, cleanup):
    """Requesting diffdock auto-adds vina_rescore_diffdock but NOT vina_rescore_boltz."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-rescore-split",
        methods_to_run=[DIFFDOCK_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert DIFFDOCK_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run


def test_both_methods_enable_both_rescores(test_input_table, cleanup):
    """Requesting both Boltz and DiffDock auto-enables both rescore tracks."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-rescore-split",
        methods_to_run=[BOLTZ_PREFIX, DIFFDOCK_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert VINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX in bulk.methods_to_run


def test_vina_alone_does_not_enable_any_rescore(test_input_table, cleanup):
    """Plain Vina docking does not auto-enable either rescore."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-rescore-split",
        methods_to_run=[VINA_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert VINA_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run


def test_rescore_scoring_functions_emit_distinct_columns():
    """The two guild_scoring entry points return distinct score columns."""
    from guild.constants.bulk import BATCH_FOLDER, COMBINATIONS_TO_RUN_KEY
    from guild.docking.boltz import vina_rescore_boltz_guild_scoring
    from guild.docking.diffdock import vina_rescore_diffdock_guild_scoring

    # Empty batch (no outputs on disk) → both functions return frames with
    # their respective score columns, no overlap.
    empty_batch = {BATCH_FOLDER: "/nonexistent", COMBINATIONS_TO_RUN_KEY: []}

    boltz_df = vina_rescore_boltz_guild_scoring(empty_batch)
    diffdock_df = vina_rescore_diffdock_guild_scoring(empty_batch)

    assert VINA_RESCORE_BOLTZ_SCORE in boltz_df.columns
    assert VINA_RESCORE_DIFFDOCK_SCORE not in boltz_df.columns
    assert VINA_RESCORE_DIFFDOCK_SCORE in diffdock_df.columns
    assert VINA_RESCORE_BOLTZ_SCORE not in diffdock_df.columns
