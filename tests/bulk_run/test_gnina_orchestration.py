"""
Tests for the gnina integration in the bulk pipeline:

- gnina is registered as a standalone method (no auto-enable rescore rules).
- `gnina_guild_scoring` returns the expected columns including the
  side-channel CNN score, and `vina_guild_scoring` does not leak gnina columns.
- Regression: COMBINATION_ID must equal "combination" (bulk constant), not
  "combination_id" (interactions constant), to avoid KeyError in scoring merges.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import (
    ALL_AVAILABLE_METHODS,
    GNINA_CNN_SCORE,
    GNINA_PREFIX,
    GNINA_SCORE,
    SCORES_DICTIONARY,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX,
    VINA_SCORE,
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
    test_project = Path.cwd() / "data" / "test-gnina"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def test_gnina_registered_as_available_method():
    """gnina is in the list of methods picked up when none are specified."""
    assert GNINA_PREFIX in ALL_AVAILABLE_METHODS


def test_gnina_score_dictionary_entry():
    assert SCORES_DICTIONARY[GNINA_PREFIX] == GNINA_SCORE


def test_gnina_alone_does_not_enable_any_rescore(test_input_table, cleanup):
    """gnina is a standalone docker; selecting it doesn't add any vina_rescore_* tracks."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-gnina",
        methods_to_run=[GNINA_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert GNINA_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run


def test_gnina_guild_scoring_emits_score_and_cnn_columns():
    """gnina_guild_scoring returns both gnina_score and gnina_cnn_score; vina_guild_scoring doesn't."""
    from guild.constants.bulk import BATCH_FOLDER, COMBINATIONS_TO_RUN_KEY
    from guild.docking.gnina import gnina_guild_scoring
    from guild.docking.vina import vina_guild_scoring

    empty_batch = {BATCH_FOLDER: "/nonexistent", COMBINATIONS_TO_RUN_KEY: []}

    gnina_df = gnina_guild_scoring(empty_batch)
    vina_df = vina_guild_scoring(empty_batch)

    assert GNINA_SCORE in gnina_df.columns
    assert GNINA_CNN_SCORE in gnina_df.columns
    assert VINA_SCORE not in gnina_df.columns

    assert VINA_SCORE in vina_df.columns
    assert GNINA_SCORE not in vina_df.columns
    assert GNINA_CNN_SCORE not in vina_df.columns


def test_gnina_guild_scoring_combination_id_column_matches_bulk_constant():
    """
    Regression: gnina_guild_scoring must emit COMBINATION_ID == 'combination'
    (from guild.constants.bulk).  A prior bug imported COMBINATION_ID from
    guild.constants.interactions ('combination_id') which shadowed the bulk
    constant in bulk.py, causing KeyError on every scoring merge.
    """
    from guild.constants.bulk import (
        BATCH_FOLDER,
        COMBINATION_ID,
        COMBINATIONS_TO_RUN_KEY,
    )
    from guild.docking.gnina import gnina_guild_scoring

    # Verify the bulk constant itself has the correct value.
    assert (
        COMBINATION_ID == "combination"
    ), f"guild.constants.bulk.COMBINATION_ID must be 'combination', got {COMBINATION_ID!r}"

    # Verify gnina_guild_scoring emits a column with that name.
    empty_batch = {BATCH_FOLDER: "/nonexistent", COMBINATIONS_TO_RUN_KEY: []}
    df = gnina_guild_scoring(empty_batch)
    assert "combination" in df.columns, (
        f"gnina_guild_scoring returned columns {list(df.columns)!r}; "
        "'combination' is missing — COMBINATION_ID shadowing bug re-introduced?"
    )
    assert "combination_id" not in df.columns, (
        "gnina_guild_scoring emitted 'combination_id' instead of 'combination' — "
        "COMBINATION_ID import is being shadowed by guild.constants.interactions"
    )


def test_vina_guild_scoring_combination_id_column_matches_bulk_constant():
    """
    Same regression check for vina_guild_scoring — ensures the scoring merge
    key ('combination') is consistent across all scoring functions consumed by
    _process_batch_scoring.
    """
    from guild.constants.bulk import (
        BATCH_FOLDER,
        COMBINATIONS_TO_RUN_KEY,
    )
    from guild.docking.vina import vina_guild_scoring

    empty_batch = {BATCH_FOLDER: "/nonexistent", COMBINATIONS_TO_RUN_KEY: []}
    df = vina_guild_scoring(empty_batch)
    assert "combination" in df.columns, (
        f"vina_guild_scoring returned columns {list(df.columns)!r}; " "'combination' is missing"
    )
    assert "combination_id" not in df.columns
