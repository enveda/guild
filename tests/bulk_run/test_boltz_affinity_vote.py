"""
Tests that ``boltz_affinity`` (Boltz-2's own affinity head) is auto-added to
``methods_to_run`` whenever ``boltz`` is requested, and only then -- mirroring
the ``vina_rescore_*`` / ``gnina_rescore_*`` auto-additions in
test_vina_rescore_split.py / test_gnina_rescore_split.py. It needs no docking
step of its own: the value is already parsed by boltz_guild_scoring alongside
boltz_score, so it has no entry in run_docking's method_runners map and is a
no-op there, the same way the rescore tracks are.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.bulk import (
    RANKS_DICTIONARY,
    RP_SCORES_DICTIONARY,
    SCORES_TO_USE_DICTIONARY,
)
from guild.constants.guild import (
    BOLTZ_AFFINITY_PREFIX,
    BOLTZ_AFFINITY_SCORE,
    BOLTZ_PREFIX,
    GNINA_RESCORE_BOLTZ_PREFIX,
    VINA_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX,
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
    test_project = Path.cwd() / "data" / "test-boltz-affinity-vote"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def test_boltz_auto_enables_boltz_affinity(test_input_table, cleanup):
    """Requesting boltz auto-adds boltz_affinity alongside its two rescores."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-boltz-affinity-vote",
        methods_to_run=[BOLTZ_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert BOLTZ_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert GNINA_RESCORE_BOLTZ_PREFIX in bulk.methods_to_run
    assert BOLTZ_AFFINITY_PREFIX in bulk.methods_to_run


def test_boltz_affinity_not_added_without_boltz(test_input_table, cleanup):
    """Requesting Vina alone never pulls in boltz_affinity."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-boltz-affinity-vote",
        methods_to_run=[VINA_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert BOLTZ_AFFINITY_PREFIX not in bulk.methods_to_run


def test_boltz_affinity_has_no_docking_runner(test_input_table, cleanup):
    """boltz_affinity must be absent from run_docking's method_runners map --
    it rides along with boltz_guild_scoring and needs no docking step, the
    same way vina_rescore_* is a no-op during docking."""
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-boltz-affinity-vote",
        methods_to_run=[BOLTZ_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert BOLTZ_AFFINITY_PREFIX in bulk.methods_to_run

    import inspect

    source = inspect.getsource(bulk.run_docking)
    assert "BOLTZ_AFFINITY_PREFIX" not in source


def test_boltz_affinity_registered_in_ranking_dictionaries():
    """available_methods_preparation indexes RANKS_DICTIONARY and
    SCORES_TO_USE_DICTIONARY by every entry in methods_to_run -- since
    boltz_affinity is now auto-added there, both must have an entry for it
    or a real bulk run raises KeyError the first time boltz is requested."""
    assert BOLTZ_AFFINITY_PREFIX in RANKS_DICTIONARY
    assert BOLTZ_AFFINITY_PREFIX in RP_SCORES_DICTIONARY
    assert BOLTZ_AFFINITY_PREFIX in SCORES_TO_USE_DICTIONARY
    assert BOLTZ_AFFINITY_SCORE in SCORES_TO_USE_DICTIONARY[BOLTZ_AFFINITY_PREFIX]
