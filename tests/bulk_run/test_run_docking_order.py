"""
Tests that ``BulkRun.run_docking`` dispatches each batch's method blocks in
the order the user supplied via ``methods_to_run`` (rather than a hardcoded
order).
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import (
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
    GNINA_PREFIX,
    KARMADOCK_PREFIX,
    VINA_PREFIX,
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
    test_project = Path.cwd() / "data" / "test-run-docking-order"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def _run_and_capture(test_input_table, methods):
    """
    Build a BulkRun, mock out every per-method helper plus ``_prepare_docking``,
    and return the order in which the helpers were called for the (single)
    batch.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-run-docking-order",
        methods_to_run=methods,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )

    call_order = []

    def make_recorder(name):
        def _record(self, current_batch, current_batch_folder):
            call_order.append(name)

        return _record

    with (
        patch.object(BulkRun, "_prepare_docking", lambda self: None),
        patch.object(BulkRun, "_run_boltz_for_batch", make_recorder(BOLTZ_PREFIX)),
        patch.object(BulkRun, "_run_vina_for_batch", make_recorder(VINA_PREFIX)),
        patch.object(BulkRun, "_run_gnina_for_batch", make_recorder(GNINA_PREFIX)),
        patch.object(BulkRun, "_run_karmadock_for_batch", make_recorder(KARMADOCK_PREFIX)),
        patch.object(BulkRun, "_run_diffdock_for_batch", make_recorder(DIFFDOCK_PREFIX)),
    ):
        bulk.run_docking()

    return call_order, bulk.methods_to_run


def test_user_order_vina_then_boltz(test_input_table, cleanup):
    """``[vina, boltz]`` dispatches Vina before Boltz."""
    order, expanded = _run_and_capture(test_input_table, [VINA_PREFIX, BOLTZ_PREFIX])

    # methods_to_run is auto-extended with vina_rescore_boltz when boltz is
    # requested, but that prefix has no docking-time runner, so it doesn't
    # show up in the call sequence.
    assert order == [VINA_PREFIX, BOLTZ_PREFIX]
    # Sanity: the auto-added rescore prefix is in the expanded list.
    assert any("vina_rescore_boltz" == m for m in expanded)


def test_user_order_boltz_then_vina(test_input_table, cleanup):
    """``[boltz, vina]`` dispatches Boltz before Vina."""
    order, _ = _run_and_capture(test_input_table, [BOLTZ_PREFIX, VINA_PREFIX])
    # First two entries (skipping the auto-added rescore which is a no-op).
    docking_order = [m for m in order if m in {BOLTZ_PREFIX, VINA_PREFIX}]
    assert docking_order == [BOLTZ_PREFIX, VINA_PREFIX]


def test_five_methods_in_explicit_order(test_input_table, cleanup):
    """A custom interleaved order is respected end-to-end."""
    desired = [
        GNINA_PREFIX,
        DIFFDOCK_PREFIX,
        VINA_PREFIX,
        KARMADOCK_PREFIX,
        BOLTZ_PREFIX,
    ]
    order, _ = _run_and_capture(test_input_table, desired)

    # Filter out any non-docking prefixes (vina_rescore_* auto-additions).
    docking_order = [m for m in order if m in set(desired)]
    assert docking_order == desired


def test_rescore_prefixes_do_not_dispatch_runners(test_input_table, cleanup):
    """vina_rescore_* prefixes are skipped during the docking phase."""
    order, expanded = _run_and_capture(test_input_table, [DIFFDOCK_PREFIX])
    # DiffDock alone auto-adds vina_rescore_diffdock; it must not produce a
    # docking-helper call (the rescore happens in run_guild_scoring).
    assert "vina_rescore_diffdock" in expanded
    assert order == [DIFFDOCK_PREFIX]
