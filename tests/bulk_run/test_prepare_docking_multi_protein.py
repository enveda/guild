"""
Tests for ``BulkRun._prepare_docking`` row-selection behavior.

Multi-protein Boltz runs were failing because the prep pool only Guild-inited
``combinations_table.iloc[[0]]`` for non-KarmaDock methods. Protein prep
writes per-protein artifacts (e.g. ``{batch}/proteins/<protein_idx>_single_chain_clean.pdb``),
so for batches whose combinations span multiple distinct ``protein_config_id``
values, only the first protein got its cleaned PDB. Boltz then silently fell
back to the raw input PDB and produced empty manifests.

The fix Guild-inits one row per unique ``protein_config_id`` for non-KarmaDock
methods; KarmaDock keeps preparing every row because its downstream worker
needs the per-ligand data tree.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import (
    BOLTZ_PREFIX,
    KARMADOCK_PREFIX,
    PROTEIN_CONF_ID,
)

TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"


@pytest.fixture
def multi_protein_table():
    """Four rows across two distinct ``protein_config_id`` values."""
    base = pd.read_csv(TEST_DATA_DIR / "bulk_dummy.csv")
    protein_path = str(TEST_DATA_DIR / base["protein_path"].iloc[0])

    template = base.iloc[0].to_dict()
    rows = []
    # Two proteins × two ligands each.
    for protein_conf_id in ("3pbl-A-ETQ-A", "3pbl-B-ETQ-A"):
        for ligand_id, smiles in (
            ("lig1", "CC(=COC=O)CCC1=C(C)CCCC1(C)C"),
            ("lig2", "CCO"),
        ):
            row = template.copy()
            row["protein_config_id"] = protein_conf_id
            row["ligand_id"] = ligand_id
            row["smiles"] = smiles
            row["protein_path"] = protein_path
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def cleanup():
    yield
    test_project = Path.cwd() / "data" / "test-prepare-multi-protein"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


class _SyncExecutor:
    """Drop-in for ``ProcessPoolExecutor`` that runs `submit` jobs inline,
    avoiding the pickling boundary so tests can use local closures."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = _CompletedFuture()
        try:
            future._result = fn(*args, **kwargs)
        except Exception as e:  # pragma: no cover — surface in test failure
            future._exc = e
        return future


class _CompletedFuture:
    _result = None
    _exc = None

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._result

    def cancel(self):
        pass


def _captured_prep_params(table, methods):
    """
    Build a BulkRun, replace the prep pool's executor with a synchronous
    stand-in, and capture the params handed to ``_prepare_single_batch_worker``.
    """
    bulk = BulkRun(
        input_table=table,
        project_name="test-prepare-multi-protein",
        methods_to_run=methods,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )

    captured = []

    def _capture(params):
        captured.append(params)
        return params.get("batch_name")

    # as_completed is just `iter()` for our sync fake.
    with (
        patch.object(BulkRun, "_prepare_single_batch_worker", staticmethod(_capture)),
        patch("guild.bulk.ProcessPoolExecutor", _SyncExecutor),
        patch("guild.bulk.as_completed", lambda fs, **kw: list(fs)),
    ):
        bulk._prepare_docking()

    return captured


def test_non_karmadock_preps_one_row_per_unique_protein(multi_protein_table, cleanup):
    """Boltz/Vina/DiffDock/gnina: prep tasks = unique proteins, not 1 and not N rows."""
    captured = _captured_prep_params(multi_protein_table, [BOLTZ_PREFIX])

    prepped_protein_ids = {p["protein_idx"] for p in captured}
    expected = set(multi_protein_table[PROTEIN_CONF_ID].unique())

    assert len(captured) == len(expected), (
        f"Expected one prep task per unique protein ({len(expected)}), "
        f"got {len(captured)}: {[p['protein_idx'] for p in captured]}"
    )
    assert prepped_protein_ids == expected


def test_karmadock_still_preps_every_row(multi_protein_table, cleanup):
    """KarmaDock: must prep every (protein, ligand) row because its per-ligand
    data tree is staged here, not by a downstream worker."""
    captured = _captured_prep_params(multi_protein_table, [KARMADOCK_PREFIX])

    assert len(captured) == len(multi_protein_table)
    captured_pairs = {(p["protein_idx"], p["ligand_idx"]) for p in captured}
    expected_pairs = set(
        zip(
            multi_protein_table[PROTEIN_CONF_ID],
            multi_protein_table["ligand_id"],
            strict=True,
        )
    )
    assert captured_pairs == expected_pairs
