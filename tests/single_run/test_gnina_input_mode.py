"""
Tests for the ``gnina_input_mode`` knob.

- ``deploy_gnina`` should accept SDF + PDB inputs and pass them through
  to gnina's CLI without forcing PDBQT validation.
- The existing PDBQT call path should be unchanged.
- ``BulkRun`` should silently downgrade ``"sdf"`` to ``"pdbqt"`` when
  Vina or a Vina-rescore is co-requested, logging a warning.
"""

import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import (
    BOLTZ_PREFIX,
    GNINA_PREFIX,
    VINA_PREFIX,
)
from guild.docking import gnina as gnina_module
from guild.docking.gnina import deploy_gnina

TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"


@pytest.fixture
def fake_inputs(tmp_path: Path):
    """Create minimal placeholder receptor + ligand files for both formats."""
    pdb = tmp_path / "receptor.pdb"
    pdb.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n")

    sdf = tmp_path / "ligand.sdf"
    sdf.write_text("dummy\n\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n")

    pdbqt_r = tmp_path / "receptor.pdbqt"
    pdbqt_r.write_text(
        "ROOT\n"
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00    +0.000 N\n"
        "ENDROOT\nTORSDOF 0\n"
    )

    pdbqt_l = tmp_path / "ligand.pdbqt"
    pdbqt_l.write_text(
        "ROOT\n"
        "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 C\n"
        "ENDROOT\nTORSDOF 0\n"
    )

    return {
        "pdb": str(pdb), "sdf": str(sdf),
        "pdbqt_receptor": str(pdbqt_r), "pdbqt_ligand": str(pdbqt_l),
        "out_pdbqt": str(tmp_path / "out.pdbqt"),
        "out_scores": str(tmp_path / "scores.txt"),
    }


def _fake_completed(stdout=""):
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = 0
    return m


_FAKE_GNINA_STDOUT = (
    "mode |  affinity  | CNN     | CNN\n"
    "     | (kcal/mol) | pose-score| affinity\n"
    "-----+------------+----------+----------\n"
    "    1     -8.345      0.789      6.234\n"
)


def test_sdf_mode_emits_sdf_and_pdb_argv(fake_inputs, tmp_path):
    """SDF + PDB inputs land in argv unmodified; PDBQT validation is skipped."""
    captured_argv = {}

    def _capture(argv, **kwargs):
        captured_argv["argv"] = list(argv)
        return _fake_completed(_FAKE_GNINA_STDOUT)

    with patch.object(gnina_module.subprocess, "run", side_effect=_capture):
        deploy_gnina(
            receptor=fake_inputs["pdb"],
            ligand=fake_inputs["sdf"],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            output_pdbqt=fake_inputs["out_pdbqt"],
            output_scores=fake_inputs["out_scores"],
            use_gpu=False,
        )

    argv = captured_argv["argv"]
    assert fake_inputs["pdb"] in argv, f"pdb receptor missing from argv: {argv}"
    assert fake_inputs["sdf"] in argv, f"sdf ligand missing from argv: {argv}"
    # gnina's CLI passes the path right after --receptor / --ligand
    assert argv[argv.index("--receptor") + 1] == fake_inputs["pdb"]
    assert argv[argv.index("--ligand") + 1] == fake_inputs["sdf"]


def test_pdbqt_mode_unchanged(fake_inputs):
    """PDBQT path still emits the PDBQT argv and validation runs (no crash)."""
    captured_argv = {}

    def _capture(argv, **kwargs):
        captured_argv["argv"] = list(argv)
        return _fake_completed(_FAKE_GNINA_STDOUT)

    with patch.object(gnina_module.subprocess, "run", side_effect=_capture):
        deploy_gnina(
            receptor=fake_inputs["pdbqt_receptor"],
            ligand=fake_inputs["pdbqt_ligand"],
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            output_pdbqt=fake_inputs["out_pdbqt"],
            output_scores=fake_inputs["out_scores"],
            use_gpu=False,
        )

    argv = captured_argv["argv"]
    assert argv[argv.index("--receptor") + 1] == fake_inputs["pdbqt_receptor"]
    assert argv[argv.index("--ligand") + 1] == fake_inputs["pdbqt_ligand"]


@pytest.fixture
def bulk_table():
    df = pd.read_csv(TEST_DATA_DIR / "bulk_dummy.csv")
    df["protein_path"] = str(TEST_DATA_DIR / df["protein_path"].iloc[0])
    return df


@pytest.fixture
def cleanup_bulk():
    yield
    test_project = Path.cwd() / "data" / "test-gnina-input-mode"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def test_bulkrun_downgrades_sdf_to_pdbqt_when_vina_also_requested(
    bulk_table, cleanup_bulk, caplog
):
    """Co-requesting Vina forces gnina back to PDBQT mode with a warning."""
    caplog.set_level(logging.WARNING, logger="guild.bulk")

    bulk = BulkRun(
        input_table=bulk_table,
        project_name="test-gnina-input-mode",
        methods_to_run=[VINA_PREFIX, GNINA_PREFIX],
        gnina_input_mode="sdf",
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )

    assert bulk.gnina_input_mode == "pdbqt"
    assert any(
        "SDF-mode requested for gnina but Vina" in rec.message
        for rec in caplog.records
    ), f"expected downgrade warning, got: {[r.message for r in caplog.records]}"


def test_bulkrun_keeps_sdf_when_gnina_alone(bulk_table, cleanup_bulk):
    """Gnina-only run with sdf preserves the requested mode."""
    bulk = BulkRun(
        input_table=bulk_table,
        project_name="test-gnina-input-mode",
        methods_to_run=[GNINA_PREFIX],
        gnina_input_mode="sdf",
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert bulk.gnina_input_mode == "sdf"


def test_bulkrun_keeps_sdf_when_gnina_with_boltz(bulk_table, cleanup_bulk):
    """Boltz doesn't require PDBQT, so gnina+boltz still honours sdf-mode."""
    bulk = BulkRun(
        input_table=bulk_table,
        project_name="test-gnina-input-mode",
        methods_to_run=[GNINA_PREFIX, BOLTZ_PREFIX],
        gnina_input_mode="sdf",
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    # BulkRun auto-adds vina_rescore_boltz when boltz is requested; that DOES
    # require PDBQT, so gnina downgrades. Surfaces an important user-facing
    # consequence of the auto-rescore: combining gnina-sdf with boltz also
    # falls back to PDBQT (because of the implicit vina_rescore_boltz).
    assert bulk.gnina_input_mode == "pdbqt"
