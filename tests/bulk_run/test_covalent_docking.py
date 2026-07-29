"""
Tests for the gnina covalent docking feature.

Covers:
- ``deploy_gnina`` adds the ``--covalent_*`` flags only when both the receptor
  atom and the ligand SMARTS pattern are supplied, and toggles
  ``--covalent_optimize_lig`` correctly.
- ``_row_covalent`` extracts the per-row CSV spec, returning ``(None, None)``
  when either column is absent / empty.

No Docker or gnina binary required — subprocess.run is mocked.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from guild.bulk import _row_covalent
from guild.constants.guild import COVALENT_LIG_SMARTS, COVALENT_REC_ATOM


def _fake_run_factory(captured_argv):
    def fake_run(argv, **kwargs):
        captured_argv.extend(argv)
        result = MagicMock()
        result.returncode = 0
        result.stdout = "mode |  affinity\n   1 |     -7.5\n"
        result.stderr = ""
        return result

    return fake_run


def _run_deploy_gnina(captured_argv, **covalent_kwargs):
    from guild.docking.gnina import deploy_gnina

    with (
        patch("guild.docking.gnina.subprocess.run", side_effect=_fake_run_factory(captured_argv)),
        patch("guild.docking.gnina._validate_pdbqt"),
        patch("guild.docking.gnina._ensure_openbabel_plugin_shim", return_value=None),
        patch("guild.docking.gnina.parse_gnina_stdout", return_value=[(1, -7.5, 0.5, -6.0)]),
        patch("guild.docking.gnina.write_subprocess_log"),
        patch("builtins.open", MagicMock()),
    ):
        deploy_gnina(
            receptor="receptor.pdb",
            ligand="ligand.sdf",
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            output_pdbqt="out.sdf",
            output_scores="scores.txt",
            use_gpu=False,
            **covalent_kwargs,
        )


# ---------------------------------------------------------------------------
# deploy_gnina covalent flag forwarding
# ---------------------------------------------------------------------------


def test_no_covalent_flags_by_default():
    argv = []
    _run_deploy_gnina(argv)
    assert "--covalent_rec_atom" not in argv
    assert "--covalent_lig_atom_pattern" not in argv
    assert "--covalent_optimize_lig" not in argv


def test_covalent_flags_added_when_both_set():
    argv = []
    _run_deploy_gnina(
        argv,
        covalent_rec_atom="A:145:SG",
        covalent_lig_atom_pattern="C#N",
    )
    assert "--covalent_rec_atom" in argv
    assert argv[argv.index("--covalent_rec_atom") + 1] == "A:145:SG"
    assert "--covalent_lig_atom_pattern" in argv
    assert argv[argv.index("--covalent_lig_atom_pattern") + 1] == "C#N"
    assert "--covalent_bond_order" in argv
    # optimize defaults to True
    assert "--covalent_optimize_lig" in argv


def test_covalent_flags_absent_when_only_rec_atom_set():
    argv = []
    _run_deploy_gnina(argv, covalent_rec_atom="A:145:SG")
    assert "--covalent_rec_atom" not in argv


def test_covalent_flags_absent_when_only_pattern_set():
    argv = []
    _run_deploy_gnina(argv, covalent_lig_atom_pattern="C#N")
    assert "--covalent_lig_atom_pattern" not in argv


def test_covalent_optimize_lig_can_be_disabled():
    argv = []
    _run_deploy_gnina(
        argv,
        covalent_rec_atom="A:145:SG",
        covalent_lig_atom_pattern="C#N",
        covalent_optimize_lig=False,
    )
    assert "--covalent_rec_atom" in argv
    assert "--covalent_optimize_lig" not in argv


# ---------------------------------------------------------------------------
# _row_covalent CSV extraction
# ---------------------------------------------------------------------------


def _row(**cols):
    return pd.DataFrame([cols]).iloc[0]


def test_row_covalent_both_present():
    row = _row(**{COVALENT_REC_ATOM: "A:145:SG", COVALENT_LIG_SMARTS: "C#N"})
    assert _row_covalent(row) == ("A:145:SG", "C#N")


def test_row_covalent_columns_absent():
    row = _row(smiles="CCO")
    assert _row_covalent(row) == (None, None)


def test_row_covalent_empty_values():
    row = _row(**{COVALENT_REC_ATOM: "", COVALENT_LIG_SMARTS: "  "})
    assert _row_covalent(row) == (None, None)


def test_row_covalent_nan_values():
    row = _row(**{COVALENT_REC_ATOM: float("nan"), COVALENT_LIG_SMARTS: float("nan")})
    assert _row_covalent(row) == (None, None)


def test_row_covalent_partial():
    row = _row(**{COVALENT_REC_ATOM: "A:145:SG"})
    assert _row_covalent(row) == ("A:145:SG", None)


# ---------------------------------------------------------------------------
# add_covalent_conect
# ---------------------------------------------------------------------------


def _write_mock_complex(path: str, rec_serial: int, lig_serial: int, lig_dist: float) -> None:
    """Write a minimal complex PDB with one ATOM and one HETATM at a given distance apart."""
    # Receptor atom at origin.
    rec_x, rec_y, rec_z = 0.0, 0.0, 0.0
    # Ligand (warhead) placed along X axis at lig_dist.
    lig_x = rec_x + lig_dist

    with open(path, "w") as fh:
        fh.write(
            f"ATOM  {rec_serial:5d}  SG  CYS A 145    "
            f"{rec_x:8.3f}{rec_y:8.3f}{rec_z:8.3f}  1.00  0.00           S  \n"
        )
        fh.write(
            f"HETATM{lig_serial:5d}  C   LIG Z   1    "
            f"{lig_x:8.3f}{rec_y:8.3f}{rec_z:8.3f}  1.00  0.00           C  \n"
        )
        fh.write("END\n")


def test_add_covalent_conect_writes_conect_records(tmp_path):
    """Happy path: warhead at ~1.8 Å from SG → two CONECT lines added before END."""
    from guild.transformers.pdb import add_covalent_conect

    pdb = str(tmp_path / "complex.pdb")
    _write_mock_complex(pdb, rec_serial=49, lig_serial=500, lig_dist=1.8)

    add_covalent_conect(pdb, "A:145:SG")

    content = open(pdb).read()
    assert "CONECT   49  500" in content
    assert "CONECT  500   49" in content
    # END must still be present and be the last non-empty line.
    last = [line.strip() for line in content.splitlines() if line.strip()][-1]
    assert last == "END"


def test_add_covalent_conect_no_conect_when_too_far(tmp_path):
    """Warhead further than 3 Å → CONECT not written, no exception raised."""
    from guild.transformers.pdb import add_covalent_conect

    pdb = str(tmp_path / "complex.pdb")
    _write_mock_complex(pdb, rec_serial=49, lig_serial=500, lig_dist=5.0)

    add_covalent_conect(pdb, "A:145:SG")  # should not raise

    content = open(pdb).read()
    assert "CONECT" not in content


def test_add_covalent_conect_raises_on_missing_receptor_atom(tmp_path):
    """Spec that doesn't match any ATOM record → ValueError."""
    import pytest

    from guild.transformers.pdb import add_covalent_conect

    pdb = str(tmp_path / "complex.pdb")
    _write_mock_complex(pdb, rec_serial=49, lig_serial=500, lig_dist=1.8)

    with pytest.raises(ValueError, match="not found"):
        add_covalent_conect(pdb, "B:999:SG")


def test_add_covalent_conect_raises_on_malformed_spec(tmp_path):
    import pytest

    from guild.transformers.pdb import add_covalent_conect

    pdb = str(tmp_path / "complex.pdb")
    _write_mock_complex(pdb, rec_serial=49, lig_serial=500, lig_dist=1.8)

    with pytest.raises(ValueError, match="malformed"):
        add_covalent_conect(pdb, "A:145")
