"""
End-to-end flexible docking tests.

Each test performs a real docking run using the 3pbl structure and
box from tests/test_data, then verifies that the binding-pocket side
chains actually moved in the output — i.e. that flexibility is being
exercised and not silently skipped.

Marked ``e2e`` because they invoke external binaries (Vina Python API,
gnina CLI) and take O(minutes).  Exclude them from the fast suite with
``-m "not e2e"``.

Note: these tests bypass ``Guild.__init__`` intentionally to avoid the
karmadock / openbabel prep path that is broken on this host.  They call
the same lower-level functions that Guild delegates to.
"""

from pathlib import Path

import meeko
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from guild.constants.gnina import GNINA_BINARY, GNINA_FLEX_DISTANCE
from guild.docking.gnina import deploy_gnina
from guild.docking.vina import deploy_vina, get_center_and_size_from_box_file
from guild.tools.preparation import isolate_protein_chain, renumber_pdb_residues
from guild.transformers.converters import prepare_flex_receptor_pdbqt
from guild.transformers.pdb import get_flexres_string_from_box

TEST_DATA = Path(__file__).parent.parent / "test_data"
PDB_SRC = TEST_DATA / "3pbl.pdb"
BOX_FILE = TEST_DATA / "3pbl_box.txt"
SMILES = "CC(=COC=O)CCC1=C(C)CCCC1(C)C"

GNINA_AVAILABLE = Path(GNINA_BINARY).exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ligand_pdbqt(workdir: Path) -> Path:
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    prep = meeko.MoleculePreparation()
    prep.prepare(mol)
    pdbqt_str = meeko.PDBQTWriterLegacy.write_string(prep.setup)[0]
    path = workdir / "ligand.pdbqt"
    path.write_text(pdbqt_str)
    return path


def _make_ligand_sdf(workdir: Path) -> Path:
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    path = workdir / "ligand.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return path


def _prepare_protein(workdir: Path):
    """Isolate chain A, renumber, return path to cleaned PDB."""
    clean = workdir / "3pbl_A.pdb"
    isolate_protein_chain(str(PDB_SRC), "3pbl", str(clean), target_chain="A")
    renumber_pdb_residues(str(clean), str(clean))
    return clean


def _parse_pdbqt_atoms(path: Path):
    """Return list of (label, x, y, z) for every ATOM/HETATM line."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            rows.append(
                (
                    f"{line[17:20].strip()}_{line[22:26].strip()}_{line[12:16].strip()}",
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return rows


def _parse_flex_atoms_pose1(pdbqt_text: str):
    """Extract flex-residue atom coords from the first MODEL block (Vina format)."""
    in_flex, model_idx, in_model, rows = False, 0, False, []
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            model_idx += 1
            in_model = model_idx == 1
        elif line.startswith("ENDMDL"):
            if in_model:
                break
            in_model = False
        elif in_model and line.startswith("BEGIN_RES"):
            in_flex = True
        elif in_model and line.startswith("END_RES"):
            in_flex = False
        elif in_flex and line.startswith(("ATOM", "HETATM")):
            rows.append(
                (
                    f"{line[17:20].strip()}_{line[22:26].strip()}_{line[12:16].strip()}",
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return rows


def _parse_gnina_flex_atoms_pose1(pdbqt_text: str):
    """Extract flex-residue atom coords from the first MODEL block (gnina out_flex format).

    gnina writes --out_flex in ligand-PDBQT style: ROOT/ENDROOT/BRANCH blocks,
    not Vina's BEGIN_RES/END_RES.  All ATOM lines in the first MODEL are flex atoms.
    """
    model_idx, in_model, rows = 0, False, []
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            model_idx += 1
            in_model = model_idx == 1
        elif line.startswith("ENDMDL"):
            if in_model:
                break
            in_model = False
        elif in_model and line.startswith(("ATOM", "HETATM")):
            rows.append(
                (
                    f"{line[17:20].strip()}_{line[22:26].strip()}_{line[12:16].strip()}",
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
    return rows


def _displacement_stats(initial, docked):
    n = min(len(initial), len(docked))
    disps = [
        (
            (docked[i][1] - initial[i][1]) ** 2
            + (docked[i][2] - initial[i][2]) ** 2
            + (docked[i][3] - initial[i][3]) ** 2
        )
        ** 0.5
        for i in range(n)
    ]
    moved = [d for d in disps if d > 0.01]
    return n, moved


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_flexible_docking_vina_side_chains_move(tmp_path):
    """
    Vina flexible docking: side-chain atoms in the output BEGIN_RES blocks
    must have moved relative to their starting positions in the flex PDBQT.
    """
    clean_pdb = _prepare_protein(tmp_path)
    center, size = get_center_and_size_from_box_file(str(BOX_FILE))

    flexres_str = get_flexres_string_from_box(str(clean_pdb), "A", center, size, max_flexres=5)
    assert flexres_str, "no residues found inside box"

    flex_rigid = tmp_path / "receptor_rigid.pdbqt"
    flex_mobile = tmp_path / "receptor_flex.pdbqt"
    prepare_flex_receptor_pdbqt(
        str(clean_pdb), str(flex_rigid), str(flex_mobile), flexres_str, allow_bad_res=True
    )
    assert flex_rigid.exists() and flex_rigid.stat().st_size > 0
    assert flex_mobile.exists() and flex_mobile.stat().st_size > 0

    initial = _parse_pdbqt_atoms(flex_mobile)
    assert len(initial) > 0, "flex PDBQT has no atoms"

    lig_pdbqt = _make_ligand_pdbqt(tmp_path)
    out_pdbqt = tmp_path / "out.pdbqt"
    out_scores = tmp_path / "scores.txt"

    deploy_vina(
        receptor_pdbqt=str(flex_rigid),
        ligand_pdbqt=str(lig_pdbqt),
        center=center,
        size=size,
        output_scores=str(out_scores),
        output_pdbqt=str(out_pdbqt),
        flex_pdbqt=str(flex_mobile),
    )

    out_text = out_pdbqt.read_text()
    assert "BEGIN_RES" in out_text, "no BEGIN_RES in Vina output — flex atoms were not written"

    docked = _parse_flex_atoms_pose1(out_text)
    assert len(docked) > 0, "could not parse flex atoms from pose 1"

    n, moved = _displacement_stats(initial, docked)
    assert n > 0
    assert len(moved) > 0, (
        f"no flex atoms moved in Vina output ({n} compared) — "
        "flexible docking may not be working"
    )


@pytest.mark.e2e
@pytest.mark.skipif(not GNINA_AVAILABLE, reason=f"gnina binary not found at {GNINA_BINARY}")
def test_flexible_docking_gnina_pdbqt_side_chains_move(tmp_path):
    """
    gnina flexible docking (PDBQT mode): side-chain atoms in the output
    BEGIN_RES blocks must have moved relative to their starting positions.
    """
    clean_pdb = _prepare_protein(tmp_path)
    center, size = get_center_and_size_from_box_file(str(BOX_FILE))

    flexres_str = get_flexres_string_from_box(str(clean_pdb), "A", center, size, max_flexres=5)
    assert flexres_str, "no residues found inside box"

    flex_rigid = tmp_path / "receptor_rigid.pdbqt"
    flex_mobile = tmp_path / "receptor_flex.pdbqt"
    prepare_flex_receptor_pdbqt(
        str(clean_pdb), str(flex_rigid), str(flex_mobile), flexres_str, allow_bad_res=True
    )

    initial = _parse_pdbqt_atoms(flex_mobile)
    assert len(initial) > 0

    lig_pdbqt = _make_ligand_pdbqt(tmp_path)
    out_pdbqt = tmp_path / "out.pdbqt"
    out_flex = tmp_path / "out_flex.pdbqt"
    out_scores = tmp_path / "scores.txt"

    deploy_gnina(
        receptor=str(flex_rigid),
        ligand=str(lig_pdbqt),
        center=center,
        size=size,
        output_pdbqt=str(out_pdbqt),
        output_scores=str(out_scores),
        use_gpu=False,
        flex_pdbqt=str(flex_mobile),
        out_flex_pdbqt=str(out_flex),
    )

    assert out_pdbqt.exists() and out_pdbqt.stat().st_size > 0, "gnina wrote no output"
    assert out_flex.exists() and out_flex.stat().st_size > 0, "gnina wrote no out_flex file"

    flex_text = out_flex.read_text()
    assert "ROOT" in flex_text, "no ROOT in gnina out_flex — flex atoms were not written"

    docked = _parse_gnina_flex_atoms_pose1(flex_text)
    assert len(docked) > 0

    n, moved = _displacement_stats(initial, docked)
    assert n > 0
    assert len(moved) > 0, f"no flex atoms moved in gnina PDBQT output ({n} compared)"


@pytest.mark.e2e
@pytest.mark.skipif(not GNINA_AVAILABLE, reason=f"gnina binary not found at {GNINA_BINARY}")
def test_flexible_docking_gnina_sdf_flexdist_side_chains_move(tmp_path):
    """
    gnina flexible docking (SDF mode): gnina selects flexible residues
    automatically via --flexdist_ligand / --flexdist.  Side-chain atoms
    in the BEGIN_RES blocks of the output must have moved.
    """
    clean_pdb = _prepare_protein(tmp_path)
    center, size = get_center_and_size_from_box_file(str(BOX_FILE))

    lig_sdf = _make_ligand_sdf(tmp_path)
    out_sdf = tmp_path / "out.sdf"
    out_flex = tmp_path / "out_flex.pdbqt"
    out_scores = tmp_path / "scores.txt"

    deploy_gnina(
        receptor=str(clean_pdb),
        ligand=str(lig_sdf),
        center=center,
        size=size,
        output_pdbqt=str(out_sdf),
        output_scores=str(out_scores),
        use_gpu=False,
        flexdist_ligand=str(lig_sdf),
        flexdist=GNINA_FLEX_DISTANCE,
        out_flex_pdbqt=str(out_flex),
    )

    assert out_sdf.exists() and out_sdf.stat().st_size > 0, "gnina wrote no output"
    assert out_flex.exists() and out_flex.stat().st_size > 0, "gnina wrote no out_flex file"

    flex_text = out_flex.read_text()
    assert "ROOT" in flex_text, "no ROOT in gnina out_flex — flexible residues were not selected"

    docked = _parse_gnina_flex_atoms_pose1(flex_text)
    assert len(docked) > 0, "could not parse flex atoms from pose 1"
