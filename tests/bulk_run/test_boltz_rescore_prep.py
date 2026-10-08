"""
Tests for Boltz rescore input preparation: Meeko's clash-residue retry and the
SMILES-templated, hydrogenated ligand.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from guild.docking.boltz import _prepare_boltz_ligand
from guild.transformers.converters import (
    _residue_spec,
    _residues_near,
    protein_pdb_to_pdbqt,
)

MEEKO_CLASH = (
    "matched with excess inter-residue bond(s): A:69\n"
    "matched with excess inter-residue bond(s): A:72\n"
    "RuntimeError: Expected 2 paddings for (A:69, A:72) with bonds [(1, 18)], but got 0\n"
)


def _pdb_line(serial, name, resname, chain, resnum, xyz, element="C", record="ATOM"):
    x, y, z = xyz
    return (
        f"{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resnum:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


@pytest.fixture
def receptor(tmp_path):
    lines = [
        _pdb_line(1, "CA", "ASN", "A", 69, (30.0, 0.0, 0.0)),
        _pdb_line(2, "CA", "ARG", "A", 72, (31.0, 0.0, 0.0)),
        _pdb_line(3, "CA", "GLY", "A", 10, (2.0, 0.0, 0.0)),
    ]
    path = tmp_path / "receptor.pdb"
    path.write_text("".join(lines) + "END\n")
    return path


def _completed(returncode, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_residue_spec_groups_by_chain():
    assert _residue_spec([("A", "69"), ("A", "72"), ("B", "5")]) == "A:69,72,B:5"


def test_residues_near_uses_any_atom(receptor):
    ligand = np.array([[0.0, 0.0, 0.0]])
    residues = [("A", "69"), ("A", "10")]
    assert _residues_near(str(receptor), residues, ligand, 8.0) == [("A", "10")]


def test_success_needs_no_retry(receptor):
    with patch("guild.transformers.converters.subprocess.run", return_value=_completed(0)) as run:
        protein_pdb_to_pdbqt(str(receptor), delete_clashing_residues=True)
    assert run.call_count == 1


def test_failure_raises_meeko_output_by_default(receptor):
    with patch(
        "guild.transformers.converters.subprocess.run", return_value=_completed(1, MEEKO_CLASH)
    ):
        with pytest.raises(RuntimeError, match="excess inter-residue bond"):
            protein_pdb_to_pdbqt(str(receptor))


def test_clash_retry_deletes_reported_residues(receptor):
    calls = [_completed(1, MEEKO_CLASH), _completed(0)]
    with patch("guild.transformers.converters.subprocess.run", side_effect=calls) as run:
        protein_pdb_to_pdbqt(
            str(receptor),
            delete_clashing_residues=True,
            protected_coordinates=np.array([[0.0, 0.0, 0.0]]),
        )
    retry_cmd = run.call_args_list[1].args[0]
    assert retry_cmd[-2:] == ["--delete_residues", "A:69,72"]


def test_clash_near_ligand_is_not_deleted(receptor):
    ligand_next_to_clash = np.array([[30.5, 0.0, 0.0]])
    with patch(
        "guild.transformers.converters.subprocess.run", return_value=_completed(1, MEEKO_CLASH)
    ) as run:
        with pytest.raises(RuntimeError, match="within 8.0 Å of the ligand"):
            protein_pdb_to_pdbqt(
                str(receptor),
                delete_clashing_residues=True,
                protected_coordinates=ligand_next_to_clash,
            )
    assert run.call_count == 1


def test_failed_retry_raises(receptor):
    calls = [_completed(1, MEEKO_CLASH), _completed(1, "still broken")]
    with patch("guild.transformers.converters.subprocess.run", side_effect=calls):
        with pytest.raises(RuntimeError, match="even without residue"):
            protein_pdb_to_pdbqt(str(receptor), delete_clashing_residues=True)


def _complex_with_ligand(tmp_path: Path, smiles: str, squash: bool = False) -> Path:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    mol = Chem.RemoveHs(mol)
    positions = mol.GetConformer().GetPositions()
    if squash:
        # Put two non-bonded heavy atoms on top of each other, as a distorted
        # prediction would; bond inference then disagrees with the SMILES.
        positions[-1] = positions[0] + 0.3
    lines = [_pdb_line(1, "CA", "GLY", "A", 1, (20.0, 20.0, 20.0))]
    for idx, atom in enumerate(mol.GetAtoms(), start=2):
        name = f"{atom.GetSymbol()}{idx}"
        lines.append(
            _pdb_line(idx, name, "LIG", "L", 1, positions[idx - 2], atom.GetSymbol(), "HETATM")
        )
    path = tmp_path / "complex.pdb"
    path.write_text("".join(lines) + "END\n")
    return path


def test_boltz_ligand_gets_smiles_bond_orders_and_hydrogens(tmp_path):
    smiles = "OC(=O)c1ccccc1"
    complex_pdb = _complex_with_ligand(tmp_path, smiles)
    sdf, heavy = _prepare_boltz_ligand(str(complex_pdb), str(tmp_path / "ligand.pdb"), smiles)
    mol = Chem.MolFromMolFile(sdf, removeHs=False)
    assert heavy.shape == (9, 3)
    assert sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()) == 6
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_boltz_ligand_that_does_not_match_smiles_is_rejected(tmp_path):
    smiles = "OCCCCCCO"
    complex_pdb = _complex_with_ligand(tmp_path, smiles, squash=True)
    with pytest.raises(ValueError, match="does not match its SMILES"):
        _prepare_boltz_ligand(str(complex_pdb), str(tmp_path / "ligand.pdb"), smiles)
