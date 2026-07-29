"""
Tests for ``guild.transformers.pdb.covalent_rec_atom_exists``.

Reuses the 3pbl PDB fixture. Pure Python + BioPython — no Docker / gnina.
"""

from pathlib import Path

from guild.transformers.pdb import covalent_rec_atom_exists

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
PDB = str(TEST_DATA_DIR / "3pbl.pdb")


def test_existing_atom_returns_true():
    # 3pbl chain A residue 32 is a TYR with backbone N / CA atoms.
    assert covalent_rec_atom_exists(PDB, "A:32:CA") is True
    assert covalent_rec_atom_exists(PDB, "A:32:N") is True


def test_missing_atom_returns_false():
    # TYR has no SG atom.
    assert covalent_rec_atom_exists(PDB, "A:32:SG") is False


def test_missing_residue_returns_false():
    assert covalent_rec_atom_exists(PDB, "A:999999:CA") is False


def test_missing_chain_returns_false():
    assert covalent_rec_atom_exists(PDB, "Z:32:CA") is False


def test_malformed_spec_returns_false():
    assert covalent_rec_atom_exists(PDB, "A:32") is False
    assert covalent_rec_atom_exists(PDB, "A:notanint:CA") is False
    assert covalent_rec_atom_exists(PDB, "garbage") is False
