"""
Tests for Guild file preparation (ligand and protein processing).
"""

import os
import shutil
from pathlib import Path

import pytest

from guild.run import Guild

# Get paths relative to test file
TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"


@pytest.fixture
def test_data_paths():
    """Fixture providing paths to test data files."""
    return {
        "pdb": str(TEST_DATA_DIR / "3pbl.pdb"),
        "box": str(TEST_DATA_DIR / "3pbl_box.txt"),
    }


@pytest.fixture
def cleanup_temp_dir():
    """Fixture to clean up temporary test output directory."""
    yield
    data_dir = Path.cwd() / "data"
    if data_dir.exists():
        for test_project in data_dir.glob("test_*"):
            shutil.rmtree(test_project, ignore_errors=True)


def test_ligand_file_generation(test_data_paths, cleanup_temp_dir):
    """
    Test that ligand files are generated correctly.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="ethanol",
        protein_idx="3pbl",
        protein_file=test_data_paths["pdb"],
        project_name="test-ligand-prep",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify ligand files were created
    assert os.path.exists(dw.ligand_sdf), "Ligand SDF should be generated"
    assert os.path.exists(dw.ligand_pdb), "Ligand PDB should be generated"
    assert os.path.exists(dw.ligand_pdbqt), "Ligand PDBQT should be generated"

    # Verify files have content
    assert os.path.getsize(dw.ligand_sdf) > 0
    assert os.path.getsize(dw.ligand_pdb) > 0
    assert os.path.getsize(dw.ligand_pdbqt) > 0

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_protein_file_relocation(test_data_paths, cleanup_temp_dir):
    """
    Test that protein file is relocated to project directory.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="test_lig",
        protein_idx="3pbl",
        protein_file=test_data_paths["pdb"],
        project_name="test-protein-reloc",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify protein was copied
    assert os.path.exists(dw.local_protein), "Raw protein file should exist"
    assert os.path.getsize(dw.local_protein) > 0

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_protein_chain_isolation(test_data_paths, cleanup_temp_dir):
    """
    Test that protein chain is isolated correctly.
    """
    dw = Guild(
        ligand_smile="CC(=COC=O)CCC1=C(C)CCCC1(C)C",
        ligand_idx="test_lig",
        protein_idx="3pbl-A-ETQ-A",
        protein_file=test_data_paths["pdb"],
        project_name="test-chain-iso",
        protein_chain="A",
        original_ligand="ETQ",
        original_ligand_chain="A",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify single chain protein was created
    assert os.path.exists(dw.single_chain_protein), "Single chain protein should be extracted"
    assert os.path.getsize(dw.single_chain_protein) > 0

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_protein_cleaning(test_data_paths, cleanup_temp_dir):
    """
    Test that protein is cleaned correctly.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="test_lig",
        protein_idx="3pbl",
        protein_file=test_data_paths["pdb"],
        project_name="test-protein-clean",
        protein_chain="A",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify cleaned protein exists
    assert os.path.exists(dw.cleaned_protein), "Cleaned protein should exist"
    assert os.path.getsize(dw.cleaned_protein) > 0

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_karmadock_file_preparation(test_data_paths, cleanup_temp_dir):
    """
    Test that KarmaDock-specific files are prepared.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="test_lig",
        protein_idx="3pbl",
        protein_file=test_data_paths["pdb"],
        project_name="test-karmadock-prep",
        protein_chain="A",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify KarmaDock files
    assert os.path.exists(dw.karmadock_ligand), "KarmaDock ligand SDF should exist"
    assert os.path.exists(dw.karmadock_mol2), "KarmaDock MOL2 should exist"
    assert os.path.exists(dw.karmadock_protein), "KarmaDock protein should exist"

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_ligand_extraction_with_original(test_data_paths, cleanup_temp_dir):
    """
    Test ligand extraction when original ligand is specified.
    """
    dw = Guild(
        ligand_smile="CC(=COC=O)CCC1=C(C)CCCC1(C)C",
        ligand_idx="test_lig",
        protein_idx="3pbl-A-ETQ-A",
        protein_file=test_data_paths["pdb"],
        project_name="test-lig-extract",
        protein_chain="A",
        original_ligand="ETQ",
        original_ligand_chain="A",
        use_gpu=False,
        is_bulk=False,
    )

    # When original ligand is specified, ligand should be extracted from PDB
    assert os.path.exists(dw.ligand_pdb)
    assert os.path.exists(dw.single_chain_protein)

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
