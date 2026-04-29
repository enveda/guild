"""
Tests for Guild initialization and basic setup.
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
        "csv": str(TEST_DATA_DIR / "bulk_dummy.csv"),
    }


@pytest.fixture
def cleanup_temp_dir():
    """Fixture to clean up temporary test output directory."""
    yield
    # Clean up any test projects created
    data_dir = Path.cwd() / "data"
    if data_dir.exists():
        for test_project in data_dir.glob("test_*"):
            shutil.rmtree(test_project, ignore_errors=True)


def test_guild_basic_initialization(test_data_paths, cleanup_temp_dir):
    """
    Test basic Guild initialization with minimal parameters.
    """
    dw = Guild(
        ligand_smile="CCO",  # Simple ethanol
        ligand_idx="ethanol",
        protein_idx="3pbl",
        protein_file=test_data_paths["pdb"],
        project_name="test-init-basic",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify object creation
    assert dw is not None
    assert dw.protein_idx == "3pbl"
    assert dw.ligand_idx == "ethanol"
    assert dw.original_ligand_smile == "CCO"
    assert dw.use_gpu is False

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_guild_full_initialization(test_data_paths, cleanup_temp_dir):
    """
    Test Guild initialization with all parameters.
    """
    dw = Guild(
        ligand_smile="CC(=COC=O)CCC1=C(C)CCCC1(C)C",
        ligand_idx="test_ligand",
        protein_idx="3pbl-A-ETQ-A",
        protein_file=test_data_paths["pdb"],
        project_name="test-init-full",
        protein_chain="A",
        original_ligand="ETQ",
        original_ligand_chain="A",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify all parameters were set
    assert dw.protein_chain == "A"
    assert dw.original_ligand == "ETQ"
    assert dw.original_ligand_chain == "A"
    assert dw.combination_id == "3pbl-A-ETQ-A_test_ligand"

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_guild_paths_set_correctly(test_data_paths, cleanup_temp_dir):
    """
    Test that all internal paths are set correctly during initialization.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="test_lig",
        protein_idx="test_prot",
        protein_file=test_data_paths["pdb"],
        project_name="test-paths",
        use_gpu=False,
        is_bulk=False,
    )

    # Check path attributes exist and are strings
    assert isinstance(dw.project_dir, str)
    assert isinstance(dw.ligand_dir, str)
    assert isinstance(dw.protein_dir, str)
    assert isinstance(dw.vina_dir, str)
    assert isinstance(dw.karmadock_root, str)
    assert isinstance(dw.diffdock_dir, str)
    assert isinstance(dw.boltz_dir, str)

    # Check paths contain expected components
    assert "test-paths" in dw.project_dir
    assert "ligands" in dw.ligand_dir
    assert "proteins" in dw.protein_dir
    assert "vina" in dw.vina_dir

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_guild_directory_creation(test_data_paths, cleanup_temp_dir):
    """
    Test that all required directories are created during initialization.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="test_lig",
        protein_idx="test_prot",
        protein_file=test_data_paths["pdb"],
        project_name="test-dirs",
        use_gpu=False,
        is_bulk=False,
    )

    # Verify all required directories exist
    assert os.path.exists(dw.project_dir)
    assert os.path.exists(dw.ligand_dir)
    assert os.path.exists(dw.protein_dir)
    assert os.path.exists(dw.vina_dir)
    assert os.path.exists(dw.boxes_dir)
    assert os.path.exists(dw.vina_boxes_dir)
    assert os.path.exists(dw.karmadock_root)
    assert os.path.exists(dw.karmadock_data_dir)
    assert os.path.exists(dw.diffdock_dir)
    assert os.path.exists(dw.boltz_dir)

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


def test_guild_pdb_file_extension_check(test_data_paths, cleanup_temp_dir):
    """
    Test that Guild correctly identifies PDB file extension.
    """
    dw = Guild(
        ligand_smile="CCO",
        ligand_idx="test_lig",
        protein_idx="test_prot",
        protein_file=test_data_paths["pdb"],
        project_name="test-extension",
        use_gpu=False,
        is_bulk=False,
    )

    assert dw.protein_file_extension == "pdb"

    # Cleanup
    if os.path.exists(dw.project_dir):
        shutil.rmtree(dw.project_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
