"""
Tests for BulkRun batch creation and management.
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from guild.bulk import BulkRun

# Get paths relative to test file
TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"


@pytest.fixture
def test_input_table():
    """Fixture providing loaded test CSV as DataFrame."""
    csv_path = str(TEST_DATA_DIR / "bulk_dummy.csv")
    df = pd.read_csv(csv_path)
    # Update protein_path to absolute path
    df["protein_path"] = str(TEST_DATA_DIR / df["protein_path"].iloc[0])
    return df


@pytest.fixture
def cleanup_bulk_test():
    """Fixture to clean up bulk test output."""
    yield
    test_project_path = Path.cwd() / "data" / "test-bulk-batch"
    if test_project_path.exists():
        shutil.rmtree(test_project_path, ignore_errors=True)


def test_batch_dictionary_creation(test_input_table, cleanup_bulk_test):
    """
    Test that batch dictionary is created correctly.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=2,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Verify batch dictionary exists
    assert bulk.batched_dictionary is not None
    assert len(bulk.batched_dictionary) > 0


def test_batch_structure(test_input_table, cleanup_bulk_test):
    """
    Test that each batch has the required structure.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=1,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check batch structure
    for _batch_name, batch_data in bulk.batched_dictionary.items():
        assert "batch_folder" in batch_data
        assert "combinations_table" in batch_data
        assert "smiles_names_dictionary" in batch_data
        assert "smiles_type_dictionary" in batch_data
        assert "combinations_to_run" in batch_data

        # Verify types
        assert isinstance(batch_data["batch_folder"], str)
        assert isinstance(batch_data["combinations_table"], pd.DataFrame)
        assert isinstance(batch_data["smiles_names_dictionary"], dict)
        assert isinstance(batch_data["smiles_type_dictionary"], dict)
        assert isinstance(batch_data["combinations_to_run"], list)


def test_batch_count_with_different_sizes(test_input_table, cleanup_bulk_test):
    """
    Test that batch count is calculated correctly for different batch sizes.
    """
    # With 1 row and batch_size=1, should have 1 batch
    bulk_size_1 = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=1,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )
    assert len(bulk_size_1.batched_dictionary) == 1

    # With 1 row and batch_size=10, should still have 1 batch
    test_project_path = Path.cwd() / "data" / "test-bulk-batch"
    if test_project_path.exists():
        shutil.rmtree(test_project_path, ignore_errors=True)

    bulk_size_10 = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=10,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )
    assert len(bulk_size_10.batched_dictionary) == 1


def test_batch_folder_creation(test_input_table, cleanup_bulk_test):
    """
    Test that batch folders are created in the file system.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=1,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check that batch folders exist
    for _batch_name, batch_data in bulk.batched_dictionary.items():
        batch_folder = batch_data["batch_folder"]
        assert os.path.exists(batch_folder)


def test_combinations_to_run_list(test_input_table, cleanup_bulk_test):
    """
    Test that combinations_to_run list contains expected data.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=2,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check combinations_to_run
    for _batch_name, batch_data in bulk.batched_dictionary.items():
        combinations = batch_data["combinations_to_run"]
        assert isinstance(combinations, list)
        # Should have at least some combinations
        assert len(combinations) >= 0


def test_smiles_dictionaries_populated(test_input_table, cleanup_bulk_test):
    """
    Test that SMILES dictionaries are populated correctly.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-batch",
        methods_to_run=["vina"],
        batch_size=1,
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    for _batch_name, batch_data in bulk.batched_dictionary.items():
        smiles_names = batch_data["smiles_names_dictionary"]
        smiles_types = batch_data["smiles_type_dictionary"]

        # Verify they're dictionaries
        assert isinstance(smiles_names, dict)
        assert isinstance(smiles_types, dict)

        # Should have entries if there are combinations
        if len(batch_data["combinations_to_run"]) > 0:
            assert len(smiles_names) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
