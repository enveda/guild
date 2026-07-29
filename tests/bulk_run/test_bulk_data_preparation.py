"""
Tests for BulkRun data preparation and processing.
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
    test_project_path = Path.cwd() / "data" / "test-bulk-data"
    if test_project_path.exists():
        shutil.rmtree(test_project_path, ignore_errors=True)


def test_protein_path_mapping(test_input_table, cleanup_bulk_test):
    """
    Test that protein paths are correctly mapped from protein_id.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check protein path mapper was created
    assert hasattr(bulk, "protein_path_mapper")
    assert len(bulk.protein_path_mapper) > 0
    assert "3pbl" in bulk.protein_path_mapper

    # Verify mapping is correct
    expected_path = str(TEST_DATA_DIR / "3pbl.pdb")
    assert bulk.protein_path_mapper["3pbl"] == expected_path


def test_smiles_cleaning(test_input_table, cleanup_bulk_test):
    """
    Test that SMILES strings are cleaned (salt stripping).
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check that SMILES in all_combinations_table were processed
    assert "smiles" in bulk.all_combinations_table.columns
    # Should not have any None/NaN values
    assert bulk.all_combinations_table["smiles"].notna().all()


def test_all_combinations_table_creation(test_input_table, cleanup_bulk_test):
    """
    Test that all_combinations_table is created correctly.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Verify table exists
    assert bulk.all_combinations_table is not None
    assert isinstance(bulk.all_combinations_table, pd.DataFrame)
    assert len(bulk.all_combinations_table) > 0

    # Check required columns exist
    assert "protein_config_id" in bulk.all_combinations_table.columns
    assert "smiles" in bulk.all_combinations_table.columns
    assert "ligand_id" in bulk.all_combinations_table.columns
    assert "protein_path" in bulk.all_combinations_table.columns


def test_all_combinations_file_saved(test_input_table, cleanup_bulk_test):
    """
    Test that all_combinations CSV file is saved to disk.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check that file exists
    assert os.path.exists(bulk.all_combinations_path)

    # Verify we can read it back
    saved_df = pd.read_csv(bulk.all_combinations_path)
    assert len(saved_df) >= 1
    assert "protein_config_id" in saved_df.columns
    assert "smiles" in saved_df.columns


def test_no_decoys_mode(test_input_table, cleanup_bulk_test):
    """
    Test that no decoys are added when use_decoys=False.
    """
    input_row_count = len(test_input_table)

    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Should only have original rows (no decoys added)
    assert len(bulk.all_combinations_table) == input_row_count


def test_no_known_binders_mode(test_input_table, cleanup_bulk_test):
    """
    Test that no known binders are added when use_known_binders=False.
    """
    input_row_count = len(test_input_table)

    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Should only have original rows (no known binders added)
    assert len(bulk.all_combinations_table) == input_row_count


def test_existing_scores_start_empty(test_input_table, cleanup_bulk_test):
    """
    Test that a fresh BulkRun starts with an empty existing-scores frame.

    The scoring path merges against this frame, so it has to be a real empty
    DataFrame rather than None.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Should have empty existing scores dataframe
    assert bulk.existing_rp_scores is not None
    assert isinstance(bulk.existing_rp_scores, pd.DataFrame)
    assert len(bulk.existing_rp_scores) == 0


def test_protein_path_replacement_in_table(test_input_table, cleanup_bulk_test):
    """
    Test that protein_path column is updated with mapped paths.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-data",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check that protein_path column contains the full path
    protein_paths = bulk.all_combinations_table["protein_path"].unique()
    for path in protein_paths:
        # Should be an absolute path or contain the test data directory
        assert os.path.isabs(path) or str(TEST_DATA_DIR) in path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
