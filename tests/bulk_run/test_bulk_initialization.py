"""
Tests for BulkRun initialization and basic setup.
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
def test_csv_path():
    """Fixture providing path to test CSV file."""
    return str(TEST_DATA_DIR / "bulk_dummy.csv")


@pytest.fixture
def test_input_table(test_csv_path):
    """Fixture providing loaded test CSV as DataFrame."""
    df = pd.read_csv(test_csv_path)
    # Update protein_path to absolute path
    df["protein_path"] = str(TEST_DATA_DIR / df["protein_path"].iloc[0])
    return df


@pytest.fixture
def cleanup_bulk_test():
    """Fixture to clean up bulk test output."""
    yield
    test_project_path = Path.cwd() / "data" / "test-bulk-init"
    if test_project_path.exists():
        shutil.rmtree(test_project_path, ignore_errors=True)


def test_bulk_run_minimal_initialization(test_input_table, cleanup_bulk_test):
    """
    Test BulkRun initialization with minimal settings.
    No decoys, no known binders.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-init",
        methods_to_run=["vina"],
        batch_size=2,
        use_decoys=False,
        use_known_binders=False,
        n_workers=1,
        use_gpu=False,
    )

    # Verify object creation
    assert bulk is not None
    assert bulk.project_name == "test-bulk-init"
    assert bulk.methods_to_run == ["vina"]
    assert bulk.use_decoys is False
    assert bulk.use_known_binders is False
    assert bulk.n_workers == 1


def test_bulk_run_directory_creation(test_input_table, cleanup_bulk_test):
    """
    Test that all required directories are created.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-init",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Verify directories were created
    assert os.path.exists(bulk.project_folder)
    assert os.path.exists(bulk.batches_folder)
    assert os.path.exists(bulk.plots_folder)


def test_bulk_run_paths_configuration(test_input_table, cleanup_bulk_test):
    """
    Test that all internal paths are configured correctly.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-init",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Check path attributes
    assert isinstance(bulk.project_folder, str)
    assert isinstance(bulk.batches_folder, str)
    assert isinstance(bulk.plots_folder, str)
    assert isinstance(bulk.rp_scores_path, str)
    assert isinstance(bulk.all_combinations_path, str)

    # Check paths contain expected components
    assert "test-bulk-init" in bulk.project_folder
    assert "batches" in bulk.batches_folder
    assert "plots" in bulk.plots_folder


def test_bulk_run_rejects_underscore_in_project_name(test_input_table):
    """
    Test that BulkRun rejects project names containing underscores.
    """
    with pytest.raises(ValueError, match="cannot contain underscores"):
        BulkRun(
            input_table=test_input_table,
            project_name="invalid_project_name",
            methods_to_run=["vina"],
            use_decoys=False,
            use_known_binders=False,
            use_gpu=False,
        )


def test_bulk_run_default_methods(test_input_table, cleanup_bulk_test):
    """
    Test that default methods are set when None is provided.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-init",
        methods_to_run=None,  # Should default to all available
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    # Should have some methods configured
    assert bulk.methods_to_run is not None
    assert len(bulk.methods_to_run) > 0


def test_bulk_run_gpu_configuration(test_input_table, cleanup_bulk_test):
    """
    Test GPU configuration setting.
    """
    bulk_no_gpu = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-init",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
    )

    assert bulk_no_gpu.use_gpu is False


def test_bulk_run_worker_count_default(test_input_table, cleanup_bulk_test):
    """
    Test that worker count defaults to CPU count when None.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-bulk-init",
        methods_to_run=["vina"],
        use_decoys=False,
        use_known_binders=False,
        n_workers=None,  # Should default to cpu_count
        use_gpu=False,
    )

    # Should have set to some positive number
    assert bulk.n_workers > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
