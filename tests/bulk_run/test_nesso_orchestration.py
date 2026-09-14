"""
Tests for the Nesso-1 integration in the bulk pipeline:

- nesso is registered as a standalone method (no auto-enable rescore rules,
  unlike Boltz's ipTM-confidence score which triggers vina_rescore_boltz).
- ``generate_nesso_yaml`` emits Nesso's minimal schema — no msa/templates/
  constraints keys, since Nesso accepts neither MSA, structural templates,
  nor pocket constraints.
- ``nesso_guild_scoring`` reads ``affinity.json`` correctly, including the
  side-channel binder-probability and entropy columns, and tolerates a
  missing/invalid file without raising.
- ``deploy_nesso`` invokes a single directory-mode subprocess call (never
  per-ligand — that would throw away Nesso's whole throughput advantage).
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from guild.bulk import BulkRun
from guild.constants.bulk import COMBINATION_ID as BULK_COMBINATION_ID
from guild.constants.guild import (
    ALL_AVAILABLE_METHODS,
    NESSO_BINDER_PROBABILITY,
    NESSO_ENTROPY_PL,
    NESSO_PREFIX,
    NESSO_SCORE,
    SCORES_DICTIONARY,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX,
)
from guild.docking.nesso import (
    deploy_nesso,
    generate_nesso_yaml,
    nesso_guild_scoring,
)

TEST_DIR = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_DIR / "test_data"


@pytest.fixture
def test_input_table():
    df = pd.read_csv(TEST_DATA_DIR / "bulk_dummy.csv")
    df["protein_path"] = str(TEST_DATA_DIR / df["protein_path"].iloc[0])
    return df


@pytest.fixture
def cleanup():
    yield
    test_project = Path.cwd() / "data" / "test-nesso"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_nesso_registered_as_available_method():
    """nesso is in the list of methods picked up when none are specified."""
    assert NESSO_PREFIX in ALL_AVAILABLE_METHODS


def test_nesso_score_dictionary_entry():
    assert SCORES_DICTIONARY[NESSO_PREFIX] == NESSO_SCORE


def test_nesso_alone_does_not_enable_any_rescore(test_input_table, cleanup):
    """
    nesso is a standalone potency predictor, not a confidence score — unlike
    Boltz, selecting it must not auto-add any vina_rescore_* track.
    """
    bulk = BulkRun(
        input_table=test_input_table,
        project_name="test-nesso",
        methods_to_run=[NESSO_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )
    assert NESSO_PREFIX in bulk.methods_to_run
    assert VINA_RESCORE_BOLTZ_PREFIX not in bulk.methods_to_run
    assert VINA_RESCORE_DIFFDOCK_PREFIX not in bulk.methods_to_run


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------
def test_generate_nesso_yaml_minimal_schema(tmp_path):
    """
    Nesso accepts only protein sequence + ligand SMILES — no msa, templates,
    or constraints keys should ever appear, unlike generate_boltz_yaml.
    """
    output_file = tmp_path / "complex.yaml"
    generate_nesso_yaml(
        protein_sequence="MKTAYIAKQR",
        protein_chain="A",
        ligand_smiles="CCO",
        ligand_id="L",
        output_file=output_file,
    )

    with open(output_file) as f:
        data = yaml.safe_load(f)

    assert "msa" not in str(data)
    assert "templates" not in data
    assert "constraints" not in data

    sequences = data["sequences"]
    assert {"protein": {"id": "A", "sequence": "MKTAYIAKQR"}} in sequences
    assert {"ligand": {"id": "L", "smiles": "CCO"}} in sequences

    assert data["properties"] == [{"affinity": {"binder": "L"}}]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _batch_dict_for(combinations_df, batch_folder):
    from guild.constants.bulk import BATCH_FOLDER, COMBINATIONS_TABLE_KEY

    return {
        BATCH_FOLDER: str(batch_folder),
        COMBINATIONS_TABLE_KEY: combinations_df,
    }


def test_nesso_guild_scoring_reads_affinity_json(tmp_path):
    combinations_df = pd.DataFrame({"protein_config_id": ["P1"], "ligand_id": ["lig1"]})
    predictions_dir = tmp_path / "nesso" / "predictions" / "P1_lig1"
    predictions_dir.mkdir(parents=True)
    with open(predictions_dir / "affinity.json", "w") as f:
        json.dump(
            {
                "affinity_pred_value": -0.75,
                "affinity_probability_binary": 0.62,
                "entropy_crop_pl": 0.31,
            },
            f,
        )

    result = nesso_guild_scoring(_batch_dict_for(combinations_df, tmp_path))

    assert len(result) == 1
    row = result.iloc[0]
    assert row[NESSO_SCORE] == pytest.approx(-0.75)
    assert row[NESSO_BINDER_PROBABILITY] == pytest.approx(0.62)
    assert row[NESSO_ENTROPY_PL] == pytest.approx(0.31)
    assert row[BULK_COMBINATION_ID] == "P1_lig1"


def test_nesso_guild_scoring_missing_file_is_skipped_not_raised(tmp_path):
    combinations_df = pd.DataFrame({"protein_config_id": ["P1"], "ligand_id": ["lig1"]})
    # No affinity.json written anywhere under tmp_path.
    result = nesso_guild_scoring(_batch_dict_for(combinations_df, tmp_path))

    assert result.empty
    assert list(result.columns) == [
        "ligand_id",
        NESSO_SCORE,
        NESSO_BINDER_PROBABILITY,
        NESSO_ENTROPY_PL,
        "protein_config_id",
        BULK_COMBINATION_ID,
    ]


def test_nesso_guild_scoring_invalid_json_is_skipped_not_raised(tmp_path):
    combinations_df = pd.DataFrame({"protein_config_id": ["P1"], "ligand_id": ["lig1"]})
    predictions_dir = tmp_path / "nesso" / "predictions" / "P1_lig1"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "affinity.json").write_text("not valid json")

    result = nesso_guild_scoring(_batch_dict_for(combinations_df, tmp_path))
    assert result.empty


def test_nesso_guild_scoring_mixed_valid_and_missing(tmp_path):
    combinations_df = pd.DataFrame(
        {
            "protein_config_id": ["P1", "P1"],
            "ligand_id": ["lig1", "lig2"],
        }
    )
    good_dir = tmp_path / "nesso" / "predictions" / "P1_lig1"
    good_dir.mkdir(parents=True)
    with open(good_dir / "affinity.json", "w") as f:
        json.dump({"affinity_pred_value": 1.1}, f)
    # lig2 has no output at all.

    result = nesso_guild_scoring(_batch_dict_for(combinations_df, tmp_path))

    assert len(result) == 1
    assert result.iloc[0]["ligand_id"] == "lig1"
    assert result.iloc[0][NESSO_SCORE] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Deployment — must batch the whole directory in one subprocess call
# ---------------------------------------------------------------------------
def test_deploy_nesso_invokes_directory_mode_once(tmp_path):
    """
    deploy_nesso must call the CLI exactly once over the input directory —
    per-ligand invocation would pay model-load cost per complex and defeat
    the entire point of using Nesso over Boltz-2.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    fake_result = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("guild.docking.nesso.subprocess.run", return_value=fake_result) as mock_run:
        deploy_nesso(input_dir, out_dir=tmp_path / "out", use_gpu=False)

    assert mock_run.call_count == 1
    argv = mock_run.call_args.args[0]
    assert argv[1] == "predict"
    assert argv[2] == str(input_dir)
    assert "--accelerator" in argv
    assert "cpu" in argv
    # CPU path must disable cuEquivariance kernels.
    assert "--no_kernels" in argv


def test_deploy_nesso_gpu_path_omits_no_kernels_by_default(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    fake_result = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("guild.docking.nesso.subprocess.run", return_value=fake_result) as mock_run:
        deploy_nesso(input_dir, out_dir=tmp_path / "out", use_gpu=True)

    argv = mock_run.call_args.args[0]
    assert "gpu" in argv
    assert "--no_kernels" not in argv
