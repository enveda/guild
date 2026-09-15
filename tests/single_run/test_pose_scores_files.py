"""
Tests for the batch-level all-poses score files (``vina_scores.txt`` /
``gnina_scores.txt``). Unlike ``guild_scores.txt``, which keeps only the
single best pose per combination, these keep the full per-pose distribution
so it doesn't require re-parsing every per-combination score file.
"""

import pandas as pd
import pytest

from guild.constants.guild import (
    GNINA_CNN_SCORE,
    GNINA_SCORE,
    POSE,
    VINA_SCORE,
)
from guild.docking.gnina import write_gnina_pose_scores_file
from guild.docking.vina import write_vina_pose_scores_file


def _make_batch(tmp_path, method_folder):
    batch_folder = tmp_path / "batch_1"
    (batch_folder / method_folder).mkdir(parents=True)
    return batch_folder


def test_write_vina_pose_scores_file_keeps_every_pose(tmp_path):
    batch_folder = _make_batch(tmp_path, "vina")
    (batch_folder / "vina" / "6CTA-A_lig1.txt").write_text("0: -8.345\n1: -7.910\n2: -6.220\n")

    batch_dictionary = {
        "batch_folder": str(batch_folder),
        "combinations_table": pd.DataFrame([{"protein_config_id": "6CTA-A", "ligand_id": "lig1"}]),
    }

    poses_df = write_vina_pose_scores_file(batch_dictionary)

    assert len(poses_df) == 3
    assert list(poses_df[POSE]) == [0, 1, 2]
    assert poses_df[VINA_SCORE].tolist() == pytest.approx([-8.345, -7.910, -6.220])

    written = (batch_folder / "vina_scores.txt").read_text()
    assert "vina_score" in written
    assert written.count("\n") == 4  # header + 3 poses (+ trailing newline)


def test_write_gnina_pose_scores_file_keeps_every_pose_and_cnn_score(tmp_path):
    batch_folder = _make_batch(tmp_path, "gnina")
    (batch_folder / "gnina" / "6CTA-A_lig1.txt").write_text(
        "1: -8.345\t0.7891\n2: -7.910\t0.6512\n3: -6.220\t0.5001\n"
    )

    batch_dictionary = {
        "batch_folder": str(batch_folder),
        "combinations_table": pd.DataFrame([{"protein_config_id": "6CTA-A", "ligand_id": "lig1"}]),
    }

    poses_df = write_gnina_pose_scores_file(batch_dictionary)

    assert len(poses_df) == 3
    assert poses_df[GNINA_SCORE].tolist() == pytest.approx([-8.345, -7.910, -6.220])
    assert poses_df[GNINA_CNN_SCORE].tolist() == pytest.approx([0.7891, 0.6512, 0.5001])

    written = (batch_folder / "gnina_scores.txt").read_text()
    assert "gnina_cnn_score" in written


def test_missing_combination_score_file_is_skipped_not_fatal(tmp_path):
    """A combination with no per-combination score file on disk (e.g. that
    docking task failed) is silently skipped, not a hard error — same
    resilience as vina_guild_scoring/gnina_guild_scoring."""
    batch_folder = _make_batch(tmp_path, "vina")

    batch_dictionary = {
        "batch_folder": str(batch_folder),
        "combinations_table": pd.DataFrame(
            [{"protein_config_id": "6CTA-A", "ligand_id": "never_ran"}]
        ),
    }

    poses_df = write_vina_pose_scores_file(batch_dictionary)
    assert poses_df.empty
    assert list(poses_df.columns) == [
        "combination",
        "protein_config_id",
        "ligand_id",
        POSE,
        VINA_SCORE,
    ]


def test_pose_scores_file_covers_whole_batch_not_just_combinations_to_run(tmp_path):
    """The aggregate is built from the batch's full combinations table, not
    COMBINATIONS_TO_RUN_KEY — so it stays complete across resumed runs where
    only some combinations are newly scored this call."""
    batch_folder = _make_batch(tmp_path, "vina")
    (batch_folder / "vina" / "6CTA-A_lig1.txt").write_text("0: -8.0\n")
    (batch_folder / "vina" / "6CTA-A_lig2.txt").write_text("0: -7.0\n")

    batch_dictionary = {
        "batch_folder": str(batch_folder),
        # Both combinations are in the table even though only one might have
        # been part of "combinations_to_run" for this particular call.
        "combinations_table": pd.DataFrame(
            [
                {"protein_config_id": "6CTA-A", "ligand_id": "lig1"},
                {"protein_config_id": "6CTA-A", "ligand_id": "lig2"},
            ]
        ),
    }

    poses_df = write_vina_pose_scores_file(batch_dictionary)
    assert set(poses_df["ligand_id"]) == {"lig1", "lig2"}
