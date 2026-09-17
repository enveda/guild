"""
Non-physical raw score visibility.

A positive Vina-family energy (or one with an absurd magnitude) is never
nulled or clamped by default -- the published case-study numbers were
generated with those values left in the table, so changing them
retroactively would change results nobody asked to re-derive. These tests
cover the two things run_guild_scoring does instead: log the count
(_log_score_physicality_summary) and, only when explicitly asked
(exclude_non_physical=True), null them out (_null_non_physical_scores).
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from guild.bulk import BulkRun
from guild.constants.guild import GNINA_PREFIX, VINA_PREFIX

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
    test_project = Path.cwd() / "data" / "test-score-physicality"
    if test_project.exists():
        shutil.rmtree(test_project, ignore_errors=True)


def _bulk(test_input_table):
    return BulkRun(
        input_table=test_input_table,
        project_name="test-score-physicality",
        methods_to_run=[VINA_PREFIX, GNINA_PREFIX],
        use_decoys=False,
        use_known_binders=False,
        use_gpu=False,
        n_workers=1,
    )


class TestLogScorePhysicalitySummary:
    def test_non_physical_scores_are_logged(self, test_input_table, cleanup, caplog):
        bulk = _bulk(test_input_table)
        raw_scores = pd.DataFrame(
            {"vina_score": [-9.0, 5.0, -43_851_078, -1.5], "gnina_score": [-8.0, -7.0, -6.0, -5.0]}
        )
        with caplog.at_level("WARNING", logger="guild.bulk"):
            bulk._log_score_physicality_summary(raw_scores)

        assert "vina: 2/4" in caplog.text
        assert "gnina:" not in caplog.text  # all-physical method logs nothing

    def test_all_physical_scores_are_silent(self, test_input_table, cleanup, caplog):
        bulk = _bulk(test_input_table)
        raw_scores = pd.DataFrame({"vina_score": [-9.0, -8.5, -1.0]})
        with caplog.at_level("WARNING", logger="guild.bulk"):
            bulk._log_score_physicality_summary(raw_scores)
        assert caplog.text == ""

    def test_missing_scores_are_not_counted_against_physicality(self, test_input_table, cleanup, caplog):
        bulk = _bulk(test_input_table)
        raw_scores = pd.DataFrame({"vina_score": [-9.0, None, None]})
        with caplog.at_level("WARNING", logger="guild.bulk"):
            bulk._log_score_physicality_summary(raw_scores)
        assert caplog.text == ""


class TestNullNonPhysicalScores:
    def test_non_physical_values_are_nulled(self, test_input_table, cleanup):
        bulk = _bulk(test_input_table)
        raw_scores = pd.DataFrame({"vina_score": [-9.0, 5.0, -43_851_078, -1.5]})
        result = bulk._null_non_physical_scores(raw_scores)
        assert result["vina_score"].tolist()[0] == -9.0
        assert pd.isna(result["vina_score"].iloc[1])
        assert pd.isna(result["vina_score"].iloc[2])
        assert result["vina_score"].tolist()[3] == -1.5

    def test_maximising_method_scores_are_untouched(self, test_input_table, cleanup):
        """karmadock_score is 'maximum'-direction — nothing here is non-physical."""
        bulk = BulkRun(
            input_table=test_input_table,
            project_name="test-score-physicality",
            methods_to_run=["karmadock"],
            use_decoys=False,
            use_known_binders=False,
            use_gpu=False,
            n_workers=1,
        )
        raw_scores = pd.DataFrame({"karmadock_score": [0.9, 5.0, 1_000_000.0]})
        result = bulk._null_non_physical_scores(raw_scores)
        assert result["karmadock_score"].tolist() == [0.9, 5.0, 1_000_000.0]

    def test_original_frame_is_not_mutated(self, test_input_table, cleanup):
        bulk = _bulk(test_input_table)
        raw_scores = pd.DataFrame({"vina_score": [5.0]})
        bulk._null_non_physical_scores(raw_scores)
        assert raw_scores["vina_score"].iloc[0] == 5.0

    def test_disabled_by_default(self):
        """
        run_guild_scoring must not null anything unless exclude_non_physical
        is explicitly passed — the default preserves published case-study
        numbers exactly as scored.
        """
        import inspect

        signature = inspect.signature(BulkRun.run_guild_scoring)
        assert signature.parameters["exclude_non_physical"].default is False
