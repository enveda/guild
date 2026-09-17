"""
CLI wiring tests for ``scripts/run_guild.py``.

``resolve_pipeline_steps`` is a pure function and tested directly.
``main()`` is exercised with ``guild.bulk`` (and the other guild modules it
patches paths on) pre-seeded into ``sys.modules`` as stand-ins, so these
tests never need the real torch/rdkit/plip dependency stack — only the
routing from CLI flags to ``BulkRun`` method calls is under test here, and
that is the part this suite previously left completely uncovered (the
PoseBusters step existed only as a method reachable from the test suite,
never from the CLI).
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_guild  # noqa: E402


def _args(**overrides):
    defaults = {
        "plip_only": False,
        "posebusters_only": False,
        "no_plip": False,
        "no_posebusters": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# resolve_pipeline_steps — pure routing logic
# ---------------------------------------------------------------------------
class TestResolvePipelineSteps:
    def test_default_runs_both_analyses_after_docking(self):
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(_args())
        assert (skip, run_plip, run_posebusters) == (False, True, True)

    def test_no_plip_skips_only_plip(self):
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(_args(no_plip=True))
        assert (skip, run_plip, run_posebusters) == (False, False, True)

    def test_no_posebusters_skips_only_posebusters(self):
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(
            _args(no_posebusters=True)
        )
        assert (skip, run_plip, run_posebusters) == (False, True, False)

    def test_plip_only_skips_docking_and_runs_only_plip(self):
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(_args(plip_only=True))
        assert (skip, run_plip, run_posebusters) == (True, True, False)

    def test_posebusters_only_skips_docking_and_runs_only_posebusters(self):
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(
            _args(posebusters_only=True)
        )
        assert (skip, run_plip, run_posebusters) == (True, False, True)

    def test_both_only_flags_rerun_both_analyses_without_docking(self):
        """
        Each "-only" flag re-runs its own analysis; combining them re-runs both
        without re-docking, rather than one flag silently overriding the other.
        """
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(
            _args(plip_only=True, posebusters_only=True)
        )
        assert (skip, run_plip, run_posebusters) == (True, True, True)

    def test_no_posebusters_is_moot_once_plip_only_already_skips_it(self):
        skip, run_plip, run_posebusters = run_guild.resolve_pipeline_steps(
            _args(plip_only=True, no_posebusters=False)
        )
        assert run_posebusters is False


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------
class TestArgParsing:
    def test_posebusters_config_defaults_to_dock(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_guild.py", "-p", "x", "-c", "combos.csv"])
        args = run_guild.parse_args()
        assert args.posebusters_config == "dock"

    def test_posebusters_config_accepts_dock_fast(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_guild.py", "-p", "x", "-c", "combos.csv", "--posebusters-config", "dock_fast"],
        )
        args = run_guild.parse_args()
        assert args.posebusters_config == "dock_fast"

    def test_unrecognised_posebusters_config_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_guild.py", "-p", "x", "-c", "combos.csv", "--posebusters-config", "bogus"],
        )
        with pytest.raises(SystemExit):
            run_guild.parse_args()

    def test_posebusters_only_and_no_posebusters_default_false(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_guild.py", "-p", "x", "-c", "combos.csv"])
        args = run_guild.parse_args()
        assert args.posebusters is False
        assert args.no_posebusters is False
        assert args.posebusters_only is False

    def test_exclude_non_physical_defaults_false(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_guild.py", "-p", "x", "-c", "combos.csv"])
        args = run_guild.parse_args()
        assert args.exclude_non_physical is False


# ---------------------------------------------------------------------------
# main() — routes CLI flags to BulkRun calls
# ---------------------------------------------------------------------------
class TestMainWiresPosebusters:
    @pytest.fixture(autouse=True)
    def _restore_stderr(self):
        # main() unconditionally reassigns sys.stderr on its way out to silence
        # adlfs/fsspec finalizer noise; put the real one back afterwards.
        original = sys.stderr
        yield
        sys.stderr = original

    def _combos(self, tmp_path):
        combos_path = tmp_path / "combos.csv"
        pd.DataFrame({"protein_config_id": ["p1"], "ligand_id": ["l1"], "smiles": ["CCO"]}).to_csv(
            combos_path, index=False
        )
        return combos_path

    def _run_main(self, monkeypatch, tmp_path, argv_extra):
        combos_path = self._combos(tmp_path)

        fake_bulk_module = ModuleType("guild.bulk")
        mock_bulk_instance = MagicMock()
        mock_bulk_instance.rp_scores_df = None
        fake_bulk_module.BulkRun = MagicMock(return_value=mock_bulk_instance)

        monkeypatch.setitem(sys.modules, "guild.bulk", fake_bulk_module)
        monkeypatch.setitem(
            sys.modules, "guild.constants.system", ModuleType("guild.constants.system")
        )
        monkeypatch.setitem(sys.modules, "guild.run", ModuleType("guild.run"))
        monkeypatch.setitem(sys.modules, "guild.docking", ModuleType("guild.docking"))
        monkeypatch.setitem(
            sys.modules, "guild.docking.diffdock", ModuleType("guild.docking.diffdock")
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_guild.py", "-p", "testproj", "-c", str(combos_path), *argv_extra],
        )
        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

        run_guild.main()
        return mock_bulk_instance

    def test_posebusters_runs_by_default(self, monkeypatch, tmp_path):
        bulk = self._run_main(monkeypatch, tmp_path, [])
        bulk.run_pose_validity_analysis.assert_called_once_with(
            config="dock", expect_existing_scores=True
        )
        bulk.run_docking.assert_called_once()
        bulk.run_guild_scoring.assert_called_once_with(exclude_non_physical=False)
        bulk.run_interactions_analysis.assert_called_once()

    def test_exclude_non_physical_is_forwarded(self, monkeypatch, tmp_path):
        bulk = self._run_main(monkeypatch, tmp_path, ["--exclude-non-physical"])
        bulk.run_guild_scoring.assert_called_once_with(exclude_non_physical=True)

    def test_no_posebusters_skips_the_call(self, monkeypatch, tmp_path):
        bulk = self._run_main(monkeypatch, tmp_path, ["--no-posebusters"])
        bulk.run_pose_validity_analysis.assert_not_called()

    def test_posebusters_only_skips_docking_and_scoring_and_plip(self, monkeypatch, tmp_path):
        """
        --posebusters-only is the one case where a missing scores table is
        expected rather than a bug, so expect_existing_scores must flip False.
        """
        bulk = self._run_main(monkeypatch, tmp_path, ["--posebusters-only"])
        bulk.run_docking.assert_not_called()
        bulk.run_guild_scoring.assert_not_called()
        bulk.run_interactions_analysis.assert_not_called()
        bulk.run_pose_validity_analysis.assert_called_once_with(
            config="dock", expect_existing_scores=False
        )

    def test_posebusters_config_is_forwarded(self, monkeypatch, tmp_path):
        bulk = self._run_main(monkeypatch, tmp_path, ["--posebusters-config", "dock_fast"])
        bulk.run_pose_validity_analysis.assert_called_once_with(
            config="dock_fast", expect_existing_scores=True
        )

    def test_plip_only_does_not_run_posebusters(self, monkeypatch, tmp_path):
        bulk = self._run_main(monkeypatch, tmp_path, ["--plip-only"])
        bulk.run_pose_validity_analysis.assert_not_called()
        bulk.run_interactions_analysis.assert_called_once()
