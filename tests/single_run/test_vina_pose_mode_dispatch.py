"""
Tests that ``Guild.run_autodock_vina`` calls each pose-mode runner with kwargs
that runner actually accepts.

Only ``deploy_vina`` takes a flex receptor. The local and score runners do not,
so passing ``flex_pdbqt`` to them raises TypeError -- which
``run_autodock_vina`` catches and reports as a generic "error in docking",
turning every pose-guided Vina run into a silent failure. The runners are
patched with ``create_autospec`` so their real signatures are enforced and that
regression reappears as a test failure rather than a log line.

``Guild.__init__`` is bypassed deliberately: it builds project directories and
runs ligand prep, none of which this dispatch logic depends on.
"""

from unittest.mock import create_autospec, patch

import pytest

from guild.constants.poses import POSE_MODE_DOCK, POSE_MODE_LOCAL, POSE_MODE_SCORE
from guild.docking.vina import (
    deploy_vina,
    deploy_vina_local_refinement,
    deploy_vina_score,
)
from guild.run import Guild

BOX_CENTER = (1.0, 2.0, 3.0)
BOX_SIZE = (20.0, 20.0, 20.0)


def _make_guild(tmp_path, pose_mode, flexible_docking):
    """A Guild carrying only the attributes run_autodock_vina reads."""
    guild = object.__new__(Guild)
    guild.output_log_file = str(tmp_path / "time.log")
    guild.pose_mode = pose_mode
    guild.flexible_docking = flexible_docking
    guild.vina_exhaustiveness = None
    guild.protein_chains = ["A"]
    guild.cleaned_protein = str(tmp_path / "protein_cleaned.pdb")
    guild.cleaned_protein_pdbqt = str(tmp_path / "protein.pdbqt")
    guild.cleaned_protein_rigid_pdbqt = str(tmp_path / "protein_rigid.pdbqt")
    guild.cleaned_protein_flex_pdbqt = str(tmp_path / "protein_flex.pdbqt")
    guild.ligand_pdbqt = str(tmp_path / "ligand.pdbqt")
    guild.vina_box = str(tmp_path / "box.txt")
    guild.vina_output_pdbqt = str(tmp_path / "out.pdbqt")
    guild.vina_output_scores = str(tmp_path / "out.txt")
    return guild


def _run(guild, flexres_str="A:88"):
    """Invoke run_autodock_vina with every external call stubbed out."""
    runners = {
        POSE_MODE_DOCK: create_autospec(deploy_vina),
        POSE_MODE_LOCAL: create_autospec(deploy_vina_local_refinement),
        POSE_MODE_SCORE: create_autospec(deploy_vina_score),
    }
    with (
        patch("guild.run.protein_pdb_to_pdbqt"),
        patch("guild.run.prepare_flex_receptor_pdbqt"),
        patch("guild.run.get_center_and_size_from_box_file", return_value=(BOX_CENTER, BOX_SIZE)),
        patch("guild.run.get_flexres_string_from_box", return_value=flexres_str),
        patch("guild.run.os.path.exists", return_value=False),
        patch("guild.run.deploy_vina", runners[POSE_MODE_DOCK]),
        patch("guild.run.deploy_vina_local_refinement", runners[POSE_MODE_LOCAL]),
        patch("guild.run.deploy_vina_score", runners[POSE_MODE_SCORE]),
    ):
        result = guild.run_autodock_vina()
    return result, runners


@pytest.mark.parametrize("pose_mode", [POSE_MODE_LOCAL, POSE_MODE_SCORE])
@pytest.mark.parametrize("flexible_docking", [False, True])
def test_pose_guided_runners_are_called_successfully(tmp_path, pose_mode, flexible_docking):
    """
    The local/score runners must be reached with an acceptable signature whether
    or not flexible docking was also requested. Before the fix this failed for
    both values of flexible_docking, because flex_pdbqt was passed
    unconditionally.
    """
    guild = _make_guild(tmp_path, pose_mode, flexible_docking)

    result, runners = _run(guild)

    assert result == 0, "run_autodock_vina reported a docking failure"
    runner = runners[pose_mode]
    runner.assert_called_once()
    assert "flex_pdbqt" not in runner.call_args.kwargs


@pytest.mark.parametrize("pose_mode", [POSE_MODE_LOCAL, POSE_MODE_SCORE])
def test_flexible_docking_keeps_the_whole_receptor_in_pose_modes(tmp_path, pose_mode):
    """
    Requesting flexible docking alongside local/score must not hand the runner
    the rigid half of a split receptor -- the flexible side chains would simply
    be missing from the structure being docked against.
    """
    guild = _make_guild(tmp_path, pose_mode, flexible_docking=True)

    _, runners = _run(guild)

    receptor = runners[pose_mode].call_args.kwargs["receptor_pdbqt"]
    assert receptor == guild.cleaned_protein_pdbqt
    assert receptor != guild.cleaned_protein_rigid_pdbqt


def test_dock_mode_still_receives_the_flex_receptor(tmp_path):
    """The flex path itself must keep working for the mode that supports it."""
    guild = _make_guild(tmp_path, POSE_MODE_DOCK, flexible_docking=True)

    result, runners = _run(guild)

    assert result == 0
    kwargs = runners[POSE_MODE_DOCK].call_args.kwargs
    assert kwargs["flex_pdbqt"] == guild.cleaned_protein_flex_pdbqt
    assert kwargs["receptor_pdbqt"] == guild.cleaned_protein_rigid_pdbqt


def test_dock_mode_without_flexible_docking_passes_no_flex_receptor(tmp_path):
    guild = _make_guild(tmp_path, POSE_MODE_DOCK, flexible_docking=False)

    result, runners = _run(guild)

    assert result == 0
    assert runners[POSE_MODE_DOCK].call_args.kwargs["flex_pdbqt"] is None
