"""
Tools for user-supplied pose validation and resolution.
"""

import os

from guild.constants.guild import LIGAND_ID
from guild.constants.poses import POSE_FILE_EXTENSION

_MAX_MISSING_IDS_LISTED = 20


def _resolve_pose_path(poses_dir, ligand_id):
    """
    Return the path to a staged pose file for ``ligand_id`` inside
    ``poses_dir``, or ``None`` when no directory has been supplied.

    Does not raise or warn when ``poses_dir`` is set but the file is
    missing — that case is surfaced by ``_validate_poses_dir`` so the
    failure happens once, upfront, with the full list of missing IDs.
    """
    if poses_dir is None:
        return None
    candidate = os.path.join(poses_dir, f"{ligand_id}{POSE_FILE_EXTENSION}")
    return candidate if os.path.isfile(candidate) else None


def _validate_poses_dir(poses_dir, combinations_table):
    """
    Pre-flight check that every ``ligand_id`` in ``combinations_table``
    resolves to a pose file under ``poses_dir``.

    Raises ``FileNotFoundError`` with a single message listing all
    missing ligand IDs (capped at ``_MAX_MISSING_IDS_LISTED`` with a
    ``(+N more)`` suffix when truncated) so the user can fix the
    directory in one round-trip. Called before any batch folder /
    project subdirectory creation so the partial-state cleanup story is
    "no state created."
    """
    if poses_dir is None:
        return
    if not os.path.isdir(poses_dir):
        raise FileNotFoundError(
            f"POSES_DIR={poses_dir} does not exist or is not a directory. "
            "Aborting before docking — supply a valid directory of "
            f"<ligand_id>{POSE_FILE_EXTENSION} pose files or unset POSES_DIR."
        )

    unique_ligand_ids = combinations_table[LIGAND_ID].dropna().unique().tolist()
    missing = [
        str(lig_id) for lig_id in unique_ligand_ids if _resolve_pose_path(poses_dir, lig_id) is None
    ]
    if not missing:
        return

    listed = missing[:_MAX_MISSING_IDS_LISTED]
    suffix = (
        f" (+{len(missing) - _MAX_MISSING_IDS_LISTED} more)"
        if len(missing) > _MAX_MISSING_IDS_LISTED
        else ""
    )
    raise FileNotFoundError(
        f"POSES_DIR={poses_dir} is missing pose files for "
        f"{len(missing)} ligand_id(s) (expected <ligand_id>"
        f"{POSE_FILE_EXTENSION}): "
        + ", ".join(listed)
        + suffix
        + ". Aborting before docking — add the missing files or unset "
        "POSES_DIR to use SMILES→3D for all ligands."
    )
