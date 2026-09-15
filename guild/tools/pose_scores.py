"""Batch-level per-pose score tables.

``guild_scores.txt`` keeps only the single best pose per combination. These
files keep every pose the engine produced, so the full distribution does not
require re-opening each per-combination score file.
"""

import logging
from collections.abc import Callable

import pandas as pd

from guild.constants.bulk import BATCH_FOLDER, COMBINATION_ID, COMBINATIONS_TABLE_KEY
from guild.constants.guild import LIGAND_ID, POSE, PROTEIN_CONF_ID

logger = logging.getLogger(__name__)


def write_pose_scores_file(
    batch_dictionary,
    *,
    method_folder: str,
    output_file: str,
    score_columns: list[str],
    read_pose_scores: Callable[[str], pd.DataFrame],
    method_label: str,
) -> pd.DataFrame:
    """Aggregate every pose's scores across a batch into ``{batch_folder}/{output_file}``.

    Iterates the batch's full combinations table rather than
    ``COMBINATIONS_TO_RUN_KEY``, so the file stays complete across resumed
    runs: the per-combination score files persist on disk regardless of which
    combinations were newly run this call.

    :param method_folder: Per-method subfolder holding the score files.
    :param score_columns: Score columns ``read_pose_scores`` yields, appended
        after the identity columns.
    :param read_pose_scores: Reads one score file into a frame of
        ``[POSE, *score_columns]``; may raise, which is logged and skipped.
    :param method_label: Engine name, for the skip log line.
    :return: One row per pose, also written as CSV.
    """
    combinations = batch_dictionary[COMBINATIONS_TABLE_KEY][
        [PROTEIN_CONF_ID, LIGAND_ID]
    ].drop_duplicates()

    pose_frames = []
    for _, row in combinations.iterrows():
        protein_conf_id, ligand_id = row[PROTEIN_CONF_ID], row[LIGAND_ID]
        score_file = (
            f"{batch_dictionary[BATCH_FOLDER]}/{method_folder}/"
            f"{protein_conf_id}_{ligand_id}.txt"
        )
        try:
            poses_df = read_pose_scores(score_file)
        except Exception as e:
            logger.info(f"No {method_label} pose scores for {(protein_conf_id, ligand_id)}: {e}")
            continue
        if poses_df.empty:
            continue
        poses_df[COMBINATION_ID] = f"{protein_conf_id}_{ligand_id}"
        poses_df[PROTEIN_CONF_ID] = protein_conf_id
        poses_df[LIGAND_ID] = ligand_id
        pose_frames.append(poses_df)

    columns = [COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID, POSE, *score_columns]
    poses_scores_df = (
        pd.concat(pose_frames, ignore_index=True)[columns]
        if pose_frames
        else pd.DataFrame(columns=columns)
    )

    poses_scores_df.to_csv(f"{batch_dictionary[BATCH_FOLDER]}/{output_file}", index=False)
    return poses_scores_df
