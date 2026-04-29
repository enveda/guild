"""
P2Rank binding site prediction tools
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from guild.constants.p2rank import (
    P2RANK_CENTER_X,
    P2RANK_CENTER_Y,
    P2RANK_CENTER_Z,
    P2RANK_DEFAULT_POCKET_RANK,
    P2RANK_EXECUTABLE,
    P2RANK_MIN_PROBABILITY_THRESHOLD,
    P2RANK_OUTPUT_SUFFIX,
    P2RANK_POCKET_NAME,
    P2RANK_POCKET_PROBABILITY,
    P2RANK_POCKET_RANK,
    P2RANK_POCKET_SCORE,
    P2RANK_RESIDUE_IDS,
)

logger = logging.getLogger(__name__)


@dataclass
class P2RankPocket:
    """
    Data class representing a predicted binding pocket from P2Rank.
    """

    name: str
    rank: int
    score: float
    probability: float
    center_x: float
    center_y: float
    center_z: float
    residue_ids: str

    @property
    def center(self) -> Tuple[float, float, float]:
        """Return the pocket center as a tuple (x, y, z)."""
        return (self.center_x, self.center_y, self.center_z)


def run_p2rank(
    protein_pdb: str,
    output_dir: str,
    silent: bool = True,
) -> str:
    """
    Run P2Rank binding site prediction on a protein PDB file.

    :param protein_pdb: Path to the input protein PDB file
    :param output_dir: Directory to save P2Rank output files
    :param silent: If True, suppress P2Rank stdout/stderr
    :return: Path to the predictions CSV file
    :raises RuntimeError: If P2Rank execution fails
    :raises FileNotFoundError: If P2Rank executable or protein file not found
    """
    # Validate inputs
    if not os.path.exists(protein_pdb):
        raise FileNotFoundError(f"Protein PDB file not found: {protein_pdb}")

    if not os.path.exists(P2RANK_EXECUTABLE):
        raise FileNotFoundError(
            f"P2Rank executable not found at: {P2RANK_EXECUTABLE}. "
            "Please ensure P2Rank is installed in the workspace."
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build the P2Rank command
    # prank predict -f protein.pdb -o /path/to/output_dir
    command = f"{P2RANK_EXECUTABLE} predict -f {protein_pdb} -o {output_dir}"

    logger.info(f"Running P2Rank: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL if silent else None,
            stderr=subprocess.DEVNULL if silent else None,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"P2Rank failed with return code {result.returncode}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"P2Rank execution failed: {e}") from e

    # Determine the output predictions file path
    # When using `-f`, P2Rank places output directly in the output dir as:
    #   <filename_with_extension>_predictions.csv  (e.g., 1fbl.pdb_predictions.csv)
    protein_filename = Path(protein_pdb).name  # e.g. "1fbl.pdb"
    predictions_file = Path(output_dir) / f"{protein_filename}{P2RANK_OUTPUT_SUFFIX}"

    if not predictions_file.exists():
        # Fallback: some P2Rank invocations create a predict_<stem>/ subdirectory
        protein_stem = Path(protein_pdb).stem
        predictions_file_alt = (
            Path(output_dir)
            / f"predict_{protein_stem}"
            / f"{protein_filename}{P2RANK_OUTPUT_SUFFIX}"
        )
        if predictions_file_alt.exists():
            predictions_file = predictions_file_alt
        else:
            raise FileNotFoundError(
                f"P2Rank predictions file not found. Checked:\n"
                f"  {predictions_file}\n"
                f"  {predictions_file_alt}"
            )

    logger.info(f"P2Rank predictions saved to: {predictions_file}")
    return str(predictions_file)


def parse_p2rank_predictions(predictions_csv: str) -> pd.DataFrame:
    """
    Parse the P2Rank predictions CSV file.

    :param predictions_csv: Path to the P2Rank predictions CSV file
    :return: DataFrame containing parsed pocket predictions
    :raises FileNotFoundError: If predictions file not found
    :raises ValueError: If predictions file is empty or malformed
    """
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"P2Rank predictions file not found: {predictions_csv}")

    # P2Rank CSV uses spaces around commas, we need to handle this
    df = pd.read_csv(predictions_csv, skipinitialspace=True)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError(f"P2Rank predictions file is empty: {predictions_csv}")

    # Validate required columns exist
    required_columns = [P2RANK_CENTER_X, P2RANK_CENTER_Y, P2RANK_CENTER_Z, P2RANK_POCKET_RANK]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"P2Rank predictions missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Parsed {len(df)} pocket predictions from P2Rank")
    return df


def get_top_pocket(
    predictions_df: pd.DataFrame,
    pocket_rank: int = P2RANK_DEFAULT_POCKET_RANK,
    min_probability: float = P2RANK_MIN_PROBABILITY_THRESHOLD,
) -> Optional[P2RankPocket]:
    """
    Get a specific pocket from P2Rank predictions by rank.

    :param predictions_df: DataFrame from parse_p2rank_predictions
    :param pocket_rank: Rank of the pocket to retrieve (1 = top-ranked)
    :param min_probability: Minimum probability threshold for valid pocket
    :return: P2RankPocket object or None if no valid pocket found
    """
    # Filter by rank
    pocket_row = predictions_df[predictions_df[P2RANK_POCKET_RANK] == pocket_rank]

    if pocket_row.empty:
        logger.warning(f"No pocket found with rank {pocket_rank}")
        return None

    row = pocket_row.iloc[0]

    # Check probability threshold
    probability = float(row[P2RANK_POCKET_PROBABILITY])
    if probability < min_probability:
        logger.warning(
            f"Pocket {pocket_rank} probability ({probability:.3f}) is below "
            f"threshold ({min_probability}). Consider using original ligand location."
        )

    pocket = P2RankPocket(
        name=str(row[P2RANK_POCKET_NAME]).strip(),
        rank=int(row[P2RANK_POCKET_RANK]),
        score=float(row[P2RANK_POCKET_SCORE]),
        probability=probability,
        center_x=float(row[P2RANK_CENTER_X]),
        center_y=float(row[P2RANK_CENTER_Y]),
        center_z=float(row[P2RANK_CENTER_Z]),
        residue_ids=str(row[P2RANK_RESIDUE_IDS]).strip(),
    )

    logger.info(
        f"Selected P2Rank pocket: {pocket.name} with probability {pocket.probability:.3f}, "
        f"center: ({pocket.center_x:.3f}, {pocket.center_y:.3f}, {pocket.center_z:.3f})"
    )

    return pocket


def get_binding_site_center_from_p2rank(
    protein_pdb: str,
    output_dir: str,
    pocket_rank: int = P2RANK_DEFAULT_POCKET_RANK,
    min_probability: float = P2RANK_MIN_PROBABILITY_THRESHOLD,
) -> Optional[Tuple[float, float, float]]:
    """
    Run P2Rank on a protein and return the binding site center coordinates.

    This is the main entry point for integrating P2Rank with the docking workflow.
    It runs P2Rank, parses the output, and returns the center coordinates of the
    specified pocket rank.

    :param protein_pdb: Path to the input protein PDB file
    :param output_dir: Directory to save P2Rank output files
    :param pocket_rank: Rank of the pocket to use (1 = top-ranked)
    :param min_probability: Minimum probability threshold for valid pocket
    :return: Tuple of (center_x, center_y, center_z) or None if no valid pocket
    :raises RuntimeError: If P2Rank execution fails
    :raises FileNotFoundError: If required files not found
    """
    # Run P2Rank prediction
    predictions_csv = run_p2rank(protein_pdb, output_dir)

    # Parse the predictions
    predictions_df = parse_p2rank_predictions(predictions_csv)

    # Get the requested pocket
    pocket = get_top_pocket(predictions_df, pocket_rank, min_probability)

    if pocket is None:
        return None

    return pocket.center
