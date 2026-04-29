"""
P2Rank binding site prediction constants
"""

from pathlib import Path

from guild.constants.system import WORKING_DIR_PATH

"""
P2Rank installation path - assumes p2rank is installed in the workspace
"""
P2RANK_INSTALL_DIR = Path(WORKING_DIR_PATH) / "p2rank_2.4.2"
P2RANK_EXECUTABLE = P2RANK_INSTALL_DIR / "prank"

"""
P2Rank output folder names
"""
P2RANK_FOLDER = "p2rank"
P2RANK_OUTPUT_SUFFIX = "_predictions.csv"
P2RANK_VISUALIZATIONS_SUFFIX = "_predictions"

"""
P2Rank prediction output column names
"""
P2RANK_POCKET_NAME = "name"
P2RANK_POCKET_RANK = "rank"
P2RANK_POCKET_SCORE = "score"
P2RANK_POCKET_PROBABILITY = "probability"
P2RANK_CENTER_X = "center_x"
P2RANK_CENTER_Y = "center_y"
P2RANK_CENTER_Z = "center_z"
P2RANK_RESIDUE_IDS = "residue_ids"

"""
Default P2Rank settings
"""
P2RANK_DEFAULT_POCKET_RANK = 1  # Use the top-ranked pocket by default
P2RANK_MIN_PROBABILITY_THRESHOLD = 0.1  # Minimum probability threshold for valid pocket
