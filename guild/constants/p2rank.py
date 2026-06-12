"""
P2Rank binding site prediction constants
"""

from pathlib import Path

import guild as _guild_pkg

"""
P2Rank installation path — sibling of the guild/ package on disk (so
guild-internal/p2rank_2.4.2 in dev, /app/p2rank_2.4.2 in the container).
Resolving from the guild package location rather than WORKING_DIR_PATH is
deliberate: run_guild.py rebinds WORKING_DIR_PATH to the caller's workspace
for output paths, which would otherwise look for p2rank under the caller's
repo. The guild package's own location is the stable anchor.
"""
_GUILD_PKG_DIR = Path(_guild_pkg.__file__).resolve().parent
P2RANK_INSTALL_DIR = _GUILD_PKG_DIR.parent / "p2rank_2.4.2"
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
