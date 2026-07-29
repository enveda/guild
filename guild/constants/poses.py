"""
Pose-mode constants shared across docking engines (Vina, gnina, …).

See README.md §"Starting from a user-supplied pose" for full semantics.

    "dock"  — global stochastic search; supplied pose has no effect on
              search-start coordinates (used only for ligand topology).
              Default for backwards compatibility.
    "local" — engine-specific local refinement of the supplied pose
              (Vina.optimize() / gnina --local_only). The mode that
              actually honours an experimentally-informed starting pose.
    "score" — score the supplied pose as-is, no movement
              (Vina.score() / gnina --score_only).
"""

POSE_MODE_DOCK = "dock"
POSE_MODE_LOCAL = "local"
POSE_MODE_SCORE = "score"
POSE_MODES = (POSE_MODE_DOCK, POSE_MODE_LOCAL, POSE_MODE_SCORE)
DEFAULT_POSE_MODE = POSE_MODE_DOCK

# File extension expected for user-supplied pose files (one per ligand_id).
POSE_FILE_EXTENSION = ".sdf"
