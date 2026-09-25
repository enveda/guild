"""
DiffDock constants
"""

"""
Folders
"""
DIFFDOCK_RESULTS_FOLDER = "results"
DIFFDOCK_DIRECTORY = "DiffDock"

"""
Files
"""
DIFFDOCK_COMBINATIONS_FILE = "diffdock_combinations.csv"
DIFFDOCK_ARGS_FILE = "diffdock_model_args.yaml"

"""
Columns
"""
PROTEIN_PATH = "protein_path"
COMPLEX_NAME = "complex_name"
LIGAND_DESCRIPTION = "ligand_description"
PROTEIN_SEQUENCE = "protein_sequence"

"""
Pocket restriction
"""
# auto: restrict to the pocket box when one exists, otherwise blind.
# box: a pocket box is required; no box or no sample inside it is a failure.
# blind: ignore any box and keep DiffDock's top-confidence sample.
DIFFDOCK_POCKET_AUTO = "auto"
DIFFDOCK_POCKET_BOX = "box"
DIFFDOCK_POCKET_BLIND = "blind"
DIFFDOCK_POCKET_MODES = (DIFFDOCK_POCKET_AUTO, DIFFDOCK_POCKET_BOX, DIFFDOCK_POCKET_BLIND)
DEFAULT_DIFFDOCK_POCKET = DIFFDOCK_POCKET_AUTO

# Batch-dictionary key carrying the pocket mode to the scoring/complex helpers.
DIFFDOCK_POCKET_KEY = "diffdock_pocket"

# Per-combination record of which sample was chosen and why, so the score,
# both rescores, the complex PDB and PoseBusters all describe the same pose.
SELECTED_POSE_FILE = "selected_pose.json"

# Outcomes written to the diffdock_pose_selection column.
POSE_SELECTED_BOX = "box"
POSE_SELECTED_BLIND = "blind"
POSE_NO_SAMPLE_IN_BOX = "no_sample_in_box"
POSE_NO_BOX = "no_box"
POSE_NO_SAMPLES = "no_samples"

# A rescored pose with no receptor heavy atom within this distance (Å) is not
# bound to the receptor; its score is reported as NaN rather than ~0.
RECEPTOR_CONTACT_CUTOFF = 4.0

# Side-channel column in the scores table; not a score, never ranked.
DIFFDOCK_POSE_SELECTION = "diffdock_pose_selection"

# Prepared receptor written by Guild prep (receptor chain(s), altlocs
# collapsed, Vina/box frame): {proteins}/{protein_conf_id}{suffix}.pdb/.pdbqt
CLEAN_RECEPTOR_SUFFIX = "_single_chain_clean"
