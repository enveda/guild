"""
Guild-specific constants
"""

"""
Rank percentile scores columns
"""


PROTEIN_CONF_ID = "protein_config_id"
PROTEIN_ID = "protein_id"
SMILES = "smiles"
LIGAND = "ligand"
LIGAND_ID = "ligand_id"
LIGAND_CATEGORY = "ligand_category"
IS_PDB = "is_pdb"
PROTEIN_PATH = "protein_path"
PROTEIN_CHAIN = "protein_chain"
ORIGINAL_LIGAND = "original_ligand"
ORIGINAL_LIGAND_CHAIN = "original_ligand_chain"
BOX_LOCATION = "box_location"


"""
Methods constants
"""
VINA_PREFIX = "vina"
KARMADOCK_PREFIX = "karmadock"
DIFFDOCK_PREFIX = "diffdock"
BOLTZ_PREFIX = "boltz"
GNINA_PREFIX = "gnina"
# Vina re-scoring is now split by upstream pose source so a run that produces
# both Boltz and DiffDock complexes gets two distinct re-score columns.
VINA_RESCORE_BOLTZ_PREFIX = "vina_rescore_boltz"
VINA_RESCORE_DIFFDOCK_PREFIX = "vina_rescore_diffdock"
VINA_SCORE = f"{VINA_PREFIX}_score"
KARMADOCK_SCORE = f"{KARMADOCK_PREFIX}_score"
DIFFDOCK_SCORE = f"{DIFFDOCK_PREFIX}_score"
BOLTZ_SCORE = f"{BOLTZ_PREFIX}_score"
GNINA_SCORE = f"{GNINA_PREFIX}_score"
# CNNscore is a side-channel pose-confidence value emitted by gnina. It is
# saved alongside gnina_score for later analysis but does NOT participate in
# guild's rank-percentile aggregation (no entry in ALL_AVAILABLE_METHODS,
# SCORES_DICTIONARY, RANKS_DICTIONARY, or RP_SCORES_DICTIONARY).
GNINA_CNN_SCORE = f"{GNINA_PREFIX}_cnn_score"
VINA_RESCORE_BOLTZ_SCORE = f"{VINA_RESCORE_BOLTZ_PREFIX}_score"
VINA_RESCORE_DIFFDOCK_SCORE = f"{VINA_RESCORE_DIFFDOCK_PREFIX}_score"


SCORES_DICTIONARY = {
    VINA_PREFIX: VINA_SCORE,
    KARMADOCK_PREFIX: KARMADOCK_SCORE,
    DIFFDOCK_PREFIX: DIFFDOCK_SCORE,
    BOLTZ_PREFIX: BOLTZ_SCORE,
    GNINA_PREFIX: GNINA_SCORE,
    VINA_RESCORE_BOLTZ_PREFIX: VINA_RESCORE_BOLTZ_SCORE,
    VINA_RESCORE_DIFFDOCK_PREFIX: VINA_RESCORE_DIFFDOCK_SCORE,
}

ALL_AVAILABLE_METHODS = [
    VINA_PREFIX,
    KARMADOCK_PREFIX,
    DIFFDOCK_PREFIX,
    BOLTZ_PREFIX,
    GNINA_PREFIX,
]

RP_SCORES_COLUMNS = [
    PROTEIN_CONF_ID,
    SMILES,
    LIGAND,
    KARMADOCK_SCORE,
    DIFFDOCK_SCORE,
    VINA_SCORE,
    BOLTZ_SCORE,
    GNINA_SCORE,
]

"""
Folders constants
"""

PROTEINS_FOLDER = "proteins"
PLOTS_FOLDER = "plots"
BOXES_FOLDER = "boxes"
LIGANDS_FOLDER = "ligands"
DATA_FOLDER = "data"
VINA_FOLDER = VINA_PREFIX
KARMADOCK_FOLDER = KARMADOCK_PREFIX
DIFFDOCK_FOLDER = DIFFDOCK_PREFIX
BOLTZ_FOLDER = BOLTZ_PREFIX
GNINA_FOLDER = GNINA_PREFIX
VINA_RESCORE_BOLTZ_FOLDER = VINA_RESCORE_BOLTZ_PREFIX
VINA_RESCORE_DIFFDOCK_FOLDER = VINA_RESCORE_DIFFDOCK_PREFIX
MSA_FOLDER = "msa"
