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
COVALENT_REC_ATOM = "covalent_rec_atom"
COVALENT_LIG_SMARTS = "covalent_lig_smarts"
GNINA_FLEXRES = "gnina_flexres"
# Per-pose index column in the batch-level all-poses score files
# (VINA_SCORES_FILE / GNINA_SCORES_FILE in guild/constants/bulk.py).
POSE = "pose"


"""
Methods constants
"""
VINA_PREFIX = "vina"
KARMADOCK_PREFIX = "karmadock"
DIFFDOCK_PREFIX = "diffdock"
BOLTZ_PREFIX = "boltz"
GNINA_PREFIX = "gnina"
NESSO_PREFIX = "nesso"
# Vina re-scoring is now split by upstream pose source so a run that produces
# both Boltz and DiffDock complexes gets two distinct re-score columns.
VINA_RESCORE_BOLTZ_PREFIX = "vina_rescore_boltz"
VINA_RESCORE_DIFFDOCK_PREFIX = "vina_rescore_diffdock"
# gnina re-scoring mirrors the Vina rescore tracks above — same split by
# upstream pose source, additive alongside (not a replacement for) the Vina
# rescore columns.
GNINA_RESCORE_BOLTZ_PREFIX = "gnina_rescore_boltz"
GNINA_RESCORE_DIFFDOCK_PREFIX = "gnina_rescore_diffdock"
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
GNINA_RESCORE_BOLTZ_SCORE = f"{GNINA_RESCORE_BOLTZ_PREFIX}_score"
GNINA_RESCORE_DIFFDOCK_SCORE = f"{GNINA_RESCORE_DIFFDOCK_PREFIX}_score"
# Side-channel CNN confidence, same status as GNINA_CNN_SCORE above: rides
# along for analysis but is not registered in RANKS_DICTIONARY/RP_SCORES_DICTIONARY.
GNINA_RESCORE_BOLTZ_CNN_SCORE = f"{GNINA_RESCORE_BOLTZ_PREFIX}_cnn_score"
GNINA_RESCORE_DIFFDOCK_CNN_SCORE = f"{GNINA_RESCORE_DIFFDOCK_PREFIX}_cnn_score"
# Nesso-1's affinity head is a real predicted potency — log10(IC50/uM), lower
# is more potent — unlike boltz_score (ipTM confidence) or the docking ΔG
# scores. It is therefore guild's first "minimum" direction that is also a
# potency rather than an energy; see SCORES_DIRECTION_DICTIONARY in
# guild/constants/bulk.py.
NESSO_SCORE = f"{NESSO_PREFIX}_score"
# Side channels from Nesso's affinity.json: the Hit-ID binder-probability
# head and the protein-ligand interface entropy (the paper's H_PL, a
# confidence gate). Saved alongside nesso_score but, like GNINA_CNN_SCORE,
# deliberately absent from ALL_AVAILABLE_METHODS / SCORES_DICTIONARY /
# RANKS_DICTIONARY / RP_SCORES_DICTIONARY.
NESSO_BINDER_PROBABILITY = f"{NESSO_PREFIX}_binder_probability"
NESSO_ENTROPY_PL = f"{NESSO_PREFIX}_entropy_pl"
# Boltz-2's own affinity head (log10(IC50/uM), lower = more potent). Unlike
# boltz_score (an ipTM confidence), this is a genuine affinity estimate, so
# it's ranked and voted into global_rp_score under BOLTZ_AFFINITY_PREFIX,
# grouped into Boltz's own pose source (see guild/constants/bulk.py). NaN for
# a run predating this column.
BOLTZ_AFFINITY_SCORE = f"{BOLTZ_PREFIX}_affinity_score"
# Prefix key for the machinery above — not a docking method itself, no entry
# in method_runners, no docking step of its own.
BOLTZ_AFFINITY_PREFIX = "boltz_affinity"


SCORES_DICTIONARY = {
    VINA_PREFIX: VINA_SCORE,
    KARMADOCK_PREFIX: KARMADOCK_SCORE,
    DIFFDOCK_PREFIX: DIFFDOCK_SCORE,
    BOLTZ_PREFIX: BOLTZ_SCORE,
    GNINA_PREFIX: GNINA_SCORE,
    NESSO_PREFIX: NESSO_SCORE,
    VINA_RESCORE_BOLTZ_PREFIX: VINA_RESCORE_BOLTZ_SCORE,
    VINA_RESCORE_DIFFDOCK_PREFIX: VINA_RESCORE_DIFFDOCK_SCORE,
    GNINA_RESCORE_BOLTZ_PREFIX: GNINA_RESCORE_BOLTZ_SCORE,
    GNINA_RESCORE_DIFFDOCK_PREFIX: GNINA_RESCORE_DIFFDOCK_SCORE,
}

ALL_AVAILABLE_METHODS = [
    VINA_PREFIX,
    KARMADOCK_PREFIX,
    DIFFDOCK_PREFIX,
    BOLTZ_PREFIX,
    GNINA_PREFIX,
    NESSO_PREFIX,
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
    NESSO_SCORE,
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
NESSO_FOLDER = NESSO_PREFIX
VINA_RESCORE_BOLTZ_FOLDER = VINA_RESCORE_BOLTZ_PREFIX
VINA_RESCORE_DIFFDOCK_FOLDER = VINA_RESCORE_DIFFDOCK_PREFIX
GNINA_RESCORE_BOLTZ_FOLDER = GNINA_RESCORE_BOLTZ_PREFIX
GNINA_RESCORE_DIFFDOCK_FOLDER = GNINA_RESCORE_DIFFDOCK_PREFIX
MSA_FOLDER = "msa"
