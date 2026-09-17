from guild.constants.guild import (
    BOLTZ_AFFINITY_PREFIX,
    BOLTZ_AFFINITY_SCORE,
    BOLTZ_PREFIX,
    BOLTZ_SCORE,
    DIFFDOCK_PREFIX,
    DIFFDOCK_SCORE,
    GNINA_PREFIX,
    GNINA_RESCORE_BOLTZ_PREFIX,
    GNINA_RESCORE_BOLTZ_SCORE,
    GNINA_RESCORE_DIFFDOCK_PREFIX,
    GNINA_RESCORE_DIFFDOCK_SCORE,
    GNINA_SCORE,
    KARMADOCK_PREFIX,
    KARMADOCK_SCORE,
    NESSO_PREFIX,
    NESSO_SCORE,
    VINA_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_BOLTZ_SCORE,
    VINA_RESCORE_DIFFDOCK_PREFIX,
    VINA_RESCORE_DIFFDOCK_SCORE,
    VINA_SCORE,
)

"""
Files
"""
RP_SCORES_FILE = "guild_scores.txt"
GUILD_COMBINATIONS_FILE = "combinations.csv"
OUTPUT_LOG_FILE = "output.log"
BATCH_PROGRESS_LOG_FILE = "batch_progress.log"
ALL_COMBINATIONS_FILE = "all_combinations.csv"
KNOWN_BINDERS_FILE = "known_binders.csv"
# Batch-level all-poses score distributions (one row per pose per
# combination) — unlike RP_SCORES_FILE / guild_scores.txt, which keeps only
# the single best pose per combination, these keep every pose Vina/gnina
# generated, so the full distribution doesn't require re-parsing every
# per-combination score file under batches/<batch>/vina/ or /gnina/.
VINA_SCORES_FILE = "vina_scores.txt"
GNINA_SCORES_FILE = "gnina_scores.txt"


"""
Folders
"""
BATCHES_FOLDER = "batches"

"""
Bulk constants
"""
# Timeout constants (in seconds)
PREPROCESSING_TIMEOUT = 300  # 5 minutes for molecule preprocessing
DOCKING_TIMEOUT = 360  # 6 minutes for docking tasks

BATCH_FOLDER = "batch_folder"
COMBINATIONS_TABLE_KEY = "combinations_table"
PROTEINS_FOLDER_KEY = "proteins_folder"
COMBINATIONS_TO_RUN_KEY = "combinations_to_run"
UNIQUE_PROTEIN_IDS_KEY = "unique_protein_ids"
INPUT_COMBINATIONS_KEY = "input_combinations"
SMILES_NAMES_DICTIONARY_KEY = "smiles_names_dictionary"
SMILES_TYPE_DICTIONARY_KEY = "smiles_type_dictionary"
PRE_EXISTING_RP_SCORES_KEY = "pre_existing_rp_scores"
RP_SCORES_DF_KEY = "rp_scores_df"
COMBINATIONS_TO_RUN_KEY = "combinations_to_run"
METHODS_TO_SCORE_DICTIONARY_KEY = "methods_to_score_dictionary"
METHODS_TO_SORT_DICTIONARY_KEY = "methods_to_sort_dictionary"
PREVIOUS_RP_SCORES_KEY = "previous_rp_scores"
PROTEIN_SEQUENCE_KEY = "protein_sequence"
DECOYS_COMBINATIONS_KEY = "decoys_combinations"
PREVIOUS_COMBINATIONS_DF_KEY = "previous_combinations_df"
SCORES_TO_USE_KEY = "scores_to_use"
RANKS_LIST_KEY = "ranks_list"
BULK_TEMPLATE_DICTIONARY = {
    BATCH_FOLDER: None,
    COMBINATIONS_TABLE_KEY: None,
    PROTEINS_FOLDER_KEY: None,
    UNIQUE_PROTEIN_IDS_KEY: None,
    INPUT_COMBINATIONS_KEY: None,
    SMILES_NAMES_DICTIONARY_KEY: None,
    SMILES_TYPE_DICTIONARY_KEY: None,
    PRE_EXISTING_RP_SCORES_KEY: None,
    RP_SCORES_DF_KEY: None,
    COMBINATIONS_TO_RUN_KEY: None,
    PREVIOUS_COMBINATIONS_DF_KEY: None,
    DECOYS_COMBINATIONS_KEY: None,
    METHODS_TO_SCORE_DICTIONARY_KEY: None,
    METHODS_TO_SORT_DICTIONARY_KEY: None,
    PREVIOUS_RP_SCORES_KEY: None,
    SCORES_TO_USE_KEY: None,
    RANKS_LIST_KEY: None,
    PROTEIN_SEQUENCE_KEY: None,
}
"""
Folders
"""
COMBINATION_ID = "combination"

"""
Scores columns
"""
GLOBAL_RP_SCORE = "global_rp_score"
VINA_RP_SCORE = f"rp_{VINA_SCORE}"
KARMADOCK_RP_SCORE = f"rp_{KARMADOCK_SCORE}"
DIFFDOCK_RP_SCORE = f"rp_{DIFFDOCK_SCORE}"
BOLTZ_RP_SCORE = f"rp_{BOLTZ_SCORE}"
GNINA_RP_SCORE = f"rp_{GNINA_SCORE}"
NESSO_RP_SCORE = f"rp_{NESSO_SCORE}"
BOLTZ_AFFINITY_RP_SCORE = f"rp_{BOLTZ_AFFINITY_SCORE}"
VINA_RESCORE_BOLTZ_RP_SCORE = f"rp_{VINA_RESCORE_BOLTZ_SCORE}"
VINA_RESCORE_DIFFDOCK_RP_SCORE = f"rp_{VINA_RESCORE_DIFFDOCK_SCORE}"
GNINA_RESCORE_BOLTZ_RP_SCORE = f"rp_{GNINA_RESCORE_BOLTZ_SCORE}"
GNINA_RESCORE_DIFFDOCK_RP_SCORE = f"rp_{GNINA_RESCORE_DIFFDOCK_SCORE}"
RANK_VINA_SCORE = f"rank_{VINA_SCORE}"
RANK_KARMADOCK_SCORE = f"rank_{KARMADOCK_SCORE}"
RANK_DIFFDOCK_SCORE = f"rank_{DIFFDOCK_SCORE}"
RANK_BOLTZ_SCORE = f"rank_{BOLTZ_SCORE}"
RANK_GNINA_SCORE = f"rank_{GNINA_SCORE}"
RANK_NESSO_SCORE = f"rank_{NESSO_SCORE}"
RANK_BOLTZ_AFFINITY_SCORE = f"rank_{BOLTZ_AFFINITY_SCORE}"
RANK_VINA_RESCORE_BOLTZ_SCORE = f"rank_{VINA_RESCORE_BOLTZ_SCORE}"
RANK_VINA_RESCORE_DIFFDOCK_SCORE = f"rank_{VINA_RESCORE_DIFFDOCK_SCORE}"
RANK_GNINA_RESCORE_BOLTZ_SCORE = f"rank_{GNINA_RESCORE_BOLTZ_SCORE}"
RANK_GNINA_RESCORE_DIFFDOCK_SCORE = f"rank_{GNINA_RESCORE_DIFFDOCK_SCORE}"

"""
Rank percentile denominator modes
"""
# What rank / denominator divides by: molecules with a valid raw score, or
# every pair attempted for the protein. The published case-study results used
# ATTEMPTED, where ~5.6% of pairs failed to score and still counted.
DENOMINATOR_VALID = "valid"
DENOMINATOR_ATTEMPTED = "attempted"
DENOMINATOR_MODES = (DENOMINATOR_VALID, DENOMINATOR_ATTEMPTED)

SCORES_DIRECTION_DICTIONARY = {
    VINA_PREFIX: "minimum",
    KARMADOCK_PREFIX: "maximum",
    DIFFDOCK_PREFIX: "maximum",
    BOLTZ_PREFIX: "maximum",
    GNINA_PREFIX: "minimum",
    # Nesso's affinity head is log10(IC50/uM) — a predicted potency, lower is
    # stronger — unlike boltz_score, which is an ipTM confidence ("maximum").
    NESSO_PREFIX: "minimum",
    VINA_RESCORE_BOLTZ_PREFIX: "minimum",
    VINA_RESCORE_DIFFDOCK_PREFIX: "minimum",
    GNINA_RESCORE_BOLTZ_PREFIX: "minimum",
    GNINA_RESCORE_DIFFDOCK_PREFIX: "minimum",
    # Boltz-2's own affinity head is also log10(IC50/uM) — same convention as
    # Nesso, and the opposite of boltz_score.
    BOLTZ_AFFINITY_PREFIX: "minimum",
}

# Vina-family (Vina/gnina, native or rescored): binding free energy in
# kcal/mol. Nesso is also "minimum"-direction but log10(IC50/uM), a different
# quantity, so it's deliberately excluded here.
VINA_FAMILY_SCORE_METHODS = frozenset(
    {
        VINA_PREFIX,
        GNINA_PREFIX,
        VINA_RESCORE_BOLTZ_PREFIX,
        VINA_RESCORE_DIFFDOCK_PREFIX,
        GNINA_RESCORE_BOLTZ_PREFIX,
        GNINA_RESCORE_DIFFDOCK_PREFIX,
    }
)

# Plausible kcal/mol range for a Vina-family score, from a 403k-pair Vina
# case study: 93.9% of values fall inside it (6.06% were non-negative, 0.79%
# exceeded 1e6 in magnitude — near-certainly numerical/parsing failures). A
# flag, not a hard bound — never clamped or nulled by default (see
# is_physical_score in guild/tools/scores.py).
VINA_FAMILY_PLAUSIBLE_SCORE_RANGE = (-20.0, 0.0)

RANKS_DICTIONARY = {
    VINA_PREFIX: RANK_VINA_SCORE,
    KARMADOCK_PREFIX: RANK_KARMADOCK_SCORE,
    DIFFDOCK_PREFIX: RANK_DIFFDOCK_SCORE,
    BOLTZ_PREFIX: RANK_BOLTZ_SCORE,
    GNINA_PREFIX: RANK_GNINA_SCORE,
    NESSO_PREFIX: RANK_NESSO_SCORE,
    VINA_RESCORE_BOLTZ_PREFIX: RANK_VINA_RESCORE_BOLTZ_SCORE,
    VINA_RESCORE_DIFFDOCK_PREFIX: RANK_VINA_RESCORE_DIFFDOCK_SCORE,
    GNINA_RESCORE_BOLTZ_PREFIX: RANK_GNINA_RESCORE_BOLTZ_SCORE,
    GNINA_RESCORE_DIFFDOCK_PREFIX: RANK_GNINA_RESCORE_DIFFDOCK_SCORE,
    BOLTZ_AFFINITY_PREFIX: RANK_BOLTZ_AFFINITY_SCORE,
}

RP_SCORES_DICTIONARY = {
    VINA_PREFIX: VINA_RP_SCORE,
    KARMADOCK_PREFIX: KARMADOCK_RP_SCORE,
    DIFFDOCK_PREFIX: DIFFDOCK_RP_SCORE,
    BOLTZ_PREFIX: BOLTZ_RP_SCORE,
    GNINA_PREFIX: GNINA_RP_SCORE,
    NESSO_PREFIX: NESSO_RP_SCORE,
    VINA_RESCORE_BOLTZ_PREFIX: VINA_RESCORE_BOLTZ_RP_SCORE,
    VINA_RESCORE_DIFFDOCK_PREFIX: VINA_RESCORE_DIFFDOCK_RP_SCORE,
    GNINA_RESCORE_BOLTZ_PREFIX: GNINA_RESCORE_BOLTZ_RP_SCORE,
    GNINA_RESCORE_DIFFDOCK_PREFIX: GNINA_RESCORE_DIFFDOCK_RP_SCORE,
    BOLTZ_AFFINITY_PREFIX: BOLTZ_AFFINITY_RP_SCORE,
}

# Pose-confidence tracks: rank percentile is still computed, but they don't
# vote in GLOBAL_RP_SCORE (diffdock/boltz measure pose correctness, not
# binding affinity) — same status GNINA_CNN_SCORE already has.
CONFIDENCE_ONLY_METHODS = frozenset({DIFFDOCK_PREFIX, BOLTZ_PREFIX})

# Which engine's pose each scoring track judges — a rescore track belongs to
# the engine whose pose it scores, not to Vina/gnina. GLOBAL_RP_SCORE averages
# within a source before combining sources, so DiffDock/Boltz don't get 3
# votes each from their auto-added rescores.
POSE_SOURCE_DICTIONARY = {
    VINA_PREFIX: VINA_PREFIX,
    KARMADOCK_PREFIX: KARMADOCK_PREFIX,
    GNINA_PREFIX: GNINA_PREFIX,
    NESSO_PREFIX: NESSO_PREFIX,
    DIFFDOCK_PREFIX: DIFFDOCK_PREFIX,
    BOLTZ_PREFIX: BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX: DIFFDOCK_PREFIX,
    GNINA_RESCORE_DIFFDOCK_PREFIX: DIFFDOCK_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX: BOLTZ_PREFIX,
    GNINA_RESCORE_BOLTZ_PREFIX: BOLTZ_PREFIX,
    # Co-predicted with the same complex the two rescores judge — joins
    # Boltz's vote as a third estimate, not a sixth independent one.
    BOLTZ_AFFINITY_PREFIX: BOLTZ_PREFIX,
}

# How GLOBAL_RP_SCORE combines per-method percentiles: average rescores into
# their pose source, then combine sources. POSE_SOURCE_MEDIAN (default) takes
# the median across sources — 0.824 vs 0.781 AUC for the flat mean on a
# 3-target benchmark, less swayed by one aberrant vote. POSE_SOURCE is the
# plain mean; FLAT is the pre-grouping mean over every track, kept for
# reproducibility.
AGGREGATION_POSE_SOURCE = "pose_source"
AGGREGATION_POSE_SOURCE_MEDIAN = "pose_source_median"
AGGREGATION_FLAT = "flat"
AGGREGATION_MODES = (AGGREGATION_POSE_SOURCE, AGGREGATION_POSE_SOURCE_MEDIAN, AGGREGATION_FLAT)

SCORES_TO_USE_DICTIONARY = {
    VINA_PREFIX: [
        VINA_SCORE,
        RANK_VINA_SCORE,
        VINA_RP_SCORE,
    ],
    KARMADOCK_PREFIX: [
        KARMADOCK_SCORE,
        RANK_KARMADOCK_SCORE,
        KARMADOCK_RP_SCORE,
    ],
    DIFFDOCK_PREFIX: [
        DIFFDOCK_SCORE,
        RANK_DIFFDOCK_SCORE,
        DIFFDOCK_RP_SCORE,
    ],
    BOLTZ_PREFIX: [
        BOLTZ_SCORE,
        RANK_BOLTZ_SCORE,
        BOLTZ_RP_SCORE,
    ],
    GNINA_PREFIX: [
        GNINA_SCORE,
        RANK_GNINA_SCORE,
        GNINA_RP_SCORE,
    ],
    NESSO_PREFIX: [
        NESSO_SCORE,
        RANK_NESSO_SCORE,
        NESSO_RP_SCORE,
    ],
    VINA_RESCORE_BOLTZ_PREFIX: [
        VINA_RESCORE_BOLTZ_SCORE,
        RANK_VINA_RESCORE_BOLTZ_SCORE,
        VINA_RESCORE_BOLTZ_RP_SCORE,
    ],
    VINA_RESCORE_DIFFDOCK_PREFIX: [
        VINA_RESCORE_DIFFDOCK_SCORE,
        RANK_VINA_RESCORE_DIFFDOCK_SCORE,
        VINA_RESCORE_DIFFDOCK_RP_SCORE,
    ],
    GNINA_RESCORE_BOLTZ_PREFIX: [
        GNINA_RESCORE_BOLTZ_SCORE,
        RANK_GNINA_RESCORE_BOLTZ_SCORE,
        GNINA_RESCORE_BOLTZ_RP_SCORE,
    ],
    GNINA_RESCORE_DIFFDOCK_PREFIX: [
        GNINA_RESCORE_DIFFDOCK_SCORE,
        RANK_GNINA_RESCORE_DIFFDOCK_SCORE,
        GNINA_RESCORE_DIFFDOCK_RP_SCORE,
    ],
    BOLTZ_AFFINITY_PREFIX: [
        BOLTZ_AFFINITY_SCORE,
        RANK_BOLTZ_AFFINITY_SCORE,
        BOLTZ_AFFINITY_RP_SCORE,
    ],
}
