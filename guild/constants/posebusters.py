"""
PoseBusters pose-validity constants.

Column names, the status vocabulary, the two check groups (intramolecular
"mol" checks and intermolecular "dock" checks), the pose-scope vocabulary and
the ordered output schema for the project-level pose-validity table.

PoseBusters is a *validator*, not a docking method: it is deliberately absent
from ``ALL_AVAILABLE_METHODS``, ``SCORES_DICTIONARY``, ``RANKS_DICTIONARY``
and ``RP_SCORES_DICTIONARY`` in :mod:`guild.constants.guild` /
:mod:`guild.constants.bulk`, so it never gets a rank or rank-percentile
column. Its verdict reaches ``guild_scores.txt`` only as the additive
``<method>_pb_valid`` / ``<method>_pb_pose`` flag columns defined at the
bottom of this module.
"""

from guild.constants.guild import (  # noqa: F401 — re-exported for the schema
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
    GNINA_PREFIX,
    POSE,
    PROTEIN_CONF_ID,
    SMILES,
    VINA_PREFIX,
)

"""
Files
"""

POSEBUSTERS_FILE = "posebusters_validity.tsv"
POSEBUSTERS_REPORT_FILE = "posebusters_full_report.tsv"

"""
PoseBusters configuration presets
"""

# "dock" is the only correct family here: guild has no crystal reference pose,
# so "redock"/"regen" (which require mol_true) are out, and "mol" alone would
# drop every protein-ligand check.
#
# "dock_fast" is exactly "dock" minus the `internal_energy` check, which
# generates a relaxed conformer ensemble. Measured on a 23-heavy-atom ligand the
# two are close — ~420 ms vs ~405 ms per pose — because the ensemble is cheap for
# a small, fairly rigid molecule. The gap widens with ligand size and rotatable-
# bond count, so treat dock_fast as a lever to try for big/floppy ligand sets
# rather than a guaranteed win. Measure before assuming it helps.
POSEBUSTERS_CONFIG_DOCK = "dock"
POSEBUSTERS_CONFIG_DOCK_FAST = "dock_fast"
POSEBUSTERS_CONFIGS = (POSEBUSTERS_CONFIG_DOCK, POSEBUSTERS_CONFIG_DOCK_FAST)
DEFAULT_POSEBUSTERS_CONFIG = POSEBUSTERS_CONFIG_DOCK

"""
Pose scope
"""

# best     — validate only the top-ranked pose.
# escalate — validate the top pose; if it fails, walk down the ranked poses
#            until one passes or POSEBUSTERS_MAX_POSES is reached. Answers
#            "does this combination have any physically valid pose?" while
#            costing a single pose in the common case.
# all      — validate every pose up to POSEBUSTERS_MAX_POSES.
POSE_SCOPE_BEST = "best"
POSE_SCOPE_ESCALATE = "escalate"
POSE_SCOPE_ALL = "all"
POSE_SCOPES = (POSE_SCOPE_BEST, POSE_SCOPE_ESCALATE, POSE_SCOPE_ALL)
DEFAULT_POSE_SCOPE = POSE_SCOPE_ESCALATE

# Hard cap on poses validated per (combination, method), for escalate and all.
# Bounds the worst case when a check fails on every pose — see the
# `internal_energy` note above.
POSEBUSTERS_MAX_POSES = 9

# Methods that emit a `<combination>_complex.pdb` and therefore have a pose to
# validate. karmadock writes no complex PDB and nesso produces no 3D output at
# all; the vina_rescore_* / gnina_rescore_* tracks are score-only and reuse
# another method's pose.
POSEBUSTERS_SUPPORTED_METHODS = [
    VINA_PREFIX,
    GNINA_PREFIX,
    BOLTZ_PREFIX,
    DIFFDOCK_PREFIX,
]

"""
Identity columns
"""

PB_COMBINATION_ID = "combination_id"
PB_DOCKING_METHOD = "docking_method"
PB_COMPLEX_PDB = "complex_pdb"
PB_POSE = POSE  # 1-based rank of the pose this row describes

"""
Status vocabulary
"""

# PB_VALID fails CLOSED — it is never None, so `df[df.pb_valid]` can never
# admit a pose that was not actually verified. PB_STATUS is how you tell an
# invalid pose ("ok" + pb_valid False) from one that could not be checked.
PB_STATUS = "posebusters_status"
PB_ERROR = "posebusters_error"

PB_STATUS_OK = "ok"
PB_STATUS_MISSING_COMPLEX = "missing_complex"
PB_STATUS_NO_POSES = "no_poses_found"
PB_STATUS_LIGAND_BUILD_FAILED = "ligand_build_failed"
PB_STATUS_PROTEIN_BUILD_FAILED = "protein_build_failed"
# The SMILES template did not match the pose's perceived bond graph, so bond
# orders were inferred from 3D geometry instead. Intermolecular checks stay
# fully valid; the intramolecular ones weaken. Surfaced as its own status
# rather than a log line only, because the row still carries real verdicts.
PB_STATUS_TEMPLATE_FALLBACK = "ligand_template_fallback"
PB_STATUS_BUST_FAILED = "posebusters_raised"
PB_STATUS_UNAVAILABLE = "posebusters_unavailable"

"""
Verdict columns
"""

PB_VALID = "pb_valid"
PB_INTRAMOLECULAR_VALID = "pb_intramolecular_valid"
PB_INTERMOLECULAR_VALID = "pb_intermolecular_valid"
PB_N_CHECKS_FAILED = "pb_n_checks_failed"
PB_FAILED_CHECKS = "pb_failed_checks"  # ";"-joined, sorted

"""
Individual checks
"""

# Names are PoseBusters' own output column names with "-" normalised to "_"
# (see _normalise_check_name in guild/analysis/posebusters.py), so upstream's
# "non-aromatic_ring_non-flatness" becomes "non_aromatic_ring_non_flatness"
# and "protein-ligand_maximum_distance" becomes
# "protein_ligand_maximum_distance". The lists below are the 22 columns the
# "dock" config actually emits, verified against posebusters 0.6.5 — the
# upstream CLI docs omit the last two.

# Intramolecular — the `mol` check family, ligand-only sanity.
CHECK_MOL_PRED_LOADED = "mol_pred_loaded"
CHECK_SANITIZATION = "sanitization"
CHECK_INCHI_CONVERTIBLE = "inchi_convertible"
CHECK_ALL_ATOMS_CONNECTED = "all_atoms_connected"
CHECK_NO_RADICALS = "no_radicals"
CHECK_BOND_LENGTHS = "bond_lengths"
CHECK_BOND_ANGLES = "bond_angles"
CHECK_INTERNAL_STERIC_CLASH = "internal_steric_clash"
CHECK_AROMATIC_RING_FLATNESS = "aromatic_ring_flatness"
CHECK_NON_AROMATIC_RING_NON_FLATNESS = "non_aromatic_ring_non_flatness"
CHECK_DOUBLE_BOND_FLATNESS = "double_bond_flatness"
# Absent from the "dock_fast" config; stays None on those rows.
CHECK_INTERNAL_ENERGY = "internal_energy"

# Intermolecular — the extra `dock` checks, placement versus the receptor.
CHECK_MOL_COND_LOADED = "mol_cond_loaded"
CHECK_PROTEIN_LIGAND_MAX_DISTANCE = "protein_ligand_maximum_distance"
CHECK_MIN_DISTANCE_TO_PROTEIN = "minimum_distance_to_protein"
CHECK_MIN_DISTANCE_TO_ORGANIC_COFACTORS = "minimum_distance_to_organic_cofactors"
CHECK_MIN_DISTANCE_TO_INORGANIC_COFACTORS = "minimum_distance_to_inorganic_cofactors"
CHECK_MIN_DISTANCE_TO_WATERS = "minimum_distance_to_waters"
CHECK_VOLUME_OVERLAP_WITH_PROTEIN = "volume_overlap_with_protein"
CHECK_VOLUME_OVERLAP_WITH_ORGANIC_COFACTORS = "volume_overlap_with_organic_cofactors"
CHECK_VOLUME_OVERLAP_WITH_INORGANIC_COFACTORS = "volume_overlap_with_inorganic_cofactors"
CHECK_VOLUME_OVERLAP_WITH_WATERS = "volume_overlap_with_waters"

PB_INTRAMOLECULAR_CHECK_COLUMNS = [
    CHECK_MOL_PRED_LOADED,
    CHECK_SANITIZATION,
    CHECK_INCHI_CONVERTIBLE,
    CHECK_ALL_ATOMS_CONNECTED,
    CHECK_NO_RADICALS,
    CHECK_BOND_LENGTHS,
    CHECK_BOND_ANGLES,
    CHECK_INTERNAL_STERIC_CLASH,
    CHECK_AROMATIC_RING_FLATNESS,
    CHECK_NON_AROMATIC_RING_NON_FLATNESS,
    CHECK_DOUBLE_BOND_FLATNESS,
    CHECK_INTERNAL_ENERGY,
]

PB_INTERMOLECULAR_CHECK_COLUMNS = [
    CHECK_MOL_COND_LOADED,
    CHECK_PROTEIN_LIGAND_MAX_DISTANCE,
    CHECK_MIN_DISTANCE_TO_PROTEIN,
    CHECK_MIN_DISTANCE_TO_ORGANIC_COFACTORS,
    CHECK_MIN_DISTANCE_TO_INORGANIC_COFACTORS,
    CHECK_MIN_DISTANCE_TO_WATERS,
    CHECK_VOLUME_OVERLAP_WITH_PROTEIN,
    CHECK_VOLUME_OVERLAP_WITH_ORGANIC_COFACTORS,
    CHECK_VOLUME_OVERLAP_WITH_INORGANIC_COFACTORS,
    CHECK_VOLUME_OVERLAP_WITH_WATERS,
]

PB_CHECK_COLUMNS = PB_INTRAMOLECULAR_CHECK_COLUMNS + PB_INTERMOLECULAR_CHECK_COLUMNS

"""
Ordered column schema for the pose-validity DataFrame / TSV
"""

POSEBUSTERS_COLUMNS = [
    PB_COMBINATION_ID,
    PROTEIN_CONF_ID,
    SMILES,
    PB_DOCKING_METHOD,
    PB_POSE,
    PB_VALID,
    PB_INTRAMOLECULAR_VALID,
    PB_INTERMOLECULAR_VALID,
    PB_N_CHECKS_FAILED,
    PB_FAILED_CHECKS,
    PB_STATUS,
    *PB_CHECK_COLUMNS,
    PB_COMPLEX_PDB,
    PB_ERROR,
]

# Leading identity columns of the full-report table. The remaining columns are
# whatever `bust(..., full_report=True)` emits (112 of them for the "dock"
# config), which is deliberately not enumerated here — the report is a
# debugging artifact, so a posebusters upgrade that adds a measurement should
# widen the file rather than trip a schema assertion.
POSEBUSTERS_REPORT_ID_COLUMNS = [
    PB_COMBINATION_ID,
    PROTEIN_CONF_ID,
    SMILES,
    PB_DOCKING_METHOD,
    PB_POSE,
]

"""
Derived guild_scores.txt columns
"""

# Additive per-method flag columns merged into guild_scores.txt, following the
# rank_/rp_ derived-column idiom in guild/constants/bulk.py. `_pb_valid` is
# True when any validated pose passed; `_pb_pose` is the 1-based rank of the
# first pose that did (NA when none did).
POSEBUSTERS_VALID_SUFFIX = "_pb_valid"
POSEBUSTERS_POSE_SUFFIX = "_pb_pose"
POSEBUSTERS_VALID_DICTIONARY = {
    method: f"{method}{POSEBUSTERS_VALID_SUFFIX}" for method in POSEBUSTERS_SUPPORTED_METHODS
}
POSEBUSTERS_POSE_DICTIONARY = {
    method: f"{method}{POSEBUSTERS_POSE_SUFFIX}" for method in POSEBUSTERS_SUPPORTED_METHODS
}
