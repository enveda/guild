"""
GNINA constants
"""

GNINA_BINARY = "/opt/gnina/bin/gnina"

# gnina ships its own torch + CUDA 12 runtime; we put them under
# /opt/gnina/lib so they only get loaded for the gnina subprocess. Prepend
# (not replace) to the caller's LD_LIBRARY_PATH at invocation time.
GNINA_LIB_PATH = "/opt/gnina/lib"

# gnina v1.3.3+ statically links OpenBabel, but the OB *data* files
# (UFF.prm etc.) still have to live on disk. The bundle Dockerfile copies
# them here; the wrapper exports ``BABEL_DATADIR=GNINA_OB_DATA_DIR`` when
# invoking gnina so OB can resolve them at runtime (without this,
# --covalent_optimize_lig prints "Cannot open UFF.prm" and skips the
# post-bond UFF minimisation).
GNINA_OB_DATA_DIR = "/opt/gnina/share/openbabel"

GNINA_DEFAULT_EXHAUSTIVENESS = 8

GNINA_DEFAULT_NUMBER_OF_POSES = 9

# gnina's default --cnn_scoring is "rescore": Vina-style search produces poses
# and the CNN rescores the top N. Fastest mode that still surfaces a CNN score.
GNINA_DEFAULT_CNN_SCORING = "rescore"

# if using covalent docking, gnina's CNN scoring is not yet calibrated for it. Warning below is typical if not using "none"
# # Recommend running with --cnn_scoring none to avoid misleading CNN scores.
# # WARNING: CNN scoring not yet calibrated for covalent docking.  Recommend running with --cnn_scoring none
GNINA_DEFAULT_CNN_SCORING_COVALENT_ONLY = "none"

# Distance threshold (Å) for auto-selecting flexible residues in gnina SDF mode.
# Passed to gnina's --flexdist flag alongside --flexdist_ligand.
GNINA_FLEX_DISTANCE = 4.0

# Covalent docking defaults. Bond order of the receptor↔ligand covalent bond
# (single bond) and whether to UFF-optimise the ligand+residue adduct (gnina's
# --covalent_optimize_lig, recommended for sensible covalent geometry).
GNINA_COVALENT_BOND_ORDER = 1
GNINA_COVALENT_OPTIMIZE_LIG = True
