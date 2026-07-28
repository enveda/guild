"""
Protein-ligand interaction analysis constants.

Shared by the interaction-analysis backends (PLIP and ProLIF) — column names,
the controlled interaction-type vocabulary, output file names, and the detail
schema.
"""

from guild.constants.guild import (  # noqa: F401 — re-exported for detail columns
    PROTEIN_CONF_ID,
    SMILES,
)

# Ligand residue identifiers (default values for complex PDB files)
LIGAND_RESNAME = "LIG"
LIGAND_CHAIN = "Z"
LIGAND_RESSEQ = 1

# File naming
INTERACTIONS_FILE = "plip_interactions.tsv"
COMPLEX_PDB_SUFFIX = "_complex.pdb"

# Column names for interaction data
INTERACTION_COMBINATION_ID = "combination_id"
COMPLEX_PDB = "complex_pdb"
BINDING_SITE_ID = "binding_site_id"

# Hydrogen bonds
N_HBONDS = "n_hbonds"

# Hydrophobic interactions
N_HYDROPHOBIC = "n_hydrophobic"

# Pi-stacking
N_PISTACKING = "n_pistacking"

# Pi-cation
N_PICATION = "n_pication"

# Salt bridges
N_SALTBRIDGES = "n_saltbridges"

# Halogen bonds
N_HALOGEN = "n_halogen"

# Water bridges
N_WATERBRIDGES = "n_waterbridges"

# Metal complexes
N_METAL = "n_metal"

# Summary statistics
TOTAL_INTERACTIONS = "total_interactions"
N_UNIQUE_RESIDUES = "n_unique_residues"

# --- Interaction detail file ---

INTERACTION_DETAILS_FILE = "plip_interaction_details.tsv"

# Detail column names
DETAIL_DOCKING_METHOD = "docking_method"
DETAIL_SOURCE = "source"
DETAIL_INTERACTION_TYPE = "interaction_type"
DETAIL_PROT_RESNAME = "protein_residue_name"
DETAIL_PROT_RESNR = "protein_residue_number"
DETAIL_PROT_CHAIN = "protein_chain"
DETAIL_PROT_SIDECHAIN = "protein_sidechain"
DETAIL_LIG_ATOM_IDX = "ligand_atom_index"
DETAIL_LIG_ATOM_TYPE = "ligand_atom_type"
DETAIL_DISTANCE = "distance"
DETAIL_DISTANCE_AD = "distance_ad"
DETAIL_DISTANCE_AH = "distance_ah"
DETAIL_DISTANCE_AW = "distance_aw"
DETAIL_DISTANCE_DW = "distance_dw"
DETAIL_ANGLE = "angle"
DETAIL_DON_ANGLE = "don_angle"
DETAIL_ACC_ANGLE = "acc_angle"
DETAIL_OFFSET = "offset"
DETAIL_PISTACK_TYPE = "pistack_type"
DETAIL_DONOR_TYPE = "donor_type"
DETAIL_ACCEPTOR_TYPE = "acceptor_type"
DETAIL_PROTISDON = "prot_is_donor"
DETAIL_SALTBRIDGE_PROTISPOS = "prot_is_positive"
DETAIL_METAL_TYPE = "metal_type"
DETAIL_COORDINATION = "coordination_num"
DETAIL_METAL_GEOMETRY = "metal_geometry"
DETAIL_TARGET_TYPE = "target_type"

# Controlled vocabulary for interaction_type
INTERACTION_TYPE_HBOND = "hbond"
INTERACTION_TYPE_HYDROPHOBIC = "hydrophobic"
INTERACTION_TYPE_PISTACKING = "pistacking"
INTERACTION_TYPE_PICATION = "pication"
INTERACTION_TYPE_SALTBRIDGE = "saltbridge"
INTERACTION_TYPE_HALOGEN = "halogen"
INTERACTION_TYPE_WATERBRIDGE = "waterbridge"
INTERACTION_TYPE_METAL = "metal"

# Ordered column schema for the interaction detail DataFrame
DETAIL_COLUMNS = [
    INTERACTION_COMBINATION_ID,
    PROTEIN_CONF_ID,
    SMILES,
    DETAIL_DOCKING_METHOD,
    DETAIL_SOURCE,
    DETAIL_INTERACTION_TYPE,
    DETAIL_PROT_RESNAME,
    DETAIL_PROT_RESNR,
    DETAIL_PROT_CHAIN,
    DETAIL_PROT_SIDECHAIN,
    DETAIL_LIG_ATOM_IDX,
    DETAIL_LIG_ATOM_TYPE,
    DETAIL_DISTANCE,
    DETAIL_DISTANCE_AD,
    DETAIL_DISTANCE_AH,
    DETAIL_DISTANCE_AW,
    DETAIL_DISTANCE_DW,
    DETAIL_ANGLE,
    DETAIL_DON_ANGLE,
    DETAIL_ACC_ANGLE,
    DETAIL_OFFSET,
    DETAIL_PISTACK_TYPE,
    DETAIL_DONOR_TYPE,
    DETAIL_ACCEPTOR_TYPE,
    DETAIL_PROTISDON,
    DETAIL_SALTBRIDGE_PROTISPOS,
    DETAIL_METAL_TYPE,
    DETAIL_COORDINATION,
    DETAIL_METAL_GEOMETRY,
    DETAIL_TARGET_TYPE,
]

# List of all interaction count columns
INTERACTION_COUNT_COLUMNS = [
    N_HBONDS,
    N_HYDROPHOBIC,
    N_PISTACKING,
    N_PICATION,
    N_SALTBRIDGES,
    N_HALOGEN,
    N_WATERBRIDGES,
    N_METAL,
]
