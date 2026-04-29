"""
PLIP (Protein-Ligand Interaction Profiler) constants
"""

# Ligand residue identifiers (default values for complex PDB files)
PLIP_LIGAND_RESNAME = "LIG"
PLIP_LIGAND_CHAIN = "Z"
PLIP_LIGAND_RESSEQ = 1

# File naming
PLIP_INTERACTIONS_FILE = "plip_interactions.tsv"
COMPLEX_PDB_SUFFIX = "_complex.pdb"

# Column names for interaction data
PLIP_COMBINATION_ID = "combination_id"
PLIP_COMPLEX_PDB = "complex_pdb"
PLIP_BINDING_SITE_ID = "binding_site_id"

# Hydrogen bonds
PLIP_N_HBONDS = "n_hbonds"

# Hydrophobic interactions
PLIP_N_HYDROPHOBIC = "n_hydrophobic"

# Pi-stacking
PLIP_N_PISTACKING = "n_pistacking"

# Pi-cation
PLIP_N_PICATION = "n_pication"

# Salt bridges
PLIP_N_SALTBRIDGES = "n_saltbridges"

# Halogen bonds
PLIP_N_HALOGEN = "n_halogen"

# Water bridges
PLIP_N_WATERBRIDGES = "n_waterbridges"

# Metal complexes
PLIP_N_METAL = "n_metal"

# Summary statistics
PLIP_TOTAL_INTERACTIONS = "total_interactions"
PLIP_N_UNIQUE_RESIDUES = "n_unique_residues"

# List of all interaction count columns
PLIP_INTERACTION_COUNT_COLUMNS = [
    PLIP_N_HBONDS,
    PLIP_N_HYDROPHOBIC,
    PLIP_N_PISTACKING,
    PLIP_N_PICATION,
    PLIP_N_SALTBRIDGES,
    PLIP_N_HALOGEN,
    PLIP_N_WATERBRIDGES,
    PLIP_N_METAL,
]
