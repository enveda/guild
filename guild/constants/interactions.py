"""
Protein-ligand interaction analysis constants.

Backend-agnostic: shared by the interaction-analysis backends rather than tied
to PLIP, which is why the names carry no ``PLIP_`` prefix.
"""

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
