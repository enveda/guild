"""
PDB/PDBQT record-format constants
"""

# PDB columns through tempFactor. PDBQT adds partial-charge and atom-type
# columns past this point, which RDKit's PDB parser rejects.
PDB_RECORD_WIDTH = 66

# Longest bond accepted when locating a covalent receptor partner.
COVALENT_BOND_DIST_MAX = 2.5
