"""
PDB/PDBQT record-format constants
"""

# PDB columns through tempFactor. PDBQT adds partial-charge and atom-type
# columns past this point, which RDKit's PDB parser rejects.
PDB_RECORD_WIDTH = 66
