"""
Proteins constants
"""

PROTEIN_VALID_FORMATS = ["pdb"]

"""
Amino acids single letter
"""
POSSIBLE_AMINO_ACIDS_SINGLE_LETTER = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

POSSIBLE_AMINO_ACIDS_THREE_LETTER = [
    "ALA",
    "CYS",
    "ASP",
    "GLY",
    "PHE",
    "GLU",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "GLN",
    "PRO",
    "ASN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]

AMINO_ACID_GROUPS = {
    "aromatic": ["TYR", "TRP", "PHE"],
    "charged_positive": ["ARG", "LYS", "HIS"],
    "charged_negative": ["ASP", "GLU"],
    "polar": ["SER", "ASN", "GLN", "THR"],
    "non_polar": ["ALA", "VAL", "LEU", "ILE", "MET", "PRO", "GLY", "CYS"],
}

AMINO_ACID_CONVERTER = {
    "GLY": "G",
    "ALA": "A",
    "VAL": "V",
    "LEU": "L",
    "ILE": "I",
    "THR": "T",
    "SER": "S",
    "MET": "M",
    "CYS": "C",
    "PRO": "P",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "HIS": "H",
    "LYS": "K",
    "ARG": "R",
    "ASP": "D",
    "GLU": "E",
    "ASN": "N",
    "GLN": "Q",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O"}

"""
Metals list
"""
METALS_LIST = ["ZN", "MG", "MN", "FE", "CA", "NA", "K", "CO", "CU", "NI", "CD", "SR", "BA", "HG"]

"""
Waters list
"""
WATERS_LIST = ["HOH", "WAT", "H2O"]
