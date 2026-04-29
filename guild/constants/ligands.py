"""
Ligands constants
"""

# Timeout for obabel conversions (in seconds)
# With canonicalization, most molecules convert in <5s, but some complex ones need more time
OBABEL_CONVERSION_TIMEOUT = 60

LIGAND_VALID_FORMATS = ["sdf", "smi", "mol"]

LIGANDS_TO_IGNORE = {
    # Solvents
    "HOH",  # Water
    "DMS",  # Dimethyl sulfoxide
    "EDO",  # 1,2-ethanediol (ethylene glycol)
    "GOL",  # Glycerol
    "PG4",  # Tetraethylene glycol
    "NH2",  # Ammonia
    # Ions
    "NA",  # Sodium ion
    "K",  # Potassium ion
    "CL",  # Chloride ion
    "CA",  # Calcium ion
    "MG",  # Magnesium ion
    "ZN",  # Zinc ion
    "SO4",  # Sulfate ion
    "PO4",  # Phosphate ion
    "MN",  # Manganese ion
    # Buffers
    "HEM",  # Heme (often non-drug-like)
    "TRS",  # Tris buffer
    "MES",  # 2-(N-morpholino)ethanesulfonic acid
    "HEZ",  # Hepes buffer
    "BME",  # Beta-mercaptoethanol
    # Crystallization agents
    "MPD",  # 2-Methyl-2,4-pentanediol
    "PEG",  # Polyethylene glycol (generic)
    "PGE",  # Polyethylene glycol (specific)
    "ACT",  # Acetate ion
    # Miscellaneous stabilizers
    "ACE",  # Acetyl group
    "ACY",  # Acetate
    "FMT",  # Formate
    "NO3",  # Nitrate ion
    "OX",  # Oxalate
    # Exotic amino acid
    "PTR",  # Phosphotyrosine
    "SEP",  # Phosphoserine
    "TPO",  # Phosphothreonine
    "MSE",  # Selenomethionine
    # Unknown
    "UNK",  # Unknown
    # Manually curated
    "D6M",
    "OLA",  # Oleic acid
    "CLR",  # Cholesterol
    "PLM",  # Palmitic acid
    "12P",  # DODECAETHYLENE GLYCOL
    "ACM",  # Acetamide
    "BU1",  # Butanediol
    "SOC",  # glucopyranoside
    "STE",  # Stearic acid
    "OLC",  # dihydroxypropyl (9Z)-octadec-9-enoate
    "OLB",  # (2S)-2,3-dihydroxypropyl (9Z)-octadec-9-enoate
    "BGC",  # beta-D-glucopyranose
    "LMT",  # DODECYL-BETA-D-MALTOSIDE
    "CIT",  # citric acid
    "PGW",  # (1R)-2-{[(S)-{[(2S)-2,3-dihydroxypropyl]oxy}(hydroxy)phosphoryl]oxy}-1-[(hexadecanoyloxy)methyl]ethyl (9Z)-octadec-9-enoate
    "TLA",  # Tartaric acid
    "DGA",  # DIACYL GLYCEROL
    "1PE",  # PENTAETHYLENE GLYCOL
    "SIN",  # Succinic acid
    "1WV",  # {1-[(2S,3S)-2-(2,3-dihydro-1H-inden-2-ylmethyl)-3-(3,5-dimethoxy-4-methylphenyl)-3-hydroxypropyl]-4-(methoxycarbonyl)-1 H-pyrrol-3-yl}acetic acid
    "Y01",  # Cholesterol hemisuccinate
    "EDT",  # {[-(BIS-CARBOXYMETHYL-AMINO)-ETHYL]-CARBOXYMETHYL-AMINO}-ACETIC ACID
    "P6G",
    "PG6",
    "MLI",  # Malonate ion
    "PGO",  # PROPANEDIOL
}
