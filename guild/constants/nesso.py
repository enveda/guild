"""
Nesso constants

Nesso-1 (Valence Labs / Recursion, Apache-2.0, https://github.com/recursionpharma/nesso)
is a coarse-grained cofolding affinity model. Unlike Boltz-2 it takes no
protein structure, MSA, template, or pocket — only a protein sequence and a
ligand SMILES — and writes a single scalar ``affinity.json`` per complex.
"""

"""
Files
"""
NESSO_AFFINITY_FILE = "affinity.json"

"""
``affinity.json`` field names (verbatim from Nesso's output schema)
"""
# Ensemble mean, log10(IC50 / uM). Lower = stronger predicted binding.
NESSO_AFFINITY_PRED_VALUE_FIELD = "affinity_pred_value"
# Sigmoid binder/non-binder classifier probability (Hit-ID head), in [0, 1].
NESSO_AFFINITY_PROBABILITY_BINARY_FIELD = "affinity_probability_binary"
# Protein-ligand interface distogram entropy on the cropped interface — the
# paper's H_PL. Nesso's own training data is filtered at H_PL > 0.7, so this
# doubles as a usable confidence gate on individual predictions.
NESSO_ENTROPY_CROP_PL_FIELD = "entropy_crop_pl"
