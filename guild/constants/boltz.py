"""
Boltz constants
"""

"""
Files
"""
BOLTZ_YAML_FILE = "boltz.yaml"

"""
Columns
"""
BOLTZ_PAIR_CHAINS_IPTM = "pair_chains_iptm"

# Boltz-2's affinity head, which guild does not use for its primary
# ``boltz_score`` (that reads ipTM confidence instead — see
# guild/docking/boltz.py::boltz_guild_scoring). This field is read only as a
# free, same-quantity (log10(IC50/uM)) comparator for validating Nesso-1
# against — Boltz-2 already writes it on every run since
# generate_boltz_yaml/deploy_boltz request the affinity head regardless.
BOLTZ_AFFINITY_PRED_VALUE_FIELD = "affinity_pred_value"
