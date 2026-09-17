"""
Column-rename map covering the naming generations used across the Figure 2 / Figure 3
result files (see notebooks/analysis/README.md). `df.rename(columns=LEGACY_RENAME)`
maps any of them onto the current `rp_*` / `global_rp_score` names; unused entries are
no-ops since `rename` only touches columns that exist.
"""

LEGACY_RENAME = {
    "drrp_vina_score": "rp_vina_score",
    "drrp_diffdock_score": "rp_diffdock_score",
    "drrp_boltz_score": "rp_boltz_score",
    "drrp_karmadock_score": "rp_karmadock_score",
    "global_drrp_score": "global_rp_score",
    "dockwizard_vina_score": "rp_vina_score",
    "global_dockwizard_score": "global_rp_score",
    "guild_vina_score": "rp_vina_score",
    "guild_diffdock_score": "rp_diffdock_score",
    "guild_boltz_score": "rp_boltz_score",
    "guild_karmadock_score": "rp_karmadock_score",
    "guild_autodock_vina_score": "rp_vina_score",
    "global_guild_score": "global_rp_score",
}
