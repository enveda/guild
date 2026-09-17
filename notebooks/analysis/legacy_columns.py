"""
Shared column-rename map for the Figure 2 / Figure 3 notebooks.

The results files behind these figures were written by three naming
generations (see notebooks/analysis/README.md):

    generation   raw Vina score        Vina rank percentile      combined
    published    autodock_vina_score   guild_autodock_vina_score global_guild_score
    case study   vina_score            dockwizard_vina_score     global_dockwizard_score
    current      vina_score            rp_vina_score             global_rp_score

plus a fourth, `guild_vina_score` / `global_guild_score`, used by the known-binders
case-study run specifically, and the `drrp_*` names used by earlier drafts of the
rank-percentile column itself. `LEGACY_RENAME` maps every generation onto the
current `rp_*` / `global_rp_score` names so a notebook can `df.rename(columns=...)`
once at load time and use current names everywhere after. `rename` only touches
columns that are actually present, so entries for a generation a given file does
not use are harmless no-ops.
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
