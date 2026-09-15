# Legacy results fixture

`single_protein_per_family_rp_scores.txt` predates the current output format and **must not be
used as a reference for what Guild produces today**. It is kept only because existing material
points at it.

It differs from current output in two ways:

**Orientation is inverted.** In this file the best-scoring ligand's rank percentile is near `1`.
Current Guild output — and the published case-study results — put the best ligand near `0`
(`rp_score = rank / n`, rank 1 = best). Reading this file as though it used the current
convention inverts every ranking derived from it.

**Column names are stale.**

| In this file                | Today                 |
|-----------------------------|-----------------------|
| `autodock_vina_score`       | `vina_score`          |
| `rank_autodock_vina_score`  | `rank_vina_score`     |
| `guild_autodock_vina_score` | `rp_vina_score`       |
| `guild_karmadock_score`     | `rp_karmadock_score`  |
| `global_guild_score`        | `global_rp_score`     |

For the current convention see the "Rank-percentile orientation" note in the top-level
`README.md`, and `guild/tools/scores.py`.
