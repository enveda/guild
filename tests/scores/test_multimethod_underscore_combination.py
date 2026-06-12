"""
Regression test for the multi-method scoring bug where a protein_config_id
containing an underscore (e.g. ``6C97_P0``) caused:
  - KarmaDock rows to be keyed wrong (split("_") mis-parse) → they failed the
    cross-method merge and got dropped by dedup (one method per row), and
  - smiles / ligand_category to come out as "unknown".

The fix makes ``karmadock_guild_scoring`` recover the authoritative
(protein_config_id, ligand_id) from ``COMBINATIONS_TO_RUN_KEY`` instead of
string-splitting, plus a coalescing dedup and ligand_id-keyed category lookup.
"""

import pandas as pd

from guild.constants.bulk import (
    BATCH_FOLDER,
    COMBINATION_ID,
    COMBINATIONS_TO_RUN_KEY,
)
from guild.constants.guild import (
    KARMADOCK_FOLDER,
    KARMADOCK_SCORE,
    LIGAND_ID,
    PROTEIN_CONF_ID,
)
from guild.constants.karmadock import KARMADOCK_RESULTS_FOLDER
from guild.docking.karmadock import karmadock_guild_scoring

# protein_config_id with an underscore — the case that broke split("_")
PROTEIN = "6C97_P0"
LIGANDS = ["lig1", "drug_2", "pos_3"]  # ligand ids, one also containing "_"


def _write_karmadock_results(results_dir):
    """Emit a KarmaDock-style results CSV (pdb_id == combined combination id)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, lig in enumerate(LIGANDS):
        rows.append(
            {
                "pdb_id": f"{PROTEIN}_{lig}",
                "score": 10.0 + i,
                "RMSD": 1.0,
                "FF_RMSD": 1.0,
                "Aligned_RMSD": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "0.csv", index=False)


def _batch_dictionary(batch_folder):
    return {
        BATCH_FOLDER: str(batch_folder),
        COMBINATIONS_TO_RUN_KEY: [(PROTEIN, lig) for lig in LIGANDS],
    }


def test_karmadock_recovers_authoritative_keys(tmp_path):
    """KarmaDock must recover protein_config_id='6C97_P0' and the full ligand_id,
    not the split('_') mis-parse ('6C97', 'P0_lig1')."""
    results_dir = tmp_path / KARMADOCK_FOLDER / KARMADOCK_RESULTS_FOLDER
    _write_karmadock_results(results_dir)

    df = karmadock_guild_scoring(_batch_dictionary(tmp_path))

    assert set(df[PROTEIN_CONF_ID].unique()) == {PROTEIN}, (
        f"expected all rows under protein_config_id={PROTEIN!r}, "
        f"got {df[PROTEIN_CONF_ID].unique().tolist()}"
    )
    assert set(df[LIGAND_ID]) == set(LIGANDS)
    # combination id is preserved intact
    assert set(df[COMBINATION_ID]) == {f"{PROTEIN}_{lig}" for lig in LIGANDS}
    # the underscore-containing ligand id survived
    row = df[df[LIGAND_ID] == "drug_2"]
    assert len(row) == 1
    assert row.iloc[0][PROTEIN_CONF_ID] == PROTEIN


def test_karmadock_keys_merge_with_vina_keys(tmp_path):
    """KarmaDock's recovered keys must match how vina/gnina build theirs
    (protein_config_id + '_' + ligand_id), so an outer merge unites them into
    one row per combination instead of doubling."""
    results_dir = tmp_path / KARMADOCK_FOLDER / KARMADOCK_RESULTS_FOLDER
    _write_karmadock_results(results_dir)
    karma = karmadock_guild_scoring(_batch_dictionary(tmp_path))

    # vina-style frame, built the way vina_guild_scoring does
    vina = pd.DataFrame(
        {
            PROTEIN_CONF_ID: [PROTEIN] * len(LIGANDS),
            LIGAND_ID: LIGANDS,
            "vina_score": [-5.0, -6.0, -7.0],
        }
    )
    vina[COMBINATION_ID] = vina[PROTEIN_CONF_ID] + "_" + vina[LIGAND_ID]

    merged = pd.merge(
        vina,
        karma,
        on=[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID],
        how="outer",
    )

    # One row per combination, both method columns populated (no NaN).
    assert len(merged) == len(LIGANDS)
    assert merged["vina_score"].notna().all()
    assert merged[KARMADOCK_SCORE].notna().all()
