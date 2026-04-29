"""
Tools for binders
"""

import logging
import os

import pandas as pd

from guild.constants.binders import (
    PDB_ID,
)
from guild.constants.guild import (
    IS_PDB,
    PROTEIN_ID,
)

logger = logging.getLogger(__name__)


def collect_known_binders(
    input_table,
    known_binders_file_path,
    min_mol_wt=250,
    max_mol_wt=450,
    chembl_version="chembl_35",
):
    """
    Get known binders for the input table.
    :param input_table: Input table.
    :param known_binders_file_path: Path to the known binders file.
    :param min_mol_wt: Minimum molecular weight for known binders.
    :param max_mol_wt: Maximum molecular weight for known binders.
    :param chembl_version: ChEMBL version to use for known binders.
    :return: Known binders.
    """
    if os.path.exists(known_binders_file_path):
        known_binders = pd.read_csv(known_binders_file_path)
    else:
        # Lazy import to avoid ChEMBL API connection at import time
        from guild.transformers.chembl import get_known_binders

        unique_pdb_ids = input_table[PROTEIN_ID].unique()
        known_binders = get_known_binders(
            unique_pdb_ids,
            subset_size=5,
            local=True,
            min_mol_wt=min_mol_wt,
            max_mol_wt=max_mol_wt,
            chembl_version=chembl_version,
        )
        known_binders.to_csv(known_binders_file_path, index=False)

    # Subset to PDB proteins with structures
    pdb_proteins = input_table.loc[input_table[IS_PDB] == 1].copy()
    if pdb_proteins.shape[0] == 0:
        logger.info("No PDB proteins detected")
        return pd.DataFrame()

    known_binders[PDB_ID] = known_binders[PDB_ID].str.lower()
    known_binders_pdb_proteins = known_binders.loc[
        known_binders[PDB_ID].isin(pdb_proteins[PROTEIN_ID])
    ]

    return known_binders_pdb_proteins
