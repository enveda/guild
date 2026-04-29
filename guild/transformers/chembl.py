import logging
import math
import os
import random
import time
from typing import Tuple

import chembl_downloader
import pandas as pd
from tqdm import tqdm
from unipressed import IdMappingClient

from guild.constants.chembl import (
    ACTIVITY,
    BINDER_DIR,
    CHEMBL_ID,
    CHEMBL_VERSION,
    DECOYS_DIR,
    MW,
    PCHEMBL_VALUE,
    PDB_ID,
    PROTEIN_SYMBOL,
    SMILES,
    UNIPROT_ID,
)
from guild.constants.general import RANDOM_SEED
from guild.constants.system import WORKING_DIR_PATH

random.seed(RANDOM_SEED)

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'

BINDER_DIR_PATH = f"{WORKING_DIR_PATH}/guild/support/{BINDER_DIR}"
DECOYS_DIR_PATH = f"{WORKING_DIR_PATH}/guild/support/{DECOYS_DIR}"

os.makedirs(BINDER_DIR_PATH, exist_ok=True)
os.makedirs(DECOYS_DIR_PATH, exist_ok=True)


def get_gene_identifiers(pdb_list: list) -> pd.DataFrame:
    """
    Mapping the PDB IDs to UniProt BridgeDB.

    :param pdb_list: List of PDB IDs
    :return: DataFrame with the mapping of PDB IDs to gene identifiers
    """

    max_retries = 5
    retries = 0
    identifier_data = []
    while retries < max_retries:
        request = IdMappingClient.submit(source="PDB", dest="UniProtKB", ids=set(pdb_list))
        request.each_result()
        try:
            for mapping in request.each_result():
                identifier_data.append([mapping["from"], mapping["to"]])
            return pd.DataFrame(identifier_data, columns=[PDB_ID, UNIPROT_ID])
        except Exception as e:
            logger.error(f"Error fetching UniProt IDs: {e}")
            time.sleep(15)
            continue
        retries += 1
        if retries >= max_retries:
            logger.error("Max retries reached. Returning empty DataFrame.")
            identifier_data = pd.DataFrame(columns=["pdb_id", "uniprot_id"])
            identifier_data["pdb_id"] = pdb_list
            identifier_data["uniprot_id"] = None
            return identifier_data


def convert_pchembl_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert standard values to pchembl values.

    :param df: DataFrame with assay results
    """
    pchembl_values = []

    for val, unit, pchembl_val in tqdm(
        df[["standard_value", "standard_units", "pchembl_value"]].values
    ):
        val = str(val)
        if pd.isna(pchembl_val):
            if pd.isna(val) or val == "0.0":
                pchembl_values.append(0)
            elif pd.notna(val) and val.startswith("-"):  # certain neagtive vlaues found
                pchembl_values.append(0)
            elif unit == "nM" and pd.notna(val):
                pchembl_values.append(round(9 - math.log10(float(val))))
            elif unit == "uM" and pd.notna(val):
                pchembl_values.append(round(6 - math.log10(float(val))))
            elif unit == "mM" and pd.notna(val):
                pchembl_values.append(round(3 - math.log10(float(val))))
            else:
                pchembl_values.append(0)
        elif pchembl_val == "None" or pd.isna(pchembl_val):
            pchembl_values.append(0)
        else:
            pchembl_values.append(float(pchembl_val))

    df["pchembl_value"] = pchembl_values
    return df


def get_decoys_locally(min_mol_wt: int, max_mol_wt: int, version: str = None) -> pd.DataFrame:
    """Get compounds from local ChEMBL database to be part of decoys.

    :param min_mol_wt: Minimum molecular weight for compounds
    :param max_mol_wt: Maximum molecular weight for compounds
    """
    if version is None:
        _, version = chembl_downloader.download_extract_sqlite(return_version=True)
        vname = str(version).split("/")[-1].split(".")[0]
    else:
        version = version
        vname = version

    logger.warning(f"Working on ChEMBL version {version}")

    sql = """
    SELECT
            MOLECULE_DICTIONARY.molregno as chembl_id,
            COMPOUND_PROPERTIES.full_mwt as mw,
            COMPOUND_STRUCTURES.canonical_smiles as canonical_smiles,
            COMPOUND_STRUCTURES.standard_inchi as standard_inchi,
            COMPOUND_STRUCTURES.standard_inchi_key as standard_inchi_key
        FROM MOLECULE_DICTIONARY
        JOIN COMPOUND_STRUCTURES on COMPOUND_STRUCTURES.molregno == MOLECULE_DICTIONARY.molregno
        JOIN COMPOUND_PROPERTIES on COMPOUND_PROPERTIES.molregno == MOLECULE_DICTIONARY.molregno
    """

    DECOY_PATH = f"{DECOYS_DIR_PATH}/{vname}_chemreps.txt"

    if not os.path.exists(DECOY_PATH):
        compound_df = chembl_downloader.query(sql=sql)
        compound_df = compound_df[compound_df[MW] <= max_mol_wt]
        compound_df = compound_df[compound_df[MW] >= min_mol_wt]
        compound_df[CHEMBL_VERSION] = vname
        compound_df[CHEMBL_ID] = "CHEMBL" + compound_df[CHEMBL_ID].astype(str)
        compound_df.drop(columns=[MW], inplace=True)
        compound_df.to_csv(DECOY_PATH, sep="\t", index=False)
    else:
        compound_df = pd.read_csv(DECOY_PATH, sep="\t")

    return compound_df


def get_assays_locally(
    protein_list: list, min_mol_wt: int, max_mol_wt: int, version: str = None
) -> pd.DataFrame:
    """Get assay data from local ChEMBL database.

    :param protein_list: List of protein identifiers
    :param mol_wt_threshold: Molecular weight threshold for compounds
    :param version: ChEMBL version
    """
    if version is None:
        _, version = chembl_downloader.download_extract_sqlite(return_version=True)
        vname = str(version).split("/")[-1].split(".")[0]
    else:
        version = version
        vname = version
    logger.warning(f"Working on ChEMBL version {version}")

    sql = """
    SELECT
            MOLECULE_DICTIONARY.molregno as chembl_id,
            COMPOUND_PROPERTIES.full_mwt as mw,
            COMPOUND_STRUCTURES.canonical_smiles as smiles,
            ACTIVITIES.standard_relation,
            ACTIVITIES.standard_type,
            ACTIVITIES.standard_value,
            ACTIVITIES.standard_units,
            ACTIVITIES.pchembl_value,
            ASSAYS.assay_organism as ORGANISM,
            ASSAYS.assay_tax_id,
            ASSAYS.chembl_id as assay_id,
            COMPONENT_SEQUENCES.accession as protein_symbol
        FROM MOLECULE_DICTIONARY
        JOIN ACTIVITIES ON MOLECULE_DICTIONARY.molregno == ACTIVITIES.molregno
        JOIN ASSAYS ON ACTIVITIES.assay_id == ASSAYS.assay_id
        JOIN TARGET_DICTIONARY on ASSAYS.tid == TARGET_DICTIONARY.tid
        JOIN TARGET_COMPONENTS on TARGET_DICTIONARY.tid == TARGET_COMPONENTS.tid
        JOIN COMPONENT_SEQUENCES on TARGET_COMPONENTS.component_id == COMPONENT_SEQUENCES.component_id
        JOIN COMPOUND_STRUCTURES on COMPOUND_STRUCTURES.molregno == MOLECULE_DICTIONARY.molregno
        JOIN COMPOUND_PROPERTIES on COMPOUND_PROPERTIES.molregno == MOLECULE_DICTIONARY.molregno
        WHERE
            ASSAYS.assay_type in ('B', 'F')
            and ACTIVITIES.standard_value is not null
            and ACTIVITIES.standard_relation in ('=', '<', '<=')
            and ACTIVITIES.standard_type in ('AC50', 'EC50', 'IC50', 'Ki', 'MIC', 'GI50', 'TG50', 'Km', 'Kd', 'CC50', 'LC50')
            and ASSAYS.assay_organism == "Homo sapiens"
    """
    CHEMBL_ASSAY_PATH = f"{BINDER_DIR_PATH}/chembl_assay_{vname}.pq"

    if not os.path.exists(CHEMBL_ASSAY_PATH):
        assay_df = chembl_downloader.query(sql=sql)
        assay_df.to_parquet(CHEMBL_ASSAY_PATH)
    else:
        assay_df = pd.read_parquet(CHEMBL_ASSAY_PATH)

    assay_subset_df = assay_df[assay_df[PROTEIN_SYMBOL].isin(protein_list)]  # uniprot ids
    assay_subset_df = assay_subset_df[assay_subset_df[MW] <= max_mol_wt]
    assay_subset_df = assay_subset_df[assay_subset_df[MW] >= min_mol_wt]
    assay_subset_df[CHEMBL_VERSION] = vname
    return assay_subset_df


def get_known_binders(
    protein_list: list,
    subset_size: int = 10,
    local: bool = True,
    min_mol_wt: int = 250,
    max_mol_wt: int = 450,
    chembl_version: str = "chembl_35",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get known binders for proteins from ChEMBL.
    :param protein_list: List of PDB identifiers
    :param subset_size: Size of the compounds to extract from each set, default is set to 10.
    :param local: Use local ChEMBL database. The database is downloaded and extracted from the ChEMBL website.
    :param mol_wt_cutoff: Cut-off value for molecular weight. By deault, all small molecules are extracted.
    """
    identifiers_df = get_gene_identifiers(protein_list)
    CHEMBL_BINDER_PATH = f"{BINDER_DIR_PATH}/chembl_binders_{chembl_version}.tsv"

    if os.path.exists(CHEMBL_BINDER_PATH):
        assay_df_filtered = pd.read_csv(CHEMBL_BINDER_PATH, sep="\t")
    else:
        if local:
            uniprot_gene_list = set(identifiers_df[UNIPROT_ID].tolist())

            assay_df = get_assays_locally(
                uniprot_gene_list, min_mol_wt=min_mol_wt, max_mol_wt=max_mol_wt
            )
            assay_df = assay_df[assay_df[PCHEMBL_VALUE] > 0]
            assay_df[ACTIVITY] = assay_df[PCHEMBL_VALUE].apply(
                lambda x: "weak-binder" if x < 6 else "strong-binder"
            )
        # else:  # TODO: Implement Uniprot to HGNC
        #     gene_subset = identifiers_df[identifiers_df["target.source"] == "HGNC"]
        #     hgnc_gene_list = set(gene_subset["target"].tolist())

        #     id2pdb = gene_subset.set_index("target")["identifier"].to_dict()
        #     assay_df = get_assays_from_api(
        #         protein_list=hgnc_gene_list, organism="Homo sapiens"
        #     )
        #     assay_df = convert_pchembl_vals(assay_df)
        #     assay_df["smiles"] = assay_df["chembl_id"].apply(assay_df)

        assay_df_filtered = assay_df[
            [
                PROTEIN_SYMBOL,
                ACTIVITY,
                CHEMBL_ID,
                SMILES,
                PCHEMBL_VALUE,
            ]
        ]
        assay_df_filtered[CHEMBL_ID] = "CHEMBL" + assay_df_filtered[CHEMBL_ID].astype(str)
        assay_df_filtered.rename(columns={PROTEIN_SYMBOL: UNIPROT_ID}, inplace=True)

        assay_df_filtered = pd.merge(assay_df_filtered, identifiers_df, how="outer", on=UNIPROT_ID)
        assay_df_filtered.to_csv(CHEMBL_BINDER_PATH, sep="\t", index=False)

    strong_binders_list, weak_binders_list = [], []
    for current_protein in list(assay_df_filtered[PDB_ID].unique()):
        current_subset = assay_df_filtered.loc[assay_df_filtered[PDB_ID] == current_protein]
        strong_binders_subset = current_subset.loc[current_subset[ACTIVITY] == "strong-binder"]
        weak_binders_subset = current_subset.loc[current_subset[ACTIVITY] == "weak-binder"]
        if not strong_binders_subset.empty:
            strong_binders_list.append(
                strong_binders_subset.sample(
                    n=subset_size, random_state=RANDOM_SEED, replace=True
                ).drop_duplicates()
            )
        if not weak_binders_subset.empty:
            weak_binders_list.append(
                weak_binders_subset.sample(
                    n=subset_size, random_state=RANDOM_SEED, replace=True
                ).drop_duplicates()
            )
    try:
        weak_binders = pd.concat(weak_binders_list, ignore_index=True)
    except Exception as e:
        logger.error(f"Error concatenating weak binders: {e}, proceeding with empty dataframe")
        weak_binders = pd.DataFrame()

    try:
        strong_binders = pd.concat(strong_binders_list, ignore_index=True)
    except Exception as e:
        logger.error(f"Error concatenating strong binders: {e}, proceeding with empty dataframe")
        strong_binders = pd.DataFrame()

    if weak_binders.empty and strong_binders.empty:
        logger.info("No known binders found")
        return False
    known_binders = pd.concat([weak_binders, strong_binders], ignore_index=True)

    return known_binders
