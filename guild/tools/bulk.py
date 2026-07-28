"""
Tools for bulk
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from guild.constants.binders import (
    ACTIVITY,
    CANONICAL_SMILES,
    CHEMBL_ID,
    PDB_ID,
)
from guild.constants.bulk import (
    BATCH_FOLDER,
    BULK_TEMPLATE_DICTIONARY,
    # Batch dictionary keys
    COMBINATIONS_TABLE_KEY,
    COMBINATIONS_TO_RUN_KEY,
    INPUT_COMBINATIONS_KEY,
    METHODS_TO_SCORE_DICTIONARY_KEY,
    METHODS_TO_SORT_DICTIONARY_KEY,
    PRE_EXISTING_RP_SCORES_KEY,
    PREVIOUS_COMBINATIONS_DF_KEY,
    PREVIOUS_RP_SCORES_KEY,
    RANKS_DICTIONARY,
    RANKS_LIST_KEY,
    SCORES_DIRECTION_DICTIONARY,
    SCORES_TO_USE_DICTIONARY,
    SCORES_TO_USE_KEY,
    SMILES_NAMES_DICTIONARY_KEY,
    SMILES_TYPE_DICTIONARY_KEY,
    UNIQUE_PROTEIN_IDS_KEY,
)
from guild.constants.decoys import DECOYS_CATEGORY
from guild.constants.diffdock import DIFFDOCK_RESULTS_FOLDER
from guild.constants.guild import (
    BOLTZ_FOLDER,
    COVALENT_REC_ATOM,
    IS_PDB,
    LIGAND_CATEGORY,
    LIGAND_ID,
    ORIGINAL_LIGAND,
    ORIGINAL_LIGAND_CHAIN,
    PROTEIN_CHAIN,
    PROTEIN_CONF_ID,
    PROTEIN_ID,
    PROTEIN_PATH,
    RP_SCORES_COLUMNS,
    SMILES,
    VINA_FOLDER,
)
from guild.tools.binders import collect_known_binders
from guild.tools.preparation import _normalize_chain_list
from guild.transformers.converters import cif_to_pdb, sdf_to_pdb
from guild.transformers.pdb import (
    add_covalent_conect,
    build_complex_pdb,
    relabel_ligand_chain_in_pdb,
)

logger = logging.getLogger(__name__)


def kickstart_batch_dictionary(current_batch_table, batch_folder):
    """
    Kickstart the batch dictionary with the necessary information.

    :param current_batch_table: Table containing the combinations to run.
    :param batch_folder: Path to the batch folder.
    """

    batch_dictionary = BULK_TEMPLATE_DICTIONARY.copy()

    batch_dictionary[BATCH_FOLDER] = batch_folder
    batch_dictionary[COMBINATIONS_TABLE_KEY] = current_batch_table.copy()

    batch_dictionary[UNIQUE_PROTEIN_IDS_KEY] = list(current_batch_table[PROTEIN_ID].unique())
    batch_dictionary[INPUT_COMBINATIONS_KEY] = list(
        [(i, j) for i, j in current_batch_table[[PROTEIN_CONF_ID, LIGAND_ID]].values]
    )
    batch_dictionary[SMILES_NAMES_DICTIONARY_KEY] = dict(
        zip(current_batch_table[LIGAND_ID], current_batch_table[SMILES], strict=True)
    )
    batch_dictionary[SMILES_TYPE_DICTIONARY_KEY] = dict(
        zip(
            current_batch_table[SMILES],
            current_batch_table[LIGAND_CATEGORY],
            strict=True,
        )
    )
    # batch_dictionary[
    #    PROTEINS_FOLDER_KEY
    # ] = proteins_folder

    batch_dictionary[PRE_EXISTING_RP_SCORES_KEY] = None
    return batch_dictionary


def identify_previously_ran_combinations(current_batch, batch_dictionary):
    """
    Access the database of previously ran combinations and identify the new combinations to run.
    :param current_batch: Name of the batch to identify the previously ran combinations for.
    :param batch_dictionary: Dictionary containing the batch information.
    :return: Batch dictionary with the previously ran combinations identified.
    """

    batch_dictionary[PREVIOUS_RP_SCORES_KEY] = pd.DataFrame(columns=RP_SCORES_COLUMNS)

    pre_existing_combinations = []
    batch_dictionary[COMBINATIONS_TO_RUN_KEY] = []

    for current_combination in tqdm(
        batch_dictionary[INPUT_COMBINATIONS_KEY],
        desc=f"Identifying pre-existing combinations for batch {current_batch}",
    ):
        current_smiles = batch_dictionary[SMILES_NAMES_DICTIONARY_KEY][current_combination[1]]
        current_protein = current_combination[0]
        current_subset = batch_dictionary[PREVIOUS_RP_SCORES_KEY][
            (batch_dictionary[PREVIOUS_RP_SCORES_KEY][PROTEIN_CONF_ID] == current_protein)
            & (batch_dictionary[PREVIOUS_RP_SCORES_KEY][SMILES] == current_smiles)
        ].copy()
        if current_subset.shape[0] == 0:
            batch_dictionary[COMBINATIONS_TO_RUN_KEY].append(current_combination)

        else:
            pre_existing_combinations.append(current_subset)

    logger.info(
        f"Number of combinations to run: {len(batch_dictionary[COMBINATIONS_TO_RUN_KEY])}\nNumber of pre-existing combinations: {len(pre_existing_combinations)}"
    )

    if len(pre_existing_combinations) > 0:
        batch_dictionary[PREVIOUS_COMBINATIONS_DF_KEY] = pd.concat(
            pre_existing_combinations, axis=0
        )
    else:
        batch_dictionary[PREVIOUS_COMBINATIONS_DF_KEY] = None
    return batch_dictionary


def available_methods_preparation(batch_dictionary, methods_to_run):
    """
    Depending on the input methods determine the available rankings as well as which steps to run.

    :param batch_dictionary: Dictionary containing the batch information.
    :param methods_to_run: List of methods to run.
    :return: Batch dictionary with the available methods prepared.
    """

    batch_dictionary[METHODS_TO_SCORE_DICTIONARY_KEY] = SCORES_TO_USE_DICTIONARY
    batch_dictionary[METHODS_TO_SORT_DICTIONARY_KEY] = SCORES_DIRECTION_DICTIONARY

    batch_dictionary[SCORES_TO_USE_KEY] = []
    batch_dictionary[RANKS_LIST_KEY] = []

    for current_method in methods_to_run:
        batch_dictionary[RANKS_LIST_KEY].append(RANKS_DICTIONARY[current_method])
        batch_dictionary[SCORES_TO_USE_KEY] += SCORES_TO_USE_DICTIONARY[current_method]
    return batch_dictionary


def extend_all_combinations_table_with_known_binders(
    input_table, known_binders_file_path, min_mol_wt, max_mol_wt, chembl_version
):
    """
    Extend the batch dictionary with the known binders for the input proteins.

    :param input_table: Input table.
    :param known_binders_file_path: Path to the known binders file.
    :param min_mol_wt: Minimum molecular weight for known binders.
    :param max_mol_wt: Maximum molecular weight for known binders.
    :param chembl_version: ChEMBL version to use for known binders.
    :return: Batch dictionary with the known binders extended.
    """
    known_binders = collect_known_binders(
        input_table=input_table,
        known_binders_file_path=known_binders_file_path,
        min_mol_wt=min_mol_wt,
        max_mol_wt=max_mol_wt,
        chembl_version=chembl_version,
    )
    if known_binders.empty:
        logger.info("No known binders found")
        return pd.DataFrame()

    subset_df = input_table[
        [
            PROTEIN_ID,
            PROTEIN_CONF_ID,
            PROTEIN_CHAIN,
            PROTEIN_PATH,
            ORIGINAL_LIGAND,
            ORIGINAL_LIGAND_CHAIN,
            IS_PDB,
        ]
    ].drop_duplicates()

    # Subset to the necessary columns
    binder_subset = known_binders[
        [
            PDB_ID,
            CHEMBL_ID,
            SMILES,
            ACTIVITY,
        ]
    ]
    binder_subset.rename(
        columns={
            PDB_ID: PROTEIN_ID,
            CHEMBL_ID: LIGAND_ID,
            SMILES: SMILES,
            ACTIVITY: LIGAND_CATEGORY,
        },
        inplace=True,
    )

    return pd.merge(
        subset_df,
        binder_subset,
        on=PROTEIN_ID,
        how="inner",
    )[input_table.columns]


def extend_all_combinations_table_with_decoys(input_table, decoys_file_path):
    """
    Extend the all combinations table with the decoy dataset.
    When under use all the input proteins will be docked with the decoy dataset.
    :param input_table: Input table.
    :param decoys_file_path: Path to the decoy file.
    :return: All combinations table with the decoy dataset extended.
    """
    decoy_list = pd.read_csv(
        decoys_file_path,
        header=0,
        usecols=[CHEMBL_ID, CANONICAL_SMILES],
        sep="\t",
    )

    rows_to_add = []
    for chembl_id, current_smiles in tqdm(decoy_list.values, desc="Processing decoy dataset"):
        for current_protein, current_protein_configuration in zip(
            input_table[PROTEIN_ID].unique(),
            input_table[PROTEIN_CONF_ID].unique(),
            strict=True,
        ):
            current_subset = input_table[
                input_table[PROTEIN_CONF_ID] == current_protein_configuration
            ]
            current_protein_chain = current_subset[PROTEIN_CHAIN].values[0]
            current_protein_path = current_subset[PROTEIN_PATH].values[0]
            current_original_ligand = current_subset[ORIGINAL_LIGAND].values[0]
            current_original_ligand_chain = current_subset[ORIGINAL_LIGAND_CHAIN].values[0]

            rows_to_add.append(
                {
                    PROTEIN_ID: current_protein,
                    PROTEIN_CONF_ID: current_protein_configuration,
                    PROTEIN_CHAIN: current_protein_chain,
                    PROTEIN_PATH: current_protein_path,
                    ORIGINAL_LIGAND: current_original_ligand,
                    ORIGINAL_LIGAND_CHAIN: current_original_ligand_chain,
                    LIGAND_ID: chembl_id,
                    SMILES: current_smiles,
                    LIGAND_CATEGORY: DECOYS_CATEGORY,
                    IS_PDB: 0,
                }
            )
    return pd.DataFrame(rows_to_add)


def _generate_pdbqt_complex_pdbs(batch_dictionary, output_folder_name: str, method_label: str):
    """
    Merge protein PDB with docked-pose PDBQT into a complex PDB, for any
    docking method whose output is a multi-pose PDBQT in
    ``{BATCH_FOLDER}/{output_folder_name}/{protein}_{ligand}.pdbqt``.

    Used by both Vina and gnina, which share the same PDBQT output format.

    :param batch_dictionary: Standard bulk batch dictionary.
    :param output_folder_name: Per-method docking-output folder (``VINA_FOLDER``,
        ``GNINA_FOLDER``).
    :param method_label: Human-readable method name for log messages.
    """
    batch_folder = batch_dictionary[BATCH_FOLDER]
    method_folder = f"{batch_folder}/{output_folder_name}"
    proteins_folder = f"{batch_folder}/proteins"

    combinations_df = batch_dictionary[COMBINATIONS_TABLE_KEY]

    complexes_created = 0
    complexes_failed = 0

    for _, row in combinations_df.iterrows():
        protein_conf_id = row[PROTEIN_CONF_ID]
        ligand_id = row[LIGAND_ID]

        ligand_pdbqt = f"{method_folder}/{protein_conf_id}_{ligand_id}.pdbqt"
        ligand_sdf = f"{method_folder}/{protein_conf_id}_{ligand_id}.sdf"
        complex_pdb = f"{method_folder}/{protein_conf_id}_{ligand_id}_complex.pdb"

        if os.path.exists(ligand_sdf):
            ligand_poses_file = ligand_sdf
            ligand_is_pdbqt = False
        elif os.path.exists(ligand_pdbqt):
            ligand_poses_file = ligand_pdbqt
            ligand_is_pdbqt = True
        else:
            continue

        if os.path.exists(complex_pdb):
            continue

        protein_pdb_candidates = [
            f"{proteins_folder}/{protein_conf_id}_single_chain_clean.pdb",
            f"{proteins_folder}/{protein_conf_id}_clean.pdb",
            f"{proteins_folder}/{protein_conf_id}.pdb",
        ]

        protein_pdb = None
        for candidate in protein_pdb_candidates:
            if os.path.exists(candidate):
                protein_pdb = candidate
                break

        if protein_pdb is None:
            logger.warning(f"Protein PDB not found for {protein_conf_id} in {proteins_folder}")
            complexes_failed += 1
            continue

        try:
            build_complex_pdb(
                protein_pdb=protein_pdb,
                ligand_file=ligand_poses_file,
                out_complex_pdb=complex_pdb,
                ligand_is_pdbqt=ligand_is_pdbqt,
                ligand_resname="LIG",
                ligand_chain="Z",
                ligand_resseq=1,
                insert_ter_between=True,
            )
            complexes_created += 1
        except Exception as e:
            logger.warning(f"Failed to create complex for {protein_conf_id}_{ligand_id}: {e}")
            complexes_failed += 1
            continue

        # For gnina covalent runs, append CONECT records so the covalent bond
        # is visible in PyMOL and PLIP.  The spec comes from the per-row
        # combinations-CSV column; absent / NaN rows are silently skipped.
        cov_spec = row[COVALENT_REC_ATOM] if COVALENT_REC_ATOM in row.index else None
        if cov_spec is not None and not (isinstance(cov_spec, float) and pd.isna(cov_spec)):
            cov_spec = str(cov_spec).strip()
            if cov_spec:
                try:
                    add_covalent_conect(complex_pdb, cov_spec)
                except Exception as e:
                    logger.warning(
                        f"covalent CONECT: could not add bond for "
                        f"{protein_conf_id}_{ligand_id}: {e}"
                    )

    logger.info(
        f"{method_label} complex PDB generation complete: "
        f"{complexes_created} created, {complexes_failed} failed"
    )


def generate_vina_complex_pdbs(batch_dictionary):
    """
    Generate complex PDB files for all Vina docking results in a batch.
    Merges protein PDB with docked ligand PDBQT into a single complex PDB.

    :param batch_dictionary: Dictionary containing batch information including
                             BATCH_FOLDER, COMBINATIONS_TABLE_KEY, etc.
    """
    _generate_pdbqt_complex_pdbs(batch_dictionary, VINA_FOLDER, "Vina")


def generate_gnina_complex_pdbs(batch_dictionary):
    """
    Generate complex PDB files for all gnina docking results in a batch.
    Uses the same PDBQT output format as Vina.
    """
    from guild.constants.guild import GNINA_FOLDER

    _generate_pdbqt_complex_pdbs(batch_dictionary, GNINA_FOLDER, "gnina")


def generate_diffdock_complex_pdbs(batch_dictionary):
    """
    Generate PLIP-ready complex PDB files for DiffDock docking results.

    For each combination, finds the highest-confidence DiffDock SDF pose,
    converts it to PDB, extracts the matching chain from the raw protein PDB
    (preserving the crystal coordinate frame), and merges them into a single
    complex PDB suitable for PLIP analysis.

    :param batch_dictionary: Dictionary containing batch information including
                             BATCH_FOLDER, COMBINATIONS_TABLE_KEY, etc.
    """
    from guild.constants.guild import DIFFDOCK_FOLDER

    batch_folder = batch_dictionary[BATCH_FOLDER]
    diffdock_results = f"{batch_folder}/{DIFFDOCK_FOLDER}/{DIFFDOCK_RESULTS_FOLDER}"
    proteins_folder = f"{batch_folder}/proteins"
    diffdock_folder = f"{batch_folder}/{DIFFDOCK_FOLDER}"

    combinations_df = batch_dictionary[COMBINATIONS_TABLE_KEY]

    complexes_created = 0
    complexes_failed = 0

    for _, row in combinations_df.iterrows():
        protein_conf_id = row[PROTEIN_CONF_ID]
        ligand_id = row[LIGAND_ID]
        run_id = f"{protein_conf_id}_{ligand_id}"

        complex_pdb = f"{diffdock_folder}/{run_id}_complex.pdb"

        # Skip if complex already exists
        if os.path.exists(complex_pdb):
            complexes_created += 1
            continue

        # --- Find best DiffDock SDF ---
        combo_dir = os.path.join(diffdock_results, run_id)
        if not os.path.isdir(combo_dir):
            logger.warning(f"DiffDock results folder not found: {combo_dir}")
            complexes_failed += 1
            continue

        sdf_scores = {}
        for fname in os.listdir(combo_dir):
            if "_confidence" in fname and fname.endswith(".sdf"):
                try:
                    score = float(fname.split("_confidence")[1].replace(".sdf", ""))
                    sdf_scores[fname] = score
                except ValueError:
                    continue

        if not sdf_scores:
            logger.warning(f"No DiffDock SDF files found in {combo_dir}")
            complexes_failed += 1
            continue

        best_fname = max(sdf_scores, key=sdf_scores.get)
        best_sdf = os.path.join(combo_dir, best_fname)

        # --- Convert SDF → PDB (preserving 3D coordinates) ---
        ligand_pdb = os.path.join(diffdock_folder, f"{run_id}_ligand.pdb")
        try:
            sdf_to_pdb(best_sdf, ligand_pdb)
        except Exception as e:
            logger.warning(f"SDF→PDB conversion failed for {run_id}: {e}")
            complexes_failed += 1
            continue

        # --- Get protein PDB in raw coordinate frame ---
        # DiffDock outputs are in the raw PDB frame, so we must use the raw PDB.
        # Extract the single chain to match what DiffDock used.
        raw_pdb = f"{proteins_folder}/{protein_conf_id}_raw.pdb"
        if not os.path.exists(raw_pdb):
            logger.warning(f"Raw PDB not found for {protein_conf_id}: {raw_pdb}")
            complexes_failed += 1
            continue

        # Extract chain(s) from protein_conf_id. Single chain ("8gut-R-KO8-R" →
        # "R") or a comma-joined set for a multi-chain pocket ("8gut-A,B-..." →
        # ["A", "B"]); every listed chain is kept so the receptor stays intact.
        parts = protein_conf_id.split("-")
        chain_ids = _normalize_chain_list(parts[1]) if len(parts) >= 2 else ["A"]

        chain_pdb = os.path.join(diffdock_folder, f"{protein_conf_id}_chain.pdb")
        if not os.path.exists(chain_pdb):
            kept = 0
            with open(raw_pdb) as fin, open(chain_pdb, "w") as fout:
                for line in fin:
                    if line.startswith("ATOM") and len(line) > 21 and line[21] in chain_ids:
                        fout.write(line)
                        kept += 1
                fout.write("END\n")
            if kept == 0:
                logger.warning(f"No ATOM records for chain(s) {chain_ids} in {raw_pdb}")
                complexes_failed += 1
                continue

        # --- Build complex PDB ---
        try:
            build_complex_pdb(
                protein_pdb=chain_pdb,
                ligand_file=ligand_pdb,
                out_complex_pdb=complex_pdb,
                ligand_is_pdbqt=False,
                ligand_resname="LIG",
                ligand_chain="Z",
                ligand_resseq=1,
                insert_ter_between=True,
            )
            complexes_created += 1
        except Exception as e:
            logger.warning(f"Failed to create DiffDock complex for {run_id}: {e}")
            complexes_failed += 1

    logger.info(
        f"DiffDock complex PDB generation complete: {complexes_created} created, {complexes_failed} failed"
    )


def generate_boltz_complex_pdbs(batch_dictionary):
    """
    Generate PLIP-ready complex PDB files from Boltz CIF output.
    Boltz writes the full predicted complex (protein + ligand) as a single CIF file.
    This function converts each CIF to PDB and relabels the ligand chain so that
    PLIP can locate the binding site (expects HETATM LIG Z 1).

    The ligand chain "L" is the value hardcoded in bulk.py when calling generate_boltz_yaml.
    """
    import tempfile

    batch_folder = batch_dictionary[BATCH_FOLDER]
    boltz_folder = f"{batch_folder}/{BOLTZ_FOLDER}"
    combinations_df = batch_dictionary[COMBINATIONS_TABLE_KEY]

    complexes_created = 0
    complexes_failed = 0

    for _, row in combinations_df.iterrows():
        protein_conf_id = row[PROTEIN_CONF_ID]
        ligand_id = row[LIGAND_ID]
        run_id = f"{protein_conf_id}_{ligand_id}"

        cif_file = (
            f"{boltz_folder}/boltz_results_{run_id}_boltz"
            f"/predictions/{run_id}_boltz/{run_id}_boltz_model_0.cif"
        )
        complex_pdb = f"{boltz_folder}/{run_id}_complex.pdb"

        if not os.path.exists(cif_file):
            logger.warning(f"Boltz CIF not found for {run_id}: {cif_file}")
            complexes_failed += 1
            continue

        if os.path.exists(complex_pdb):
            complexes_created += 1
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp_pdb = tmp.name
            cif_to_pdb(cif_file, tmp_pdb)
            relabel_ligand_chain_in_pdb(
                input_pdb=tmp_pdb,
                output_pdb=complex_pdb,
                ligand_chain_id="L",  # always "L" — hardcoded in bulk.py generate_boltz_yaml call
            )
            os.unlink(tmp_pdb)
            complexes_created += 1
        except Exception as e:
            logger.warning(f"Failed to create Boltz complex PDB for {run_id}: {e}")
            complexes_failed += 1

    logger.info(
        f"Boltz complex PDB generation complete: {complexes_created} created, {complexes_failed} failed"
    )
