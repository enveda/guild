import logging
import math
import multiprocessing as mp
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)

import pandas as pd
from tqdm import tqdm

from guild.analysis.plip import analyze_batch_interactions
from guild.constants.bulk import (
    ALL_COMBINATIONS_FILE,
    # Batch dictionary keys
    BATCH_FOLDER,
    BATCH_PROGRESS_LOG_FILE,
    # Folders
    BATCHES_FOLDER,
    # Bulk constants
    COMBINATION_ID,
    COMBINATIONS_TABLE_KEY,
    COMBINATIONS_TO_RUN_KEY,
    # Timeout constants
    DOCKING_TIMEOUT,
    # Files
    GUILD_COMBINATIONS_FILE,
    KNOWN_BINDERS_FILE,
    OUTPUT_LOG_FILE,
    PREPROCESSING_TIMEOUT,
    RP_SCORES_FILE,
    SMILES_NAMES_DICTIONARY_KEY,
    SMILES_TYPE_DICTIONARY_KEY,
)
from guild.constants.diffdock import (
    DIFFDOCK_COMBINATIONS_FILE,
    DIFFDOCK_RESULTS_FOLDER,
)
from guild.constants.guild import (
    ALL_AVAILABLE_METHODS,
    BOLTZ_FOLDER,
    BOLTZ_PREFIX,
    DIFFDOCK_FOLDER,
    DIFFDOCK_PREFIX,
    KARMADOCK_FOLDER,
    KARMADOCK_PREFIX,
    LIGAND_CATEGORY,
    LIGAND_ID,
    MSA_FOLDER,
    ORIGINAL_LIGAND,
    ORIGINAL_LIGAND_CHAIN,
    PLOTS_FOLDER,
    PROTEIN_CHAIN,
    # Columns
    PROTEIN_CONF_ID,
    PROTEIN_ID,
    PROTEIN_PATH,
    # Scores lists
    RP_SCORES_COLUMNS,
    # Dictionaries
    SCORES_DICTIONARY,
    SMILES,
    # Folders
    VINA_FOLDER,
    # Prefixes
    VINA_PREFIX,
    VINA_RESCORE_PREFIX,
)
from guild.constants.karmadock import (
    KARMADOCK_DATA_FOLDER,
    KARMADOCK_GRAPHS_FOLDER,
    KARMADOCK_RESULTS_FOLDER,
)
from guild.constants.plip import (
    COMPLEX_PDB_SUFFIX,
    PLIP_COMBINATION_ID,
    PLIP_INTERACTIONS_FILE,
    PLIP_LIGAND_CHAIN,
    PLIP_LIGAND_RESNAME,
    PLIP_LIGAND_RESSEQ,
    PLIP_N_UNIQUE_RESIDUES,
    PLIP_TOTAL_INTERACTIONS,
)
from guild.constants.system import PROJECTS_FOLDER, WORKING_DIR_PATH
from guild.docking.boltz import (
    boltz_guild_scoring,
    deploy_boltz,
    generate_boltz_yaml,
)
from guild.docking.diffdock import (
    deploy_diffdock,
    diffdock_guild_scoring,
    write_diffdock_combinations_table,
)
from guild.docking.karmadock import deploy_karmadock, karmadock_guild_scoring
from guild.docking.vina import vina_guild_scoring, vina_rescore_guild_scoring
from guild.run import Guild
from guild.tools.bulk import (
    available_methods_preparation,
    extend_all_combinations_table_with_decoys,
    extend_all_combinations_table_with_known_binders,
    generate_diffdock_complex_pdbs,
    generate_vina_complex_pdbs,
    identify_previously_ran_combinations,
    kickstart_batch_dictionary,
)
from guild.tools.preparation import clean_smiles
from guild.tools.protein_sequence import (
    get_original_sequence_dictionary,
    process_into_fasta_string,
)
from guild.tools.scores import compute_rank_percentile_scores
from guild.transformers.msa import fetch_protein_msa
from guild.transformers.pdb import get_pocket_contacts_from_ligand
from guild.visualization.plotting import (
    bulk_plot_unique_proteins_scorings,
    bulk_plot_unique_scores,
)

logger = logging.getLogger(__name__)


class BulkRun:
    def __init__(
        self,
        input_table,
        project_name,
        methods_to_run=ALL_AVAILABLE_METHODS,
        batch_size=1000,
        decoys=None,
        use_decoys=True,
        use_known_binders=True,
        min_mol_wt=250,
        max_mol_wt=450,
        chembl_version="chembl_36",
        use_gpu=True,
        n_workers=None,
        predict_binding_pocket=False,
    ):
        """
        BulkRun class for running multiple docking simulations and performing guild scoring.
        :param input_table: Table containing the protein path, the ligand SMILES and name.
        :param project_name: Name of the project.
        :param methods_to_run: List of methods to run for docking simulations.
        :param batch_size: Number of combinations to run in a single batch.
        :param decoys: Path to the decoy file.
        :param use_decoys: Whether to use decoys.
        :param use_known_binders: Whether to use known binders.
        :param min_mol_wt: Minimum molecular weight for known binders.
        :param max_mol_wt: Maximum molecular weight for known binders.
        :param chembl_version: ChEMBL version to use for known binders.
        :param use_gpu: Use GPU.
        :param n_workers: Number of parallel workers for multiprocessing. If None, uses CPU count.
        :param predict_binding_pocket: Use P2Rank for binding site prediction instead of original ligand location.
        """

        # Pathing variables
        logger.info("Initializing BulkAnalysis object")

        self.batched_table = {}

        self.project_name = project_name

        self.min_mol_wt = min_mol_wt
        self.max_mol_wt = max_mol_wt
        self.chembl_version = chembl_version
        self.use_decoys = use_decoys
        self.use_known_binders = use_known_binders
        self.use_gpu = use_gpu
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        self.predict_binding_pocket = predict_binding_pocket

        if "_" in project_name:
            raise ValueError("Project name cannot contain underscores!")

        self._set_paths(decoys)

        if methods_to_run is None:
            logger.info("No methods to run provided, using all available methods")
            methods_to_run = ALL_AVAILABLE_METHODS

        self.methods_to_run = methods_to_run

        # Automatically enable Vina rescore when DiffDock is requested
        if DIFFDOCK_PREFIX in self.methods_to_run and VINA_RESCORE_PREFIX not in self.methods_to_run:
            self.methods_to_run = list(self.methods_to_run) + [VINA_RESCORE_PREFIX]

        logger.info(f"Methods to run: {self.methods_to_run}")

        logger.info("BulkRun object initialized")

        self._generate_all_combinations_table(input_table)
        self.existing_rp_scores = pd.DataFrame(
            columns=[COMBINATION_ID, PROTEIN_CONF_ID, SMILES] + RP_SCORES_COLUMNS
        )
        self._generate_batched_dictionary(self.all_combinations_table, batch_size=batch_size)
        logger.info("Filled batched dictionary")

    def _set_paths(self, decoys):
        """
        Set the paths for the project.
        :param decoys: Path to the decoy file.
        """
        self.home_path = WORKING_DIR_PATH
        self.project_folder = f"{PROJECTS_FOLDER}/{self.project_name}"
        self.plots_folder = f"{self.project_folder}/{PLOTS_FOLDER}"
        self.batches_folder = f"{self.project_folder}/{BATCHES_FOLDER}"
        self.msa_cache_dir = f"{self.project_folder}/{MSA_FOLDER}"
        if decoys is None:
            self.decoys = f"{self.home_path}/guild/support/chembl_35_decoys_100.tsv"
        else:
            self.decoys = decoys
        for folder in [self.project_folder, self.plots_folder, self.batches_folder]:
            os.makedirs(folder, exist_ok=True)

        self.rp_scores_path = f"{self.project_folder}/{RP_SCORES_FILE}"
        self.all_combinations_path = f"{self.project_folder}/{ALL_COMBINATIONS_FILE}"
        self.known_binders_file_path = f"{self.project_folder}/{KNOWN_BINDERS_FILE}"

    def _map_protein_paths(self, input_table):
        """
        Map the protein paths to the all combinations table.
        :param input_table: Table containing the combinations to run.
        :return: Table containing the combinations to run with the protein paths mapped.
        """
        self.protein_path_mapper = {}
        for current_protein, protein_path in zip(
            input_table[PROTEIN_ID], input_table[PROTEIN_PATH], strict=True
        ):
            self.protein_path_mapper[current_protein] = protein_path

    def _generate_all_combinations_table(self, input_table):
        """
        Generate the all combinations table.
        """
        self.all_combinations_table = input_table.copy()
        self._map_protein_paths(self.all_combinations_table)
        if self.use_known_binders:
            self.known_binders_table = extend_all_combinations_table_with_known_binders(
                input_table=self.all_combinations_table,
                known_binders_file_path=self.known_binders_file_path,
                min_mol_wt=self.min_mol_wt,
                max_mol_wt=self.max_mol_wt,
                chembl_version=self.chembl_version,
            )

            logger.info("Extended known binders table with known binders")
            self.all_combinations_table = pd.concat(
                [self.all_combinations_table, self.known_binders_table], axis=0
            )
        else:
            logger.info("Skipping known binders for known binders table, per user request")

        if self.use_decoys:
            self.all_decoys_table = extend_all_combinations_table_with_decoys(
                input_table=self.all_combinations_table,
                decoys_file_path=self.decoys,
            )
            logger.info("Extended all combinations table with decoy dataset")
            self.all_combinations_table = pd.concat(
                [self.all_combinations_table, self.all_decoys_table], axis=0
            )
        else:
            logger.info("Skipping decoy dataset for all combinations table, per user request")

        self.all_combinations_table[PROTEIN_PATH] = self.all_combinations_table[PROTEIN_ID].replace(
            self.protein_path_mapper
        )

        # stripping salt from the ligands
        self.all_combinations_table[SMILES] = self.all_combinations_table[SMILES].apply(
            lambda x: clean_smiles(x)
        )
        self.all_combinations_table.to_csv(self.all_combinations_path, index=False)
        logger.info(f"Saved all combinations table to {self.all_combinations_path}")

    def _generate_batched_dictionary(self, input_table, batch_size=1000):
        """
        Fill the batched dictionary with the necessary information.
        :param input_table: Table containing the combinations to run.
        :param batch_size: Number of combinations to run in a single batch.
        :return: Batched dictionary.
        """
        number_of_batches = math.ceil(input_table.shape[0] / batch_size)
        self.batched_dictionary = {}
        for i in range(number_of_batches):
            current_batch = f"batch_{i + 1}"
            current_batch_table = input_table.iloc[i * batch_size : (i + 1) * batch_size]
            batch_folder = f"{self.batches_folder}/{current_batch}"

            # Create batch folder for this batch
            os.makedirs(batch_folder, exist_ok=True)

            self.batched_dictionary[current_batch] = kickstart_batch_dictionary(
                current_batch_table=current_batch_table,
                batch_folder=batch_folder,
            )

            self.batched_dictionary[current_batch] = identify_previously_ran_combinations(
                current_batch=current_batch,
                batch_dictionary=self.batched_dictionary[current_batch],
            )

            logger.info(f"Identified previously ran combinations for batch {current_batch}")
            self.batched_dictionary[current_batch][COMBINATIONS_TABLE_KEY].to_csv(
                f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{GUILD_COMBINATIONS_FILE}",
                index=False,
            )
            if DIFFDOCK_PREFIX in self.methods_to_run:
                write_diffdock_combinations_table(
                    input_table=self.batched_dictionary[current_batch][COMBINATIONS_TABLE_KEY],
                    output_dir=self.batched_dictionary[current_batch][BATCH_FOLDER],
                )
                logger.info(f"Wrote diffdock combinations table for batch {current_batch}")

            # Prepare the available methods scores and rankings for the batch
            self.batched_dictionary[current_batch] = available_methods_preparation(
                batch_dictionary=self.batched_dictionary[current_batch],
                methods_to_run=self.methods_to_run,
            )
            logger.info(f"Prepared available methods for batch {current_batch}")

    @staticmethod
    def _run_single_vina_docking(task_params):
        """
        Run Vina docking for a single combination (worker function for multiprocessing).
        :param task_params: Dictionary containing all parameters for Guild initialization.
        """
        try:
            # Log SMILES before attempting docking
            logger.info(
                f"Starting docking for SMILES: {task_params['ligand_smile']} "
                f"(ligand_idx: {task_params['ligand_idx']}, protein_idx: {task_params['protein_idx']})"
            )

            guild_object = Guild(
                ligand_smile=task_params["ligand_smile"],
                ligand_idx=task_params["ligand_idx"],
                protein_idx=task_params["protein_idx"],
                protein_file=task_params["protein_file"],
                project_name=task_params["project_name"],
                protein_chain=task_params["protein_chain"],
                original_ligand=task_params["original_ligand"],
                original_ligand_chain=task_params["original_ligand_chain"],
                output_log_file=task_params["output_log_file"],
                use_gpu=task_params["use_gpu"],
                is_bulk=True,
                predict_binding_pocket=task_params.get("predict_binding_pocket", False),
            )
            guild_object.run_autodock_vina()
            return task_params["ligand_idx"], task_params["protein_idx"]
        except Exception as e:
            logger.error(
                f"Error in Vina docking for SMILES {task_params['ligand_smile']} "
                f"(ligand_idx: {task_params['ligand_idx']}): {e}"
            )
            return None

    @staticmethod
    def _prepare_single_batch_worker(batch_params):
        """
        Prepare docking for a single batch (static worker function for multiprocessing).
        :param batch_params: Dictionary containing batch parameters.
        """
        try:
            Guild(
                ligand_smile=batch_params["ligand_smile"],
                ligand_idx=batch_params["ligand_idx"],
                protein_idx=batch_params["protein_idx"],
                protein_file=batch_params["protein_file"],
                project_name=batch_params["project_name"],
                protein_chain=batch_params["protein_chain"],
                original_ligand=batch_params["original_ligand"],
                original_ligand_chain=batch_params["original_ligand_chain"],
                output_log_file=batch_params["output_log_file"],
                use_gpu=batch_params["use_gpu"],
                is_bulk=True,
                predict_binding_pocket=batch_params.get("predict_binding_pocket", False),
            )
            return batch_params["batch_name"]
        except Exception as e:
            logger.error(f"Error during preparation: {e}")
            return None

    def _prepare_docking(self):
        """
        Prepare the docking for all batches in parallel.
        """
        logger.info(f"Preparing docking with {self.n_workers} workers")

        # Prepare batch parameters for each batch
        batch_params_list = []

        for current_batch in tqdm(self.batched_dictionary, desc="Preparing docking"):

            # Skip batches with no combinations to run
            if len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY]) == 0:
                logger.info(f"Skipping {current_batch} - no new combinations to dock")
                continue

            # Check if combinations table is empty
            if self.batched_dictionary[current_batch][COMBINATIONS_TABLE_KEY].empty:
                logger.info(f"Skipping {current_batch} - empty combinations table")
                continue

            # This will download the necessary raw pdb files beforehand

            current_subset_row = self.batched_dictionary[current_batch][
                COMBINATIONS_TABLE_KEY
            ].iloc[0]
            batch_params_list.append(
                {
                    "batch_name": current_batch,
                    "ligand_smile": self.batched_dictionary[current_batch][
                        SMILES_NAMES_DICTIONARY_KEY
                    ][current_subset_row[LIGAND_ID]],
                    "ligand_idx": current_subset_row[LIGAND_ID],
                    "protein_idx": current_subset_row[PROTEIN_CONF_ID],
                    "protein_file": current_subset_row[PROTEIN_PATH],
                    "project_name": f"{self.project_name}/{BATCHES_FOLDER}/{current_batch}",
                    "protein_chain": current_subset_row[PROTEIN_CHAIN],
                    "original_ligand": current_subset_row[ORIGINAL_LIGAND],
                    "original_ligand_chain": current_subset_row[ORIGINAL_LIGAND_CHAIN],
                    "output_log_file": f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{OUTPUT_LOG_FILE}",
                    "use_gpu": self.use_gpu,
                    "predict_binding_pocket": self.predict_binding_pocket,
                }
            )

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all preparation tasks
            futures = [
                executor.submit(self._prepare_single_batch_worker, params)
                for params in batch_params_list
            ]

            # Process results as they complete, not in submission order
            for future in tqdm(
                as_completed(futures),
                desc="Preparing docking",
                total=len(futures),
            ):
                try:
                    result = future.result(timeout=PREPROCESSING_TIMEOUT)
                    if result:
                        logger.debug(f"Completed preparation for batch: {result}")
                except FuturesTimeoutError:
                    logger.error(
                        f"Preparation task timed out after {PREPROCESSING_TIMEOUT} seconds"
                    )
                    future.cancel()  # Attempt to cancel
                except Exception as e:
                    logger.error(f"Preparation task failed with error: {e}")

    def run_docking(self):
        """
        Perform docking simulations using AutoDock Vina, KarmaDock, and DiffDock.
        """
        self._prepare_docking()

        # Create main progress log file
        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"

        for current_batch in tqdm(self.batched_dictionary, desc="Running docking"):
            current_batch_folder = self.batched_dictionary[current_batch][BATCH_FOLDER]

            # Log progress to main log file
            with open(main_progress_log, "a") as f:
                f.write(f"Starting {current_batch} at {pd.Timestamp.now()}\n")

            # Skip batches with no new combinations
            if len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY]) == 0:
                logger.info(f"Skipping {current_batch} - no new combinations to dock")
                with open(main_progress_log, "a") as f:
                    f.write(
                        f"Skipped {current_batch} - no new combinations at {pd.Timestamp.now()}\n"
                    )
                continue

            if BOLTZ_PREFIX in self.methods_to_run:
                # Botlz is applied in the whole batch at once, so we need to iterate over the unique protein configuration ids
                combinations_table_variable = self.batched_dictionary[current_batch][
                    COMBINATIONS_TABLE_KEY
                ]
                for unique_protein_configuration_id in combinations_table_variable[
                    PROTEIN_CONF_ID
                ].unique():

                    current_protein_combinations = combinations_table_variable[
                        combinations_table_variable[PROTEIN_CONF_ID]
                        == unique_protein_configuration_id
                    ]
                    current_protein_chain = current_protein_combinations[PROTEIN_CHAIN].values[0]
                    current_protein_path = current_protein_combinations[PROTEIN_PATH].values[0]
                    original_sequence_dictionary = get_original_sequence_dictionary(
                        current_protein_path
                    )
                    current_protein_sequence = process_into_fasta_string(
                        original_sequence_dictionary[current_protein_chain]
                    )
                    current_protein_unique_ligands_ids = list(
                        current_protein_combinations[LIGAND_ID].unique()
                    )

                    # Fetch MSA once per protein — cached across all ligands and batches
                    msa_file = fetch_protein_msa(
                        sequence=current_protein_sequence,
                        protein_id=unique_protein_configuration_id,
                        protein_chain_id=current_protein_chain,
                        output_a3m_dir=self.msa_cache_dir,
                    )
                    logger.info(
                        f"Fetched MSA for protein configuration {unique_protein_configuration_id} (chain {current_protein_chain})"
                    )

                    # Compute binding-pocket contacts once per protein configuration
                    _orig_lig = current_protein_combinations[ORIGINAL_LIGAND].values[0]
                    _orig_lig_chain = current_protein_combinations[ORIGINAL_LIGAND_CHAIN].values[0]
                    pocket_contacts = []
                    if (
                        _orig_lig
                        and _orig_lig_chain
                        and pd.notna(_orig_lig)
                        and pd.notna(_orig_lig_chain)
                    ):
                        pocket_contacts = get_pocket_contacts_from_ligand(
                            protein_pdb=current_protein_path,
                            protein_chain=current_protein_chain,
                            original_ligand=_orig_lig,
                            original_ligand_chain=_orig_lig_chain,
                            distance_threshold=4.0,
                        )

                    # Run one Boltz job per ligand to avoid O(N²) attention memory growth
                    for current_ligand_id in current_protein_unique_ligands_ids:
                        ligand_smiles = self.batched_dictionary[current_batch][
                            SMILES_NAMES_DICTIONARY_KEY
                        ][current_ligand_id]
                        run_id = f"{unique_protein_configuration_id}_{current_ligand_id}"

                        # Saving time by skipping already run Boltz dockings - checking for one of the expected output files
                        boltz_result_fild = (
                            f"{BOLTZ_FOLDER}/predictions/{run_id}_boltz/confidence_{run_id}_boltz_model_0.json"
                        )
                        if os.path.exists(f"{current_batch_folder}/{boltz_result_fild}"):
                            logger.info(
                                f"Boltz docking already ran for {current_batch}: "
                                f"protein {unique_protein_configuration_id}, ligand {current_ligand_id}"
                            )
                            continue

                        os.makedirs(f"{current_batch_folder}/{BOLTZ_FOLDER}", exist_ok=True)
                        generate_boltz_yaml(
                            protein_sequence=current_protein_sequence,
                            protein_chain=current_protein_chain,
                            ligand_sequences=[ligand_smiles],
                            ligand_ids=["L"],
                            output_file=f"{current_batch_folder}/{BOLTZ_FOLDER}/{run_id}_boltz.yaml",
                            template_file=current_protein_path,
                            pocket_contacts=pocket_contacts if pocket_contacts else None,
                            msa_file=msa_file,
                        )

                        deploy_boltz(
                            f"{current_batch_folder}/{BOLTZ_FOLDER}/{run_id}_boltz.yaml",
                            out_dir=f"{current_batch_folder}/{BOLTZ_FOLDER}",
                            use_gpu=self.use_gpu,
                        )

                        logger.info(
                            f"Boltz docking completed for {current_batch}: "
                            f"protein {unique_protein_configuration_id}, ligand {current_ligand_id}"
                        )

            if VINA_PREFIX in self.methods_to_run:
                # Collect all combinations to run
                vina_tasks = []
                for _index, current_row in self.batched_dictionary[current_batch][
                    COMBINATIONS_TABLE_KEY
                ].iterrows():
                    protein_path = current_row[PROTEIN_PATH]
                    current_protein, current_ligand = (
                        current_row[PROTEIN_CONF_ID],
                        current_row[LIGAND_ID],
                    )
                    vina_final_path = f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{VINA_FOLDER}/{current_protein}_{current_ligand}.txt"
                    if not os.path.exists(vina_final_path):
                        vina_tasks.append(
                            {
                                "ligand_smile": self.batched_dictionary[current_batch][
                                    SMILES_NAMES_DICTIONARY_KEY
                                ][current_ligand],
                                "ligand_idx": current_ligand,
                                "protein_idx": current_protein,
                                "protein_file": protein_path,
                                "project_name": f"{self.project_name}/{BATCHES_FOLDER}/{current_batch}",
                                "protein_chain": current_row[PROTEIN_CHAIN],
                                "original_ligand": current_row[ORIGINAL_LIGAND],
                                "original_ligand_chain": current_row[ORIGINAL_LIGAND_CHAIN],
                                "output_log_file": f"{current_batch_folder}/{OUTPUT_LOG_FILE}",
                                "use_gpu": self.use_gpu,
                                "predict_binding_pocket": self.predict_binding_pocket,
                            }
                        )

                if vina_tasks:
                    logger.info(
                        f"Running {len(vina_tasks)} Vina docking tasks with {self.n_workers} workers"
                    )
                    with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                        # Submit all docking tasks - store task info for better error reporting
                        futures = {}
                        for i, task in enumerate(vina_tasks):
                            future = executor.submit(self._run_single_vina_docking, task)
                            futures[future] = (i, task)  # Store index and full task dict

                        # Process results as they complete, with per-task timeout
                        completed = 0
                        for future in tqdm(
                            as_completed(futures, timeout=None),
                            desc=f"AutoDock Vina for batch: {current_batch}",
                            total=len(futures),
                        ):
                            task_num, task_dict = futures[future]
                            try:
                                future.result(timeout=DOCKING_TIMEOUT)
                                completed += 1
                            except FuturesTimeoutError:
                                logger.warning(
                                    f"Task {task_num+1}/{len(vina_tasks)} timed out after {DOCKING_TIMEOUT} seconds "
                                    f"(SMILES: {task_dict.get('ligand_smile', 'unknown')})"
                                )
                                future.cancel()  # Attempt to cancel
                            except Exception as e:
                                logger.error(
                                    f"Task {task_num+1}/{len(vina_tasks)} failed: {e} "
                                    f"(SMILES: {task_dict.get('ligand_smile', 'unknown')})"
                                )

                        logger.info(f"Completed {completed}/{len(vina_tasks)} Vina docking tasks")
                else:
                    logger.info(f"No new Vina combinations to dock for batch {current_batch}")

                # Generate complex PDB files for Vina results
                logger.info(f"Generating complex PDB files for Vina results in {current_batch}")
                generate_vina_complex_pdbs(self.batched_dictionary[current_batch])

            # if BOLTZ_PREFIX in self.methods_to_run:
            #     # (Boltz docking loop is earlier in this function)
            #     # Generate complex PDB files for PLIP analysis
            #     logger.info(f"Generating complex PDB files for Boltz results in {current_batch}")
            #     generate_boltz_complex_pdbs(self.batched_dictionary[current_batch])

            # Generate complex PDB files for DiffDock results (for PLIP)
            if DIFFDOCK_PREFIX in self.methods_to_run:
                diffdock_results_dir = os.path.join(
                    self.batched_dictionary[current_batch][BATCH_FOLDER],
                    DIFFDOCK_FOLDER,
                    DIFFDOCK_RESULTS_FOLDER,
                )
                if os.path.isdir(diffdock_results_dir):
                    logger.info(
                        f"Generating complex PDB files for DiffDock results in {current_batch}"
                    )
                    generate_diffdock_complex_pdbs(self.batched_dictionary[current_batch])

            # Karmadock and DiffDock are triggered by a single run

            if KARMADOCK_PREFIX in self.methods_to_run:
                if (
                    os.path.exists(
                        f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_RESULTS_FOLDER}/0.csv"
                    )
                    and os.path.exists(
                        f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_RESULTS_FOLDER}/1.csv"
                    )
                    and os.path.exists(
                        f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_RESULTS_FOLDER}/2.csv"
                    )
                ):
                    logger.info(
                        f"Karmadock docking already ran for {current_batch} with {len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY])} combinations"
                    )
                else:
                    deploy_karmadock(
                        home_path=self.home_path,
                        karmadock_results_dir=f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_RESULTS_FOLDER}",
                        karmadock_graphs_dir=f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_GRAPHS_FOLDER}",
                        karmadock_data_dir=f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{KARMADOCK_FOLDER}/{KARMADOCK_DATA_FOLDER}",
                    )
                    logger.info(
                        f"Karmadock docking completed for {current_batch} with {len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY])} combinations"
                    )

            if DIFFDOCK_PREFIX in self.methods_to_run:

                batch_csv = f"{current_batch_folder}/{DIFFDOCK_COMBINATIONS_FILE}"
                batch_results_dir = (
                    f"{current_batch_folder}/{DIFFDOCK_FOLDER}/{DIFFDOCK_RESULTS_FOLDER}"
                )

                # If batch already produced results, skip
                if os.path.exists(batch_results_dir) and len(os.listdir(batch_results_dir)) > 0:
                    logger.info(f"DiffDock batch already ran for {current_batch}")
                else:
                    # Run DiffDock ONCE for the whole batch
                    deploy_diffdock(
                        home_path=self.home_path,
                        diffdock_results_dir=batch_results_dir,
                        input_csv=batch_csv,
                        # use_gpu=self.use_gpu,
                    )
                    logger.info(
                        f"DiffDock batch completed for {current_batch} with {len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY])} combinations"
                    )

            # Log batch completion to main progress log
            with open(main_progress_log, "a") as f:
                f.write(f"Completed {current_batch} at {pd.Timestamp.now()}\n")

    def _compute_global_ranks_per_protein(self, all_raw_scores):
        """
        Compute ranks and rank percentile scores PER PROTEIN across ALL data (all batches + database).
        This ensures rankings are consistent across batches - all ligands for a given protein
        are ranked together, not separately per batch.

        :param all_raw_scores: DataFrame with raw scores from all batches and database.
        :return: DataFrame with ranks and rank percentile scores added.
        """
        return compute_rank_percentile_scores(all_raw_scores, methods=self.methods_to_run)

    def _process_batch_scoring(self, batch):
        """
        Process a single batch to collect RAW scores only (no ranking yet).
        Ranking will be done globally across all batches per protein.

        :param batch: Batch to process.
        :return: Raw scores table for this batch (COMBINATION_ID, PROTEIN_ID, LIGAND_ID, raw scores).
        """
        logger.info(f"Collecting raw scores for batch {batch}")
        if len(self.batched_dictionary[batch]["combinations_to_run"]) == 0:
            logger.info("No new combinations to score")
            return None

        docked_methods = []

        if VINA_PREFIX in self.methods_to_run:
            docked_methods.append(vina_guild_scoring(self.batched_dictionary[batch]))

        if KARMADOCK_PREFIX in self.methods_to_run:
            docked_methods.append(karmadock_guild_scoring(self.batched_dictionary[batch]))

        if DIFFDOCK_PREFIX in self.methods_to_run:
            docked_methods.append(diffdock_guild_scoring(self.batched_dictionary[batch]))

        if VINA_RESCORE_PREFIX in self.methods_to_run:
            docked_methods.append(vina_rescore_guild_scoring(self.batched_dictionary[batch]))

        if BOLTZ_PREFIX in self.methods_to_run:
            docked_methods.append(boltz_guild_scoring(self.batched_dictionary[batch]))

        if not docked_methods:
            return None

        # Merge raw scores from all methods
        raw_scores_table = docked_methods[0].reset_index(drop=True)
        if len(docked_methods) > 1:
            for i in range(1, len(docked_methods)):
                raw_scores_table = pd.merge(
                    raw_scores_table.reset_index(drop=True),
                    docked_methods[i].reset_index(drop=True),
                    on=[COMBINATION_ID, PROTEIN_CONF_ID, LIGAND_ID],
                    how="outer",
                )

        # Add SMILES column for later merging
        raw_scores_table[SMILES] = raw_scores_table[LIGAND_ID].apply(
            lambda x: self.batched_dictionary[batch][SMILES_NAMES_DICTIONARY_KEY].get(x, "unknown")
        )

        # Return RAW scores only - no ranks or rank percentile scores yet
        return raw_scores_table

    def run_guild_scoring(self, n_processes=None):
        """
        Scoring function to uniformize multiple docking methods by leveraging the decoy dataset.
        Steps:
        1. Collect RAW scores from all batches (parallel)
        2. Merge with existing database scores (if any)
        3. Compute ranks PER PROTEIN across ALL data (not per batch)
        4. Compute rank percentile scores from global ranks

        :param n_processes: Number of processes to use for multiprocessing.
        """

        if n_processes is None:
            n_processes = mp.cpu_count()

        # Step 1: Collect raw scores from all batches (parallel)
        if len(self.batched_dictionary) == 0:
            logger.info("No new batches to process")
            new_raw_scores = pd.DataFrame()
        else:
            with mp.Pool(processes=n_processes) as pool:
                raw_scores_list = list(
                    tqdm(
                        pool.imap(self._process_batch_scoring, self.batched_dictionary.keys()),
                        total=len(self.batched_dictionary),
                        desc="Collecting raw scores from batches",
                    )
                )

            # Filter and concatenate
            raw_scores_list = [r for r in raw_scores_list if r is not None]
            if raw_scores_list:
                new_raw_scores = pd.concat(raw_scores_list, axis=0).reset_index(drop=True)
            else:
                new_raw_scores = pd.DataFrame()

        all_raw_scores = new_raw_scores

        if all_raw_scores.empty:
            logger.warning("No scores to process")
            return

        # Ensure ligand category is available before scoring
        smiles_type_dictionary = self.unpack_dictionary(what_to_unpack=SMILES_TYPE_DICTIONARY_KEY)
        if LIGAND_CATEGORY not in all_raw_scores.columns:
            all_raw_scores[LIGAND_CATEGORY] = all_raw_scores[SMILES].apply(
                lambda x: smiles_type_dictionary.get(x, "unknown")
            )

        # Deduplicate: keep the row that actually has a score (scored rows first, then drop dupes).
        score_cols = [SCORES_DICTIONARY[m] for m in self.methods_to_run if SCORES_DICTIONARY[m] in all_raw_scores.columns]
        if score_cols:
            all_raw_scores = (
                all_raw_scores
                .assign(_has_score=all_raw_scores[score_cols].notna().any(axis=1))
                .sort_values("_has_score", ascending=False)
                .drop_duplicates(subset=[COMBINATION_ID], keep="first")
                .drop(columns=["_has_score"])
                .reset_index(drop=True)
            )

        # Step 3: Compute ranks PER PROTEIN across ALL data
        logger.info(
            f"Computing global ranks per protein for {all_raw_scores.shape[0]} total combinations"
        )
        self.rp_scores_df = self._compute_global_ranks_per_protein(all_raw_scores)

        # Step 4: Add ligand category
        if LIGAND_CATEGORY not in self.rp_scores_df.columns:
            self.rp_scores_df[LIGAND_CATEGORY] = self.rp_scores_df[SMILES].apply(
                lambda x: smiles_type_dictionary.get(x, "unknown")
            )

        # Step 5: Save results
        self.rp_scores_df.to_csv(
            self.rp_scores_path,
            index=False,
            sep="\t",
        )
        logger.info(f"Saved rank percentile scores to {self.rp_scores_path}")

    def unpack_dictionary(self, what_to_unpack="smiles_type_dictionary"):
        """
        Unpack the smiles type dictionary.
        :param what_to_unpack: What to unpack out of the batched dictionary.
        """
        unpacked_dictionary = {}
        for current_batch in self.batched_dictionary:
            for current_key in self.batched_dictionary[current_batch][what_to_unpack].keys():
                unpacked_dictionary[current_key] = self.batched_dictionary[current_batch][
                    what_to_unpack
                ][current_key]
        return unpacked_dictionary

    def unpack_list(self, what_to_unpack="ranks_list"):
        """
        Unpack the ranks list.
        :param what_to_unpack: What to unpack out of the batched dictionary.
        """
        unpacked_list = []
        for current_batch in self.batched_dictionary:
            unpacked_list.extend(self.batched_dictionary[current_batch][what_to_unpack])
        return unpacked_list

    def plot_guild_scoring(self):
        """
        Plot the Guild scoring results.
        """
        if os.path.exists(self.rp_scores_path):
            self.rp_scores_df = pd.read_csv(self.rp_scores_path, sep="\t")
        else:
            self.rp_scores_df.copy()
        self.unpack_dictionary(what_to_unpack="smiles_type_dictionary")
        ranks_list = list(set(self.unpack_list(what_to_unpack="ranks_list")))
        bulk_plot_unique_scores(
            self.rp_scores_df, ranks_list, save_folder=self.plots_folder
        )

    def plot_unique_proteins_scorings(self, top_n_hits=5):
        """
        Plot the Guild scoring results for unique proteins.
        :param top_n_hits: Number of top hits to plot, to avoid overplotting.
        """
        ranks_list = list(set(self.unpack_list(what_to_unpack="ranks_list")))

        bulk_plot_unique_proteins_scorings(
            self.rp_scores_df,
            ranks_list,
            top_n_hits=top_n_hits,
            save_folder=self.plots_folder,
        )

    def run_interactions_analysis(self, methods_to_analyze=None):
        """
        Analyze protein-ligand interactions using PLIP for all docked complexes.
        Supports Vina, Boltz, and DiffDock docking results (complex PDB files).

        :param methods_to_analyze: List of docking methods to analyze. Defaults to methods_to_run.
        """
        if methods_to_analyze is None:
            methods_to_analyze = self.methods_to_run

        logger.info("Starting interactions analysis")

        all_interactions = []

        for current_batch in self.batched_dictionary:
            logger.info(f"Analyzing interactions for {current_batch}")

            batch_dict = self.batched_dictionary[current_batch]
            batch_folder = batch_dict[BATCH_FOLDER]
            combinations_df = batch_dict[COMBINATIONS_TABLE_KEY]

            if BOLTZ_PREFIX in methods_to_analyze:
                boltz_folder = f"{batch_folder}/{BOLTZ_FOLDER}"

                complex_metadata = []
                for _, row in combinations_df.iterrows():
                    protein_conf_id = row[PROTEIN_CONF_ID]
                    ligand_id = row[LIGAND_ID]
                    run_id = f"{protein_conf_id}_{ligand_id}"
                    complex_pdb = f"{boltz_folder}/{run_id}{COMPLEX_PDB_SUFFIX}"
                    if os.path.exists(complex_pdb):
                        smiles = batch_dict[SMILES_NAMES_DICTIONARY_KEY][ligand_id]
                        complex_metadata.append((complex_pdb, protein_conf_id, smiles))

                if complex_metadata:
                    logger.info(
                        f"Analyzing {len(complex_metadata)} Boltz complexes in {current_batch}"
                    )
                    batch_interactions = analyze_batch_interactions(
                        complex_pdb_paths=[x[0] for x in complex_metadata],
                        combination_ids=[f"{x[1]}_{x[2]}" for x in complex_metadata],
                        ligand_resname=PLIP_LIGAND_RESNAME,
                        ligand_chain=PLIP_LIGAND_CHAIN,
                        ligand_resseq=PLIP_LIGAND_RESSEQ,
                    )
                    if not batch_interactions.empty:
                        batch_interactions[PROTEIN_CONF_ID] = [x[1] for x in complex_metadata]
                        batch_interactions[SMILES] = [x[2] for x in complex_metadata]
                        all_interactions.append(batch_interactions)
                else:
                    logger.info(f"No Boltz complex PDB files found for {current_batch}")

            if VINA_PREFIX in methods_to_analyze:
                vina_folder = f"{batch_folder}/{VINA_FOLDER}"

                # Collect all complex PDB files and their IDs
                complex_paths = []
                combination_ids = []

                complex_metadata = []  # Store (complex_path, protein_conf_id, smiles)

                for _, row in combinations_df.iterrows():
                    protein_conf_id = row[PROTEIN_CONF_ID]
                    ligand_id = row[LIGAND_ID]
                    combination_id = f"{protein_conf_id}_{ligand_id}"

                    complex_pdb = f"{vina_folder}/{combination_id}{COMPLEX_PDB_SUFFIX}"

                    if os.path.exists(complex_pdb):
                        smiles = batch_dict[SMILES_NAMES_DICTIONARY_KEY][ligand_id]
                        complex_metadata.append((complex_pdb, protein_conf_id, smiles))

                if complex_metadata:
                    logger.info(
                        f"Analyzing {len(complex_metadata)} Vina complexes in {current_batch}"
                    )
                    complex_paths = [x[0] for x in complex_metadata]
                    combination_ids = [
                        f"{x[1]}_{x[2]}" for x in complex_metadata
                    ]  # temporary for analysis

                    batch_interactions = analyze_batch_interactions(
                        complex_pdb_paths=complex_paths,
                        combination_ids=combination_ids,
                        ligand_resname=PLIP_LIGAND_RESNAME,
                        ligand_chain=PLIP_LIGAND_CHAIN,
                        ligand_resseq=PLIP_LIGAND_RESSEQ,
                    )

                    if not batch_interactions.empty:
                        # Add protein_conf_id and smiles columns
                        batch_interactions[PROTEIN_CONF_ID] = [x[1] for x in complex_metadata]
                        batch_interactions[SMILES] = [x[2] for x in complex_metadata]
                        all_interactions.append(batch_interactions)
                else:
                    logger.info(f"No complex PDB files found for {current_batch}")

            if DIFFDOCK_PREFIX in methods_to_analyze:
                diffdock_folder = f"{batch_folder}/{DIFFDOCK_FOLDER}"

                complex_metadata = []
                for _, row in combinations_df.iterrows():
                    protein_conf_id = row[PROTEIN_CONF_ID]
                    ligand_id = row[LIGAND_ID]
                    run_id = f"{protein_conf_id}_{ligand_id}"
                    complex_pdb = f"{diffdock_folder}/{run_id}{COMPLEX_PDB_SUFFIX}"
                    if os.path.exists(complex_pdb):
                        smiles = batch_dict[SMILES_NAMES_DICTIONARY_KEY][ligand_id]
                        complex_metadata.append((complex_pdb, protein_conf_id, smiles))

                if complex_metadata:
                    logger.info(
                        f"Analyzing {len(complex_metadata)} DiffDock complexes in {current_batch}"
                    )
                    batch_interactions = analyze_batch_interactions(
                        complex_pdb_paths=[x[0] for x in complex_metadata],
                        combination_ids=[f"{x[1]}_{x[2]}" for x in complex_metadata],
                        ligand_resname=PLIP_LIGAND_RESNAME,
                        ligand_chain=PLIP_LIGAND_CHAIN,
                        ligand_resseq=PLIP_LIGAND_RESSEQ,
                    )
                    if not batch_interactions.empty:
                        batch_interactions[PROTEIN_CONF_ID] = [x[1] for x in complex_metadata]
                        batch_interactions[SMILES] = [x[2] for x in complex_metadata]
                        all_interactions.append(batch_interactions)
                else:
                    logger.info(f"No DiffDock complex PDB files found for {current_batch}")

        if not all_interactions:
            logger.warning("No interaction data collected")
            return None

        # Combine all batch results
        self.interactions_df = pd.concat(all_interactions, axis=0).reset_index(drop=True)

        # Rename protein_config_id to protein_conf_id if present (for consistency)
        if "protein_config_id" in self.interactions_df.columns:
            self.interactions_df.rename(
                columns={"protein_config_id": PROTEIN_CONF_ID}, inplace=True
            )

        # Reorder columns: protein_conf_id and smiles first, then drop combination_id
        cols_order = [PROTEIN_CONF_ID, SMILES] + [
            col
            for col in self.interactions_df.columns
            if col not in [PROTEIN_CONF_ID, SMILES, PLIP_COMBINATION_ID]
        ]
        df_to_save = self.interactions_df[cols_order]

        # Save to file
        interactions_path = f"{self.project_folder}/{PLIP_INTERACTIONS_FILE}"
        df_to_save.to_csv(interactions_path, sep="\t", index=False)
        logger.info(f"Saved interaction analysis to {interactions_path}")

        # Log summary statistics
        logger.info(
            f"Interaction analysis complete: {len(self.interactions_df)} complexes analyzed"
        )
        logger.info(
            f"Average interactions per complex: {self.interactions_df[PLIP_TOTAL_INTERACTIONS].mean():.2f}"
        )
        logger.info(
            f"Average unique residues per complex: {self.interactions_df[PLIP_N_UNIQUE_RESIDUES].mean():.2f}"
        )

        return self.interactions_df
