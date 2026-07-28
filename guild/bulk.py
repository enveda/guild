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
from concurrent.futures.process import BrokenProcessPool

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
)
from guild.constants.diffdock import (
    DIFFDOCK_COMBINATIONS_FILE,
    DIFFDOCK_RESULTS_FOLDER,
)
from guild.constants.guild import (
    ALL_AVAILABLE_METHODS,
    BOLTZ_FOLDER,
    BOLTZ_PREFIX,
    BOX_LOCATION,
    DIFFDOCK_FOLDER,
    DIFFDOCK_PREFIX,
    GNINA_FOLDER,
    GNINA_PREFIX,
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
    PROTEINS_FOLDER,
    # Scores lists
    RP_SCORES_COLUMNS,
    # Dictionaries
    SMILES,
    # Folders
    VINA_FOLDER,
    # Prefixes
    VINA_PREFIX,
    VINA_RESCORE_BOLTZ_PREFIX,
    VINA_RESCORE_DIFFDOCK_PREFIX,
)
from guild.constants.interactions import (
    COMPLEX_PDB_SUFFIX,
    INTERACTION_COMBINATION_ID,
    INTERACTION_COUNT_COLUMNS,
    INTERACTIONS_FILE,
    LIGAND_CHAIN,
    LIGAND_RESNAME,
    LIGAND_RESSEQ,
    N_UNIQUE_RESIDUES,
    TOTAL_INTERACTIONS,
)
from guild.constants.karmadock import (
    KARMADOCK_DATA_FOLDER,
    KARMADOCK_GRAPHS_FOLDER,
    KARMADOCK_RESULTS_FOLDER,
)
from guild.constants.system import PROJECTS_FOLDER, WORKING_DIR_PATH
from guild.docking.boltz import (
    boltz_guild_scoring,
    deploy_boltz,
    generate_boltz_yaml,
    vina_rescore_boltz_guild_scoring,
)
from guild.docking.diffdock import (
    deploy_diffdock,
    diffdock_guild_scoring,
    vina_rescore_diffdock_guild_scoring,
    write_diffdock_combinations_table,
)
from guild.docking.gnina import gnina_guild_scoring
from guild.docking.karmadock import deploy_karmadock, karmadock_guild_scoring
from guild.docking.vina import (
    get_center_and_size_from_box_file,
    vina_guild_scoring,
)
from guild.run import Guild
from guild.tools.bulk import (
    available_methods_preparation,
    extend_all_combinations_table_with_decoys,
    extend_all_combinations_table_with_known_binders,
    generate_boltz_complex_pdbs,
    generate_diffdock_complex_pdbs,
    generate_gnina_complex_pdbs,
    generate_vina_complex_pdbs,
    identify_previously_ran_combinations,
    kickstart_batch_dictionary,
)
from guild.tools.preparation import _normalize_chain_list, clean_smiles
from guild.tools.protein_sequence import (
    get_original_sequence_dictionary,
    process_into_fasta_string,
)
from guild.tools.scores import compute_rank_percentile_scores
from guild.transformers.msa import fetch_protein_msa
from guild.transformers.pdb import (
    get_pocket_contacts_from_box,
    get_pocket_contacts_from_ligand,
)
from guild.visualization.plotting import (
    bulk_plot_unique_proteins_scorings,
    bulk_plot_unique_scores,
)

logger = logging.getLogger(__name__)


def _row_box_location(row):
    """
    Extract a usable ``box_location`` path from a combinations-table row.

    Returns ``None`` when the column is absent or the value is missing / empty,
    so downstream Vina/Boltz code falls back to existing pocket-derivation logic.
    """
    if BOX_LOCATION not in row.index:
        return None
    value = row[BOX_LOCATION]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


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
        gnina_input_mode="pdbqt",
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
        :param gnina_input_mode: ``"pdbqt"`` (default — preserves current
            behaviour) or ``"sdf"``. When ``"sdf"`` AND gnina is the sole
            docking method requested (no Vina or Vina-rescore co-running),
            OpenBabel PDBQT prep is skipped and gnina runs directly on the
            RDKit SDF + cleaned PDB. If ``"sdf"`` is requested alongside
            Vina/Vina-rescore, this resolves back to ``"pdbqt"`` and a
            warning is logged — see the constructor body for the rule.
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

        # Automatically enable Vina rescore for each pose-producing method that
        # was requested. The Boltz-rescore and DiffDock-rescore tracks are
        # independent — running both methods produces both columns.
        # DiffDock gives many ranked poses; Boltz gives a confidence-only score
        # (ipTM) which doesn't reflect binding energy — re-scoring the predicted
        # complex with Vina's physics-based function provides a comparable ΔG.
        if (
            DIFFDOCK_PREFIX in self.methods_to_run
            and VINA_RESCORE_DIFFDOCK_PREFIX not in self.methods_to_run
        ):
            self.methods_to_run = list(self.methods_to_run) + [VINA_RESCORE_DIFFDOCK_PREFIX]
        if (
            BOLTZ_PREFIX in self.methods_to_run
            and VINA_RESCORE_BOLTZ_PREFIX not in self.methods_to_run
        ):
            self.methods_to_run = list(self.methods_to_run) + [VINA_RESCORE_BOLTZ_PREFIX]

        # Resolve gnina_input_mode against the (now auto-extended) methods
        # list. SDF-mode only applies when gnina is the *sole* PDBQT-relevant
        # method requested; if Vina or any of the Vina-rescore tracks are in
        # the mix, OpenBabel has to run for them anyway, so we silently fall
        # back to PDBQT for gnina (with a warning so the caller knows their
        # upstream protonation work will be overwritten by Gasteiger).
        _pdbqt_requiring = {
            VINA_PREFIX,
            VINA_RESCORE_BOLTZ_PREFIX,
            VINA_RESCORE_DIFFDOCK_PREFIX,
        }
        if (
            gnina_input_mode == "sdf"
            and GNINA_PREFIX in self.methods_to_run
            and not _pdbqt_requiring.intersection(self.methods_to_run)
        ):
            self.gnina_input_mode = "sdf"
        else:
            if gnina_input_mode == "sdf":
                logger.warning(
                    "SDF-mode requested for gnina but Vina/Vina-rescore is also "
                    "in methods_to_run — falling back to PDBQT for gnina. "
                    "OpenBabel ligand prep will still run, and any upstream "
                    "ligand protonation will be overwritten by OpenBabel's "
                    "Gasteiger pathway."
                )
            self.gnina_input_mode = "pdbqt"

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
    def _log_progress(*paths, message):
        """
        Append a timestamped line to one or more progress-log files. The
        per-batch path is ``{batch}/output.log`` (same file Guild workers
        write to via ``logging.basicConfig``, so there's one log per batch
        rather than two); the project-level path is
        ``{project}/batch_progress.log``. Failures are duplicated to both.
        ``None`` entries are skipped so callers can pass an unused slot.
        """
        line = f"{message} at {pd.Timestamp.now()}\n"
        for path in paths:
            if path is None:
                continue
            with open(path, "a") as f:
                f.write(line)

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
                box_location=task_params.get("box_location"),
                gnina_input_mode=task_params.get("gnina_input_mode", "pdbqt"),
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
    def _run_single_gnina_docking(task_params):
        """
        Run gnina docking for a single combination (worker function for multiprocessing).
        """
        try:
            logger.info(
                f"Starting gnina docking for SMILES: {task_params['ligand_smile']} "
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
                box_location=task_params.get("box_location"),
                gnina_input_mode=task_params.get("gnina_input_mode", "pdbqt"),
            )
            guild_object.run_gnina()
            return task_params["ligand_idx"], task_params["protein_idx"]
        except Exception as e:
            logger.error(
                f"Error in gnina docking for SMILES {task_params['ligand_smile']} "
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
                box_location=batch_params.get("box_location"),
                gnina_input_mode=batch_params.get("gnina_input_mode", "pdbqt"),
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

            combinations_table = self.batched_dictionary[current_batch][
                COMBINATIONS_TABLE_KEY
            ]
            # KarmaDock has no per-ligand worker downstream (deploy_karmadock
            # runs once per batch on an already-staged data dir), so the prep
            # pool must Guild-init every combination.
            # Vina, Boltz, gnina and DiffDock have per-ligand or per-batch
            # downstream workers that re-prep individual ligands themselves,
            # but they all depend on per-protein artifacts that Guild.__init__
            # writes under {batch}/proteins/{protein_idx}_single_chain_clean.pdb.
            # Those outputs are keyed on protein_idx only, so we Guild-init one
            # row per unique protein in the batch — enough to cover every
            # protein's cleaned PDB / PDBQT without wasting work on batches
            # with many ligands per protein. Picking only iloc[[0]] previously
            # left N-1 proteins unprepped in multi-protein batches, which made
            # Boltz silently fall back to the raw template PDB and fail.
            if KARMADOCK_PREFIX in self.methods_to_run:
                rows_to_prep = combinations_table.iterrows()
            else:
                rows_to_prep = combinations_table.drop_duplicates(
                    subset=[PROTEIN_CONF_ID], keep="first"
                ).iterrows()

            for _, current_row in rows_to_prep:
                batch_params_list.append(
                    {
                        "batch_name": current_batch,
                        "ligand_smile": self.batched_dictionary[current_batch][
                            SMILES_NAMES_DICTIONARY_KEY
                        ][current_row[LIGAND_ID]],
                        "ligand_idx": current_row[LIGAND_ID],
                        "protein_idx": current_row[PROTEIN_CONF_ID],
                        "protein_file": current_row[PROTEIN_PATH],
                        "project_name": f"{self.project_name}/{BATCHES_FOLDER}/{current_batch}",
                        "protein_chain": current_row[PROTEIN_CHAIN],
                        "original_ligand": current_row[ORIGINAL_LIGAND],
                        "original_ligand_chain": current_row[ORIGINAL_LIGAND_CHAIN],
                        "output_log_file": f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/{OUTPUT_LOG_FILE}",
                        "use_gpu": self.use_gpu,
                        "predict_binding_pocket": self.predict_binding_pocket,
                        "box_location": _row_box_location(current_row),
                        "gnina_input_mode": self.gnina_input_mode,
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
        Perform docking simulations using the methods listed in
        ``self.methods_to_run``. Per-batch method blocks dispatch in the
        order the user supplied — e.g. ``methods_to_run=["vina", "boltz"]``
        docks Vina first, ``["boltz", "vina"]`` docks Boltz first. Auto-added
        ``vina_rescore_*`` entries are no-ops here (their work runs during
        scoring).
        """
        self._prepare_docking()

        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"

        # Per-method per-batch runners. Methods not in this map (such as the
        # vina_rescore_* auto-additions) are skipped during docking.
        method_runners = {
            BOLTZ_PREFIX: self._run_boltz_for_batch,
            VINA_PREFIX: self._run_vina_for_batch,
            GNINA_PREFIX: self._run_gnina_for_batch,
            KARMADOCK_PREFIX: self._run_karmadock_for_batch,
            DIFFDOCK_PREFIX: self._run_diffdock_for_batch,
        }

        for current_batch in tqdm(self.batched_dictionary, desc="Running docking"):
            current_batch_folder = self.batched_dictionary[current_batch][BATCH_FOLDER]
            batch_progress_log = f"{current_batch_folder}/{OUTPUT_LOG_FILE}"

            # Mirror the batch-start marker to both logs: the project-level log
            # gives a one-line-per-batch overview; the per-batch log opens with
            # the same line so it's easy to align timing between the two.
            self._log_progress(
                main_progress_log, batch_progress_log,
                message=f"Starting {current_batch}",
            )

            if len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY]) == 0:
                logger.info(f"Skipping {current_batch} - no new combinations to dock")
                self._log_progress(
                    main_progress_log, batch_progress_log,
                    message=f"Skipped {current_batch} - no new combinations",
                )
                continue

            for method in self.methods_to_run:
                runner = method_runners.get(method)
                if runner is None:
                    continue
                runner(current_batch, current_batch_folder)

            self._log_progress(
                main_progress_log, batch_progress_log,
                message=f"Completed {current_batch}",
            )

    def _run_boltz_for_batch(self, current_batch, current_batch_folder):
        """Run Boltz docking for the batch, then generate complex PDBs."""
        batch_progress_log = f"{current_batch_folder}/{OUTPUT_LOG_FILE}"
        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"
        self._log_progress(batch_progress_log, message="Starting Boltz")

        # Boltz is applied in the whole batch at once, so we need to iterate over the unique protein configuration ids
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
            # ``protein_chain`` may name one chain ("A") or several ("A,B") for a
            # pocket that spans multiple chains. Keep only chains actually present
            # in the structure, aligned with their sequences and MSAs.
            current_protein_chains = _normalize_chain_list(
                current_protein_combinations[PROTEIN_CHAIN].values[0]
            )
            current_protein_path = current_protein_combinations[PROTEIN_PATH].values[0]
            original_sequence_dictionary = get_original_sequence_dictionary(
                current_protein_path
            )
            current_protein_chains = [
                c for c in current_protein_chains if c in original_sequence_dictionary
            ]
            current_protein_sequences = [
                process_into_fasta_string(original_sequence_dictionary[c])
                for c in current_protein_chains
            ]
            current_protein_unique_ligands_ids = list(
                current_protein_combinations[LIGAND_ID].unique()
            )

            # Fetch one MSA per chain — cached across all ligands and batches
            msa_files = [
                fetch_protein_msa(
                    sequence=sequence,
                    protein_id=unique_protein_configuration_id,
                    protein_chain_id=chain,
                    output_a3m_dir=self.msa_cache_dir,
                )
                for chain, sequence in zip(
                    current_protein_chains, current_protein_sequences, strict=True
                )
            ]
            logger.info(
                f"Fetched MSA(s) for protein configuration {unique_protein_configuration_id} "
                f"(chains {current_protein_chains})"
            )

            # Compute binding-pocket contacts once per protein configuration.
            # Precedence:  user-supplied box_location  >  original-ligand derivation.
            pocket_contacts = []
            _box_path = None
            if BOX_LOCATION in current_protein_combinations.columns:
                _box_values = [
                    _row_box_location(r)
                    for _, r in current_protein_combinations.iterrows()
                ]
                _non_empty_box_values = [b for b in _box_values if b]
                _unique_boxes = set(_non_empty_box_values)
                if _non_empty_box_values:
                    # Select deterministically using input order.
                    _box_path = _non_empty_box_values[0]
            else:
                _unique_boxes = set()

            if len(_unique_boxes) > 1:
                logger.warning(
                    f"Inconsistent box_location values for protein "
                    f"{unique_protein_configuration_id}: {sorted(_unique_boxes)}. "
                    f"Boltz pocket is per-protein; using {_box_path}."
                )
            if _box_path:
                if os.path.exists(_box_path):
                    _center, _size = get_center_and_size_from_box_file(_box_path)
                    pocket_contacts = get_pocket_contacts_from_box(
                        protein_pdb=current_protein_path,
                        protein_chain=current_protein_chains,
                        center=_center,
                        size=_size,
                    )
                    logger.info(
                        f"Derived {len(pocket_contacts)} Boltz pocket contacts from "
                        f"box {_box_path} for protein {unique_protein_configuration_id}"
                    )
                else:
                    logger.warning(
                        f"box_location {_box_path} not found on disk; falling back "
                        f"to original-ligand pocket derivation."
                    )

            if not pocket_contacts:
                _orig_lig = current_protein_combinations[ORIGINAL_LIGAND].values[0]
                _orig_lig_chain = current_protein_combinations[
                    ORIGINAL_LIGAND_CHAIN
                ].values[0]
                if (
                    _orig_lig
                    and _orig_lig_chain
                    and pd.notna(_orig_lig)
                    and pd.notna(_orig_lig_chain)
                ):
                    pocket_contacts = get_pocket_contacts_from_ligand(
                        protein_pdb=current_protein_path,
                        protein_chain=current_protein_chains,
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

                # Saving time by skipping already run Boltz dockings - checking for one of the expected output files.
                # Boltz writes outputs under {BOLTZ_FOLDER}/boltz_results_{run_id}_boltz/predictions/{run_id}_boltz/.
                boltz_result_fild = (
                    f"{BOLTZ_FOLDER}/boltz_results_{run_id}_boltz"
                    f"/predictions/{run_id}_boltz/confidence_{run_id}_boltz_model_0.json"
                )
                if os.path.exists(f"{current_batch_folder}/{boltz_result_fild}"):
                    logger.info(
                        f"Boltz docking already ran for {current_batch}: "
                        f"protein {unique_protein_configuration_id}, ligand {current_ligand_id}"
                    )
                    continue

                os.makedirs(f"{current_batch_folder}/{BOLTZ_FOLDER}", exist_ok=True)

                # Use single-chain PDB as template to avoid Boltz multi-chain parsing errors
                boltz_template_file = (
                    f"{current_batch_folder}/{PROTEINS_FOLDER}/"
                    f"{unique_protein_configuration_id}_single_chain_clean.pdb"
                )
                if not os.path.exists(boltz_template_file):
                    # Fallback to original protein path if single-chain not available
                    boltz_template_file = current_protein_path

                yaml_file = f"{current_batch_folder}/{BOLTZ_FOLDER}/{run_id}_boltz.yaml"
                boltz_out_dir = f"{current_batch_folder}/{BOLTZ_FOLDER}"

                generate_boltz_yaml(
                    protein_sequence=current_protein_sequences,
                    protein_chain=current_protein_chains,
                    ligand_sequences=[ligand_smiles],
                    ligand_ids=["L"],
                    output_file=yaml_file,
                    template_file=boltz_template_file,
                    pocket_contacts=pocket_contacts if pocket_contacts else None,
                    msa_file=msa_files,
                )

                # Per-combination subprocess log path. Boltz can exit 0 and
                # still produce an empty manifest, so we keep the transcript
                # regardless of success/failure for downstream troubleshooting.
                boltz_subprocess_log = (
                    f"{boltz_out_dir}/{run_id}.subprocess.log"
                )

                deploy_boltz(
                    yaml_file,
                    out_dir=boltz_out_dir,
                    use_gpu=self.use_gpu,
                    subprocess_log_path=boltz_subprocess_log,
                )

                # Check if Boltz produced valid output (manifest with records).
                # Template PDB parsing can fail silently in Boltz2, resulting
                # in an empty manifest.  If that happens, retry without the template.
                manifest_path = (
                    f"{boltz_out_dir}/boltz_results_{run_id}_boltz/processed/manifest.json"
                )
                if os.path.exists(manifest_path):
                    import json as _json

                    with open(manifest_path) as _mf:
                        _manifest = _json.load(_mf)
                    if not _manifest.get("records"):
                        logger.warning(
                            f"Boltz2 produced empty manifest for {run_id} "
                            "(likely template parsing failure). Retrying without template..."
                        )
                        # Remove the failed output directory
                        import shutil

                        failed_dir = f"{boltz_out_dir}/boltz_results_{run_id}_boltz"
                        if os.path.isdir(failed_dir):
                            shutil.rmtree(failed_dir)

                        # Regenerate YAML without template
                        generate_boltz_yaml(
                            protein_sequence=current_protein_sequences,
                            protein_chain=current_protein_chains,
                            ligand_sequences=[ligand_smiles],
                            ligand_ids=["L"],
                            output_file=yaml_file,
                            template_file=None,
                            pocket_contacts=pocket_contacts if pocket_contacts else None,
                            msa_file=msa_files,
                        )

                        # Retry overwrites the same subprocess.log — the
                        # latter run is the one whose outcome we report on.
                        deploy_boltz(
                            yaml_file,
                            out_dir=boltz_out_dir,
                            use_gpu=self.use_gpu,
                            subprocess_log_path=boltz_subprocess_log,
                        )

                # After any retry, verify the manifest is non-empty. If not,
                # surface this combo as a failure to both progress logs so it's
                # visible without diving into Boltz's processed/ tree.
                final_failure_reason = None
                if not os.path.exists(manifest_path):
                    final_failure_reason = "no manifest"
                else:
                    import json as _json

                    with open(manifest_path) as _mf:
                        _final_manifest = _json.load(_mf)
                    if not _final_manifest.get("records"):
                        final_failure_reason = "empty manifest after retry"
                if final_failure_reason is not None:
                    self._log_progress(
                        batch_progress_log,
                        main_progress_log,
                        message=(
                            f"FAILED Boltz {run_id} ({final_failure_reason}) "
                            f"— see {boltz_subprocess_log}"
                        ),
                    )

                logger.info(
                    f"Boltz docking completed for {current_batch}: "
                    f"protein {unique_protein_configuration_id}, ligand {current_ligand_id}"
                )

        # Generate complex PDB files — needed both for PLIP and for
        # Vina re-scoring of the predicted pose.
        logger.info(f"Generating complex PDB files for Boltz results in {current_batch}")
        generate_boltz_complex_pdbs(self.batched_dictionary[current_batch])

        self._log_progress(batch_progress_log, message="Completed Boltz")


    def _run_vina_for_batch(self, current_batch, current_batch_folder):
        """Run Vina docking for the batch, then generate complex PDBs."""
        batch_progress_log = f"{current_batch_folder}/{OUTPUT_LOG_FILE}"
        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"
        self._log_progress(batch_progress_log, message="Starting Vina")

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
                        "box_location": _row_box_location(current_row),
                        "gnina_input_mode": self.gnina_input_mode,
                    }
                )

        if vina_tasks:
            logger.info(
                f"Running {len(vina_tasks)} Vina docking tasks with {self.n_workers} workers"
            )
            # OpenBabel can raise a C-level abort during ligand prep that
            # kills the worker process. Once that happens, every still-pending
            # future in the same ProcessPoolExecutor raises BrokenProcessPool —
            # losing the rest of the batch. Recover by restarting the pool
            # with the remaining tasks, skipping the one that crashed.
            remaining = list(vina_tasks)
            completed = 0
            pool_restarts = 0
            max_pool_restarts = len(vina_tasks)  # hard upper bound

            while remaining:
                succeeded_indices = set()
                pool_broken = False

                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    future_to_task = {}
                    for idx, task in enumerate(remaining):
                        future = executor.submit(
                            self._run_single_vina_docking, task
                        )
                        future_to_task[future] = (idx, task)

                    desc_suffix = (
                        f" (restart {pool_restarts})" if pool_restarts else ""
                    )
                    for future in tqdm(
                        as_completed(future_to_task, timeout=None),
                        desc=(
                            f"AutoDock Vina for batch: {current_batch}"
                            f"{desc_suffix}"
                        ),
                        total=len(future_to_task),
                    ):
                        idx, task = future_to_task[future]
                        try:
                            future.result(timeout=DOCKING_TIMEOUT)
                            completed += 1
                            succeeded_indices.add(idx)
                        except FuturesTimeoutError:
                            logger.warning(
                                f"Task {idx + 1}/{len(remaining)} timed out after "
                                f"{DOCKING_TIMEOUT}s (SMILES: "
                                f"{task.get('ligand_smile', 'unknown')})"
                            )
                            succeeded_indices.add(idx)  # drop from retry
                            future.cancel()
                            self._log_progress(
                                batch_progress_log,
                                main_progress_log,
                                message=f"FAILED Vina {task['protein_idx']}_{task['ligand_idx']} (timeout)",
                            )
                        except BrokenProcessPool:
                            # Worker died (typically OpenBabel C++ abort).
                            # The current/queued futures all raise this;
                            # we identify the actual crasher after the loop.
                            pool_broken = True
                        except Exception as e:
                            logger.error(
                                f"Task {idx + 1}/{len(remaining)} failed: {e} "
                                f"(SMILES: {task.get('ligand_smile', 'unknown')})"
                            )
                            succeeded_indices.add(idx)  # drop from retry
                            self._log_progress(
                                batch_progress_log,
                                main_progress_log,
                                message=f"FAILED Vina {task['protein_idx']}_{task['ligand_idx']} ({e})",
                            )

                if not pool_broken:
                    break

                not_succeeded = sorted(
                    set(range(len(remaining))) - succeeded_indices
                )
                if not not_succeeded:
                    break
                crasher = remaining[not_succeeded[0]]
                self._log_progress(
                    batch_progress_log,
                    main_progress_log,
                    message=(
                        f"FAILED Vina {crasher['protein_idx']}_{crasher['ligand_idx']} "
                        "(BrokenProcessPool — presumed OpenBabel C++ abort)"
                    ),
                )
                logger.warning(
                    f"BrokenProcessPool: skipping presumed crasher "
                    f"ligand={crasher['ligand_idx']} "
                    f"SMILES={crasher.get('ligand_smile', 'unknown')}; "
                    f"restarting pool with {len(not_succeeded) - 1} remaining tasks"
                )
                remaining = [remaining[i] for i in not_succeeded[1:]]
                pool_restarts += 1
                if pool_restarts > max_pool_restarts:
                    logger.error(
                        f"Hit pool-restart cap ({max_pool_restarts}); "
                        f"bailing with {len(remaining)} tasks unfinished."
                    )
                    break

            logger.info(
                f"Completed {completed}/{len(vina_tasks)} Vina docking tasks "
                f"(pool restarts: {pool_restarts})"
            )
        else:
            logger.info(f"No new Vina combinations to dock for batch {current_batch}")

        # Generate complex PDB files for Vina results
        logger.info(f"Generating complex PDB files for Vina results in {current_batch}")
        generate_vina_complex_pdbs(self.batched_dictionary[current_batch])

        self._log_progress(batch_progress_log, message="Completed Vina")


    def _run_gnina_for_batch(self, current_batch, current_batch_folder):
        """Run gnina docking for the batch, then generate complex PDBs."""
        batch_progress_log = f"{current_batch_folder}/{OUTPUT_LOG_FILE}"
        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"
        self._log_progress(batch_progress_log, message="Starting gnina")

        gnina_tasks = []
        for _index, current_row in self.batched_dictionary[current_batch][
            COMBINATIONS_TABLE_KEY
        ].iterrows():
            protein_path = current_row[PROTEIN_PATH]
            current_protein, current_ligand = (
                current_row[PROTEIN_CONF_ID],
                current_row[LIGAND_ID],
            )
            gnina_final_path = (
                f"{self.batched_dictionary[current_batch][BATCH_FOLDER]}/"
                f"{GNINA_FOLDER}/{current_protein}_{current_ligand}.txt"
            )
            if not os.path.exists(gnina_final_path):
                gnina_tasks.append(
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
                        "box_location": _row_box_location(current_row),
                        "gnina_input_mode": self.gnina_input_mode,
                    }
                )

        if gnina_tasks:
            logger.info(
                f"Running {len(gnina_tasks)} gnina docking tasks with {self.n_workers} workers"
            )
            # gnina shares Vina's PDBQT ligand prep, so the same OpenBabel
            # C-level abort risk applies; reuse the BrokenProcessPool
            # recovery loop with a different worker function.
            remaining = list(gnina_tasks)
            completed = 0
            pool_restarts = 0
            max_pool_restarts = len(gnina_tasks)

            while remaining:
                succeeded_indices = set()
                pool_broken = False

                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    future_to_task = {}
                    for idx, task in enumerate(remaining):
                        future = executor.submit(
                            self._run_single_gnina_docking, task
                        )
                        future_to_task[future] = (idx, task)

                    desc_suffix = (
                        f" (restart {pool_restarts})" if pool_restarts else ""
                    )
                    for future in tqdm(
                        as_completed(future_to_task, timeout=None),
                        desc=(
                            f"gnina for batch: {current_batch}"
                            f"{desc_suffix}"
                        ),
                        total=len(future_to_task),
                    ):
                        idx, task = future_to_task[future]
                        try:
                            future.result(timeout=DOCKING_TIMEOUT)
                            completed += 1
                            succeeded_indices.add(idx)
                        except FuturesTimeoutError:
                            logger.warning(
                                f"gnina task {idx + 1}/{len(remaining)} timed out after "
                                f"{DOCKING_TIMEOUT}s (SMILES: "
                                f"{task.get('ligand_smile', 'unknown')})"
                            )
                            succeeded_indices.add(idx)
                            future.cancel()
                            gnina_log = (
                                f"{current_batch_folder}/{GNINA_FOLDER}/"
                                f"{task['protein_idx']}_{task['ligand_idx']}.subprocess.log"
                            )
                            self._log_progress(
                                batch_progress_log,
                                main_progress_log,
                                message=(
                                    f"FAILED gnina {task['protein_idx']}_{task['ligand_idx']} "
                                    f"(timeout) — see {gnina_log}"
                                ),
                            )
                        except BrokenProcessPool:
                            pool_broken = True
                        except Exception as e:
                            logger.error(
                                f"gnina task {idx + 1}/{len(remaining)} failed: {e} "
                                f"(SMILES: {task.get('ligand_smile', 'unknown')})"
                            )
                            succeeded_indices.add(idx)
                            gnina_log = (
                                f"{current_batch_folder}/{GNINA_FOLDER}/"
                                f"{task['protein_idx']}_{task['ligand_idx']}.subprocess.log"
                            )
                            self._log_progress(
                                batch_progress_log,
                                main_progress_log,
                                message=(
                                    f"FAILED gnina {task['protein_idx']}_{task['ligand_idx']} "
                                    f"({e}) — see {gnina_log}"
                                ),
                            )

                if not pool_broken:
                    break

                not_succeeded = sorted(
                    set(range(len(remaining))) - succeeded_indices
                )
                if not not_succeeded:
                    break
                crasher = remaining[not_succeeded[0]]
                logger.warning(
                    f"BrokenProcessPool (gnina): skipping presumed crasher "
                    f"ligand={crasher['ligand_idx']} "
                    f"SMILES={crasher.get('ligand_smile', 'unknown')}; "
                    f"restarting pool with {len(not_succeeded) - 1} remaining tasks"
                )
                gnina_crash_log = (
                    f"{current_batch_folder}/{GNINA_FOLDER}/"
                    f"{crasher['protein_idx']}_{crasher['ligand_idx']}.subprocess.log"
                )
                self._log_progress(
                    batch_progress_log,
                    main_progress_log,
                    message=(
                        f"FAILED gnina {crasher['protein_idx']}_{crasher['ligand_idx']} "
                        f"(BrokenProcessPool — presumed OpenBabel C++ abort) "
                        f"— see {gnina_crash_log} (may not exist if the crash happened pre-subprocess)"
                    ),
                )
                remaining = [remaining[i] for i in not_succeeded[1:]]
                pool_restarts += 1
                if pool_restarts > max_pool_restarts:
                    logger.error(
                        f"Hit gnina pool-restart cap ({max_pool_restarts}); "
                        f"bailing with {len(remaining)} tasks unfinished."
                    )
                    break

            logger.info(
                f"Completed {completed}/{len(gnina_tasks)} gnina docking tasks "
                f"(pool restarts: {pool_restarts})"
            )
        else:
            logger.info(f"No new gnina combinations to dock for batch {current_batch}")

        # Generate complex PDB files for gnina results (shares Vina's PDBQT format)
        logger.info(f"Generating complex PDB files for gnina results in {current_batch}")
        generate_gnina_complex_pdbs(self.batched_dictionary[current_batch])

        self._log_progress(batch_progress_log, message="Completed gnina")


    def _run_karmadock_for_batch(self, current_batch, current_batch_folder):
        """Run KarmaDock for the batch (single-shot, batch-level)."""
        batch_progress_log = f"{current_batch_folder}/{OUTPUT_LOG_FILE}"
        self._log_progress(batch_progress_log, message="Starting KarmaDock")

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

        self._log_progress(batch_progress_log, message="Completed KarmaDock")


    def _run_diffdock_for_batch(self, current_batch, current_batch_folder):
        """Run DiffDock for the batch. Generates complex PDBs on resume, then docks."""
        batch_progress_log = f"{current_batch_folder}/{OUTPUT_LOG_FILE}"
        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"
        self._log_progress(batch_progress_log, message="Starting DiffDock")

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

        batch_csv = f"{current_batch_folder}/{DIFFDOCK_COMBINATIONS_FILE}"
        batch_results_dir = (
            f"{current_batch_folder}/{DIFFDOCK_FOLDER}/{DIFFDOCK_RESULTS_FOLDER}"
        )

        # DiffDock writes rank{N}_confidence-{score}.sdf files into
        # batch_results_dir/{protein_conf_id}_{ligand_id}/. A batch is
        # only "done" when every combination has at least one such SDF.
        # Checking len(os.listdir(batch_results_dir)) > 0 is not enough:
        # an interrupted run leaves per-combo subfolders that are empty
        # (or absent), and the next run would otherwise skip DiffDock
        # and produce all-NaN scores downstream.
        combinations_df = self.batched_dictionary[current_batch][
            COMBINATIONS_TABLE_KEY
        ]

        missing = []
        for _, row in combinations_df.iterrows():
            combo_dir = os.path.join(
                batch_results_dir,
                f"{row[PROTEIN_CONF_ID]}_{row[LIGAND_ID]}",
            )
            has_pose = os.path.isdir(combo_dir) and any(
                "_confidence" in fname and fname.endswith(".sdf")
                for fname in os.listdir(combo_dir)
            )
            if not has_pose:
                missing.append((row[PROTEIN_CONF_ID], row[LIGAND_ID]))

        if not missing:
            logger.info(
                f"DiffDock batch already ran for {current_batch} "
                f"({len(combinations_df)} combinations have poses on disk)"
            )
        else:
            logger.info(
                f"DiffDock for {current_batch}: "
                f"{len(missing)}/{len(combinations_df)} combinations "
                f"missing docked poses; re-running batch"
            )
            # DiffDock runs once per batch (not per combo), so the subprocess
            # log is batch-level. All per-combo failures point at this same
            # file for context.
            diffdock_subprocess_log = (
                f"{current_batch_folder}/{DIFFDOCK_FOLDER}/_batch.subprocess.log"
            )
            deploy_diffdock(
                home_path=self.home_path,
                diffdock_results_dir=batch_results_dir,
                input_csv=batch_csv,
                subprocess_log_path=diffdock_subprocess_log,
                # use_gpu=self.use_gpu,
            )
            logger.info(
                f"DiffDock batch completed for {current_batch} with "
                f"{len(self.batched_dictionary[current_batch][COMBINATIONS_TO_RUN_KEY])} "
                f"combinations"
            )

            # Generate per-combination complex PDBs from the SDFs DiffDock
            # just wrote. The resume-path call above only fires when the
            # results dir already existed at entry, so on a fresh run we'd
            # otherwise never get complex PDBs — which then leaves PLIP
            # with nothing to analyze for DiffDock. The generator is
            # idempotent (skips combos whose *_complex.pdb already exists).
            if os.path.isdir(batch_results_dir):
                logger.info(
                    f"Generating complex PDB files for DiffDock results in {current_batch} (post-docking)"
                )
                generate_diffdock_complex_pdbs(self.batched_dictionary[current_batch])

            # Re-check which combos still lack a docked pose after the re-run
            # and surface them as explicit failures in both progress logs.
            for protein_conf_id, ligand_id in missing:
                combo_dir = os.path.join(
                    batch_results_dir, f"{protein_conf_id}_{ligand_id}"
                )
                still_missing = not (
                    os.path.isdir(combo_dir)
                    and any(
                        "_confidence" in fname and fname.endswith(".sdf")
                        for fname in os.listdir(combo_dir)
                    )
                )
                if still_missing:
                    self._log_progress(
                        batch_progress_log,
                        main_progress_log,
                        message=(
                            f"FAILED DiffDock {protein_conf_id}_{ligand_id} "
                            f"(no _confidence SDF produced) — see {diffdock_subprocess_log}"
                        ),
                    )

        self._log_progress(batch_progress_log, message="Completed DiffDock")


    def _compute_global_ranks_per_protein(self, all_raw_scores):
        """
        Compute ranks and rank percentile scores PER PROTEIN across ALL data (all batches).
        This ensures rankings are consistent across batches - all ligands for a given protein
        are ranked together, not separately per batch.

        :param all_raw_scores: DataFrame with raw scores from all batches.
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
        batch_folder = self.batched_dictionary[batch][BATCH_FOLDER]
        batch_progress_log = f"{batch_folder}/{OUTPUT_LOG_FILE}"
        self._log_progress(batch_progress_log, message="Starting scoring")

        logger.info(f"Collecting raw scores for batch {batch}")
        if len(self.batched_dictionary[batch]["combinations_to_run"]) == 0:
            logger.info("No new combinations to score")
            self._log_progress(
                batch_progress_log,
                message="Completed scoring (no new combinations)",
            )
            return None

        docked_methods = []

        if VINA_PREFIX in self.methods_to_run:
            docked_methods.append(vina_guild_scoring(self.batched_dictionary[batch]))

        if GNINA_PREFIX in self.methods_to_run:
            docked_methods.append(gnina_guild_scoring(self.batched_dictionary[batch]))

        if KARMADOCK_PREFIX in self.methods_to_run:
            docked_methods.append(karmadock_guild_scoring(self.batched_dictionary[batch]))

        if DIFFDOCK_PREFIX in self.methods_to_run:
            docked_methods.append(diffdock_guild_scoring(self.batched_dictionary[batch]))

        if VINA_RESCORE_DIFFDOCK_PREFIX in self.methods_to_run:
            docked_methods.append(
                vina_rescore_diffdock_guild_scoring(self.batched_dictionary[batch])
            )

        if VINA_RESCORE_BOLTZ_PREFIX in self.methods_to_run:
            docked_methods.append(
                vina_rescore_boltz_guild_scoring(self.batched_dictionary[batch])
            )

        if BOLTZ_PREFIX in self.methods_to_run:
            docked_methods.append(boltz_guild_scoring(self.batched_dictionary[batch]))

        if not docked_methods:
            self._log_progress(
                batch_progress_log,
                message="Completed scoring (no methods produced output)",
            )
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

        self._log_progress(batch_progress_log, message="Completed scoring")

        # Return RAW scores only - no ranks or rank percentile scores yet
        return raw_scores_table

    def run_guild_scoring(self, n_processes=None):
        """
        Scoring function to uniformize multiple docking methods by leveraging the decoy dataset.
        Steps:
        1. Collect RAW scores from all batches (parallel)
        2. Compute ranks PER PROTEIN across ALL data (not per batch)
        3. Compute rank percentile scores from global ranks

        :param n_processes: Number of processes to use for multiprocessing.
        """

        if n_processes is None:
            n_processes = mp.cpu_count()

        main_progress_log = f"{self.project_folder}/{BATCH_PROGRESS_LOG_FILE}"
        self._log_progress(main_progress_log, message="Starting scoring")

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

        # Ensure ligand category is available before scoring.
        # Key the lookup on ligand_id (authoritative per-row category from the
        # combinations table), NOT on SMILES. The SMILES-keyed dictionary both
        # (a) loses the per-row category when two ligands share a SMILES, and
        # (b) silently yields "unknown" whenever an upstream key mishap leaves
        # SMILES itself as "unknown".
        category_by_ligand = dict(
            zip(
                self.all_combinations_table[LIGAND_ID],
                self.all_combinations_table[LIGAND_CATEGORY],
                strict=False,
            )
        )
        if LIGAND_CATEGORY not in all_raw_scores.columns:
            all_raw_scores[LIGAND_CATEGORY] = (
                all_raw_scores[LIGAND_ID].map(category_by_ligand).fillna("unknown")
            )

        # Collapse any duplicate COMBINATION_ID rows by *coalescing* columns
        # (first non-null per column) rather than keeping a single row. Within a
        # batch the per-method outer-merge already yields one row per
        # combination; coalescing protects against any concatenation that pairs
        # rows carrying a complementary set of method-score columns, merging them
        # instead of letting one row win and dropping the other's scores.
        all_raw_scores = (
            all_raw_scores
            .groupby(COMBINATION_ID, as_index=False, sort=False)
            .first()
        )

        # Step 2: Compute ranks PER PROTEIN across ALL data
        logger.info(
            f"Computing global ranks per protein for {all_raw_scores.shape[0]} total combinations"
        )
        self.rp_scores_df = self._compute_global_ranks_per_protein(all_raw_scores)

        # Step 3: Add ligand category
        if LIGAND_CATEGORY not in self.rp_scores_df.columns:
            self.rp_scores_df[LIGAND_CATEGORY] = (
                self.rp_scores_df[LIGAND_ID].map(category_by_ligand).fillna("unknown")
            )

        # Step 4: Save results
        self.rp_scores_df.to_csv(
            self.rp_scores_path,
            index=False,
            sep="\t",
        )
        logger.info(f"Saved rank percentile scores to {self.rp_scores_path}")

        self._log_progress(main_progress_log, message="Completed scoring")

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
                        ligand_resname=LIGAND_RESNAME,
                        ligand_chain=LIGAND_CHAIN,
                        ligand_resseq=LIGAND_RESSEQ,
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
                        ligand_resname=LIGAND_RESNAME,
                        ligand_chain=LIGAND_CHAIN,
                        ligand_resseq=LIGAND_RESSEQ,
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
                        ligand_resname=LIGAND_RESNAME,
                        ligand_chain=LIGAND_CHAIN,
                        ligand_resseq=LIGAND_RESSEQ,
                    )
                    if not batch_interactions.empty:
                        batch_interactions[PROTEIN_CONF_ID] = [x[1] for x in complex_metadata]
                        batch_interactions[SMILES] = [x[2] for x in complex_metadata]
                        all_interactions.append(batch_interactions)
                else:
                    logger.info(f"No DiffDock complex PDB files found for {current_batch}")

            if GNINA_PREFIX in methods_to_analyze:
                gnina_folder = f"{batch_folder}/{GNINA_FOLDER}"

                complex_metadata = []
                for _, row in combinations_df.iterrows():
                    protein_conf_id = row[PROTEIN_CONF_ID]
                    ligand_id = row[LIGAND_ID]
                    combination_id = f"{protein_conf_id}_{ligand_id}"
                    complex_pdb = f"{gnina_folder}/{combination_id}{COMPLEX_PDB_SUFFIX}"
                    if os.path.exists(complex_pdb):
                        smiles = batch_dict[SMILES_NAMES_DICTIONARY_KEY][ligand_id]
                        complex_metadata.append((complex_pdb, protein_conf_id, smiles))

                if complex_metadata:
                    logger.info(
                        f"Analyzing {len(complex_metadata)} gnina complexes in {current_batch}"
                    )
                    batch_interactions = analyze_batch_interactions(
                        complex_pdb_paths=[x[0] for x in complex_metadata],
                        combination_ids=[f"{x[1]}_{x[2]}" for x in complex_metadata],
                        ligand_resname=LIGAND_RESNAME,
                        ligand_chain=LIGAND_CHAIN,
                        ligand_resseq=LIGAND_RESSEQ,
                    )
                    if not batch_interactions.empty:
                        batch_interactions[PROTEIN_CONF_ID] = [x[1] for x in complex_metadata]
                        batch_interactions[SMILES] = [x[2] for x in complex_metadata]
                        all_interactions.append(batch_interactions)
                else:
                    logger.info(f"No gnina complex PDB files found for {current_batch}")

        interactions_path = f"{self.project_folder}/{INTERACTIONS_FILE}"

        if not all_interactions:
            # Contract: external consumers (e.g. host-side notebooks that read
            # the file with pandas) should always be able to read this path
            # after a guild run, even if no method produced complex PDBs. Write
            # a header-only TSV with the columns analyze_batch_interactions
            # emits so downstream code paths are deterministic.
            logger.warning(
                "No interaction data collected — writing header-only TSV to "
                f"{interactions_path} so the contract 'file always exists' holds."
            )
            empty_cols = [
                PROTEIN_CONF_ID,
                SMILES,
                INTERACTION_COMBINATION_ID,
                *INTERACTION_COUNT_COLUMNS,
                TOTAL_INTERACTIONS,
                N_UNIQUE_RESIDUES,
            ]
            self.interactions_df = pd.DataFrame(columns=empty_cols)
            self.interactions_df.to_csv(interactions_path, sep="\t", index=False)
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
            if col not in [PROTEIN_CONF_ID, SMILES, INTERACTION_COMBINATION_ID]
        ]
        df_to_save = self.interactions_df[cols_order]

        # Save to file (interactions_path computed at the top of this method)
        df_to_save.to_csv(interactions_path, sep="\t", index=False)
        logger.info(f"Saved interaction analysis to {interactions_path}")

        # Log summary statistics
        logger.info(
            f"Interaction analysis complete: {len(self.interactions_df)} complexes analyzed"
        )
        logger.info(
            f"Average interactions per complex: {self.interactions_df[TOTAL_INTERACTIONS].mean():.2f}"
        )
        logger.info(
            f"Average unique residues per complex: {self.interactions_df[N_UNIQUE_RESIDUES].mean():.2f}"
        )

        return self.interactions_df
