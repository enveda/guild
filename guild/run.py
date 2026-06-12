"""Main module."""

import logging
import os
import subprocess
from shutil import copyfile

from guild.constants.boltz import (
    BOLTZ_YAML_FILE,
)
from guild.constants.diffdock import (
    DIFFDOCK_RESULTS_FOLDER,
)
from guild.constants.guild import (
    # Lists and dictionaries
    ALL_AVAILABLE_METHODS,
    BOLTZ_FOLDER,
    BOLTZ_PREFIX,
    BOXES_FOLDER,
    DATA_FOLDER,
    DIFFDOCK_FOLDER,
    DIFFDOCK_PREFIX,
    GNINA_FOLDER,
    GNINA_PREFIX,
    KARMADOCK_FOLDER,
    KARMADOCK_PREFIX,
    LIGANDS_FOLDER,
    MSA_FOLDER,
    PLOTS_FOLDER,
    PROTEINS_FOLDER,
    # Folders
    VINA_FOLDER,
    # Prefixes
    VINA_PREFIX,
)
from guild.constants.karmadock import (
    KARMADOCK_DATA_FOLDER,
    KARMADOCK_GRAPHS_FOLDER,
    KARMADOCK_RESULTS_FOLDER,
)
from guild.constants.p2rank import (
    P2RANK_DEFAULT_POCKET_RANK,
    P2RANK_FOLDER,
    P2RANK_MIN_PROBABILITY_THRESHOLD,
)
from guild.constants.system import SHELL_SILENCER, WORKING_DIR_PATH
from guild.constants.vina import VINA_BOXES_FOLDER
from guild.docking.boltz import deploy_boltz, generate_boltz_yaml
from guild.docking.diffdock import deploy_diffdock_single
from guild.docking.gnina import deploy_gnina
from guild.docking.karmadock import deploy_karmadock
from guild.docking.vina import (
    deploy_vina,
    generate_vina_box,
    get_center_and_size_from_box_file,
)
from guild.tools.p2rank import get_binding_site_center_from_p2rank
from guild.tools.preparation import (
    _normalize_chain_list,
    clean_receptor,
    get_protein_chain,
    isolate_protein_chain,
)
from guild.tools.protein_sequence import (
    get_original_sequence_dictionary,
    process_into_fasta_string,
)
from guild.tools.utils import timeit
from guild.transformers.converters import (
    ligand_pdb_to_pdbqt,
    protein_pdb_to_pdbqt,
    sdf_to_pdb,
    smiles_to_sdf,
    smiles_to_sdf_karmadock,
)
from guild.transformers.msa import fetch_protein_msa
from guild.transformers.pdb import (
    LigandIdentifier,
    get_pocket_contacts_from_box,
    get_pocket_contacts_from_ligand,
)

logger = logging.getLogger(__name__)


class Guild:
    """
    Guild class for docking ligands to proteins.
    """

    def __init__(
        self,
        ligand_smile: str,
        ligand_idx: str,
        protein_idx: str,
        protein_file: str,
        project_name: str,
        protein_chain: str = None,
        original_ligand: str = None,
        original_ligand_chain: str = None,
        output_log_file: str = None,
        use_gpu: bool = True,
        is_bulk: bool = False,
        predict_binding_pocket: bool = False,
        box_location: str = None,
        gnina_input_mode: str = "pdbqt",
    ):
        """
        Start the Guild for docking ligands to proteins.
        :param ligand_smile: The SMILES string of the ligand.
        :param ligand_idx: The index of the ligand.
        :param protein_idx: The index of the protein.
        :param protein_file: The path to the protein data file.
        :param project_name: The name of the project.
        :param protein_chain: The chain to be used for docking. If not provided, the first chain is used.
        :param original_ligand: The original ligand in the protein data. If not provided, the first ligand is used.
        :param original_ligand_chain: The chain of the original ligand in the protein data. If not provided, the first chain is used.
        :param output_log_file: The file to save the log. If not provided, the log is saved in the project directory.
        :param use_gpu: Use GPU.
        :param is_bulk: Whether the run is part of a bulk run.
        :param predict_binding_pocket: Use P2Rank for binding site prediction instead of original ligand location.
        :param box_location: Path to a user-supplied Vina box file (center_{x,y,z} + size_{x,y,z}).
                             When set, takes precedence over P2Rank / original-ligand pocket derivation
                             for both Vina (used as the Vina box) and Boltz (residues inside the box
                             become the pocket contact constraint).
        """
        self.original_ligand_smile = ligand_smile
        self.ligand_idx = ligand_idx

        self.original_protein_data = protein_file
        self.protein_idx = protein_idx

        self.protein_file_extension = (
            self.original_protein_data.split("/")[-1].split(".")[1].lower()
        )
        self.combination_id = f"{self.protein_idx}_{self.ligand_idx}"

        # Resolve the protein chain(s). ``protein_chain`` may be a single ID
        # ("A"), a list, or a comma-separated string ("A,B") for a pocket that
        # spans multiple chains (e.g. a dimer interface). ``self.protein_chains``
        # is the canonical list; ``self.protein_chain`` keeps the first chain as
        # the "primary" for single-chain code paths (box / original-ligand
        # pocket derivation, MSA/file naming).
        self.protein_chains = _normalize_chain_list(protein_chain)
        if not self.protein_chains:
            self.protein_chains = [
                get_protein_chain(self.original_protein_data, self.protein_idx)
            ]
        self.protein_chain = self.protein_chains[0]

        self.original_ligand = original_ligand
        self.original_ligand_chain = original_ligand_chain

        self.project_name = project_name
        self.home_path = WORKING_DIR_PATH
        self.use_gpu = use_gpu
        self.is_bulk = is_bulk
        self.predict_binding_pocket = predict_binding_pocket
        # ``gnina_input_mode == "sdf"`` skips OpenBabel-driven PDBQT prep for
        # both the ligand (in ``_prepare_ligand``) and the receptor (in
        # ``run_gnina``). BulkRun resolves the effective mode globally —
        # falling back to "pdbqt" when Vina/Vina-rescore is also requested —
        # so by the time we get here the value is final.
        self.gnina_input_mode = gnina_input_mode
        # Validate box_location up-front. An unreadable path is treated as
        # "not provided" so the normal P2Rank / original-ligand fallback chain
        # still kicks in instead of silently leaving Vina/Boltz without a
        # pocket. Downstream code can then trust ``self.box_location is not
        # None`` as the single source of truth for "user supplied a box".
        if box_location:
            if os.path.exists(box_location):
                self.box_location = box_location
            else:
                logger.warning(
                    f"box_location {box_location} does not exist on disk; "
                    "falling back to P2Rank / original-ligand for the binding pocket."
                )
                self.box_location = None
        else:
            self.box_location = None
        self._set_paths(home_path=self.home_path, output_log_file=output_log_file)

        self._create_directories()
        logger.info("Successfully created directories.")

        self._start_log_file()

        # Write to the log file
        logger.info(f"Project directory created at: {self.project_dir}")

        assert self.protein_file_extension == "pdb", "Only PDB files are supported for now."

        try:
            self._relocate_files()
            logger.info("Files relocated to the appropriate directories.")
        except Exception as e:
            logger.error(f"Error in relocating files: {e}")

        try:
            self._prepare_ligand()
            logger.info("Ligand data prepared.")
        except Exception as e:
            logger.error(f"Error in preparing ligand data: {e}")

        try:
            self._prepare_protein(target_chain=self.protein_chains)
            logger.info(f"Protein data prepared for chain(s) {self.protein_chains}.")
        except Exception as e:
            logger.error(f"Error in preparing protein data: {e}")

        try:
            self._prepare_ligand_for_karmadock()
            logger.info("KarmaDock ligand prepared.")
        except Exception as e:
            logger.error(f"Error preparing KarmaDock ligand: {e}")
            raise

    def _set_paths(self, home_path: str = None, output_log_file: str = None):
        """
        Set the paths for the project.
        """
        if home_path is not None:
            self.home_path = home_path
        else:
            self.home_path = os.getcwd()
        self.output_log_file = output_log_file

        # Project directories
        self.project_dir = f"{self.home_path}/data/{self.project_name}"
        self.plots_dir = f"{self.project_dir}/{PLOTS_FOLDER}"
        self.boxes_dir = f"{self.project_dir}/{BOXES_FOLDER}"
        self.vina_boxes_dir = f"{self.boxes_dir}/{VINA_BOXES_FOLDER}"
        self.ligand_dir = f"{self.project_dir}/{LIGANDS_FOLDER}"
        self.protein_dir = f"{self.project_dir}/{PROTEINS_FOLDER}"
        self.local_protein = f"{self.protein_dir}/{self.protein_idx}_raw.pdb"
        self.vina_box = f"{self.vina_boxes_dir}/{self.protein_idx}_{self.ligand_idx}.txt"
        self.msa_cache_dir = f"{self.project_dir}/{MSA_FOLDER}"

        # Files to be used for docking
        self.protein_file = f"{self.protein_dir}/{self.protein_idx}.pdb"
        self.single_chain_protein = f"{self.protein_dir}/{self.protein_idx}_single_chain.pdb"
        self.cleaned_protein = self.single_chain_protein.replace(".pdb", "_clean.pdb")
        self.cleaned_protein_pdbqt = self.cleaned_protein.replace(".pdb", ".pdbqt")

        self.ligand_pdb = f"{self.ligand_dir}/{self.ligand_idx}_ligand.pdb"
        self.ligand_pdbqt = self.ligand_pdb.replace(".pdb", ".pdbqt")
        self.ligand_mol2 = f"{self.ligand_dir}/{self.ligand_idx}_ligand.mol2"
        self.ligand_sdf = f"{self.ligand_dir}/{self.ligand_idx}_ligand.sdf"

        # Autodock directories
        self.vina_dir = f"{self.project_dir}/{VINA_FOLDER}"
        self.vina_output_pdbqt = f"{self.vina_dir}/{self.protein_idx}_{self.ligand_idx}.pdbqt"
        self.vina_output_scores = f"{self.vina_dir}/{self.protein_idx}_{self.ligand_idx}.txt"

        # GNINA directories — reuses the Vina box file (same format) and the
        # PDBQT-prepped receptor/ligand produced for Vina.
        self.gnina_dir = f"{self.project_dir}/{GNINA_FOLDER}"
        self.gnina_output_pdbqt = f"{self.gnina_dir}/{self.protein_idx}_{self.ligand_idx}.pdbqt"
        self.gnina_output_scores = f"{self.gnina_dir}/{self.protein_idx}_{self.ligand_idx}.txt"

        # KarmaDock directories
        self.karmadock_root = f"{self.project_dir}/{KARMADOCK_FOLDER}"
        self.karmadock_graphs_dir = f"{self.karmadock_root}/{KARMADOCK_GRAPHS_FOLDER}"
        self.karmadock_results_dir = f"{self.karmadock_root}/{KARMADOCK_RESULTS_FOLDER}"
        self.karmadock_data_dir = f"{self.karmadock_root}/{KARMADOCK_DATA_FOLDER}"
        self.karmadock_data_sub_dir = f"{self.karmadock_data_dir}/{self.combination_id}"
        self.karmadock_ligand = f"{self.karmadock_data_sub_dir}/{self.combination_id}_ligand.sdf"
        self.karmadock_protein = f"{self.karmadock_data_sub_dir}/{self.combination_id}_protein.pdb"
        self.karmadock_mol2 = f"{self.karmadock_data_sub_dir}/{self.combination_id}_ligand.mol2"

        # DiffDock directories
        self.diffdock_dir = f"{self.project_dir}/{DIFFDOCK_FOLDER}"
        self.diffdock_results_dir = f"{self.diffdock_dir}/{DIFFDOCK_RESULTS_FOLDER}"

        # Boltz directories only for single runs
        self.boltz_dir = f"{self.project_dir}/{BOLTZ_FOLDER}"
        if not self.is_bulk:
            self.boltz_subdir = f"{self.boltz_dir}/{self.protein_idx}_{self.ligand_idx}"
            self.boltz_yaml = (
                f"{self.boltz_dir}/{self.protein_idx}_{self.ligand_idx}_{BOLTZ_YAML_FILE}"
            )

        # P2Rank directories for binding site prediction
        self.p2rank_dir = f"{self.project_dir}/{P2RANK_FOLDER}"

    def _start_log_file(self):
        """
        Start the log file for the project.
        """
        if self.output_log_file is None:
            self.output_log_file = f"{self.project_dir}/output.log"

        logging.basicConfig(
            filename=self.output_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _create_directories(self):
        """
        Create the directories for the project.
        """

        # Start making project directory
        for dir_name in [
            f"{self.home_path}/{DATA_FOLDER}",
            self.project_dir,
            self.plots_dir,
            self.boxes_dir,
            self.vina_boxes_dir,
            self.ligand_dir,
            self.protein_dir,
            self.vina_dir,
            self.gnina_dir,
            self.karmadock_root,
            self.karmadock_data_dir,
            self.karmadock_data_sub_dir,
            self.karmadock_graphs_dir,
            self.karmadock_results_dir,
            self.diffdock_dir,
            self.diffdock_results_dir,
            self.boltz_dir,
            self.p2rank_dir,
        ]:
            os.makedirs(dir_name, exist_ok=True)

        if not self.is_bulk:
            os.makedirs(self.boltz_subdir, exist_ok=True)

    def _refine_protein_data(self):
        """
        Refine the protein data to only include the required chain and ligand.
        Generates the refined protein, ligand, and box files.
        """
        current_object = LigandIdentifier(
            self.local_protein,
            usable_chain=self.protein_chain,
            heteroatoms=[self.original_ligand],
            heteroatoms_chain=self.original_ligand_chain,
            output_ligand_file=self.ligand_pdb,
            output_protein_file=self.single_chain_protein,
            config_file_name=self.vina_box,
        )

        current_object.run()

    def _relocate_files(self):
        """
        Relocate the files to the appropriate directories. Some of the methods require the files to be in specific directories.
        """

        # Copying the protein data to the project directory
        if not os.path.exists(self.local_protein):
            copyfile(self.original_protein_data, self.local_protein)

        # User-supplied Vina box short-circuits both _refine_protein_data (which
        # would otherwise derive the box from the original co-crystal ligand) and
        # P2Rank prediction further down. The path was validated in __init__, so
        # self.box_location is guaranteed to be an existing file when non-None.
        if self.box_location:
            os.makedirs(os.path.dirname(self.vina_box), exist_ok=True)
            copyfile(self.box_location, self.vina_box)
            logger.info(f"Seeded Vina box from user-supplied {self.box_location}")
            return

        if (
            (self.protein_chain is not None)
            and (self.original_ligand is not None)
            and (self.original_ligand_chain is not None)
        ):
            self._refine_protein_data()

    def _prepare_ligand(self):
        """
        Preprocessing the ligand data for docking.

        When ``self.gnina_input_mode == "sdf"`` the OpenBabel-backed
        ``ligand_pdb_to_pdbqt`` step (and its prerequisite ``sdf_to_pdb``)
        are skipped: gnina reads the RDKit-generated SDF directly, and no
        other method requires the PDBQT in this run (BulkRun has already
        downgraded to PDBQT if Vina or a Vina-rescore was co-requested).
        """
        if os.path.exists(self.ligand_sdf):
            logger.info(f"Ligand {self.ligand_idx} already prepared.")
            return  # Skip preparation if already done

        smiles_to_sdf(
            smiles=self.original_ligand_smile,
            sdf=self.ligand_sdf,
        )

        if self.gnina_input_mode == "sdf":
            logger.info(
                f"Skipping OpenBabel PDBQT prep for ligand {self.ligand_idx} "
                "(gnina_input_mode='sdf')."
            )
            return

        sdf_to_pdb(
            sdf=self.ligand_sdf,
            pdb=self.ligand_pdb,
        )

        ligand_pdb_to_pdbqt(pdb=self.ligand_pdb)

        # Generate ligand-specific box based on this decoy's radius of gyration.
        # Keep the binding site center, but adjust size for this specific ligand.
        # A user-supplied box (self.box_location) defines both center AND size
        # explicitly, so leave it untouched — only auto-derived boxes (P2Rank or
        # original-ligand) get resized to the ligand's radius of gyration.
        if os.path.exists(self.vina_box) and not self.box_location:
            center, _ = get_center_and_size_from_box_file(self.vina_box)
            generate_vina_box(
                input_x=center[0],
                input_y=center[1],
                input_z=center[2],
                ligand_smiles=self.original_ligand_smile,
                output_file=self.vina_box,
            )
            logger.info(
                f"Generated ligand-specific box for {self.ligand_idx} based on radius of gyration"
            )

    def _prepare_ligand_for_karmadock(self):
        """
        Prepare ligand specifically for KarmaDock (MOL2 conversion).
        Only called when KarmaDock is actually being used.

        KarmaDock derives the binding pocket from residues within 12 Å of the
        *input ligand's* coordinates (see ``KarmaDock/utils/pre_processing.py``
        ``get_pocket()``). After generating the SDF/mol2 from SMILES we
        therefore re-centre them on the Vina box if one is available — that
        way KarmaDock picks the user-supplied (or P2Rank- / original-ligand-
        derived) pocket instead of wherever obabel's 3D embedding lands.
        """
        if os.path.exists(self.karmadock_mol2):
            logger.info(f"KarmaDock ligand {self.ligand_idx} already prepared.")
            return

        smiles_to_sdf_karmadock(smiles=self.original_ligand_smile, sdf_path=self.karmadock_ligand)
        smiles_2_mol2 = (
            f'obabel "{self.karmadock_ligand}" -O "{self.karmadock_mol2}" --gen3d {SHELL_SILENCER}'
        )

        result = subprocess.run(
            smiles_2_mol2, shell=True, check=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(
                f"Error occurred: obabel command failed with return code {result.returncode}"
            )
        else:
            logger.info("Command for converting SMILES to MOL2 succeeded")

        # Re-centre on the binding pocket. By this point self.vina_box has been
        # populated by box_location (_relocate_files) and/or P2Rank
        # (_prepare_protein) and/or _refine_protein_data (original ligand), so
        # any available pocket signal drives KarmaDock's get_pocket() correctly.
        if os.path.isfile(self.vina_box):
            from guild.docking.karmadock import center_karmadock_ligand_on_box

            center_karmadock_ligand_on_box(
                sdf_path=self.karmadock_ligand,
                mol2_path=self.karmadock_mol2,
                box_file=self.vina_box,
            )

    def _prepare_protein(self, target_chain=None):
        """
        Preprocessing the protein data for docking.

        :param target_chain: Chain ID, list of chain IDs, or comma-separated
                             string to be used for docking. If not provided, the
                             first chain is used. Multiple chains are kept
                             together so a multi-chain pocket survives prep.
        """

        # Isolate the protein chain(s)
        if not os.path.exists(self.single_chain_protein):
            isolate_protein_chain(
                input_file=self.local_protein,
                input_name=self.protein_idx,
                output_file=self.single_chain_protein,
                target_chain=target_chain,
            )
        else:
            logger.info(f"Protein {self.protein_idx} already prepared.")

        if not os.path.exists(self.cleaned_protein):
            clean_receptor(
                input_pdb=self.single_chain_protein,
                output_pdb=self.cleaned_protein,
                keep_metals=True,
            )
        else:
            logger.info(f"Protein {self.protein_idx} already cleaned.")

        # Copy the protein data to the karmadock data directory
        if not os.path.exists(self.karmadock_protein):
            copyfile(self.cleaned_protein, self.karmadock_protein)
        else:
            logger.info(f"Protein {self.protein_idx} already copied to karmadock data directory.")

        # Get the original sequence of the protein, one per requested chain.
        # ``self.original_sequence`` keeps the primary chain's sequence for any
        # single-chain consumer; ``self.original_sequences`` is the per-chain
        # list (same order as ``self.protein_chains``) used by Boltz.
        original_sequence_dictionary = get_original_sequence_dictionary(self.cleaned_protein)
        self.sequence_chains = [
            chain for chain in self.protein_chains if chain in original_sequence_dictionary
        ]
        self.original_sequences = [
            process_into_fasta_string(original_sequence_dictionary[chain])
            for chain in self.sequence_chains
        ]
        self.original_sequence = self.original_sequences[0] if self.original_sequences else ""

        # If P2Rank binding site prediction is enabled, run it after protein is prepared.
        # A user-supplied box overrides P2Rank — it already populated self.vina_box.
        if self.predict_binding_pocket and not self.box_location:
            try:
                self._run_p2rank_binding_site_prediction()
                logger.info("P2Rank binding site prediction completed.")
            except Exception as e:
                logger.error(f"P2Rank binding site prediction failed: {e}")
                logger.warning("Falling back to original ligand location for binding site.")

    def _run_p2rank_binding_site_prediction(self):
        """
        Run P2Rank binding site prediction and generate Vina box from predicted pocket.

        This method runs P2Rank on the cleaned protein structure and uses the
        predicted binding pocket center to generate the Vina box instead of
        relying on the original ligand location.
        """
        logger.info(f"Running P2Rank binding site prediction for {self.protein_idx}")

        # Run P2Rank and get the binding site center
        binding_site_center = get_binding_site_center_from_p2rank(
            protein_pdb=self.cleaned_protein,
            output_dir=self.p2rank_dir,
            pocket_rank=P2RANK_DEFAULT_POCKET_RANK,
            min_probability=P2RANK_MIN_PROBABILITY_THRESHOLD,
        )

        if binding_site_center is None:
            raise ValueError(
                f"P2Rank did not find a valid binding pocket for {self.protein_idx}. "
                "Consider using original ligand location."
            )

        center_x, center_y, center_z = binding_site_center
        logger.info(
            f"P2Rank predicted binding site center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})"
        )

        # Generate the Vina box using the P2Rank-predicted center
        # The box size is calculated from the ligand radius of gyration
        generate_vina_box(
            input_x=center_x,
            input_y=center_y,
            input_z=center_z,
            ligand_smiles=self.original_ligand_smile,
            output_file=self.vina_box,
        )

        logger.info(f"Generated Vina box from P2Rank binding site prediction at {self.vina_box}")

    @timeit()
    def run_autodock_vina(self):
        """
        Run autock vina for docking the ligand to the protein. Output is saved in the autodock directory, inside the project folder.
        """
        failed_steps = 0

        # Prepare the receptor
        try:
            protein_pdb_to_pdbqt(
                input_pdb=self.cleaned_protein,
                output_pdbqt=self.cleaned_protein_pdbqt,
                allow_bad_res=True,
            )
            logger.info("Autodock Vina: receptor prepared.")
        except Exception as e:
            logger.error(f"Autodock Vina: error in preparing receptor: {e}")
            failed_steps += 1

        if os.path.exists(self.vina_output_pdbqt):
            logger.info("Autodock Vina: output file already exists.")
            return failed_steps
        vina_box_center, vina_box_size = get_center_and_size_from_box_file(self.vina_box)

        try:
            deploy_vina(
                receptor_pdbqt=self.cleaned_protein_pdbqt,
                ligand_pdbqt=self.ligand_pdbqt,
                center=vina_box_center,
                size=vina_box_size,
                output_scores=self.vina_output_scores,
                output_pdbqt=self.vina_output_pdbqt,
            )
        except Exception as e:
            logger.error(f"Autodock Vina: error in docking: {e}")
            failed_steps += 1

        return 0 if failed_steps == 0 else 1

    @timeit()
    def run_gnina(self):
        """
        Run gnina docking. In the default ``pdbqt`` mode (re)prepares the
        receptor PDBQT and feeds gnina the PDBQT receptor + ligand. In
        ``sdf`` mode, skips the OpenBabel-backed PDBQT prep entirely and
        feeds gnina the PDB receptor + SDF ligand directly (gnina sniffs
        the format from the extension).
        """
        failed_steps = 0

        if self.gnina_input_mode == "sdf":
            receptor_path = self.cleaned_protein  # PDB
            ligand_path = self.ligand_sdf  # SDF (RDKit-generated upstream)
            logger.info("gnina: SDF-mode — skipping PDBQT prep.")
        else:
            # Receptor PDBQT prep — same as Vina. Run independently of
            # run_autodock_vina because gnina can be requested without vina.
            try:
                protein_pdb_to_pdbqt(
                    input_pdb=self.cleaned_protein,
                    output_pdbqt=self.cleaned_protein_pdbqt,
                    allow_bad_res=True,
                )
                logger.info("gnina: receptor prepared.")
            except Exception as e:
                logger.error(f"gnina: error in preparing receptor: {e}")
                failed_steps += 1
            receptor_path = self.cleaned_protein_pdbqt
            ligand_path = self.ligand_pdbqt

        # Require a *non-empty* pose file: older runs (before the OpenBabel
        # plugin fix) left 0-byte pdbqts behind, and treating those as "done"
        # would permanently skip regenerating usable poses on re-run.
        try:
            output_ok = (
                os.path.isfile(self.gnina_output_pdbqt)
                and os.path.getsize(self.gnina_output_pdbqt) > 0
            )
        except OSError:
            output_ok = False

        if output_ok:
            logger.info("gnina: output file already exists.")
            return failed_steps

        gnina_box_center, gnina_box_size = get_center_and_size_from_box_file(self.vina_box)

        # Per-combination subprocess transcript — predictable path that the
        # orchestrator points users at when this combo fails. See
        # guild/tools/subprocess_log.py for the format.
        subprocess_log = (
            f"{self.gnina_dir}/{self.protein_idx}_{self.ligand_idx}.subprocess.log"
        )
        try:
            deploy_gnina(
                receptor=receptor_path,
                ligand=ligand_path,
                center=gnina_box_center,
                size=gnina_box_size,
                output_pdbqt=self.gnina_output_pdbqt,
                output_scores=self.gnina_output_scores,
                use_gpu=self.use_gpu,
                subprocess_log_path=subprocess_log,
            )
        except Exception as e:
            logger.error(f"gnina: error in docking: {e}")
            failed_steps += 1

        return 0 if failed_steps == 0 else 1

    @timeit()
    def run_karmadock(self):
        """
        Run KarmaDock for docking the ligand to the protein. Output is saved in the karmadock directory, inside the project folder.
        """

        deploy_karmadock(
            home_path=self.home_path,
            karmadock_results_dir=self.karmadock_results_dir,
            karmadock_graphs_dir=self.karmadock_graphs_dir,
            karmadock_data_dir=self.karmadock_data_dir,
        )

    @timeit()
    def run_diffdock(self):
        """
        Run DiffDock for docking the ligand to the protein. Output is saved in the diffdock directory, inside the project folder.
        """
        deploy_diffdock_single(
            home_path=self.home_path,
            combination_id=self.combination_id,
            cleaned_protein=self.cleaned_protein,
            original_ligand_smile=self.original_ligand_smile,
            project_dir=self.project_dir,
            diffdock_results_dir=self.diffdock_results_dir,
            # use_gpu=self.use_gpu,
        )

    @timeit()
    def run_boltz(self):
        """
        Run Boltz for docking the ligand to the protein. Output is saved in the boltz directory, inside the project folder.
        """
        # Fetch one MSA per chain (cache-keyed by chain), aligned with
        # self.sequence_chains / self.original_sequences for the Boltz YAML.
        msa_files = [
            fetch_protein_msa(
                sequence=sequence,
                protein_id=self.protein_idx,
                protein_chain_id=chain,
                output_a3m_dir=self.msa_cache_dir,
            )
            for chain, sequence in zip(
                self.sequence_chains, self.original_sequences, strict=True
            )
        ]
        print(f"MSA file(s) for Boltz: {msa_files}")

        # self.box_location is the single source of truth for "user supplied a
        # box" (validated in __init__). Don't fall back to checking self.vina_box
        # — that file may have been generated from the original co-crystal
        # ligand via _refine_protein_data, in which case it isn't actually the
        # user-supplied pocket and shouldn't drive the Boltz pocket_contacts.
        # Pocket contacts are collected across all docking chains so an
        # interface pocket spanning multiple chains is fully constrained.
        if self.box_location:
            center, size = get_center_and_size_from_box_file(self.vina_box)
            pocket_contacts = get_pocket_contacts_from_box(
                protein_pdb=self.local_protein,
                protein_chain=self.sequence_chains,
                center=center,
                size=size,
            )
            print(
                f"Identified {len(pocket_contacts)} pocket contact residues for Boltz "
                f"constraints from user-supplied box."
            )
        elif self.original_ligand is not None and self.original_ligand_chain is not None:
            pocket_contacts = get_pocket_contacts_from_ligand(
                protein_pdb=self.local_protein,
                protein_chain=self.sequence_chains,
                original_ligand=self.original_ligand,
                original_ligand_chain=self.original_ligand_chain,
                distance_threshold=4.0,
            )
            print(
                f"Identified {len(pocket_contacts)} pocket contact residues for Boltz constraints."
            )
        else:
            pocket_contacts = None

        generate_boltz_yaml(
            protein_sequence=self.original_sequences,
            protein_chain=self.sequence_chains,
            ligand_sequences=[self.original_ligand_smile],
            ligand_ids=["L"],  # Boltz expects ligand Id to be < 5 characters
            output_file=self.boltz_yaml,
            template_file=self.local_protein,
            msa_file=msa_files,
            pocket_contacts=pocket_contacts,
        )
        deploy_boltz(self.boltz_yaml, out_dir=self.boltz_dir, use_gpu=self.use_gpu)

    def dock(self, box_location: str = "", methods: list = ALL_AVAILABLE_METHODS):
        """
        Dock the ligand to the protein.
        """
        # AutoDock Vina
        self.vina_box = box_location
        if VINA_PREFIX in methods:
            report_vina = self.run_autodock_vina()
            if report_vina == 0:
                logger.info("Vina: completed.")
            else:
                logger.error(f"Vina: failed. Failed steps: {report_vina} / {len(methods)}")

        # KarmaDock
        if KARMADOCK_PREFIX in methods:
            self.run_karmadock()
            logger.info("KarmaDock: completed.")

        # DiffDock
        if DIFFDOCK_PREFIX in methods:
            self.run_diffdock()
            logger.info("DiffDock: completed.")

        # Boltz
        if BOLTZ_PREFIX in methods:
            self.run_boltz()
            logger.info("Boltz: completed.")

        # GNINA
        if GNINA_PREFIX in methods:
            report_gnina = self.run_gnina()
            if report_gnina == 0:
                logger.info("gnina: completed.")
            else:
                logger.error(f"gnina: failed. Failed steps: {report_gnina} / {len(methods)}")
