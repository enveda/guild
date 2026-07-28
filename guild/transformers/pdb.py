import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Iterable, Optional, Tuple

import numpy as np
from Bio.PDB import PDBIO, Chain, Model, PDBParser, Structure
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem

from guild.constants.ligands import LIGANDS_TO_IGNORE
from guild.docking.vina import generate_vina_box
from guild.tools.preparation import _normalize_chain_list

# Suppress PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logger = logging.getLogger(__name__)

# Maximum distance (Å) to search for a warhead atom when adding a covalent CONECT.
_COVALENT_BOND_DIST_MAX = 2.5


def calculate_centroid(atom_list):
    """
    Calculate the centroid of a list of atoms. This will be used to center molecule.
    :param atom_list: List of atoms
    """
    x, y, z = 0.0, 0.0, 0.0
    for atom in atom_list:
        coord = atom.coord
        x += coord[0]
        y += coord[1]
        z += coord[2]
    num_atoms = len(atom_list)
    return x / num_atoms, y / num_atoms, z / num_atoms


class LigandIdentifier:
    """
    This class is used to identify the ligand in a PDB file and extract it from the rest of the protein.
    Furthermore, it calculates the center of the chain and the center of the ligand.
    It also writes the ligand and the protein to separate PDB files.
    Can be used to generate a config file for the box in Autodock.
    """

    def __init__(
        self,
        input_pdb_file,
        usable_chain="",
        heteroatoms=None,
        heteroatoms_chain=None,
        output_ligand_file="ligand.pdb",
        output_protein_file="protein.pdb",
        config_file_name="config_box.txt",
        box_size=20,
    ):
        """
        :param input_pdb_file: The input PDB file
        :param usable_chain: The chain to be used. If not specified, the first chain is used.
        :param heteroatoms: The heteroatoms to be considered as ligand. Default is an empty list.
        :param heteroatoms_chains: The chains to be considered for heteroatoms. Default is an empty string. If nothing is provided, usable_chain is used.
        :param output_ligand_file: The output file for the ligand. Default is "ligand.pdb". Provide full path.
        :param output_protein_file: The output file for the protein. Default is "protein.pdb". Provide full path.
        :param config_file_name: The output file for the config file. Default is "config_box.txt". Provide full path.
        :param box_size: The size of the box to be used in Autodock. Default is 20.
        """
        if heteroatoms is None:
            heteroatoms = [""]
        self.input_pdb_file = input_pdb_file
        self.input_object_name = input_pdb_file.split(".")[0]
        self.heteroatoms = heteroatoms
        self.parser = PDBParser()
        self.structure = self.parser.get_structure(self.input_object_name, self.input_pdb_file)
        self.usable_chain = usable_chain
        self.output_ligand_file = output_ligand_file

        self.output_protein_file = output_protein_file
        self.config_file_name = config_file_name
        self.box_size = box_size
        if self.usable_chain == "":
            # If no chain is specified, the first chain is used
            self.chain = self.structure[0][0]
        else:
            self.chain = self.structure[0][self.usable_chain]
        if heteroatoms_chain is None:
            self.heteroatoms_usable_chain = self.usable_chain
        else:
            self.heteroatoms_usable_chain = heteroatoms_chain
        self.heteroatoms_chain = self.structure[0][self.heteroatoms_usable_chain]

    def _residue_center(self, input_residue):
        """
        Calculate the center of a residue
        :param input_residue: The input residue, which is a Bio.PDB.Residue object
        """
        coordinates = []
        for atom in input_residue:
            coordinates.append(atom.get_coord())
        coordinates_array = np.vstack(coordinates)
        return np.mean(coordinates_array, axis=0)

    def extract_chain_center(self):
        """
        Extract the center of the chain
        """
        # Center the chain in the origin
        atom_list = [atom for atom in self.chain.get_atoms()]
        centroid = calculate_centroid(atom_list)
        for atom in atom_list:
            atom.set_coord(atom.coord - centroid)

        all_coordinates = []
        for residue in self.chain:
            all_coordinates.append(self._residue_center(residue))
        coordinates_array = np.vstack(all_coordinates)
        self.chain_center_coordinates = np.mean(coordinates_array, axis=0)

    def write_ligand(self):
        """
        Write the ligand to a PDB file
        """
        structure = Structure.Structure("ligand")
        model = Model.Model(0)
        chain = Chain.Chain(self.heteroatoms_usable_chain)
        chain.add(self.closest_molecule)
        model.add(chain)
        structure.add(model)
        io = PDBIO()
        io.set_structure(structure)
        io.save(self.output_ligand_file)
        opened_molecule = Chem.MolFromPDBFile(self.output_ligand_file)
        self.smiles = Chem.MolToSmiles(opened_molecule)

    def write_protein(self):
        """
        Write the single chain protein to a PDB file
        """
        structure = Structure.Structure("protein")
        model = Model.Model(0)
        chain = Chain.Chain(self.usable_chain)
        for residue in self.chain:
            if residue.get_resname() not in self.heteroatoms:
                chain.add(residue)
        model.add(chain)
        structure.add(model)
        io = PDBIO()
        io.set_structure(structure)
        io.save(self.output_protein_file)

    def calculate_closest_molecule(self):
        """
        Calculate the closest molecule to the chain center, which is considered as the ligand
        """
        self.closest_molecule = None
        closest_distance = float("inf")
        for current_residue in self.heteroatoms_chain:
            if current_residue.get_resname() in self.heteroatoms:
                current_residue_center = self._residue_center(current_residue)
                distance = np.linalg.norm(self.chain_center_coordinates - current_residue_center)
                if distance < closest_distance:
                    self.closest_molecule = current_residue
                    closest_distance = distance

    def write_config_box_file(self):
        """
        Write the config file for Autodock using radius of gyration from ligand SMILES
        """
        self.closest_molecule_center = self._residue_center(self.closest_molecule)

        # Use generate_vina_box which calculates box size from ligand SMILES
        generate_vina_box(
            input_x=self.closest_molecule_center[0],
            input_y=self.closest_molecule_center[1],
            input_z=self.closest_molecule_center[2],
            ligand_smiles=self.smiles,
            output_file=self.config_file_name,
        )
        logger.info(
            f"Box file generated for {self.input_pdb_file} at {self.config_file_name} using ligand radius of gyration"
        )

    def run(self):
        """
        Run the class in its entirety
        """
        self.extract_chain_center()
        self.calculate_closest_molecule()
        self.write_ligand()
        self.write_protein()
        self.write_config_box_file()


def filter_ligands(pdb_file):
    """
    Filters and lists ligands in a PDB file.

    :param pdb_file: Path to the PDB file
    :return: Tuple containing chain ID and residue name of the first identified ligand, or (
    """
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    ligands = []  # To store ligand residues

    # Iterate over all residues in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue is a heteroatom and not water
                if residue.id[0] != " " and residue.resname not in LIGANDS_TO_IGNORE:
                    ligands.append(residue)

    if ligands:
        for ligand in ligands:
            return ligand.get_full_id()[2], ligand.resname
    else:
        return None, None


### Merge protein and ligand into single file


def _is_atom_line(line: str) -> bool:
    return line.startswith("ATOM  ") or line.startswith("HETATM")


def _is_ter_or_end(line: str) -> bool:
    s = line.strip()
    return s == "TER" or s == "END" or s == "ENDMDL"


def _max_atom_serial(pdb_lines: Iterable[str]) -> int:
    mx = 0
    for line in pdb_lines:
        if _is_atom_line(line) and len(line) >= 11:
            try:
                serial = int(line[6:11])
                mx = max(mx, serial)
            except ValueError:
                pass
    return mx


def _format_pdb_atom_line(
    *,
    record: str,
    serial: int,
    name: str,
    altloc: str,
    resname: str,
    chain: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    occ: float,
    temp: float,
    element: str,
    charge: str = "",
) -> str:
    """
    PDB fixed-column formatting.
    """
    record = (record[:6]).ljust(6)
    name = name[:4].ljust(4)
    altloc = altloc[:1] if altloc else " "
    resname = resname[:3].rjust(3)
    chain = chain[:1] if chain else " "
    icode = icode[:1] if icode else " "
    element = element[:2].rjust(2) if element else "  "
    charge = charge[:2].rjust(2) if charge else "  "

    return (
        f"{record}"
        f"{serial:5d} "
        f"{name}{altloc}"
        f"{resname} "
        f"{chain}"
        f"{resseq:4d}{icode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{temp:6.2f}"
        f"          "
        f"{element}{charge}\n"
    )


def _parse_pdb_atom_fields(
    line: str,
) -> Tuple[str, int, str, str, str, str, int, str, float, float, float, float, float, str, str]:
    """
    Parse the commonly used PDB ATOM/HETATM fields from fixed columns.
    Returns:
      record, serial, name, altloc, resname, chain, resseq, icode, x, y, z, occ, temp, element, charge
    """
    record = line[0:6].strip() if len(line) >= 6 else "ATOM"
    serial = int(line[6:11]) if len(line) >= 11 and line[6:11].strip() else 0
    name = line[12:16] if len(line) >= 16 else " C  "
    altloc = line[16:17] if len(line) >= 17 else " "
    resname = line[17:20] if len(line) >= 20 else "LIG"
    chain = line[21:22] if len(line) >= 22 else " "
    resseq = int(line[22:26]) if len(line) >= 26 and line[22:26].strip() else 1
    icode = line[26:27] if len(line) >= 27 else " "
    x = float(line[30:38]) if len(line) >= 38 else 0.0
    y = float(line[38:46]) if len(line) >= 46 else 0.0
    z = float(line[46:54]) if len(line) >= 54 else 0.0
    occ = float(line[54:60]) if len(line) >= 60 and line[54:60].strip() else 1.00
    temp = float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else 0.00
    element = line[76:78].strip() if len(line) >= 78 else ""
    charge = line[78:80].strip() if len(line) >= 80 else ""
    return (
        record,
        serial,
        name,
        altloc,
        resname,
        chain,
        resseq,
        icode,
        x,
        y,
        z,
        occ,
        temp,
        element,
        charge,
    )


def _convert_pdbqt_to_pdb(ligand_pdbqt: str, out_pdb: str) -> None:
    """
    Convert PDBQT -> PDB, keeping only the top-ranked pose (MODEL 1).

    Vina/gnina PDBQT output is score-sorted, so MODEL 1 is always the
    best-scoring pose. Without restricting to it, a multi-pose PDBQT
    converts to a PDB with every pose's atoms stacked into the same
    ligand residue — e.g. 8 poses x 46 atoms = 368 "atoms" all sharing
    resname/resseq, which downstream RDKit template matching
    (AssignBondOrdersFromTemplate in PLIP/prolif analysis) cannot
    resolve and can hang for a very long time trying.

    Prefer OpenBabel CLI (obabel). If not available, fall back to stripping
    extra PDBQT columns (best-effort).
    """
    obabel = shutil.which("obabel")
    if obabel:
        # -f 1 -l 1: only convert the first model (top pose).
        cmd = [obabel, ligand_pdbqt, "-O", out_pdb, "-f", "1", "-l", "1"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"obabel failed converting PDBQT->PDB.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )
        return

    # Fallback: treat PDBQT as PDB and strip after column 66 (keep coords/occ/temp).
    # Stop at the first ENDMDL so only MODEL 1 (the top pose) is kept.
    with (
        open(ligand_pdbqt, "r", encoding="utf-8", errors="replace") as fin,
        open(out_pdb, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if line.startswith("ENDMDL"):
                break
            if _is_atom_line(line):
                # Keep up to tempFactor (col 66), then try to preserve element if present.
                base = line[:66]
                # If element exists in PDBQT line near the end, keep it in cols 77-78.
                element = ""
                if len(line) >= 78:
                    element = line[76:78].strip()
                # Pad to 76 then add element.
                base = base.rstrip("\n")
                base = base.ljust(76)
                base = base + (element.rjust(2) if element else "  ")
                fout.write(base + "\n")
            elif not _is_ter_or_end(line):
                # Ignore other records for ligand conversion
                continue


def _sdf_to_pdb_rdkit(sdf_path: str, out_pdb: str) -> None:
    """
    Convert SDF -> PDB using RDKit (not OpenBabel).
    RDKit assigns unique atom names (C1, C2, …) and writes CONECT records for
    every bond — both required because viewers (e.g. PyMOL) disable auto-bonding
    when any explicit CONECT is present.
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        # Fallback for molecules with unusual valences (e.g. gnina covalent output)
        suppl2 = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        mol = next((m for m in suppl2 if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not parse any molecule from {sdf_path}")
    Chem.MolToPDBFile(mol, out_pdb)


def build_complex_pdb(
    protein_pdb: str,
    ligand_file: str,
    out_complex_pdb: str,
    *,
    ligand_is_pdbqt: Optional[bool] = None,
    ligand_resname: str = "LIG",
    ligand_chain: str = "Z",
    ligand_resseq: int = 1,
    insert_ter_between: bool = True,
) -> str:
    """
    Build a single "complex" PDB file suitable for PLIP from:
      - protein PDB
      - ligand PDB, PDBQT, or SDF

    The ligand will be rewritten as HETATM with (resname, chain, resseq),
    and atom serials will be renumbered to follow the protein serials.

    Returns the output path.
    """
    if ligand_is_pdbqt is None:
        ligand_is_pdbqt = ligand_file.lower().endswith(".pdbqt")
    ligand_is_sdf = ligand_file.lower().endswith(".sdf")

    # Read protein, keep non-END records
    with open(protein_pdb, "r", encoding="utf-8", errors="replace") as f:
        protein_lines = [ln for ln in f if not _is_ter_or_end(ln)]

    # Determine where to start ligand serials
    start_serial = _max_atom_serial(protein_lines) + 1

    shared_kwargs = dict(
        protein_lines=protein_lines,
        out_complex_pdb=out_complex_pdb,
        start_serial=start_serial,
        ligand_resname=ligand_resname,
        ligand_chain=ligand_chain,
        ligand_resseq=ligand_resseq,
        insert_ter_between=insert_ter_between,
    )

    if ligand_is_pdbqt:
        with tempfile.TemporaryDirectory() as td:
            ligand_pdb = os.path.join(td, "ligand.pdb")
            _convert_pdbqt_to_pdb(ligand_file, ligand_pdb)
            _write_complex(ligand_pdb=ligand_pdb, **shared_kwargs)
    elif ligand_is_sdf:
        with tempfile.TemporaryDirectory() as td:
            ligand_pdb = os.path.join(td, "ligand.pdb")
            _sdf_to_pdb_rdkit(ligand_file, ligand_pdb)
            _write_complex(ligand_pdb=ligand_pdb, **shared_kwargs)
    else:
        _write_complex(ligand_pdb=ligand_file, **shared_kwargs)

    return out_complex_pdb


def _write_complex(
    *,
    protein_lines: list[str],
    ligand_pdb: str,
    out_complex_pdb: str,
    start_serial: int,
    ligand_resname: str,
    ligand_chain: str,
    ligand_resseq: int,
    insert_ter_between: bool,
) -> None:
    # Read ligand: keep atom records and any CONECT records
    with open(ligand_pdb, "r", encoding="utf-8", errors="replace") as f:
        ligand_lines = f.readlines()

    ligand_atom_lines = [ln for ln in ligand_lines if _is_atom_line(ln)]
    ligand_conect_lines = [ln for ln in ligand_lines if ln.startswith("CONECT")]

    if not ligand_atom_lines:
        raise ValueError(f"No ATOM/HETATM records found in ligand file: {ligand_pdb}")

    with open(out_complex_pdb, "w", encoding="utf-8") as out:
        # Write protein lines as-is
        for ln in protein_lines:
            out.write(ln if ln.endswith("\n") else (ln + "\n"))

        if insert_ter_between:
            out.write("TER\n")

        # Rewrite ligand lines as HETATM, with controlled residue/chain/serial.
        # Track old→new serial mapping so CONECT records can be remapped below.
        first_ligand_serial = int(_parse_pdb_atom_fields(ligand_atom_lines[0])[1])
        serial_offset = start_serial - first_ligand_serial

        serial = start_serial
        for ln in ligand_atom_lines:
            (
                _record,
                _old_serial,
                name,
                altloc,
                _resname,
                _chain,
                _resseq,
                icode,
                x,
                y,
                z,
                occ,
                temp,
                element,
                charge,
            ) = _parse_pdb_atom_fields(ln)

            # If element is missing, try to infer from atom name (very rough but helps)
            if not element:
                # Strip digits/spaces, take first 1-2 letters
                guess = "".join([c for c in name.strip() if c.isalpha()])
                element = (guess[:2] if guess else "").capitalize()

            out.write(
                _format_pdb_atom_line(
                    record="HETATM",
                    serial=serial,
                    name=name,
                    altloc=altloc,
                    resname=ligand_resname,
                    chain=ligand_chain,
                    resseq=ligand_resseq,
                    icode=icode,
                    x=x,
                    y=y,
                    z=z,
                    occ=occ,
                    temp=temp,
                    element=element,
                    charge=charge,
                )
            )
            serial += 1

        # Carry internal ligand CONECT records with remapped serials.
        # When any CONECT exists for an atom, PyMOL uses only explicit CONECTs
        # for that atom and disables auto-bonding — so all bonds must be listed.
        for ln in ligand_conect_lines:
            raw = ln[6:].rstrip("\n")
            serials = []
            for i in range(0, len(raw), 5):
                chunk = raw[i : i + 5].strip()
                if chunk:
                    try:
                        serials.append(int(chunk) + serial_offset)
                    except ValueError:
                        pass
            if serials:
                out.write("CONECT" + "".join(f"{s:5d}" for s in serials) + "\n")

        out.write("END\n")


def get_pocket_contacts_from_ligand(
    protein_pdb: str,
    protein_chain,
    original_ligand: str,
    original_ligand_chain: str,
    distance_threshold: float = 6.0,
) -> list:
    """
    Return a list of [chain_id, residue_index] protein residue contacts that lie
    within ``distance_threshold`` Å of any atom of ``original_ligand`` (by residue
    name). ``residue_index`` is 1-based and contiguous *within each chain* (Boltz
    schema indexing), not the raw PDB residue number. Suitable for use in a Boltz
    pocket-constraint ``contacts`` list.

    :param protein_pdb: Path to the crystal-structure PDB that contains both
                        protein and the reference ligand.
    :param protein_chain: Chain ID, list of chain IDs, or comma-separated string
                          (e.g. ``"A,B"``) of the protein receptor. Contacts are
                          collected from every listed chain, each indexed
                          independently from 1, so a pocket spanning multiple
                          chains is fully captured.
    :param original_ligand: Residue name of the reference ligand (e.g. ``"LIG"``).
    :param original_ligand_chain: Chain ID that contains the reference ligand.
    :param distance_threshold: Maximum heavy-atom distance (Å) to include a
                               protein residue as a contact (default 6.0).
    :return: List of ``[chain_id, residue_index]`` pairs, or ``[]`` if
             the ligand residue cannot be found.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", protein_pdb)
    model = structure[0]

    # Collect all atoms belonging to the reference ligand
    ligand_atoms = []
    if original_ligand_chain in [c.id for c in model]:
        for residue in model[original_ligand_chain]:
            if residue.get_resname().strip() == original_ligand.strip() and residue.id[0] != " ":
                ligand_atoms.extend(list(residue.get_atoms()))

    if not ligand_atoms:
        logger.warning(
            f"get_pocket_contacts_from_ligand: no atoms found for ligand "
            f"'{original_ligand}' in chain '{original_ligand_chain}' of {protein_pdb}. "
            f"Pocket constraint will be skipped."
        )
        return []

    ligand_coords = np.array([a.get_coord() for a in ligand_atoms])
    available = [c.id for c in model]

    # Find protein residues with at least one atom within distance_threshold,
    # indexing each chain independently (Boltz expects per-chain 1-based indices).
    contacts = []
    for chain_id in _normalize_chain_list(protein_chain):
        if chain_id not in available:
            logger.warning(
                f"get_pocket_contacts_from_ligand: protein chain '{chain_id}' "
                f"not found in {protein_pdb}."
            )
            continue
        seen = set()
        sequence_index = 0
        for residue in model[chain_id]:
            if residue.id[0] != " ":  # skip heteroatoms / water on the protein chain
                continue
            sequence_index += 1
            if sequence_index in seen:
                continue
            for atom in residue:
                diffs = ligand_coords - atom.get_coord()
                dists = np.linalg.norm(diffs, axis=1)
                if np.any(dists <= distance_threshold):
                    contacts.append([chain_id, sequence_index])
                    seen.add(sequence_index)
                    break  # one close atom is enough to include this residue

    logger.debug(
        f"get_pocket_contacts_from_ligand: found {len(contacts)} contact residues "
        f"within {distance_threshold} Å of '{original_ligand}' in {protein_pdb}."
    )
    return contacts


def get_pocket_contacts_from_box(
    protein_pdb: str,
    protein_chain,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
) -> list:
    """
    Return ``[[chain_id, residue_index], ...]`` for protein residues whose Cα atom
    lies inside the axis-aligned box defined by ``center`` and ``size``.
    ``residue_index`` is 1-based and contiguous *within each chain* (Boltz schema
    indexing). Same return shape as :func:`get_pocket_contacts_from_ligand`.

    :param protein_pdb: Path to the protein PDB.
    :param protein_chain: Chain ID, list of chain IDs, or comma-separated string
                          (e.g. ``"A,B"``) of the protein receptor. Residues are
                          collected from every listed chain, each indexed
                          independently from 1, so an interface pocket spanning
                          multiple chains is fully captured.
    :param center: ``(x, y, z)`` of the box center.
    :param size: ``(sx, sy, sz)`` full edge lengths of the box.
    :return: List of ``[chain_id, residue_index]`` pairs, or ``[]`` if no
             listed chain is found or no residue Cα falls inside the box.
    """
    cx, cy, cz = center
    sx, sy, sz = size
    half = np.array([sx / 2.0, sy / 2.0, sz / 2.0])
    center_arr = np.array([cx, cy, cz])

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", protein_pdb)
    model = structure[0]
    available = [c.id for c in model]

    contacts = []
    for chain_id in _normalize_chain_list(protein_chain):
        if chain_id not in available:
            logger.warning(
                f"get_pocket_contacts_from_box: protein chain '{chain_id}' "
                f"not found in {protein_pdb}."
            )
            continue
        sequence_index = 0
        for residue in model[chain_id]:
            if residue.id[0] != " ":  # skip heteroatoms / water on the protein chain
                continue
            sequence_index += 1
            if "CA" not in residue:
                continue
            ca_coord = residue["CA"].get_coord()
            if np.all(np.abs(ca_coord - center_arr) <= half):
                contacts.append([chain_id, sequence_index])

    logger.debug(
        f"get_pocket_contacts_from_box: found {len(contacts)} contact residues "
        f"inside box center={center} size={size} in {protein_pdb}."
    )
    return contacts


def get_flexres_string_from_box(
    protein_pdb: str,
    protein_chain,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    max_flexres: int | None = None,
) -> str | None:
    """
    Return an AutoDock-format flexible-residue string (e.g. ``"A:1_A:3_B:7"``)
    for protein residues whose Cα lies inside the docking box.

    Uses the same Cα-in-AABB check as :func:`get_pocket_contacts_from_box` but
    returns ``chain:resSeq`` tokens joined by ``_`` rather than Boltz-schema
    ``[chain, index]`` pairs.  Residue numbers are read directly from the PDB
    record (``residue.id[1]``), which for ``cleaned_protein`` (produced by
    ``isolate_protein_chain`` + ``renumber_pdb_residues``) are 1-based per-chain
    integers — consistent with what ``mk_prepare_receptor.py --flexres`` expects.

    :param protein_pdb: Path to the cleaned protein PDB.
    :param protein_chain: Chain ID, list of chain IDs, or comma-separated string.
    :param center: ``(x, y, z)`` of the box center.
    :param size: ``(sx, sy, sz)`` full edge lengths of the box.
    :param max_flexres: If set, keep only the ``max_flexres`` residues whose Cα
        is closest to the box center. Vina/gnina search time grows sharply with
        the number of flexible residues; values above ~8 become impractical.
    :return: AutoDock flexres string, or ``None`` if no residues fall inside the box.
    """
    cx, cy, cz = center
    sx, sy, sz = size
    half = np.array([sx / 2.0, sy / 2.0, sz / 2.0])
    center_arr = np.array([cx, cy, cz])

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", protein_pdb)
    model = structure[0]
    available = [c.id for c in model]

    hits = []  # (distance_to_center, token)
    for chain_id in _normalize_chain_list(protein_chain):
        if chain_id not in available:
            logger.warning(
                f"get_flexres_string_from_box: chain '{chain_id}' not found in {protein_pdb}."
            )
            continue
        for residue in model[chain_id]:
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            ca_coord = residue["CA"].get_coord()
            if np.all(np.abs(ca_coord - center_arr) <= half):
                dist = float(np.linalg.norm(ca_coord - center_arr))
                hits.append((dist, f"{chain_id}:{residue.id[1]}"))

    if not hits:
        logger.debug(
            f"get_flexres_string_from_box: no residues inside box center={center} "
            f"size={size} in {protein_pdb}."
        )
        return None

    if max_flexres is not None and len(hits) > max_flexres:
        hits.sort(key=lambda x: x[0])
        hits = hits[:max_flexres]
        logger.debug(f"get_flexres_string_from_box: capped to {max_flexres} closest residues.")

    flexres_str = "_".join(token for _, token in hits)
    logger.debug(f"get_flexres_string_from_box: {len(hits)} flexible residues: {flexres_str}")
    return flexres_str


def covalent_rec_atom_exists(protein_pdb: str, spec: str) -> bool:
    """
    Check that a gnina ``chain:resnum:atomname`` covalent receptor-atom spec
    resolves to a real atom in ``protein_pdb``.

    gnina silently mis-bonds (or aborts) when ``--covalent_rec_atom`` points at
    a non-existent atom, so guild validates the spec against the *cleaned*
    protein up-front. The residue number is read from ``residue.id[1]`` — the
    same per-chain 1-based numbering produced by ``renumber_pdb_residues`` and
    used by :func:`get_flexres_string_from_box`.

    :param protein_pdb: Path to the cleaned protein PDB gnina will dock against.
    :param spec: ``chain:resnum:atomname`` string, e.g. ``"A:145:SG"``.
    :return: ``True`` if the chain, residue, and atom all exist; ``False``
        otherwise (including malformed specs).
    """
    parts = spec.split(":")
    if len(parts) != 3:
        logger.warning(
            f"covalent_rec_atom_exists: malformed spec '{spec}' "
            "(expected chain:resnum:atomname)."
        )
        return False
    chain_id, resnum_str, atom_name = parts[0], parts[1], parts[2].strip()
    try:
        resnum = int(resnum_str)
    except ValueError:
        logger.warning(f"covalent_rec_atom_exists: non-integer resnum in '{spec}'.")
        return False

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", protein_pdb)
    model = structure[0]
    if chain_id not in [c.id for c in model]:
        logger.warning(f"covalent_rec_atom_exists: chain '{chain_id}' not found in {protein_pdb}.")
        return False
    for residue in model[chain_id]:
        if residue.id[1] == resnum:
            if atom_name in residue:
                return True
            logger.warning(
                f"covalent_rec_atom_exists: atom '{atom_name}' not in "
                f"{chain_id}:{resnum} of {protein_pdb}."
            )
            return False
    logger.warning(
        f"covalent_rec_atom_exists: residue {chain_id}:{resnum} not found in {protein_pdb}."
    )
    return False


def add_covalent_conect(complex_pdb: str, covalent_rec_atom: str) -> None:
    """
    Append ``CONECT`` records to a complex PDB for a covalent gnina pose.

    gnina positions the warhead atom at covalent-bond distance (~1.8 Å) but
    does not write explicit connectivity records.  Calling this function after
    :func:`build_complex_pdb` makes the bond visible in PyMOL and PLIP without
    any manual post-processing.

    The receptor attachment atom is located by matching the
    ``chain:resnum:atomname`` spec against ``ATOM`` records.  The warhead is
    the nearest ``HETATM`` (chain Z) atom — gnina enforces a bond-length
    geometry, so it should always be within 2.5 Å.  If no HETATM is found
    within that threshold the function logs a warning and returns without
    modifying the file.

    :param complex_pdb: Path to the complex PDB produced by :func:`build_complex_pdb`.
    :param covalent_rec_atom: Receptor atom spec, ``chain:resnum:atomname``
        (e.g. ``"A:145:SG"``).  Must match the spec that was passed to gnina.
    :raises ValueError: If the spec is malformed or the receptor atom is not
        found in the complex PDB.
    """
    parts = covalent_rec_atom.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"add_covalent_conect: malformed spec '{covalent_rec_atom}' "
            "(expected chain:resnum:atomname)."
        )
    chain_id, resnum_str, atom_name = parts[0], parts[1], parts[2].strip()
    try:
        resnum = int(resnum_str)
    except ValueError as exc:
        raise ValueError(
            f"add_covalent_conect: non-integer resnum in '{covalent_rec_atom}'."
        ) from exc

    with open(complex_pdb, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # Locate the receptor ATOM record.
    rec_serial = None
    rec_coords = None
    for ln in lines:
        if not ln.startswith("ATOM"):
            continue
        # PDB fixed-column format:
        # cols 13-16 atom name, 22 chain, 23-26 resseq
        ln_chain = ln[21:22].strip()
        try:
            ln_resseq = int(ln[22:26].strip())
        except ValueError:
            continue
        ln_atom = ln[12:16].strip()
        if ln_chain == chain_id and ln_resseq == resnum and ln_atom == atom_name:
            try:
                rec_serial = int(ln[6:11].strip())
                rec_coords = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            except (ValueError, IndexError):
                pass
            break

    if rec_serial is None or rec_coords is None:
        raise ValueError(
            f"add_covalent_conect: receptor atom '{covalent_rec_atom}' "
            f"not found in {complex_pdb}."
        )

    # Find the closest HETATM (ligand chain Z) within 2.5 Å.
    best_serial = None
    best_dist = float("inf")
    for ln in lines:
        if not ln.startswith("HETATM"):
            continue
        try:
            lig_serial = int(ln[6:11].strip())
            coords = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        except (ValueError, IndexError):
            continue
        dist = float(np.linalg.norm(coords - rec_coords))
        if dist < best_dist:
            best_dist = dist
            best_serial = lig_serial

    if best_serial is None or best_dist > _COVALENT_BOND_DIST_MAX:
        logger.warning(
            f"add_covalent_conect: no HETATM within {_COVALENT_BOND_DIST_MAX} Å of "
            f"'{covalent_rec_atom}' in {complex_pdb} "
            f"(closest={best_dist:.2f} Å) — CONECT not written."
        )
        return

    # Insert CONECT records before the final END line.
    conect_a = f"CONECT{rec_serial:5d}{best_serial:5d}\n"
    conect_b = f"CONECT{best_serial:5d}{rec_serial:5d}\n"
    if lines and lines[-1].strip() == "END":
        lines = lines[:-1] + [conect_a, conect_b, "END\n"]
    else:
        lines += [conect_a, conect_b]

    with open(complex_pdb, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    logger.debug(
        f"add_covalent_conect: wrote CONECT {rec_serial}↔{best_serial} "
        f"({best_dist:.2f} Å) in {complex_pdb}."
    )


def relabel_ligand_chain_in_pdb(
    input_pdb: str,
    output_pdb: str,
    ligand_chain_id: str = "L",
    new_resname: str = "LIG",
    new_chain: str = "Z",
    new_resseq: int = 1,
) -> str:
    """
    Post-processes a full-complex PDB produced from a Boltz CIF conversion.
    Every ATOM/HETATM record on `ligand_chain_id` is rewritten as a HETATM entry
    with (new_resname, new_chain, new_resseq) so PLIP can locate the binding site.
    Protein ATOM records are written unchanged.

    :param input_pdb: Path to the full-complex PDB from CIF conversion.
    :param output_pdb: Path for the PLIP-ready output PDB.
    :param ligand_chain_id: Chain ID used for the ligand in the Boltz CIF (default "L").
    :param new_resname: Residue name to assign to the ligand (default "LIG").
    :param new_chain: Chain ID to assign to the ligand (default "Z").
    :param new_resseq: Residue sequence number for the ligand (default 1).
    :return: output_pdb path.
    """
    with open(input_pdb, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    out_lines = []
    resname_padded = new_resname.ljust(3)[:3]
    resseq_str = str(new_resseq).rjust(4)

    for line in lines:
        record = line[:6].strip()
        if record in ("ATOM", "HETATM") and len(line) > 21 and line[21] == ligand_chain_id:
            # PDB fixed-width columns:
            # 1-6  record type (6)   7-11 serial (5)  12 blank  13-16 atom name (4)
            # 17 altLoc (1)  18-20 resName (3)  21 blank  22 chainID (1)
            # 23-26 resSeq (4)  rest unchanged
            line = (
                f"HETATM{line[6:12]}{line[12:17]}{resname_padded} "
                f"{new_chain}{resseq_str}{line[26:]}"
            )
        out_lines.append(line)

    with open(output_pdb, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)

    return output_pdb
