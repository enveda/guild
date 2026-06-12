"""
Tools for preparing receptors for docking.
"""

import logging
import warnings

from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser, StructureBuilder
from Bio.PDB.PDBIO import Select
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from guild.constants.proteins import METALS_LIST, WATERS_LIST

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=BiopythonWarning)


class KeepAll(Select):
    def __init__(self, keep):
        self.keep = keep  # set of (chain_id, (' ',resseq,icode)) to keep

    def accept_chain(self, chain):
        return 1

    def accept_residue(self, residue):
        key = (residue.get_parent().id, residue.id)
        return 1 if key in self.keep else 0

    def accept_atom(self, atom):
        return 1


def strip_pdb_none(output_pdb: str):
    with open(output_pdb, "r") as f:
        lines = f.readlines()
    with open(output_pdb, "w") as f:
        for line in lines:
            # Replace any " None" between B-factor and element columns with spaces
            if line.startswith(("ATOM", "HETATM")):
                line = line.replace(" None", "     ")
            f.write(line)


def _is_metal_res(res):
    return len(res) == 1 and (res.child_list[0].element or "").upper() in METALS_LIST


def _pick_altloc_atoms(res, prefer_altloc="A", occ_floor=0.01):
    chosen = {}
    for a in res:
        name = a.get_name().strip()
        occ = a.get_occupancy() or 0.0
        if occ < occ_floor:  # drop zero/near-zero occupancy
            continue
        alt = (a.get_altloc() or "").strip()
        cur = chosen.get(name)
        take = False
        if cur is None:
            take = True
        else:
            cur_alt = (cur.get_altloc() or "").strip()
            cur_occ = cur.get_occupancy() or 0.0
            # prefer requested altloc, else higher occupancy
            if prefer_altloc and alt == prefer_altloc and cur_alt != prefer_altloc:
                take = True
            elif occ > cur_occ:
                take = True
        if take:
            chosen[name] = a
    return list(chosen.values())


def clean_receptor(
    input_pdb: str,
    output_pdb: str,
    prefer_altloc: str = "A",
    keep_resnames=None,
    keep_metals: bool = True,
):
    keep_resnames = set(keep_resnames or [])
    parser = PDBParser(QUIET=True)
    in_struct = parser.get_structure("receptor", input_pdb)
    in_model = next(in_struct.get_models())

    # build output
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure("clean")
    structure_builder.init_model(0)
    out_struct = structure_builder.get_structure()
    out_model = out_struct[0]

    def ensure_chain(cid: str):
        structure_builder.init_chain(cid)  # (re)select chain in builder
        return out_model[cid] if cid in out_model else out_model.child_list[-1]

    for chain in in_model:
        cid_base = chain.id or "A"
        cid = cid_base
        out_chain = ensure_chain(cid)

        cid_base = chain.id or "A"
        cid = None
        out_chain = None

        for res in chain:
            cid = cid_base
            out_chain = ensure_chain(cid)
            resname = res.get_resname().upper()
            if resname in WATERS_LIST:
                continue
            het = res.id[0] != " "
            if het and not ((_is_metal_res(res) and keep_metals) or (resname in keep_resnames)):
                continue

            picked = _pick_altloc_atoms(res, prefer_altloc=prefer_altloc, occ_floor=0.0)
            if not picked:
                picked = list(res)

            out_chain = ensure_chain(cid)
            structure_builder.init_residue(res.get_resname(), res.id[0], res.id[1], res.id[2])
            out_res = out_chain.child_list[-1]
            out_res.child_list[:] = []
            for current_atom in picked:
                atom_copy = current_atom.copy()
                # altloc/occ normalize
                alt = (atom_copy.get_altloc() or "").strip()
                if alt and alt != prefer_altloc:
                    atom_copy.set_altloc(" ")
                if getattr(atom_copy, "segid", None) is None:
                    atom_copy.segid = "    "
                out_res.add(atom_copy)

    io = PDBIO()
    io.set_structure(out_struct)  # write OUTPUT structure
    io.save(output_pdb)
    with open(output_pdb) as f:
        if not any(line.startswith("ATOM") for line in f):
            raise RuntimeError("Cleaned PDB has no ATOM records.")
    strip_pdb_none(output_pdb)
    return output_pdb


class KeepProteinAndMetals(Select):
    def accept_residue(self, residue):
        name = residue.resname.upper()
        het = residue.id[0] != " "
        if name in WATERS_LIST:
            return 0
        if not het:
            return 1  # standard residues
        # keep single-atom metals
        return len(residue) == 1 and (residue.child_list[0].element or "").upper() in METALS_LIST


def _normalize_chain_list(chains):
    """
    Normalize a chain specification into a list of chain-ID strings.

    Accepts a single chain ID (``"A"``), a comma-separated string (``"A,B"``),
    a list/tuple of IDs, or ``None`` (→ empty list). Whitespace is stripped and
    empty tokens dropped, so ``"A, B"`` and ``["A", "B"]`` both yield
    ``["A", "B"]``. This is the single place that defines how the
    ``protein_chain`` CSV column encodes multiple chains.
    """
    if chains is None:
        return []

    # Treat pandas/NumPy missing values (NaN, <NA>) as "no chain specified".
    try:
        import pandas as pd  # type: ignore

        if pd.isna(chains):
            return []
    except Exception:
        pass

    if isinstance(chains, str):
        return [c.strip() for c in chains.split(",") if c.strip()]

    # Iterable of chain IDs (list/tuple/set/etc). If it's a non-iterable scalar,
    # treat it as a single token.
    try:
        return [str(c).strip() for c in chains if str(c).strip()]
    except TypeError:
        token = str(chains).strip()
        return [token] if token else []


def get_protein_chain(input_file, input_name):
    """
    Get the chain ID of the first chain in a PDB file.
    :param input_file: The path to the input PDB file.
    :param input_name: The name of the input PDB file.
    :return: The chain ID of the first chain in the PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(input_name, input_file)
    model = structure[0]

    # find the requested chain (or default to first)
    chain_to_write = next(model.get_chains())
    logger.warning(
        f"No chain specified for {input_file}. Defaulting to first chain with ID '{chain_to_write.id}'."
    )

    return chain_to_write.id


def isolate_protein_chain(input_file, input_name, output_file, target_chain=None):
    """
    Isolate one or more protein chains by their chain IDs (e.g., 'A' or ['A', 'B']).
    Falls back to the first chain if ``target_chain`` is None or none of the
    requested chains are present. Preserves chain IDs so a multi-chain pocket
    (e.g. a dimer interface) stays intact downstream. (HETATM cleanup is done by ``clean_receptor``.)
    :param input_file: The path to the input PDB file.
    :param input_name: The name of the input PDB file.
    :param output_file: The path to the output PDB file.
    :param target_chain: Chain ID, list of chain IDs, or comma-separated string
                         (e.g. ``"A,B"``). If not provided, the first chain is used.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(input_name, input_file)
    model = structure[0]

    available = [ch.id for ch in model.get_chains()]
    requested = _normalize_chain_list(target_chain)
    chains_to_write = [c for c in requested if c in available]
    if not chains_to_write:
        # Fall back to the first chain if nothing was requested or matched.
        chains_to_write = [next(model.get_chains()).id]
        logger.warning(
            f"No requested chain {requested} found in {input_file}. "
            f"Defaulting to first chain '{chains_to_write[0]}'."
        )

    # 🔧 rebuild a proper structure hierarchy, preserving every requested chain ID
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure("filtered")
    structure_builder.init_model(0)
    for chain_id in chains_to_write:
        structure_builder.structure[0].add(model[chain_id].copy())  # preserve chain ID

    io = PDBIO()
    io.set_structure(structure_builder.get_structure())
    io.save(output_file)
    renumber_pdb_residues(output_file, output_file)


def renumber_pdb_residues(input_pdb, output_pdb, start_index=1):
    """
    Renumber the residues in a PDB file.

    :param input_pdb: The path to the input PDB file.
    :param output_pdb: The path to the output PDB file.
    :param start_index: The starting index for the renumbering.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("renumbered", input_pdb)
    for model in structure:
        for chain in model:
            # Renumber per chain so each chain is 1-based. Boltz pocket
            # contacts index residues per chain, so continuous numbering
            # across chains would misalign multi-chain constraints.
            i = start_index
            for residue in chain:
                res_id = residue.id
                residue.id = (res_id[0], i, res_id[2])
                i += 1
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def clean_smiles(smiles: str):
    """
    Clean and canonicalize a SMILES string by:
    1. Stripping salts (if possible)
    2. Canonicalizing the structure (speeds up obabel conversion 10-100x)

    :param smiles: The input SMILES string.
    :return: The cleaned and canonicalized SMILES string.
    """
    try:
        # Try to strip salts from the ligand
        remover = SaltRemover()
        stripped_mol = remover.StripMol(Chem.MolFromSmiles(smiles))
    except Exception as e:
        logging.warning(f"Could not strip salts from SMILES: {smiles}. Error: {e}")
        try:
            # Fallback: parse without salt removal
            stripped_mol = Chem.MolFromSmiles(smiles)
        except Exception as e2:
            logging.warning(f"Could not parse SMILES: {smiles}. Error: {e2}")
            return smiles

    # Canonicalize SMILES - normalizes stereochemistry and structure
    return Chem.MolToSmiles(stripped_mol, canonical=True)
