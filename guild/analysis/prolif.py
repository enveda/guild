"""
ProLIF (Protein-Ligand Interaction Fingerprinter) analysis tools.

Uses fp.generate(..., metadata=True) to access the full IFP geometry dict rather
than the boolean fingerprint — each interaction record includes distance and
type-specific angles alongside residue/atom identifiers.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

from guild.constants.guild import PROTEIN_CONF_ID, SMILES
from guild.constants.interactions import (
    DETAIL_ACC_ANGLE,
    DETAIL_ANGLE,
    DETAIL_COLUMNS,
    DETAIL_DISTANCE,
    DETAIL_DOCKING_METHOD,
    DETAIL_DON_ANGLE,
    DETAIL_INTERACTION_TYPE,
    DETAIL_LIG_ATOM_IDX,
    DETAIL_OFFSET,
    DETAIL_PISTACK_TYPE,
    DETAIL_PROT_CHAIN,
    DETAIL_PROT_RESNAME,
    DETAIL_PROT_RESNR,
    DETAIL_SOURCE,
    INTERACTION_COMBINATION_ID,
    INTERACTION_TYPE_HALOGEN,
    INTERACTION_TYPE_HBOND,
    INTERACTION_TYPE_HYDROPHOBIC,
    INTERACTION_TYPE_METAL,
    INTERACTION_TYPE_PICATION,
    INTERACTION_TYPE_PISTACKING,
    INTERACTION_TYPE_SALTBRIDGE,
    INTERACTION_TYPE_WATERBRIDGE,
    LIGAND_RESNAME,
)

logger = logging.getLogger(__name__)

# ProLIF interaction class name → guild controlled vocabulary
_PROLIF_TYPE_MAP: Dict[str, str] = {
    "HBDonor": INTERACTION_TYPE_HBOND,
    "HBAcceptor": INTERACTION_TYPE_HBOND,
    "Hydrophobic": INTERACTION_TYPE_HYDROPHOBIC,
    "VdWContact": INTERACTION_TYPE_HYDROPHOBIC,
    "FaceToFace": INTERACTION_TYPE_PISTACKING,
    "EdgeToFace": INTERACTION_TYPE_PISTACKING,
    "PiStacking": INTERACTION_TYPE_PISTACKING,
    "CationPi": INTERACTION_TYPE_PICATION,
    "PiCation": INTERACTION_TYPE_PICATION,
    "Cationic": INTERACTION_TYPE_SALTBRIDGE,
    "Anionic": INTERACTION_TYPE_SALTBRIDGE,
    "XBDonor": INTERACTION_TYPE_HALOGEN,
    "XBAcceptor": INTERACTION_TYPE_HALOGEN,
    "WaterBridge": INTERACTION_TYPE_WATERBRIDGE,
    "MetalDonor": INTERACTION_TYPE_METAL,
    "MetalAcceptor": INTERACTION_TYPE_METAL,
}

# ProLIF class names that produce a face/T-shaped pistack_type label
_PISTACK_TYPE_MAP: Dict[str, str] = {
    "FaceToFace": "F",
    "EdgeToFace": "T",
}


def analyze_prolif_interactions(
    complex_pdb_path: str,
    combination_id: str,
    protein_conf_id: str,
    smiles: str,
    docking_method: str,
    ligand_resname: str = LIGAND_RESNAME,
) -> List[Dict]:
    """
    Extract per-interaction records from a complex PDB using ProLIF.

    Uses fp.generate(..., metadata=True) to access IFP geometry — each record
    includes distance and type-specific angles (DHA for H-bonds, plane_angle for
    pi-stacking, AXD/XAR for halogen bonds, etc.) as well as ligand atom index.

    Columns with no ProLIF equivalent (distance_ad, distance_ah, distance_aw,
    distance_dw, ligand_atom_type, protein_sidechain, donor_type, acceptor_type,
    prot_is_donor, prot_is_positive, metal_type, coordination_num, metal_geometry,
    target_type) are absent from each returned dict and become None in the DataFrame.

    :param complex_pdb_path: Path to the protein-ligand complex PDB
    :param combination_id: Unique combination identifier
    :param protein_conf_id: Protein configuration ID
    :param smiles: Ligand SMILES string
    :param docking_method: Docking method prefix (e.g. "vina", "boltz")
    :param ligand_resname: Residue name of the ligand in the PDB (default "LIG")
    :return: List of interaction record dicts, or [] on failure / missing deps
    """
    try:
        import prolif
        from prolif.molecule import Molecule
    except ImportError:
        logger.warning(
            "prolif not available — skipping ProLIF analysis. "
            "Ensure 'prolif>=2.0.0,<3' is listed in pyproject.toml dependencies."
        )
        return []

    import os

    if not os.path.exists(complex_pdb_path):
        logger.warning(f"Complex PDB file not found: {complex_pdb_path}")
        return []

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        pdb_text = open(complex_pdb_path).read()

        # Split PDB into per-residue blocks for ligand and protein separately.
        # Build ligand mol via SMILES template (preserves bond orders and avoids
        # the MDAnalysis vdW-radii crash on halogens).
        lig_lines, prot_lines = [], []
        for line in pdb_text.splitlines(keepends=True):
            if line.startswith(("ATOM  ", "HETATM")):
                resname = line[17:20].strip()
                if resname == ligand_resname:
                    lig_lines.append(line)
                else:
                    prot_lines.append(line)
            elif line.startswith(("CONECT", "END")):
                continue

        if not lig_lines or not prot_lines:
            logger.warning(
                f"ProLIF: empty ligand ({len(lig_lines)} lines) or protein "
                f"({len(prot_lines)} lines) in {complex_pdb_path}"
            )
            return []

        # Ligand: parse heavy-atom PDB block, assign bond orders from SMILES
        lig_pdb_block = "".join(lig_lines) + "END\n"
        lig_raw = Chem.MolFromPDBBlock(lig_pdb_block, removeHs=False, sanitize=False)
        if lig_raw is None:
            logger.warning(f"ProLIF: RDKit failed to parse ligand block in {complex_pdb_path}")
            return []
        tmpl = Chem.MolFromSmiles(smiles)
        if tmpl is not None:
            try:
                lig_rdmol = AllChem.AssignBondOrdersFromTemplate(tmpl, lig_raw)
                Chem.SanitizeMol(lig_rdmol)
            except Exception:
                lig_rdmol = lig_raw
                Chem.SanitizeMol(
                    lig_rdmol,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                )
        else:
            lig_rdmol = lig_raw

        # Protein: parse as a single PDB block
        prot_pdb_block = "".join(prot_lines) + "END\n"
        prot_rdmol = Chem.MolFromPDBBlock(prot_pdb_block, removeHs=False, sanitize=False)
        if prot_rdmol is None:
            logger.warning(f"ProLIF: RDKit failed to parse protein block in {complex_pdb_path}")
            return []
        Chem.SanitizeMol(
            prot_rdmol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )

        lig_mol = Molecule(lig_rdmol)
        prot_mol = Molecule(prot_rdmol)

        fp = prolif.Fingerprint()
        ifp = fp.generate(lig_mol, prot_mol, metadata=True)

        records = []
        base = {
            INTERACTION_COMBINATION_ID: combination_id,
            PROTEIN_CONF_ID: protein_conf_id,
            SMILES: smiles,
            DETAIL_DOCKING_METHOD: docking_method,
            DETAIL_SOURCE: "prolif",
        }

        for idata in ifp.interactions():
            meta = idata.metadata
            class_name = idata.interaction
            itype = _PROLIF_TYPE_MAP.get(class_name, class_name.lower())

            # Resolve primary angle: DHA for H-bonds, plane_angle for pi, etc.
            angle = meta.get("DHA_angle") or meta.get("angle") or meta.get("plane_angle")

            # Ligand atom index from parent_indices (index in the full complex)
            lig_parent = meta.get("parent_indices", {}).get("ligand", ())
            lig_atom_idx = lig_parent[0] if lig_parent else None

            record = {
                **base,
                DETAIL_INTERACTION_TYPE: itype,
                DETAIL_PROT_RESNAME: getattr(idata.protein, "name", None),
                DETAIL_PROT_RESNR: getattr(idata.protein, "number", None),
                DETAIL_PROT_CHAIN: getattr(idata.protein, "chain", None),
                DETAIL_LIG_ATOM_IDX: lig_atom_idx,
                DETAIL_DISTANCE: meta.get("distance"),
                DETAIL_ANGLE: angle,
            }

            # Halogen bond angles
            if class_name in ("XBDonor", "XBAcceptor"):
                record[DETAIL_DON_ANGLE] = meta.get("AXD_angle")
                record[DETAIL_ACC_ANGLE] = meta.get("XAR_angle")

            # Pi-stacking geometry and subtype
            if class_name in _PISTACK_TYPE_MAP:
                record[DETAIL_PISTACK_TYPE] = _PISTACK_TYPE_MAP[class_name]
                record[DETAIL_OFFSET] = meta.get("normal_to_centroid_angle")
            elif class_name in ("CationPi", "PiCation"):
                record[DETAIL_OFFSET] = meta.get("normal_to_centroid_angle")

            records.append(record)

        logger.info(f"ProLIF: {len(records)} interactions found in {complex_pdb_path}")
        return records

    except Exception as e:
        logger.error(f"ProLIF analysis failed for {complex_pdb_path}: {type(e).__name__}: {e}")
        return []


def get_batch_prolif_interactions(
    complex_metadata: List[Tuple[str, str, str, str, str]],
    ligand_resname: str = LIGAND_RESNAME,
) -> pd.DataFrame:
    """
    Collect per-interaction ProLIF records for a batch of complex PDBs.

    :param complex_metadata: List of (path, combination_id, protein_conf_id, smiles, docking_method)
    :param ligand_resname: Residue name of the ligand in each PDB
    :return: DataFrame with DETAIL_COLUMNS schema; empty DataFrame if nothing succeeds.
    """
    all_records = []
    for path, combination_id, protein_conf_id, smiles, docking_method in complex_metadata:
        records = analyze_prolif_interactions(
            path,
            combination_id,
            protein_conf_id,
            smiles,
            docking_method,
            ligand_resname=ligand_resname,
        )
        all_records.extend(records)

    if not all_records:
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    df = pd.DataFrame(all_records)
    for col in DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[DETAIL_COLUMNS]
