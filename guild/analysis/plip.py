"""
PLIP (Protein-Ligand Interaction Profiler) analysis tools
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from plip.structure.preparation import PDBComplex

from guild.constants.guild import PROTEIN_CONF_ID, SMILES
from guild.constants.interactions import (
    COMPLEX_PDB_SUFFIX,
    DETAIL_ACC_ANGLE,
    DETAIL_ACCEPTOR_TYPE,
    DETAIL_ANGLE,
    DETAIL_COLUMNS,
    DETAIL_COORDINATION,
    DETAIL_DISTANCE,
    DETAIL_DISTANCE_AD,
    DETAIL_DISTANCE_AH,
    DETAIL_DISTANCE_AW,
    DETAIL_DISTANCE_DW,
    DETAIL_DOCKING_METHOD,
    DETAIL_DON_ANGLE,
    DETAIL_DONOR_TYPE,
    DETAIL_INTERACTION_TYPE,
    DETAIL_LIG_ATOM_IDX,
    DETAIL_LIG_ATOM_TYPE,
    DETAIL_METAL_GEOMETRY,
    DETAIL_METAL_TYPE,
    DETAIL_OFFSET,
    DETAIL_PISTACK_TYPE,
    DETAIL_PROT_CHAIN,
    DETAIL_PROT_RESNAME,
    DETAIL_PROT_RESNR,
    DETAIL_PROT_SIDECHAIN,
    DETAIL_PROTISDON,
    DETAIL_SALTBRIDGE_PROTISPOS,
    DETAIL_SOURCE,
    DETAIL_TARGET_TYPE,
    INTERACTION_COMBINATION_ID,
    INTERACTION_TYPE_HALOGEN,
    INTERACTION_TYPE_HBOND,
    INTERACTION_TYPE_HYDROPHOBIC,
    INTERACTION_TYPE_METAL,
    INTERACTION_TYPE_PICATION,
    INTERACTION_TYPE_PISTACKING,
    INTERACTION_TYPE_SALTBRIDGE,
    INTERACTION_TYPE_WATERBRIDGE,
    LIGAND_CHAIN,
    LIGAND_RESNAME,
    LIGAND_RESSEQ,
    N_HALOGEN,
    N_HBONDS,
    N_HYDROPHOBIC,
    N_METAL,
    N_PICATION,
    N_PISTACKING,
    N_SALTBRIDGES,
    N_UNIQUE_RESIDUES,
    N_WATERBRIDGES,
    TOTAL_INTERACTIONS,
)

logger = logging.getLogger(__name__)


def analyze_protein_ligand_interactions(
    complex_pdb_path: str,
    ligand_resname: str = LIGAND_RESNAME,
    ligand_chain: str = LIGAND_CHAIN,
    ligand_resseq: int = LIGAND_RESSEQ,
) -> Optional[Dict]:
    """
    Analyze protein-ligand interactions using PLIP.

    :param complex_pdb_path: Path to the complex PDB file
    :param ligand_resname: Residue name of the ligand (default: from constants)
    :param ligand_chain: Chain ID of the ligand (default: from constants)
    :param ligand_resseq: Residue sequence number of the ligand (default: from constants)
    :return: Dictionary containing interaction data, or None if analysis fails
    """
    if not os.path.exists(complex_pdb_path):
        logger.warning(f"Complex PDB file not found: {complex_pdb_path}")
        return None

    try:
        # Load PDB complex
        mol = PDBComplex()
        mol.load_pdb(complex_pdb_path)

        # Construct binding site identifier: HetID:Chain:Position
        bsid = f"{ligand_resname}:{ligand_chain}:{ligand_resseq}"

        # Analyze interactions
        mol.analyze()

        # Check if binding site was found
        if bsid not in mol.interaction_sets:
            logger.warning(f"Binding site {bsid} not found in {complex_pdb_path}")
            # Try to find any binding site
            if mol.interaction_sets:
                bsid = list(mol.interaction_sets.keys())[0]
                logger.info(f"Using alternative binding site: {bsid}")
            else:
                logger.warning(f"No binding sites found in {complex_pdb_path}")
                return None

        interactions = mol.interaction_sets[bsid]

        # Extract interaction data using correct PLIP API attributes
        # Hydrogen bonds: combine ligand donor and protein donor
        all_hbonds = list(interactions.hbonds_ldon) + list(interactions.hbonds_pdon)
        hbond_residues = [hb.resnr for hb in all_hbonds]

        # Hydrophobic interactions
        hydrophobic_residues = [hc.resnr for hc in interactions.hydrophobic_contacts]

        # Pi-stacking
        pistacking_residues = [ps.resnr for ps in interactions.pistacking]

        # Pi-cation: combine ligand aromatic and protein aromatic
        all_pication = list(interactions.pication_laro) + list(interactions.pication_paro)
        pication_residues = [pc.resnr for pc in all_pication]

        # Salt bridges: combine ligand negative and protein negative
        all_saltbridges = list(interactions.saltbridge_lneg) + list(interactions.saltbridge_pneg)
        saltbridge_residues = [sb.resnr for sb in all_saltbridges]

        # Halogen bonds
        halogen_residues = [hb.resnr for hb in interactions.halogen_bonds]

        # Water bridges
        waterbridge_residues = [wb.resnr for wb in interactions.water_bridges]

        # Metal complexes
        metal_residues = [mc.resnr for mc in interactions.metal_complexes]

        interaction_data = {
            # Hydrogen bonds
            N_HBONDS: len(all_hbonds),
            # Hydrophobic interactions
            N_HYDROPHOBIC: len(interactions.hydrophobic_contacts),
            # Pi-stacking
            N_PISTACKING: len(interactions.pistacking),
            # Pi-cation interactions
            N_PICATION: len(all_pication),
            # Salt bridges
            N_SALTBRIDGES: len(all_saltbridges),
            # Halogen bonds
            N_HALOGEN: len(interactions.halogen_bonds),
            # Water bridges
            N_WATERBRIDGES: len(interactions.water_bridges),
            # Metal complexes
            N_METAL: len(interactions.metal_complexes),
            # Total interactions count
            TOTAL_INTERACTIONS: sum(
                [
                    len(all_hbonds),
                    len(interactions.hydrophobic_contacts),
                    len(interactions.pistacking),
                    len(all_pication),
                    len(all_saltbridges),
                    len(interactions.halogen_bonds),
                    len(interactions.water_bridges),
                    len(interactions.metal_complexes),
                ]
            ),
        }

        # Count unique interacting residues
        all_residues = set(
            hbond_residues
            + hydrophobic_residues
            + pistacking_residues
            + pication_residues
            + saltbridge_residues
            + halogen_residues
            + waterbridge_residues
            + metal_residues
        )
        interaction_data[N_UNIQUE_RESIDUES] = len(all_residues)

        logger.info(
            f"Successfully analyzed {complex_pdb_path}: {interaction_data[TOTAL_INTERACTIONS]} total interactions"
        )
        return interaction_data

    except FileNotFoundError:
        logger.error(f"PDB file not found: {complex_pdb_path}")
        return None
    except KeyError as e:
        logger.error(
            f"Failed to analyze {complex_pdb_path}: Binding site or interaction type not found - {e}"
        )
        return None
    except AttributeError as e:
        logger.error(f"Failed to analyze {complex_pdb_path}: PLIP API attribute error - {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error analyzing {complex_pdb_path}: {type(e).__name__}: {e}")
        return None


def analyze_batch_interactions(
    complex_pdb_paths: List[str],
    combination_ids: Optional[List[str]] = None,
    ligand_resname: str = LIGAND_RESNAME,
    ligand_chain: str = LIGAND_CHAIN,
    ligand_resseq: int = LIGAND_RESSEQ,
) -> pd.DataFrame:
    """
    Analyze multiple protein-ligand complexes and return results as DataFrame.

    :param complex_pdb_paths: List of paths to complex PDB files
    :param combination_ids: Optional list of combination IDs corresponding to each complex
    :param ligand_resname: Residue name of the ligand
    :param ligand_chain: Chain ID of the ligand
    :param ligand_resseq: Residue sequence number of the ligand
    :return: DataFrame with interaction data for all complexes
    """
    results = []

    for idx, complex_path in enumerate(complex_pdb_paths):
        interaction_data = analyze_protein_ligand_interactions(
            complex_path,
            ligand_resname=ligand_resname,
            ligand_chain=ligand_chain,
            ligand_resseq=ligand_resseq,
        )

        if interaction_data is not None:
            # Add combination ID if provided
            if combination_ids is not None and idx < len(combination_ids):
                interaction_data[INTERACTION_COMBINATION_ID] = combination_ids[idx]
            else:
                # Extract from filename
                base_name = os.path.basename(complex_path).replace(COMPLEX_PDB_SUFFIX, "")
                interaction_data[INTERACTION_COMBINATION_ID] = base_name

            results.append(interaction_data)

    if not results:
        logger.warning("No successful interaction analyses")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns to put combination_id first
    cols = [INTERACTION_COMBINATION_ID] + [
        col for col in df.columns if col != INTERACTION_COMBINATION_ID
    ]
    df = df[cols]

    return df


def get_detailed_interactions(
    complex_pdb_path: str,
    combination_id: str,
    protein_conf_id: str,
    smiles: str,
    docking_method: str,
    ligand_resname: str = LIGAND_RESNAME,
    ligand_chain: str = LIGAND_CHAIN,
    ligand_resseq: int = LIGAND_RESSEQ,
) -> List[Dict]:
    """
    Extract one record per interaction from a complex PDB using PLIP.

    Sparse columns (e.g. pistack_type on a hydrophobic row) are omitted from each
    dict and filled with None when the batch function builds the DataFrame.

    :param complex_pdb_path: Path to the protein-ligand complex PDB
    :param combination_id: Unique combination identifier
    :param protein_conf_id: Protein configuration ID
    :param smiles: Ligand SMILES string
    :param docking_method: Docking method prefix (e.g. "vina", "boltz")
    :param ligand_resname: Residue name of the ligand in the PDB (default "LIG")
    :param ligand_chain: Chain ID of the ligand in the PDB (default "Z")
    :param ligand_resseq: Residue sequence number of the ligand (default 1)
    :return: List of interaction record dicts matching DETAIL_COLUMNS, or [] on failure
    """
    if not os.path.exists(complex_pdb_path):
        logger.warning(f"Complex PDB file not found: {complex_pdb_path}")
        return []

    try:
        mol = PDBComplex()
        mol.load_pdb(complex_pdb_path)
        mol.analyze()

        bsid = f"{ligand_resname}:{ligand_chain}:{ligand_resseq}"
        if bsid not in mol.interaction_sets:
            if mol.interaction_sets:
                bsid = list(mol.interaction_sets.keys())[0]
                logger.info(f"Using alternative binding site {bsid} in {complex_pdb_path}")
            else:
                logger.warning(f"No binding sites found in {complex_pdb_path}")
                return []

        ix = mol.interaction_sets[bsid]
        records = []

        base = {
            INTERACTION_COMBINATION_ID: combination_id,
            PROTEIN_CONF_ID: protein_conf_id,
            SMILES: smiles,
            DETAIL_DOCKING_METHOD: docking_method,
            DETAIL_SOURCE: "plip",
        }

        # Hydrogen bonds
        for hb in list(ix.hbonds_ldon) + list(ix.hbonds_pdon):
            lig_atom_idx = hb.a_orig_idx if hb.protisdon else hb.d_orig_idx
            lig_atom_type = hb.atype if hb.protisdon else hb.dtype
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_HBOND,
                    DETAIL_PROT_RESNAME: hb.restype,
                    DETAIL_PROT_RESNR: hb.resnr,
                    DETAIL_PROT_CHAIN: hb.reschain,
                    DETAIL_PROT_SIDECHAIN: hb.sidechain,
                    DETAIL_LIG_ATOM_IDX: lig_atom_idx,
                    DETAIL_LIG_ATOM_TYPE: lig_atom_type,
                    DETAIL_DISTANCE_AH: hb.distance_ah,
                    DETAIL_DISTANCE_AD: hb.distance_ad,
                    DETAIL_ANGLE: hb.angle,
                    DETAIL_DONOR_TYPE: hb.dtype,
                    DETAIL_ACCEPTOR_TYPE: hb.atype,
                    DETAIL_PROTISDON: hb.protisdon,
                }
            )

        # Hydrophobic contacts
        for hc in ix.hydrophobic_contacts:
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_HYDROPHOBIC,
                    DETAIL_PROT_RESNAME: hc.restype,
                    DETAIL_PROT_RESNR: hc.resnr,
                    DETAIL_PROT_CHAIN: hc.reschain,
                    DETAIL_LIG_ATOM_IDX: hc.ligatom_orig_idx,
                    DETAIL_DISTANCE: hc.distance,
                }
            )

        # Pi-stacking
        for ps in ix.pistacking:
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_PISTACKING,
                    DETAIL_PROT_RESNAME: ps.restype,
                    DETAIL_PROT_RESNR: ps.resnr,
                    DETAIL_PROT_CHAIN: ps.reschain,
                    DETAIL_DISTANCE: ps.distance,
                    DETAIL_ANGLE: ps.angle,
                    DETAIL_OFFSET: ps.offset,
                    DETAIL_PISTACK_TYPE: ps.type,
                }
            )

        # Pi-cation
        for pc in list(ix.pication_laro) + list(ix.pication_paro):
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_PICATION,
                    DETAIL_PROT_RESNAME: pc.restype,
                    DETAIL_PROT_RESNR: pc.resnr,
                    DETAIL_PROT_CHAIN: pc.reschain,
                    DETAIL_DISTANCE: pc.distance,
                    DETAIL_OFFSET: pc.offset,
                }
            )

        # Salt bridges
        for sb in list(ix.saltbridge_lneg) + list(ix.saltbridge_pneg):
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_SALTBRIDGE,
                    DETAIL_PROT_RESNAME: sb.restype,
                    DETAIL_PROT_RESNR: sb.resnr,
                    DETAIL_PROT_CHAIN: sb.reschain,
                    DETAIL_DISTANCE: sb.distance,
                    DETAIL_SALTBRIDGE_PROTISPOS: sb.protispos,
                }
            )

        # Halogen bonds
        for hb in ix.halogen_bonds:
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_HALOGEN,
                    DETAIL_PROT_RESNAME: hb.restype,
                    DETAIL_PROT_RESNR: hb.resnr,
                    DETAIL_PROT_CHAIN: hb.reschain,
                    DETAIL_PROT_SIDECHAIN: hb.sidechain,
                    DETAIL_LIG_ATOM_IDX: hb.don_orig_idx,
                    DETAIL_DISTANCE: hb.distance,
                    DETAIL_DON_ANGLE: hb.don_angle,
                    DETAIL_ACC_ANGLE: hb.acc_angle,
                    DETAIL_DONOR_TYPE: hb.donortype,
                    DETAIL_ACCEPTOR_TYPE: hb.acctype,
                }
            )

        # Water bridges
        for wb in ix.water_bridges:
            lig_atom_idx = wb.a_orig_idx if wb.protisdon else wb.d_orig_idx
            lig_atom_type = wb.atype if wb.protisdon else wb.dtype
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_WATERBRIDGE,
                    DETAIL_PROT_RESNAME: wb.restype,
                    DETAIL_PROT_RESNR: wb.resnr,
                    DETAIL_PROT_CHAIN: wb.reschain,
                    DETAIL_LIG_ATOM_IDX: lig_atom_idx,
                    DETAIL_LIG_ATOM_TYPE: lig_atom_type,
                    DETAIL_DISTANCE_AW: wb.distance_aw,
                    DETAIL_DISTANCE_DW: wb.distance_dw,
                    DETAIL_ANGLE: wb.w_angle,
                }
            )

        # Metal complexes
        for mc in ix.metal_complexes:
            records.append(
                {
                    **base,
                    DETAIL_INTERACTION_TYPE: INTERACTION_TYPE_METAL,
                    DETAIL_PROT_RESNAME: mc.restype,
                    DETAIL_PROT_RESNR: mc.resnr,
                    DETAIL_PROT_CHAIN: mc.reschain,
                    DETAIL_DISTANCE: mc.distance,
                    DETAIL_METAL_TYPE: mc.metal_type,
                    DETAIL_COORDINATION: mc.coordination_num,
                    DETAIL_METAL_GEOMETRY: mc.geometry,
                    DETAIL_TARGET_TYPE: mc.target_type,
                }
            )

        return records

    except Exception as e:
        logger.error(f"Detail extraction failed for {complex_pdb_path}: {type(e).__name__}: {e}")
        return []


def get_batch_detailed_interactions(
    complex_metadata: List[Tuple[str, str, str, str, str]],
    ligand_resname: str = LIGAND_RESNAME,
    ligand_chain: str = LIGAND_CHAIN,
    ligand_resseq: int = LIGAND_RESSEQ,
) -> pd.DataFrame:
    """
    Collect per-interaction PLIP detail records for a batch of complex PDBs.

    :param complex_metadata: List of (path, combination_id, protein_conf_id, smiles, docking_method)
    :param ligand_resname: Residue name of the ligand in each PDB (default "LIG")
    :param ligand_chain: Chain ID of the ligand in each PDB (default "Z")
    :param ligand_resseq: Residue sequence number of the ligand (default 1)
    :return: DataFrame with DETAIL_COLUMNS schema; empty DataFrame if nothing succeeds.
    """
    all_records = []
    for path, combination_id, protein_conf_id, smiles, docking_method in complex_metadata:
        records = get_detailed_interactions(
            path,
            combination_id,
            protein_conf_id,
            smiles,
            docking_method,
            ligand_resname=ligand_resname,
            ligand_chain=ligand_chain,
            ligand_resseq=ligand_resseq,
        )
        all_records.extend(records)

    if not all_records:
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    df = pd.DataFrame(all_records)
    for col in DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[DETAIL_COLUMNS]
