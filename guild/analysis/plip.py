"""
PLIP (Protein-Ligand Interaction Profiler) analysis tools
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from plip.structure.preparation import PDBComplex

from guild.constants.interactions import (
    COMPLEX_PDB_SUFFIX,
    INTERACTION_COMBINATION_ID,
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
    cols = [INTERACTION_COMBINATION_ID] + [col for col in df.columns if col != INTERACTION_COMBINATION_ID]
    df = df[cols]

    return df
