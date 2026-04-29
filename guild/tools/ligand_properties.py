"""
Ligand properties tools
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from tqdm import tqdm

RDLogger.DisableLog("rdApp.info")
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
np.random.seed(42)


def assign_properties(input_df):
    """
    Assign molecular weights to a list of smiles

    :param input_df: DataFrame with the molecule id and smiles as columns. The first column is the molecule id and the second column is the smiles.
    :return: DataFrame with the molecule id, smiles, and the assigned properties
    """
    results = {
        "id": [],
        "molecular_weight": [],
        "logp": [],
        "topological_surface_area_mapping": [],
        "fsp3": [],
        "scaffold_smiles": [],
        "n_hba": [],
        "n_hbd": [],
        "ro5_fulfilled": [],
        "max_ring_size": [],
    }

    for mol_id, smiles in tqdm(input_df.values, total=len(input_df), desc="Assigning properties"):
        try:
            current_molecule = Chem.MolFromSmiles(smiles)
        except Exception:
            current_molecule = None
            results["id"].append(mol_id)
            results["molecular_weight"].append(None)
            results["logp"].append(None)
            results["topological_surface_area_mapping"].append(None)
            results["fsp3"].append(None)
            results["scaffold_smiles"].append(None)
            results["n_hba"].append(None)
            results["n_hbd"].append(None)
            results["ro5_fulfilled"].append(None)
            continue
        try:
            results["molecular_weight"].append(MolWt(current_molecule))
        except Exception:
            results["molecular_weight"].append(None)
        try:
            results["logp"].append(Descriptors.MolLogP(current_molecule))
        except Exception:
            results["logp"].append(None)
        try:
            results["topological_surface_area_mapping"].append(
                Chem.QED.properties(current_molecule).PSA
            )
        except Exception:
            results["topological_surface_area_mapping"].append(None)
        try:
            results["fsp3"].append(Chem.Lipinski.FractionCSP3(current_molecule))
        except Exception:
            results["fsp3"].append(None)
        try:
            murcko_smiles = GetScaffoldForMol(current_molecule)
            results["scaffold_smiles"].append(Chem.MolToSmiles(murcko_smiles))
        except Exception:
            results["scaffold_smiles"].append(None)
        try:
            results["n_hba"].append(Descriptors.NumHAcceptors(current_molecule))
        except Exception:
            results["n_hba"].append(None)
        try:
            results["n_hbd"].append(Descriptors.NumHDonors(current_molecule))
        except Exception:
            results["n_hbd"].append(None)
        try:
            results["ro5_fulfilled"].append(
                assign_ro5_status(
                    results["molecular_weight"][-1],
                    results["n_hba"][-1],
                    results["n_hbd"][-1],
                    results["logp"][-1],
                )
            )
        except Exception:
            results["ro5_fulfilled"].append(None)
        try:
            results["max_ring_size"].append(get_largest_ring_size(current_molecule))
        except Exception:
            results["max_ring_size"].append(None)

        results["id"].append(mol_id)

    return pd.DataFrame(results)


def assign_ro5_status(molecular_weight, n_hba, n_hbd, logp):
    """
    Test if input molecule (molecular_weight, n_hba, n_hbd, logp) fulfills Lipinski's rule of five.

    :param molecular_weight: molecular weight of the molecule
    :param n_hba: number of hydrogen bond acceptors
    :param n_hbd: number of hydrogen bond donors
    :param logp: logP of the molecule
    :return: Whether the molecule fulfills Lipinski's rule of five.
    """
    # RDKit molecule from SMILES
    conditions = [molecular_weight <= 500, n_hba <= 10, n_hbd <= 5, logp <= 5]
    ro5_fulfilled = sum(conditions) >= 3
    return ro5_fulfilled


def get_largest_ring_size(input_molecule):
    ri = input_molecule.GetRingInfo()
    atom_rings = ri.AtomRings()

    max_ring_size = 0
    for ring in atom_rings:
        max_ring_size = max(max_ring_size, len(ring))
    return max_ring_size


def get_largest_ring_size_from_smiles(input_smiles):
    """
    Get the largest ring size from a SMILES string

    :param input_smiles: SMILES string
    :return: Largest ring size
    """
    try:
        input_molecule = Chem.MolFromSmiles(input_smiles)
        return get_largest_ring_size(input_molecule)
    except Exception:
        return None


def compound_filter(
    input_df,
    min_MW,
    max_MW,
    scaffold_col="scaffold_smiles",
    ro5_fulfilled_col="ro5_fulfilled",
    largest_ring_size_col="largest_ring_size",
    max_per_scaffold=5,
    seed=42,
    MW_column="MW",
    max_ring_size=10,
):
    """
    Filter compounds based on molecular weight and scaffold.

    :param input_df: DataFrame with the compounds
    :param min_MW: Minimum molecular weight
    :param max_MW: Maximum molecular weight
    :param scaffold_col: Column with the scaffold
    :param ro5_fulfilled_col: Column with the RO5 fulfilled
    :param largest_ring_size_col: Column with the largest ring size
    :param max_per_scaffold: Maximum number of compounds per scaffold
    :param seed: Random seed
    :param MW_column: Column with the molecular weight
    :param max_ring_size: Maximum ring size
    returns: DataFrame with the filtered compounds
    """

    # Set random seed
    np.random.seed(seed)

    # Create a copy of the dataframe
    input_df = input_df.copy()
    print("===Filtering by MW===")
    print(f"Filtering {len(input_df)} compounds")
    input_df = input_df[
        (input_df[MW_column] >= min_MW) & (input_df[MW_column] <= max_MW)
    ].reset_index(drop=True)
    print(f"After filtering, {len(input_df)} compounds")

    print("===Applying RO5 filter===")
    input_df[ro5_fulfilled_col] = input_df[ro5_fulfilled_col].astype(bool)
    input_df = input_df[input_df[ro5_fulfilled_col]]
    print(f"After filtering, {len(input_df)} compounds")

    print("===Applying max ring size filter===")
    input_df = input_df[input_df[largest_ring_size_col] <= max_ring_size]
    print(f"After filtering, {len(input_df)} compounds")

    print("===Limiting compounds per scaffold===")
    print(f"Max per scaffold: {max_per_scaffold}")
    scaffold_counts = input_df[scaffold_col].value_counts()
    print(f"Original scaffold counts: {scaffold_counts.head(5)}")
    # Group by scaffold
    scaffold_groups = input_df.groupby(scaffold_col)

    # Select compounds
    selected_compounds = []
    for _scaffold, group in tqdm(
        scaffold_groups,
        total=len(scaffold_groups),
        desc="Limiting compounds per scaffold",
    ):
        # If scaffold group is within limit, keep all compounds
        if len(group) <= max_per_scaffold:
            selected_compounds.append(group)
        else:
            # Randomly sample max_per_scaffold compounds
            sampled = group.sample(max_per_scaffold)
            selected_compounds.append(sampled)

    # Combine selected compounds
    result_df = pd.concat(selected_compounds).reset_index(drop=True)
    final_scaffold_counts = result_df[scaffold_col].value_counts()
    print(f"Final scaffold counts: {final_scaffold_counts.head(5)}")
    return result_df


def _prep_df(
    df,
    mw_column="molecular_weight",
    lipinski_column="ro5_fulfilled",
    mw_min=250,
    mw_max=450,
):
    """
    Basic preparation of a dataframe for molecular weight distribution matching.

    :param df: DataFrame with the compounds
    :param mw_column: Column with the molecular weight
    :param lipinski_column: Column with the Lipinski rule of five fulfilled
    :param mw_min: Minimum molecular weight
    :param mw_max: Maximum molecular weight
    :return: DataFrame with the prepared compounds
    """
    df = df.copy()
    df = df[df[lipinski_column] & (df[mw_column] >= mw_min) & (df[mw_column] <= mw_max)]

    # Drop rows with missing molecular weights
    df = df.dropna(subset=[mw_column])

    return df


def _create_bin_classes(df1, df2, mw_column="molecular_weight", n_bins=10):
    """
    Create bins for molecular weight distribution matching.

    :param df: DataFrame with the compounds
    :param mw_column: Column with the molecular weight
    :param n_bins: Number of bins
    :return: DataFrame with the bins
    """
    # Determine common min and max for binning
    min_mw = min(df1[mw_column].min(), df2[mw_column].min())
    max_mw = max(df1[mw_column].max(), df2[mw_column].max())

    # Create bins
    bins = np.linspace(min_mw, max_mw, n_bins + 1)

    # Add bin labels to each dataframe
    df1["mw_bin"] = pd.cut(df1[mw_column], bins=bins)
    df2["mw_bin"] = pd.cut(df2[mw_column], bins=bins)

    # Count samples in each bin for both dataframes
    df1_bin_counts = df1["mw_bin"].value_counts().sort_index()
    df2_bin_counts = df2["mw_bin"].value_counts().sort_index()

    return df1_bin_counts, df2_bin_counts


def _calculate_bin_proportions(df1_bin_counts, df2_bin_counts, common_bins, target_size):
    """Calculate the relative proportions for each bin (based on minimum counts).
    :param df1: DataFrame with the compounds
    :param df2: DataFrame with the compounds
    :param mw_column: Column with the molecular weight
    :param n_bins: Number of bins
    :return: Dictionary with the bin proportions
    """
    # Calculate the relative proportions for each bin (based on minimum counts)
    bin_proportions = {}
    total_min_count = 0

    for bin_label in common_bins:
        min_count = min(df1_bin_counts[bin_label], df2_bin_counts[bin_label])
        bin_proportions[bin_label] = min_count
        total_min_count += min_count

    # Calculate scaling factor to reach target_size
    scaling_factor = target_size / total_min_count

    # Calculate target counts for each bin
    target_counts = {}
    for bin_label, min_count in bin_proportions.items():
        # Initial target based on scaling
        target_count = int(min_count * scaling_factor)

        # Ensure we don't exceed what's available in either dataset
        # Take the minimum of target and available counts
        target_counts[bin_label] = min(
            target_count, df1_bin_counts[bin_label], df2_bin_counts[bin_label]
        )

    # We might not hit exactly target_size due to rounding, so adjust
    total_allocated = sum(target_counts.values())

    # If we need more samples, add them to bins with capacity
    remaining = target_size - total_allocated
    if remaining > 0:
        # Sort bins by available capacity (min of df1 and df2 minus current allocation)
        bin_capacity = {}
        for bin_label in target_counts:
            df1_capacity = df1_bin_counts[bin_label] - target_counts[bin_label]
            df2_capacity = df2_bin_counts[bin_label] - target_counts[bin_label]
            bin_capacity[bin_label] = min(df1_capacity, df2_capacity)

        # Sort bins by capacity
        sorted_bins = sorted(bin_capacity.items(), key=lambda x: x[1], reverse=True)

        # Distribute remaining samples to bins with capacity
        for bin_label, capacity in sorted_bins:
            if remaining <= 0:
                break
            add_count = min(capacity, remaining)
            target_counts[bin_label] += add_count
            remaining -= add_count

    # If we still haven't hit target_size, we'll need to reduce counts from some bins
    if remaining < 0:
        # Sort bins by count in descending order
        sorted_counts = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)

        # Reduce counts from bins with highest allocation
        for bin_label, count in sorted_counts:
            if remaining >= 0:
                break
            reduce_by = min(count - 1, -remaining)  # Ensure we don't remove all from a bin
            target_counts[bin_label] -= reduce_by
            remaining += reduce_by

    return target_counts


def _multiple_mw_distributions(
    df_1, df_2, mw_column="molecular_weight", n_bins=10, target_size=1000
):
    """
    Generate multiple datasets with equal molecular weight distributions within or between two dataframes.
    :param df_1: DataFrame with the compounds
    :param df_2: DataFrame with the compounds
    :param mw_column: Column with the molecular weight
    :param lipinski_column: Column with the Lipinski rule of five fulfilled
    :param mw_min: Minimum molecular weight
    :param mw_max: Maximum molecular weight
    :param n_bins: Number of bins
    :param target_size: Number of samples to generate for each dataset
    :return: List of DataFrames with equal MW distributions
    """
    # Determine common binning
    # Identify bins where both datasets have samples
    df1_bin_counts, df2_bin_counts = _create_bin_classes(df_1, df_2, mw_column, n_bins)
    common_bins = df1_bin_counts.index.intersection(df2_bin_counts.index)

    if len(common_bins) == 0:
        raise ValueError("No overlapping molecular weight bins between datasets")

    target_counts = _calculate_bin_proportions(
        df1_bin_counts, df2_bin_counts, common_bins, target_size
    )

    # Sample from each bin
    resampled_df1 = []
    resampled_df2 = []

    for bin_label, sample_count in target_counts.items():
        if sample_count < 0:
            continue

        # Get compounds in this bin
        df1_in_bin = df_1[df_1["mw_bin"] == bin_label]
        df2_in_bin = df_2[df_2["mw_bin"] == bin_label]

        # Sample the specified number
        df1_sample = df1_in_bin.sample(sample_count)
        df2_sample = df2_in_bin.sample(sample_count)

        # Add to our results
        resampled_df1.append(df1_sample)
        resampled_df2.append(df2_sample)

    # Combine all bins
    resampled_df1 = pd.concat(resampled_df1)
    resampled_df2 = pd.concat(resampled_df2)

    # Verify we have exactly target_size samples
    assert (
        len(resampled_df1) == target_size
    ), f"Expected {target_size} samples, got {len(resampled_df1)}"
    assert (
        len(resampled_df2) == target_size
    ), f"Expected {target_size} samples, got {len(resampled_df2)}"

    # Drop the bin column
    resampled_df1 = resampled_df1.drop(columns=["mw_bin"])
    resampled_df2 = resampled_df2.drop(columns=["mw_bin"])

    return resampled_df1, resampled_df2


def generate_equal_mw_distributions(
    df_1,
    df_2=None,
    mw_column="molecular_weight",
    lipinski_column="ro5_fulfilled",
    mw_min=250,
    mw_max=450,
    n_bins=10,
    target_size=1000,
):
    """
    Generate datasets with equal molecular weight distributions within or between two dataframes.

    :param df_1: DataFrame with the compounds
    :param df_2: DataFrame with the compounds (optional)
    :param mw_column: Column with the molecular weight
    :param lipinski_column: Column with the Lipinski rule of five fulfilled
    :param mw_min: Minimum molecular weight
    :param mw_max: Maximum molecular weight
    :param n_bins: Number of bins
    :param target_size: Number of samples to generate for each dataset
    :return: DataFrame(s) with equal MW distributions
    """

    # Create a copy of the input dataframes
    tmp_df1 = _prep_df(df_1, mw_column, lipinski_column, mw_min, mw_max)

    # Check if we have enough data
    if len(df_1) < target_size:
        raise ValueError(f"Not enough compounds in datasets to reach target_size of {target_size}")

    if df_2 is not None:
        tmp_df2 = _prep_df(df_2, mw_column, lipinski_column, mw_min, mw_max)

        # Check if we have enough data
        if len(df_2) < target_size:
            raise ValueError(
                f"Not enough compounds in datasets to reach target_size of {target_size}"
            )

        return _multiple_mw_distributions(
            df_1=tmp_df1,
            df_2=tmp_df2,
            mw_column=mw_column,
            n_bins=n_bins,
            target_size=target_size,
        )
    else:
        # Determine common min and max for binning
        min_mw = tmp_df1[mw_column].min()
        max_mw = tmp_df1[mw_column].max()

        # Create bins
        bins = np.linspace(min_mw, max_mw, n_bins + 1)

        # Add bin labels to each dataframe
        tmp_df1["mw_bin"] = pd.cut(tmp_df1[mw_column], bins=bins)
        tmp_df1 = tmp_df1.dropna(subset=["mw_bin"])

        # Sample from each bin
        resampled_decoys = []

        mol_per_bin = int(target_size / n_bins)

        for bin_label in tmp_df1["mw_bin"].unique():
            decoys_in_bin = tmp_df1[tmp_df1["mw_bin"] == bin_label]
            # Sample the specified number
            decoys_sample = decoys_in_bin.sample(mol_per_bin)

            # Add to our results
            resampled_decoys.append(decoys_sample)

        # Combine all bins
        resampled_decoys_df = pd.concat(resampled_decoys)

        # Verify we have exactly target_size samples
        assert (
            len(resampled_decoys_df) == target_size
        ), f"Expected {target_size} samples, got {len(resampled_decoys_df)}"

        # Drop the bin column
        resampled_decoys_df = resampled_decoys_df.drop(columns=["mw_bin"])

        return resampled_decoys_df


def _get_3d_mol_from_smiles(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    """
    Build a single reasonable 3D conformer for radius of gyration (ETKDG + MMFF/UFF).
    :param smiles: SMILES string
    :param seed: Random seed
    :return: RDKit Mol with 3D conformer
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)  # needed for sane 3D
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True

    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        # One more try with random coords (sometimes helps)
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            raise RuntimeError(f"RDKit failed to embed 3D conformer for SMILES: {smiles}")

    # Try MMFF, fall back to UFF
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    return mol


def radius_of_gyration_rdkit(
    mol: Chem.Mol,
    heavy_only: bool = True,
    mass_weighted: bool = False,
) -> float:
    """
    Computes radius of gyration from the current conformer on mol object.
    PAPER: https://link.springer.com/article/10.1186/s13321-015-0067-5

    Paper uses geometric center (unweighted) and heavy atoms (in many RMSD/contact metrics they use heavy atoms).
    If you want to match paper more closely, use: heavy_only=True, mass_weighted=False.
    :param mol: RDKit Mol with 3D conformer
    :param heavy_only: Whether to consider only heavy atoms
    :param mass_weighted: Whether to weight by atomic mass
    :return: Radius of gyration
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers. Embed or read a 3D structure first.")

    conf = mol.GetConformer()
    idxs = []
    masses = []
    for a in mol.GetAtoms():
        if heavy_only and a.GetAtomicNum() == 1:
            continue
        idxs.append(a.GetIdx())
        masses.append(a.GetMass())

    if not idxs:
        raise ValueError("No atoms selected for Rg (check heavy_only).")

    coords = np.array([list(conf.GetAtomPosition(i)) for i in idxs], dtype=float)

    if mass_weighted:
        w = np.array(masses, dtype=float)
        w /= w.sum()
        center = (coords * w[:, None]).sum(axis=0)
        radius_of_gyration2 = (w * ((coords - center) ** 2).sum(axis=1)).sum()
    else:
        center = coords.mean(axis=0)
        radius_of_gyration2 = ((coords - center) ** 2).sum(axis=1).mean()

    return float(np.sqrt(radius_of_gyration2))


def radius_of_gyration_from_smiles(
    smiles: str,
    heavy_only: bool = True,
    mass_weighted: bool = False,
) -> float:
    """
    Compute radius of gyration from SMILES string.
    Radius of gyration (radius of gyration, Rg) is a measure of the compactness of a molecule.:

    Given N atoms with Cartesian coordinates r_i = (x_i, y_i, z_i),
    the geometric center r_c is:

        r_c = (1/N) * Σ_i r_i

    The radius of gyration is defined as:

        Rg = sqrt( (1/N) * Σ_i || r_i - r_c ||^2 )

    where || · || denotes the Euclidean norm.

    Rg has units of length (e.g. Å) and represents the root-mean-square
    distance of atoms from the molecular centroid.

    :param smiles: SMILES string
    :param heavy_only: Whether to consider only heavy atoms
    :param mass_weighted: Whether to weight by atomic mass
    :return: Radius of gyration
    """
    mol = _get_3d_mol_from_smiles(smiles)
    return radius_of_gyration_rdkit(mol, heavy_only=heavy_only, mass_weighted=mass_weighted)


def vina_box_edge_from_radius_of_gyration(radius_of_gyration: float, ratio: float = 0.35) -> float:
    """
    box_edge = radius_of_gyration / ratio.
    Paper optimum ratio is 0.35. In the abstract it states 2.9, but that regards the entire search space, not the edge:
    if radius_of_gyration / box_edge = 0.35, then box_edge = radius_of_gyration / 0.35 = 2.857 * radius_of_gyration.
    :param radius_of_gyration: Radius of gyration
    """
    if radius_of_gyration <= 0:
        raise ValueError("Radius of gyration must be > 0")
    return float(radius_of_gyration / ratio)
