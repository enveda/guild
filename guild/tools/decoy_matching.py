"""
Property-matched decoy selection and a descriptor-only baseline (R2-7).

R2-7 asks for Figure 3 repeated on a decoy panel matched to each target's binders on
size/shape, plus a no-docking-score control (the unmatched pool isn't size-matched: a
heavy-atom-count baseline alone reaches AUC 0.865). A decoy is "matched" if it's within
tolerance of a binder on every descriptor in :data:`MATCH_DESCRIPTORS`, and not simply a
close analogue of one (ECFP4 Tanimoto below a threshold to every binder for that target).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from guild.tools.ligand_properties import assign_properties

MATCH_DESCRIPTORS = (
    "molecular_weight",
    "logp",
    "n_hba",
    "n_hbd",
    "n_rotatable_bonds",
    "net_charge",
)

# Absolute tolerance per descriptor; a decoy matches if all are satisfied at once.
# Not literature-mandated, just a reasonable default -- overridable per call.
DEFAULT_MATCH_TOLERANCES: dict[str, float] = {
    "molecular_weight": 50.0,
    "logp": 1.0,
    "n_hba": 1,
    "n_hbd": 1,
    "n_rotatable_bonds": 2,
    "net_charge": 0,
}

DEFAULT_TANIMOTO_THRESHOLD = 0.35
ECFP4_RADIUS = 2  # ECFP4 = Morgan fingerprint, radius 2 (diameter 4)
ECFP4_N_BITS = 2048


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
def assign_matching_properties(input_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptors needed to property-match a decoy to a known binder: weight, logP,
    HBA, HBD via :func:`assign_properties`, plus rotatable bonds and net charge here."""
    result = assign_properties(input_df)

    rotatable_bonds: list[float | None] = []
    net_charges: list[int | None] = []
    for smiles in input_df.iloc[:, 1]:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            rotatable_bonds.append(None)
            net_charges.append(None)
            continue
        rotatable_bonds.append(Descriptors.NumRotatableBonds(mol))
        net_charges.append(Chem.GetFormalCharge(mol))

    result["n_rotatable_bonds"] = rotatable_bonds
    result["net_charge"] = net_charges
    return result


def ecfp4_fingerprint(smiles: str):
    """ECFP4 fingerprint (Morgan, radius 2) for one molecule, or None if unparsed."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, ECFP4_RADIUS, nBits=ECFP4_N_BITS)


def max_tanimoto_similarity(query_fp, reference_fps: Sequence) -> float:
    """Highest ECFP4 Tanimoto similarity between one fingerprint and a set of
    others (None entries skipped); nan if either side is empty."""
    valid_refs = [fp for fp in reference_fps if fp is not None]
    if query_fp is None or not valid_refs:
        return float("nan")
    return max(DataStructs.TanimotoSimilarity(query_fp, ref) for ref in valid_refs)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _within_tolerance(binder_row, decoy_row, tolerances: dict[str, float]) -> bool:
    for descriptor, tolerance in tolerances.items():
        b, d = binder_row[descriptor], decoy_row[descriptor]
        if pd.isna(b) or pd.isna(d) or abs(b - d) > tolerance:
            return False
    return True


def match_decoys_to_binders(
    binders: pd.DataFrame,
    decoys: pd.DataFrame,
    target_col: str = "protein_config_id",
    id_col: str = "ligand_id",
    smiles_col: str = "smiles",
    tolerances: dict[str, float] | None = None,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
) -> pd.DataFrame:
    """
    Per-target property-matched, structurally-distinct decoy selection (R2-7).

    A decoy matches if it's within tolerance of at least one binder for its target on
    every descriptor, and is rejected regardless if its ECFP4 Tanimoto similarity to
    *any* binder for that target is >= tanimoto_threshold (so it isn't just a close
    analogue of the binder it matched). Returns the matched subset of decoys plus a
    matched_binder_id column.
    """
    tolerances = dict(DEFAULT_MATCH_TOLERANCES if tolerances is None else tolerances)

    binders = _ensure_descriptors(binders, id_col, smiles_col)
    decoys = _ensure_descriptors(decoys, id_col, smiles_col)

    kept_rows = []
    for target, target_binders in binders.groupby(target_col):
        target_decoys = decoys[decoys[target_col] == target]
        if target_decoys.empty or target_binders.empty:
            continue

        binder_fps = list(target_binders["_ecfp4"])
        for _, decoy_row in target_decoys.iterrows():
            if max_tanimoto_similarity(decoy_row["_ecfp4"], binder_fps) >= tanimoto_threshold:
                continue
            for _, binder_row in target_binders.iterrows():
                if _within_tolerance(binder_row, decoy_row, tolerances):
                    kept = decoy_row.copy()
                    kept["matched_binder_id"] = binder_row[id_col]
                    kept_rows.append(kept)
                    break

    if not kept_rows:
        return decoys.iloc[0:0].assign(matched_binder_id=pd.Series(dtype=object))
    return pd.DataFrame(kept_rows).drop(columns=["_ecfp4"]).reset_index(drop=True)


def _ensure_descriptors(frame: pd.DataFrame, id_col: str, smiles_col: str) -> pd.DataFrame:
    """Add MATCH_DESCRIPTORS + a fingerprint column if not already present."""
    frame = frame.copy()
    missing = [d for d in MATCH_DESCRIPTORS if d not in frame.columns]
    if missing:
        computed = assign_matching_properties(frame[[id_col, smiles_col]])
        for descriptor in missing:
            frame[descriptor] = computed[descriptor].to_numpy()
    if "_ecfp4" not in frame.columns:
        frame["_ecfp4"] = frame[smiles_col].map(ecfp4_fingerprint)
    return frame


# ---------------------------------------------------------------------------
# Descriptor-only baseline
# ---------------------------------------------------------------------------
def _binary_auc(active_values: np.ndarray, decoy_values: np.ndarray) -> float:
    """AUC (P(active ranks higher than decoy), ties handled by average rank)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.concatenate([np.ones(len(active_values)), np.zeros(len(decoy_values))])
    y_score = np.concatenate([active_values, decoy_values])
    mask = ~np.isnan(y_score)
    if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[mask], y_score[mask]))


def descriptor_only_auc(
    binders: pd.DataFrame,
    decoys: pd.DataFrame,
    id_col: str = "ligand_id",
    smiles_col: str = "smiles",
    descriptors: Sequence[str] = MATCH_DESCRIPTORS,
) -> pd.DataFrame:
    """
    Binder-vs-decoy AUC from physicochemical descriptors alone, no docking score --
    the R2-7 control for whether Figure 3's separation is binding signal or a
    property confound. One row per descriptor, reported standalone rather than only
    inside a combined model, so a single-descriptor confound stays visible.
    """
    binders = _ensure_descriptors(binders, id_col, smiles_col)
    decoys = _ensure_descriptors(decoys, id_col, smiles_col)

    rows = []
    for descriptor in descriptors:
        auc = _binary_auc(
            binders[descriptor].to_numpy(dtype=float),
            decoys[descriptor].to_numpy(dtype=float),
        )
        rows.append({"descriptor": descriptor, "auc": auc})
    return pd.DataFrame(rows)


def combined_descriptor_auc(
    binders: pd.DataFrame,
    decoys: pd.DataFrame,
    id_col: str = "ligand_id",
    smiles_col: str = "smiles",
    descriptors: Sequence[str] = MATCH_DESCRIPTORS,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Binder-vs-decoy AUC from all descriptors combined (logistic regression, no
    docking score), cross-validated so it isn't inflated by fitting and scoring on
    the same rows. nan if there are too few rows to fold or a fold is single-class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    binders = _ensure_descriptors(binders, id_col, smiles_col)
    decoys = _ensure_descriptors(decoys, id_col, smiles_col)

    frame = pd.concat(
        [
            binders[list(descriptors)].assign(_label=1),
            decoys[list(descriptors)].assign(_label=0),
        ],
        ignore_index=True,
    ).dropna()
    if frame["_label"].nunique() < 2 or len(frame) < 2 * n_splits:
        return float("nan")

    features = frame[list(descriptors)].to_numpy(dtype=float)
    labels = frame["_label"].to_numpy(dtype=int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_predict(
        LogisticRegression(), features, labels, cv=cv, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, scores))
