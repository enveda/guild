import os
from pathlib import Path

os.environ.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")

import numpy as np
import pandas as pd
from posebusters import PoseBusters

from guild.constants.interactions import LIGAND_RESNAME
from guild.analysis.posebusters import _pose_mols
from guild.tools.pose_molecules import split_complex_records, mol_from_pdb_block

PROJECT_ROOT = Path("/workspace")
DATA_ROOT = PROJECT_ROOT / "data" / "small-example-all-methods"

TARGETS = [
    {
        "protein_config_id": "7v3z-A-9GF-A",
        "raw_pdb": PROJECT_ROOT / "data" / "7v3z.pdb",
        "raw_chain": "A",
        "resname": "9GF",
        "combination_id": "7v3z-A-9GF-A_9GF_native",
        "smiles": "CCCCCCC(C)(C)c1ccc([C@@H]2C[C@H](O)CC[C@H]2CCCO)c(O)c1",
        "batch_folder": str(DATA_ROOT / "batches" / "batch_83"),
    },
    {
        "protein_config_id": "6ot0-R-CO1-R",
        "raw_pdb": PROJECT_ROOT / "data" / "6ot0.pdb",
        "raw_chain": "R",
        "resname": "CO1",
        "combination_id": "6ot0-R-CO1-R_CO1_native",
        "smiles": "C[C@H](CC[C@@H]1OC1(C)C)[C@H]2CC[C@H]3[C@@H]4CC=C5C[C@@H](O)CC[C@]5(C)[C@H]4CC[C@]23C",
        "batch_folder": str(DATA_ROOT / "batches" / "batch_84"),
    },
    {
        "protein_config_id": "8gdc-R-P2E-R",
        "raw_pdb": PROJECT_ROOT / "data" / "8gdc.pdb",
        "raw_chain": "R",
        "resname": "P2E",
        "combination_id": "8gdc-R-P2E-R_P2E_native",
        "smiles": "CCCCC[C@H](O)/C=C/[C@H]1[C@H](O)CC(=O)[C@@H]1C\\C=C/CCCC(O)=O",
        "batch_folder": str(DATA_ROOT / "batches" / "batch_84"),
    },
]

METHODS = ["vina", "gnina", "karmadock", "diffdock", "boltz"]


def read_ca_coords(path, chain_filter=None):
    # Dedupe by residue number, keeping the first occurrence (guild's own prep
    # collapses alternate side-chain conformers to a single one, preferring
    # altloc 'A' -- confirmed against the run's own logged warning about
    # 8gdc R:163 HIS having an A/B altloc; the raw PDB still carries both).
    coords = []
    seen_resnum = set()
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                if chain_filter and line[21] != chain_filter:
                    continue
                resnum = line[22:27]  # includes insertion code
                if resnum in seen_resnum:
                    continue
                seen_resnum.add(resnum)
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(coords)


def kabsch_transform(mobile, target):
    """Rotation+translation that maps `mobile` (N,3) onto `target` (N,3), in
    residue-sequence order (robust to residue renumbering across guild's
    receptor-prep stages, since chain length/order is preserved even when the
    numbering isn't -- confirmed empirically: same atom count, sequential).
    """
    assert mobile.shape == target.shape, f"CA count mismatch: {mobile.shape} vs {target.shape}"
    mobile_c = mobile - mobile.mean(axis=0)
    target_c = target - target.mean(axis=0)
    H = mobile_c.T @ target_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = target.mean(axis=0) - R @ mobile.mean(axis=0)
    return R, t


def apply_transform_to_pdb_block(lines, R, t):
    out = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            xyz = R @ np.array([x, y, z]) + t
            newline = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
            out.append(newline)
        else:
            out.append(line)
    return out


def crystal_ligand_lines(raw_pdb_path, resname):
    lines = []
    with open(raw_pdb_path) as fh:
        for line in fh:
            if line.startswith("HETATM") and line[17:20].strip() == resname:
                lines.append(line)
    if not lines:
        raise ValueError(f"no HETATM lines for {resname} in {raw_pdb_path}")
    return lines


def complex_pdb_path(batch_folder, method, combination_id):
    return f"{batch_folder}/{method}/{combination_id}_complex.pdb"


def receptor_path_from_complex(complex_pdb, out_path, ligand_resname=LIGAND_RESNAME):
    text = Path(complex_pdb).read_text(encoding="utf-8", errors="replace")
    _, protein_lines = split_complex_records(text, ligand_resname)
    if not protein_lines:
        raise ValueError(f"no protein atoms left in {complex_pdb} after excluding {ligand_resname}")
    with open(out_path, "w") as fh:
        fh.writelines(protein_lines)
        fh.write("END\n")
    return out_path


def main():
    buster = PoseBusters(config="redock")
    rows = []
    printed_columns = False

    for target in TARGETS:
        raw_ca = read_ca_coords(target["raw_pdb"], chain_filter=target["raw_chain"])
        raw_ligand_lines = crystal_ligand_lines(target["raw_pdb"], target["resname"])

        for method in METHODS:
            combo_id = target["combination_id"]
            cpdb = complex_pdb_path(target["batch_folder"], method, combo_id)
            record = {"protein_config_id": target["protein_config_id"], "method": method}

            if not os.path.exists(cpdb):
                rows.append({**record, "status": "missing_complex_pdb", "path_checked": cpdb})
                continue

            complex_ca = read_ca_coords(cpdb)
            if complex_ca.shape != raw_ca.shape:
                rows.append({
                    **record, "status": "ca_count_mismatch",
                    "reason": f"raw has {raw_ca.shape[0]} CA, complex has {complex_ca.shape[0]}",
                })
                continue

            R, t = kabsch_transform(raw_ca, complex_ca)
            rmsd_fit = float(np.sqrt(np.mean(np.sum((raw_ca @ R.T + t - complex_ca) ** 2, axis=1))))
            transformed_ligand_lines = apply_transform_to_pdb_block(raw_ligand_lines, R, t)
            ligand_block = "".join(transformed_ligand_lines) + "END\n"

            mol_true, true_reason, true_fallback = mol_from_pdb_block(ligand_block, target["smiles"])
            if mol_true is None:
                rows.append({**record, "status": "crystal_ligand_build_failed", "reason": true_reason})
                continue

            poses = _pose_mols(
                complex_pdb_path=cpdb,
                batch_folder=target["batch_folder"],
                docking_method=method,
                combination_id=combo_id,
                smiles=target["smiles"],
                ligand_resname=LIGAND_RESNAME,
                max_poses=1,
            )
            if not poses or poses[0][0] is None:
                reason = poses[0][1] if poses else "no pose records found"
                rows.append({**record, "status": "no_usable_pose", "reason": reason})
                continue

            mol_pred, pose_reason, pred_fallback = poses[0]
            receptor_tmp = f"/tmp/{target['protein_config_id']}_{method}_receptor.pdb"
            try:
                receptor_path_from_complex(cpdb, receptor_tmp)
            except Exception as error:
                rows.append({**record, "status": "receptor_build_failed", "reason": str(error)})
                continue

            try:
                frame = buster.bust(
                    mol_pred=mol_pred, mol_true=mol_true, mol_cond=receptor_tmp, full_report=True
                )
            except Exception as error:
                rows.append({**record, "status": "posebusters_raised", "reason": str(error)})
                continue

            if frame is None or frame.empty:
                rows.append({**record, "status": "posebusters_empty_frame"})
                continue

            raw = frame.reset_index(drop=True).iloc[0].to_dict()
            if not printed_columns:
                print("PoseBusters redock columns:", sorted(raw.keys()))
                printed_columns = True

            rmsd = raw.get("rmsd")
            within_2A = raw.get("rmsd_≤_2å")

            rows.append({
                **record,
                "status": "ok",
                "kabsch_fit_rmsd_ca": rmsd_fit,  # sanity check: should be ~0
                "pred_template_fallback": pred_fallback,
                "true_template_fallback": true_fallback,
                "mol_true_loaded": raw.get("mol_true_loaded"),
                "mol_pred_loaded": raw.get("mol_pred_loaded"),
                "mol_cond_loaded": raw.get("mol_cond_loaded"),
                "inchi_crystal_valid": raw.get("inchi_crystal_valid"),
                "inchi_docked_valid": raw.get("inchi_docked_valid"),
                "rmsd": rmsd,
                "kabsch_rmsd": raw.get("kabsch_rmsd"),
                "within_2A": within_2A,
                "stereochemistry_preserved": raw.get("stereochemistry_preserved"),
            })

    df = pd.DataFrame(rows)
    out_path = DATA_ROOT / "native_ligand_rmsd.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()
