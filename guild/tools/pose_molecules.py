"""Read docked poses out of PDB/PDBQT/SDF text into RDKit molecules.

Shared by the pose analyses (PoseBusters, ProLIF, PLIP) and Vina rescoring.
Imports only constants so all of them can reach it; transformers/pdb.py would
be the natural home but imports guild.docking.vina.
"""

from guild.constants.pdb import PDB_RECORD_WIDTH


def is_atom_record(line: str) -> bool:
    return line.startswith(("ATOM  ", "HETATM"))


def residue_name(line: str) -> str:
    # Right-justified under three characters, so strip beats a pad-compare.
    return line[17:20].strip()


def split_complex_records(pdb_text: str, ligand_resname: str) -> tuple[list[str], list[str]]:
    """Partition a complex PDB's coordinate records into (ligand, protein).

    Keyed on residue name, not chain ID -- cif_to_pdb may rename Boltz's
    ligand chain.
    """
    ligand: list[str] = []
    protein: list[str] = []
    for line in pdb_text.splitlines(keepends=True):
        if is_atom_record(line) and len(line) > 20:
            target = ligand if residue_name(line) == ligand_resname else protein
            target.append(line)
    return ligand, protein


def split_pdbqt_models(pdbqt_path: str) -> list[str]:
    # Best-first: Vina and gnina write poses score-sorted.
    blocks: list[str] = []
    current: list[str] = []
    with open(pdbqt_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                current = []
            elif line.startswith("ENDMDL"):
                if current:
                    blocks.append("".join(current) + "END\n")
                current = []
            elif line.startswith(("ATOM", "HETATM")):
                current.append(line[:PDB_RECORD_WIDTH].rstrip("\n") + "\n")

    if not blocks and current:
        blocks.append("".join(current) + "END\n")
    return blocks


def mol_from_pdb_block(pdb_block: str, smiles: str) -> tuple[object | None, str | None, bool]:
    """Build a ligand from a PDB block, taking bond orders from ``smiles``.

    Complex PDBs carry no ligand CONECT records, so RDKit would otherwise infer
    bonds from the very geometry under suspicion.

    :return: ``(mol, reason, used_template_fallback)``; ``mol`` is None only
        when the block itself is unusable.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if not pdb_block.strip():
        return None, "empty ligand PDB block", False

    raw = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw is None:
        return None, "RDKit could not parse the ligand PDB block", False

    template = Chem.MolFromSmiles(smiles) if smiles else None
    if template is None:
        reason = f"unusable SMILES template: {smiles!r}"
    elif template.GetNumHeavyAtoms() != raw.GetNumHeavyAtoms():
        # AssignBondOrdersFromTemplate accepts a partial match silently and
        # leaves the rest geometry-inferred. Heavy atoms only, since
        # protonation may differ.
        reason = (
            f"SMILES template has {template.GetNumHeavyAtoms()} heavy atoms but the "
            f"pose has {raw.GetNumHeavyAtoms()} — template does not describe this ligand"
        )
    else:
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, raw)
            Chem.SanitizeMol(mol)
            return mol, None, False
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"

    try:
        Chem.SanitizeMol(
            raw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception as error:
        return None, f"sanitization failed after template fallback: {error}", False
    return raw, reason, True


def mols_from_sdf(sdf_path: str, smiles: str) -> list[tuple[object | None, str | None, bool]]:
    """Read every record of an SDF as a ligand, in file order.

    SDFs carry their own bond orders, so these skip the SMILES template. A
    record that will not sanitize is re-read unsanitized and rebuilt from it.
    """
    from rdkit import Chem

    mols: list[tuple[object | None, str | None, bool]] = []
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    fallback_records = None
    for index, mol in enumerate(supplier):
        if mol is not None:
            mols.append((mol, None, False))
            continue

        if fallback_records is None:
            fallback = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            # Unfiltered, so indices match the sanitized supplier's.
            fallback_records = list(fallback)

        record = fallback_records[index] if index < len(fallback_records) else None
        if record is None:
            mols.append((None, f"unreadable SDF record {index}", False))
            continue

        try:
            block = Chem.MolToPDBBlock(record)
        except Exception as error:
            mols.append((None, f"unreadable SDF record {index}: {error}", False))
            continue
        mols.append(mol_from_pdb_block(block, smiles))
    return mols
