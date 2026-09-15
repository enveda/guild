"""Read docked poses out of PDB/PDBQT/SDF text into RDKit molecules.

Shared by the pose-consuming analyses (PoseBusters, ProLIF, PLIP) and by the
Vina rescoring path, which all need the same two primitives: split a complex
PDB into ligand and receptor records, and rebuild a ligand whose bond orders
come from SMILES rather than from geometry.

This module imports nothing from ``guild`` beyond constants, so every one of
those callers can use it. ``guild/transformers/pdb.py`` would otherwise be the
natural home, but it imports ``guild.docking.vina`` and so cannot be imported
back from there.
"""

from guild.constants.pdb import PDB_RECORD_WIDTH


def is_atom_record(line: str) -> bool:
    """True for ATOM/HETATM coordinate records."""
    return line.startswith(("ATOM  ", "HETATM"))


def residue_name(line: str) -> str:
    """Residue name from columns 18-20.

    Stripped rather than compared against a padded literal: the field is
    right-justified for names shorter than three characters, so a padded
    comparison misses them.
    """
    return line[17:20].strip()


def split_complex_records(pdb_text: str, ligand_resname: str) -> tuple[list[str], list[str]]:
    """Partition a complex PDB's coordinate records into (ligand, protein).

    Residue name is the reliable ligand marker, not chain ID — ``cif_to_pdb``
    may rename Boltz's ligand chain.
    """
    ligand: list[str] = []
    protein: list[str] = []
    for line in pdb_text.splitlines(keepends=True):
        if is_atom_record(line) and len(line) > 20:
            target = ligand if residue_name(line) == ligand_resname else protein
            target.append(line)
    return ligand, protein


def split_pdbqt_models(pdbqt_path: str) -> list[str]:
    """Split a multi-model Vina/gnina PDBQT into per-pose PDB text blocks.

    Vina and gnina write poses score-sorted, so the result is best-first. A
    file with no MODEL records yields one block.

    Distinct from ``guild.transformers.pdb._convert_pdbqt_to_pdb``, which keeps
    only MODEL 1: analyses that judge every pose need them all.
    """
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
    """Build an RDKit ligand from a PDB block, taking bond orders from ``smiles``.

    Complex PDBs carry no ligand CONECT records, so RDKit would otherwise infer
    bonds from 3D distance — and geometry is exactly what pose validation puts
    under suspicion.

    :return: ``(mol, reason, used_template_fallback)``. ``mol`` is None only
        when the block itself is unusable. ``used_template_fallback`` is True
        when the SMILES did not match and bonds came from geometry instead.
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
        # AssignBondOrdersFromTemplate does not require the template to describe
        # the whole molecule: given a partial match it succeeds silently and
        # leaves the unmatched bonds geometry-inferred, which is
        # indistinguishable from success at the call site. Heavy atoms only —
        # protonation differences are expected, a different skeleton is not.
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

    # Geometry-perceived fallback: intermolecular checks stay valid, the
    # intramolecular ones are judged against inferred bonds. Callers promote
    # this to its own status.
    try:
        Chem.SanitizeMol(
            raw, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception as error:
        return None, f"sanitization failed after template fallback: {error}", False
    return raw, reason, True


def mols_from_sdf(sdf_path: str, smiles: str) -> list[tuple[object | None, str | None, bool]]:
    """Read every record of an SDF as a ligand, preserving file order.

    An SDF already carries bond orders, so these poses skip the SMILES template
    and are immune to template-mismatch weakening. A record that will not
    sanitize (e.g. gnina covalent output, with unusual valences) is re-read
    unsanitized and rebuilt through the template.
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
            # list(), not a filtered comprehension: indices must stay aligned
            # with the sanitized supplier's, or a record that is unreadable
            # even unsanitized shifts every later lookup onto the wrong pose.
            fallback_records = list(fallback)

        if index >= len(fallback_records):
            mols.append((None, f"unreadable SDF record {index}", False))
            continue

        record = fallback_records[index]
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
