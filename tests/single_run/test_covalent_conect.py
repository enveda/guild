"""
Tests for ``guild.transformers.pdb.add_covalent_conect`` and the residue-class
filtering in ``covalent_rec_atom_exists``.

Both use small synthetic PDBs rather than the 3pbl fixture, because the point is
to reproduce collisions that a well-behaved structure does not contain: a
non-ligand HETATM closer to the reactive residue than the warhead, and a water
renumbered onto a protein residue's number.
"""

from guild.transformers.pdb import add_covalent_conect, covalent_rec_atom_exists


def _atom(serial, name, resname, chain, resseq, xyz, record="ATOM  ", element=None):
    """Render one fixed-column PDB record."""
    x, y, z = xyz
    element = element or name[0]
    return (
        f"{record}{serial:>5} {name:<4}{'':1}{resname:>3} {chain}{resseq:>4}{'':4}"
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{0.00:>6.2f}{'':10}{element:>2}\n"
    )


def _write(tmp_path, records, name="complex.pdb"):
    path = tmp_path / name
    path.write_text("".join(records) + "END\n")
    return str(path)


def _conect_partners(path):
    """Map each CONECT record's first serial to the serials it bonds to."""
    partners = {}
    for line in open(path):
        if line.startswith("CONECT"):
            a, b = int(line[6:11]), int(line[11:16])
            partners.setdefault(a, set()).add(b)
    return partners


def test_prefers_ligand_chain_over_closer_receptor_hetatm(tmp_path):
    """
    A cofactor HETATM sitting closer to the reactive atom than the warhead must
    not win. build_complex_pdb copies receptor HETATM records through verbatim,
    so this is the realistic failure: metals and ordered waters coordinate
    reactive residues at bonding-like distances.
    """
    records = [
        _atom(1, "SG", "CYS", "A", 145, (0.0, 0.0, 0.0)),
        # Zn at 1.0 A -- nearer than the warhead, and not the ligand.
        _atom(2, "ZN", "ZN", "A", 300, (1.0, 0.0, 0.0), record="HETATM", element="ZN"),
        # Warhead on the ligand chain at 1.8 A, a real covalent bond length.
        _atom(3, "C1", "LIG", "Z", 1, (1.8, 0.0, 0.0), record="HETATM"),
    ]
    path = _write(tmp_path, records)

    add_covalent_conect(path, "A:145:SG")

    partners = _conect_partners(path)
    assert partners[1] == {3}, "CONECT should reach the ligand-chain warhead, not the Zn"
    assert partners[3] == {1}


def test_no_conect_written_when_ligand_chain_is_out_of_range(tmp_path):
    """
    A nearby non-ligand HETATM must not mask the absence of a bonded warhead --
    the function should decline to write rather than invent a bond.
    """
    records = [
        _atom(1, "SG", "CYS", "A", 145, (0.0, 0.0, 0.0)),
        _atom(2, "ZN", "ZN", "A", 300, (1.0, 0.0, 0.0), record="HETATM", element="ZN"),
        # Ligand present but far away: not covalently bound.
        _atom(3, "C1", "LIG", "Z", 1, (9.0, 0.0, 0.0), record="HETATM"),
    ]
    path = _write(tmp_path, records)

    add_covalent_conect(path, "A:145:SG")

    assert _conect_partners(path) == {}


def test_custom_ligand_chain_is_honoured(tmp_path):
    records = [
        _atom(1, "SG", "CYS", "A", 145, (0.0, 0.0, 0.0)),
        _atom(2, "C1", "LIG", "L", 1, (1.8, 0.0, 0.0), record="HETATM"),
    ]
    path = _write(tmp_path, records)

    add_covalent_conect(path, "A:145:SG", ligand_chain="L")

    assert _conect_partners(path)[1] == {2}


def test_rec_atom_exists_ignores_water_sharing_a_residue_number(tmp_path):
    """
    renumber_pdb_residues renumbers every residue in a chain, so a water can
    carry a protein residue's number. The water is written first here so it is
    reached first during iteration.
    """
    records = [
        _atom(1, "O", "HOH", "A", 145, (5.0, 5.0, 5.0), record="HETATM"),
        _atom(2, "SG", "CYS", "A", 145, (0.0, 0.0, 0.0)),
    ]
    path = _write(tmp_path, records, name="receptor.pdb")

    assert covalent_rec_atom_exists(path, "A:145:SG") is True
    # The water's own atom must not validate as a receptor attachment point.
    assert covalent_rec_atom_exists(path, "A:145:O") is False
