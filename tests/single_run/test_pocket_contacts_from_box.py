"""
Tests for ``guild.transformers.pdb.get_pocket_contacts_from_box``.
"""

from pathlib import Path

from guild.docking.vina import get_center_and_size_from_box_file
from guild.transformers.pdb import get_pocket_contacts_from_box

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
PDB = str(TEST_DATA_DIR / "3pbl.pdb")
BOX = str(TEST_DATA_DIR / "3pbl_box.txt")


def test_returns_residues_inside_box():
    center, size = get_center_and_size_from_box_file(BOX)
    contacts = get_pocket_contacts_from_box(
        protein_pdb=PDB,
        protein_chain="A",
        center=center,
        size=size,
    )
    assert contacts, "expected at least one residue Cα inside the box"
    # Schema: list of [chain_id, 1-based contiguous sequence index]
    for chain, idx in contacts:
        assert chain == "A"
        assert isinstance(idx, int) and idx >= 1


def test_empty_when_box_is_far_away():
    contacts = get_pocket_contacts_from_box(
        protein_pdb=PDB,
        protein_chain="A",
        center=(10000.0, 10000.0, 10000.0),
        size=(1.0, 1.0, 1.0),
    )
    assert contacts == []


def test_empty_when_chain_missing():
    contacts = get_pocket_contacts_from_box(
        protein_pdb=PDB,
        protein_chain="Z",
        center=(0.0, 0.0, 0.0),
        size=(20.0, 20.0, 20.0),
    )
    assert contacts == []


def test_multi_chain_indexes_each_chain_from_one():
    """A pocket spanning two chains must return contacts from both, with each
    chain's residue index restarting at 1 (Boltz per-chain indexing)."""
    # A large box centred on the structure should catch residues on both chains.
    center, _ = get_center_and_size_from_box_file(BOX)
    contacts = get_pocket_contacts_from_box(
        protein_pdb=PDB,
        protein_chain="A,B",
        center=center,
        size=(200.0, 200.0, 200.0),
    )
    chains = {c for c, _ in contacts}
    assert chains == {"A", "B"}, f"expected contacts on both chains, got {chains}"
    # Each chain is indexed independently from 1.
    for chain in ("A", "B"):
        idxs = [idx for c, idx in contacts if c == chain]
        assert min(idxs) == 1, f"chain {chain} should be 1-based, got min {min(idxs)}"


def test_multi_chain_is_union_of_single_chains():
    """contacts('A,B') == contacts('A') + contacts('B'), so the list form and
    the comma-string form are equivalent and chains compose cleanly."""
    center, _ = get_center_and_size_from_box_file(BOX)
    size = (200.0, 200.0, 200.0)
    only_a = get_pocket_contacts_from_box(PDB, "A", center, size)
    only_b = get_pocket_contacts_from_box(PDB, "B", center, size)
    both_str = get_pocket_contacts_from_box(PDB, "A,B", center, size)
    both_list = get_pocket_contacts_from_box(PDB, ["A", "B"], center, size)
    assert both_str == both_list
    assert both_str == only_a + only_b


def test_cuboid_axis_independence():
    """An asymmetric box must include residues outside the smaller axis if they
    sit along a larger one — cuboid AABB, not min-axis sphere."""
    center, _ = get_center_and_size_from_box_file(BOX)
    narrow_cubic = get_pocket_contacts_from_box(
        protein_pdb=PDB, protein_chain="A", center=center, size=(2.0, 2.0, 2.0)
    )
    elongated_x = get_pocket_contacts_from_box(
        protein_pdb=PDB, protein_chain="A", center=center, size=(40.0, 2.0, 2.0)
    )
    assert len(elongated_x) >= len(narrow_cubic), (
        "elongating the X axis must not shrink the contact set "
        f"(narrow={len(narrow_cubic)}, elongated={len(elongated_x)})"
    )
