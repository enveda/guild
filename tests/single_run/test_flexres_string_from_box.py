"""
Tests for ``guild.transformers.pdb.get_flexres_string_from_box``.

Reuses the same 3pbl PDB and box file as test_pocket_contacts_from_box.py.
No Docker or external tools required — pure Python + BioPython.
"""

from pathlib import Path

from guild.docking.vina import get_center_and_size_from_box_file
from guild.transformers.pdb import get_flexres_string_from_box

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
PDB = str(TEST_DATA_DIR / "3pbl.pdb")
BOX = str(TEST_DATA_DIR / "3pbl_box.txt")


def test_returns_string_for_box_with_residues():
    center, size = get_center_and_size_from_box_file(BOX)
    result = get_flexres_string_from_box(
        protein_pdb=PDB,
        protein_chain="A",
        center=center,
        size=size,
    )
    assert result is not None, "expected at least one residue Cα inside the box"
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_is_chain_colon_resnum():
    """Every token must be <chain>:<integer> joined by underscores."""
    center, size = get_center_and_size_from_box_file(BOX)
    result = get_flexres_string_from_box(PDB, "A", center, size)
    assert result is not None
    for token in result.split("_"):
        chain, resnum = token.split(":")
        assert chain == "A"
        assert resnum.isdigit() and int(resnum) >= 1


def test_returns_none_when_box_is_far_away():
    result = get_flexres_string_from_box(
        protein_pdb=PDB,
        protein_chain="A",
        center=(10000.0, 10000.0, 10000.0),
        size=(1.0, 1.0, 1.0),
    )
    assert result is None


def test_returns_none_when_chain_missing():
    center, size = get_center_and_size_from_box_file(BOX)
    result = get_flexres_string_from_box(
        protein_pdb=PDB,
        protein_chain="Z",
        center=center,
        size=size,
    )
    assert result is None


def test_large_box_covers_more_residues_than_small_box():
    center, _ = get_center_and_size_from_box_file(BOX)
    small = get_flexres_string_from_box(PDB, "A", center, (2.0, 2.0, 2.0))
    large = get_flexres_string_from_box(PDB, "A", center, (40.0, 40.0, 40.0))
    small_count = len(small.split("_")) if small else 0
    large_count = len(large.split("_")) if large else 0
    assert large_count >= small_count


def test_multi_chain_includes_both_chains():
    """A large box spanning both chains must produce tokens for A and B."""
    center, _ = get_center_and_size_from_box_file(BOX)
    result = get_flexres_string_from_box(PDB, "A,B", center, (200.0, 200.0, 200.0))
    assert result is not None
    chains_found = {token.split(":")[0] for token in result.split("_")}
    assert "A" in chains_found
    assert "B" in chains_found


def test_multi_chain_list_equals_comma_string():
    """list form and comma-string form must produce identical results."""
    center, size = get_center_and_size_from_box_file(BOX)
    via_str = get_flexres_string_from_box(PDB, "A,B", center, size)
    via_list = get_flexres_string_from_box(PDB, ["A", "B"], center, size)
    assert via_str == via_list


def test_consistent_residue_count_with_pocket_contacts_from_box():
    """get_flexres_string_from_box and get_pocket_contacts_from_box must identify
    the same *number* of residues for the same box.

    Note: the two functions use different residue numbering schemes intentionally.
    ``get_pocket_contacts_from_box`` uses a per-chain sequential counter (Boltz
    schema, gap-free starting at 1), while ``get_flexres_string_from_box`` reads
    ``residue.id[1]`` (the original PDB resSeq, which may have gaps). The counts
    must agree but the actual numbers may differ on PDBs with residue gaps.
    """
    from guild.transformers.pdb import get_pocket_contacts_from_box

    center, size = get_center_and_size_from_box_file(BOX)
    contacts = get_pocket_contacts_from_box(PDB, "A", center, size)
    flexres = get_flexres_string_from_box(PDB, "A", center, size)

    if not contacts:
        assert flexres is None
        return

    assert flexres is not None
    assert len(flexres.split("_")) == len(contacts)
