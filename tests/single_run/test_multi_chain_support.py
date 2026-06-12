"""
Tests for multi-chain receptor support: chain isolation keeps every requested
chain, residue renumbering restarts per chain, and the Boltz YAML emits one
``protein`` block per chain. See README "Multi-chain binding pocket".
"""

from pathlib import Path

import yaml
from Bio.PDB import PDBParser

from guild.docking.boltz import generate_boltz_yaml
from guild.tools.preparation import _normalize_chain_list, isolate_protein_chain

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
PDB = str(TEST_DATA_DIR / "3pbl.pdb")  # has chains A and B


def _chains_and_resnums(pdb_path):
    model = PDBParser(QUIET=True).get_structure("x", pdb_path)[0]
    return {ch.id: [res.id[1] for res in ch] for ch in model}


def test_normalize_chain_list_forms():
    assert _normalize_chain_list("A") == ["A"]
    assert _normalize_chain_list("A,B") == ["A", "B"]
    assert _normalize_chain_list("A, B ") == ["A", "B"]
    assert _normalize_chain_list(["A", "B"]) == ["A", "B"]
    assert _normalize_chain_list(None) == []


def test_isolate_keeps_all_requested_chains(tmp_path):
    out = tmp_path / "multi.pdb"
    isolate_protein_chain(PDB, "3pbl", str(out), target_chain="A,B")
    chains = _chains_and_resnums(str(out))
    assert set(chains) == {"A", "B"}, f"expected both chains, got {set(chains)}"


def test_renumber_restarts_per_chain(tmp_path):
    """Each isolated chain must be 1-based independently — required so Boltz
    pocket contacts ([chain, index]) line up with each chain's sequence."""
    out = tmp_path / "multi.pdb"
    isolate_protein_chain(PDB, "3pbl", str(out), target_chain="A,B")
    chains = _chains_and_resnums(str(out))
    for chain_id, resnums in chains.items():
        assert resnums[0] == 1, f"chain {chain_id} should start at residue 1"
        # Residues are contiguous within the chain.
        assert resnums == list(range(1, len(resnums) + 1))


def test_single_chain_still_works(tmp_path):
    out = tmp_path / "single.pdb"
    isolate_protein_chain(PDB, "3pbl", str(out), target_chain="A")
    chains = _chains_and_resnums(str(out))
    assert set(chains) == {"A"}


def test_boltz_yaml_single_chain_unchanged(tmp_path):
    out = tmp_path / "single.yaml"
    generate_boltz_yaml(
        protein_sequence="MKT",
        protein_chain="A",
        ligand_sequences=["CCO"],
        ligand_ids=["L"],
        output_file=str(out),
        template_file=None,
    )
    doc = yaml.safe_load(out.read_text())
    proteins = [s for s in doc["sequences"] if "protein" in s]
    assert len(proteins) == 1
    assert proteins[0]["protein"]["id"] == "A"
    assert proteins[0]["protein"]["sequence"] == "MKT"


def test_boltz_yaml_multi_chain(tmp_path):
    out = tmp_path / "multi.yaml"
    generate_boltz_yaml(
        protein_sequence=["MKT", "GGS"],
        protein_chain=["A", "B"],
        ligand_sequences=["CCO"],
        ligand_ids=["L"],
        output_file=str(out),
        template_file=PDB,
        msa_file=["a.a3m", "b.a3m"],
        pocket_contacts=[["A", 1], ["B", 2]],
    )
    doc = yaml.safe_load(out.read_text())
    proteins = [s["protein"] for s in doc["sequences"] if "protein" in s]
    assert [p["id"] for p in proteins] == ["A", "B"]
    assert [p["sequence"] for p in proteins] == ["MKT", "GGS"]
    assert [p["msa"] for p in proteins] == ["a.a3m", "b.a3m"]
    # Template constrains both chains.
    template = doc["templates"][0]
    assert template["chain_id"] == ["A", "B"]
    assert template["template_id"] == ["A1", "B1"]
    # Pocket constraint spans both chains.
    pocket = doc["constraints"][0]["pocket"]
    assert pocket["contacts"] == [["A", 1], ["B", 2]]


def test_boltz_yaml_chain_sequence_mismatch_raises(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        generate_boltz_yaml(
            protein_sequence=["MKT"],
            protein_chain=["A", "B"],
            ligand_sequences=["CCO"],
            ligand_ids=["L"],
            output_file=str(tmp_path / "bad.yaml"),
        )
