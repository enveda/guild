"""
Regression test for altloc handling in ``guild.tools.preparation.clean_receptor``.

A receptor residue resolved with alternate side-chain conformers (altloc
A/B) must be collapsed to a single conformer per atom name. Left
unresolved, downstream flexres tooling (gnina's ``--flexres``) hard-fails
with "Multiple copies of residue ... I can't handle this situation." for
any receptor+flexres combination that includes one of the altloc'd
residues.
"""

from pathlib import Path

from guild.tools.preparation import clean_receptor, detect_altloc_residues

ALTLOC_PDB = """\
ATOM      1  N   MET A  65      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  MET A  65      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  CB  MET A  65      12.000  10.000  10.000  1.00 20.00           C
ATOM      4  CG AMET A  65      13.000  10.000  10.000  0.60 20.00           C
ATOM      5  CG BMET A  65      13.500  10.500  10.500  0.40 20.00           C
ATOM      6  SD AMET A  65      14.000  10.000  10.000  0.60 20.00           S
ATOM      7  SD BMET A  65      14.500  10.500  10.500  0.40 20.00           S
ATOM      8  CE AMET A  65      15.000  10.000  10.000  0.60 20.00           C
ATOM      9  CE BMET A  65      15.500  10.500  10.500  0.40 20.00           C
END
"""


def test_altloc_residue_collapses_to_one_conformer_per_atom(tmp_path, caplog):
    input_pdb = tmp_path / "altloc.pdb"
    output_pdb = tmp_path / "altloc_clean.pdb"
    input_pdb.write_text(ALTLOC_PDB)

    clean_receptor(str(input_pdb), str(output_pdb), keep_metals=True)

    lines = [line for line in Path(output_pdb).read_text().splitlines() if line.startswith("ATOM")]
    atom_names = [line[12:16].strip() for line in lines]

    # Exactly one atom per name -- no duplicate CG/SD/CE from the discarded
    # altloc conformer.
    assert sorted(atom_names) == ["CA", "CB", "CE", "CG", "N", "SD"]

    # The kept conformer is the requested altloc (occupancy 0.60, altloc A
    # coordinates), and the altloc code itself is cleared since only one
    # conformer remains.
    cg_line = next(line for line in lines if line[12:16].strip() == "CG")
    assert cg_line[16] == " "
    assert float(cg_line[30:38]) == 13.000


def test_altloc_residue_is_reported(tmp_path, caplog):
    input_pdb = tmp_path / "altloc.pdb"
    output_pdb = tmp_path / "altloc_clean.pdb"
    input_pdb.write_text(ALTLOC_PDB)

    with caplog.at_level("WARNING"):
        clean_receptor(str(input_pdb), str(output_pdb), keep_metals=True)

    assert any("alternate side-chain conformers" in r.message for r in caplog.records)
    assert any("A:65MET" in r.message for r in caplog.records)


def test_no_warning_when_no_altloc(tmp_path, caplog):
    plain_pdb = """\
ATOM      1  N   MET A  65      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  MET A  65      11.000  10.000  10.000  1.00 20.00           C
END
"""
    input_pdb = tmp_path / "plain.pdb"
    output_pdb = tmp_path / "plain_clean.pdb"
    input_pdb.write_text(plain_pdb)

    with caplog.at_level("WARNING"):
        clean_receptor(str(input_pdb), str(output_pdb), keep_metals=True)

    assert not any("alternate side-chain conformers" in r.message for r in caplog.records)


def test_detect_altloc_residues_reports_affected_residue(tmp_path):
    input_pdb = tmp_path / "altloc.pdb"
    input_pdb.write_text(ALTLOC_PDB)

    findings = detect_altloc_residues(str(input_pdb))

    assert len(findings) == 1
    assert "A:65 MET" in findings[0]
    assert "A/B" in findings[0]


def test_detect_altloc_residues_empty_when_no_altloc(tmp_path):
    plain_pdb = """\
ATOM      1  N   MET A  65      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  MET A  65      11.000  10.000  10.000  1.00 20.00           C
END
"""
    input_pdb = tmp_path / "plain.pdb"
    input_pdb.write_text(plain_pdb)

    assert detect_altloc_residues(str(input_pdb)) == []
