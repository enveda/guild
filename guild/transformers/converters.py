"""
Convert between different file formats.
- SDF to PDB
- SDF to PDBQT
- PDB to PDBQT
- PDB to SDF
"""

import os
import shutil
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from guild.constants.ligands import OBABEL_CONVERSION_TIMEOUT
from guild.constants.system import SHELL_SILENCER

# warnings.filterwarnings("ignore", category=DeprecationWarning)


def smiles_to_sdf_karmadock(smiles: str, sdf_path: str):
    """
    Generate a KarmaDock-compatible SDF using OpenBabel.
    OpenBabel's --gen3d with minimization produces clean, compatible output.

    :param smiles: SMILES string
    :param sdf_path: Output SDF file path
    :return: None
    """
    # Use OpenBabel for robust SMILES→SDF conversion
    # --gen3d: generate 3D coordinates
    # -h: add hydrogens
    # NOTE: Removed --minimize to prevent hangs on complex molecules
    obabel_cmd = f'obabel -:"{smiles}" -O "{sdf_path}" --gen3d -h {SHELL_SILENCER}'

    try:
        _ = subprocess.run(
            obabel_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=OBABEL_CONVERSION_TIMEOUT,
        )

        if not os.path.exists(sdf_path):
            raise RuntimeError(f"OpenBabel failed to create SDF file: {sdf_path}")

        with open(sdf_path, "r") as f:
            content = f.read()

        if not content or len(content) < 50:
            raise ValueError(f"Generated SDF file appears empty or malformed: {sdf_path}")

    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"OpenBabel SMILES conversion timed out for: {smiles}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel SMILES conversion failed for {smiles}: {e.stderr}") from e


def sdf_to_pdb(sdf: str, pdb: str):
    """
    Convert SDF to PDB using OpenBabel (simpler and more robust than RDKit).

    :param sdf: SDF file to be converted
    :param pdb: PDB file to be written
    :return: None
    """
    if not os.path.exists(sdf):
        raise FileNotFoundError(f"SDF file not found: {sdf}")

    obabel_cmd = f'obabel "{sdf}" -O "{pdb}" {SHELL_SILENCER}'

    try:
        _ = subprocess.run(
            obabel_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=OBABEL_CONVERSION_TIMEOUT,
        )

        if not os.path.exists(pdb):
            raise RuntimeError(f"OpenBabel failed to create PDB file: {pdb}")

    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"OpenBabel SDF→PDB conversion timed out for: {sdf}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel SDF→PDB conversion failed for {sdf}: {e.stderr}") from e


def cif_to_pdb(cif: str, pdb: str):
    """
    Convert mmCIF to PDB using OpenBabel.
    Used to convert Boltz structure output (CIF) for downstream PLIP analysis.

    :param cif: Path to CIF file
    :param pdb: Path to output PDB file
    """
    if not os.path.exists(cif):
        raise FileNotFoundError(f"CIF file not found: {cif}")

    obabel_cmd = f'obabel "{cif}" -O "{pdb}" {SHELL_SILENCER}'

    try:
        subprocess.run(
            obabel_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=OBABEL_CONVERSION_TIMEOUT,
        )
        if not os.path.exists(pdb):
            raise RuntimeError(f"OpenBabel failed to create PDB file: {pdb}")
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"OpenBabel CIF→PDB conversion timed out for: {cif}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel CIF→PDB conversion failed for {cif}: {e.stderr}") from e


def align_sdf(template_mol: str, align_mol: str, output_name: str = None):
    """
    Aligns two molecules using RDKit, and writes the aligned molecule to a SDF file.

    :param template_mol: SDF file path of the template molecule
    :param align_mol: SDF file path to the molecule to be aligned
    :param output_name: name of the output file, if None, no file is written
    return: aligned molecule
    """
    mol1 = Chem.SDMolSupplier(template_mol)[0]
    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.SDMolSupplier(align_mol)[0]
    mol2 = Chem.AddHs(mol2)
    aligned_object = rdMolAlign.GetO3A(mol2, mol1)
    aligned_object.Align()

    output_pdb_name = output_name + ".pdb"
    output_sdf_name = output_name + ".sdf"

    if output_name is not None:
        Chem.MolToPDBFile(mol2, output_pdb_name)
        with Chem.SDWriter(output_sdf_name) as writer:
            writer.write(mol2)

    return mol2


def _sdf_has_degenerate_coords(sdf: str) -> bool:
    """
    Detect a V2000 SDF whose atoms all sit at the same (usually 0,0,0) point.

    OpenBabel's ``--gen3d`` can fail its distance-geometry embedding for a
    molecule with over-constrained stereochemistry, print an error to
    stderr, and *still* exit 0 while writing a placeholder molecule with
    every atom at the origin. That passes a plain file-size check, so we
    check the actual coordinates.
    """
    with open(sdf, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if len(lines) < 4:
        return True
    try:
        n_atoms = int(lines[3][:3])
    except (ValueError, IndexError):
        return False
    coords = set()
    for line in lines[4 : 4 + n_atoms]:
        coords.add(line[:30])  # x, y, z columns (10 chars each)
        if len(coords) > 1:
            return False
    return True


def smiles_to_sdf(smiles: str, sdf: str = None):
    """
    Convert SMILES to 3D SDF using OpenBabel, falling back to RDKit ETKDG
    if OpenBabel's distance-geometry embedding fails.

    OpenBabel handles: parsing, 3D generation, H addition, and minimization.

    KNOWN LIMITATION (flagged, not fixed here): the embedded conformer is placed in an
    arbitrary local coordinate frame with NO reference to any receptor/box -- this
    function has no receptor context to position against. This silently breaks gnina's
    sdf-mode automatic flexible-residue selection (``Guild(flexible_docking=True,
    gnina_input_mode="sdf")`` -> ``--flexdist_ligand``/``--flexdist`` in
    guild/run.py and guild/docking/gnina.py), which measures "residues within N Å of the
    ligand" using THIS function's output coordinates. For any ligand built from a bare
    SMILES (i.e. not a supplied/aligned pose), those coordinates are nowhere near the
    real binding site, so --flexdist_ligand locks onto whatever happens to be near the
    ligand's arbitrary embedding origin -- confirmed in practice to pick residues on an
    unrelated chain and then fail to find any valid pose at all ("ligand outside box" on
    effectively every sample), not a real docking failure. Rigid docking and the
    box-based pdbqt-mode flex derivation are unaffected (they don't use the ligand's
    absolute position). Workaround: use the explicit ``flexres_gnina``/``gnina_flexres``
    override (bypasses --flexdist_ligand entirely) instead of automatic sdf-mode
    flexible_docking for any ligand that isn't pose-supplied.

    :param smiles: SMILES string of the molecule to be converted
    :param sdf: Name of the output file
    :return: None
    """
    if sdf is None:
        sdf = "output.sdf"

    # Use OpenBabel for robust SMILES→SDF conversion with 3D generation
    # --gen3d: generate 3D coordinates
    # -h: add hydrogens
    # NOTE: Removed --minimize to prevent hangs on complex molecules
    obabel_cmd = f'obabel -:"{smiles}" -O "{sdf}" --gen3d -h {SHELL_SILENCER}'

    try:
        _ = subprocess.run(
            obabel_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=OBABEL_CONVERSION_TIMEOUT,
        )

        # Validate output file was created and has content
        if not os.path.exists(sdf):
            raise RuntimeError(f"OpenBabel failed to create SDF file: {sdf}")

        with open(sdf, "r") as f:
            content = f.read()

        if not content or len(content) < 50:  # SDF files should have substantial content
            raise ValueError(f"Generated SDF file appears empty or malformed: {sdf}")

        if _sdf_has_degenerate_coords(sdf):
            # OpenBabel's DG embedder can fail on over-constrained
            # stereochemistry (conflicting/unembeddable defined stereocenters
            # + double bonds) yet still exit 0 with a placeholder molecule at
            # the origin. RDKit's ETKDG embedder handles these cases.
            molecule = _molecule_rdkit_processing(Chem.MolFromSmiles(smiles))
            with Chem.SDWriter(sdf) as writer:
                writer.write(molecule)

    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"OpenBabel SMILES conversion timed out for: {smiles}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel SMILES conversion failed for {smiles}: {e.stderr}") from e


def ligand_pdb_to_pdbqt(pdb: str, timeout: int = 300):
    """
    Convert PDB to PDBQT using OpenBabel (adds Gasteiger charges automatically).

    :param pdb: PDB file to be converted
    :param timeout: Maximum time in seconds for conversion (default 5 minutes)
    """
    # Validate PDB file exists and has content
    if not os.path.exists(pdb):
        raise FileNotFoundError(f"PDB file not found: {pdb}")

    with open(pdb, "r") as f:
        pdb_lines = f.readlines()

    atom_count = sum(
        1 for line in pdb_lines if line.startswith("ATOM") or line.startswith("HETATM")
    )
    if atom_count == 0:
        raise ValueError(f"PDB file has no ATOM records: {pdb}")

    output_pdbqt = pdb.replace(".pdb", ".pdbqt")
    # OpenBabel automatically adds Gasteiger charges for PDBQT format
    openbabel_string = f'obabel "{pdb}" -O "{output_pdbqt}" {SHELL_SILENCER}'

    try:
        _ = subprocess.run(
            openbabel_string,
            shell=True,
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        # Validate output PDBQT was created and has content
        if not os.path.exists(output_pdbqt):
            raise RuntimeError(f"OpenBabel failed to create output file: {output_pdbqt}")

        with open(output_pdbqt, "r") as f:
            pdbqt_lines = f.readlines()

        pdbqt_atoms = sum(
            1 for line in pdbqt_lines if line.startswith("ATOM") or line.startswith("HETATM")
        )
        if pdbqt_atoms == 0:
            raise ValueError(f"Generated PDBQT has no ATOM records: {output_pdbqt}")

        # OpenBabel sometimes writes non-ligand records such as COMPND/AUTHOR
        # before ROOT, which Vina's PDBQT parser rejects with "Unknown or
        # inappropriate tag". Strip everything except the tags Vina accepts in a
        # ligand PDBQT, including REMARK.
        # Match on the leading whitespace-separated token — ``line[:6]`` would
        # truncate the 7/9-char tags ENDROOT/ENDBRANCH/TORSDOF, silently
        # dropping the very lines that close the torsion tree. Vina tolerates
        # the truncated result, but gnina rejects it as malformed.
        _VINA_LIGAND_TAGS = {
            "REMARK",
            "ROOT",
            "ENDROOT",
            "BRANCH",
            "ENDBRANCH",
            "TORSDOF",
            "ATOM",
            "HETATM",
            "END",
        }
        filtered = [
            line
            for line in pdbqt_lines
            if line.strip() and line.split(maxsplit=1)[0] in _VINA_LIGAND_TAGS
        ]
        with open(output_pdbqt, "w") as f:
            f.writelines(filtered)

    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"OpenBabel conversion timed out after {timeout} seconds for {pdb}"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel conversion failed for {pdb}: {e.stderr}") from e


def sdf_to_pdbqt(sdf: str, pdbqt: str = None, timeout: int = OBABEL_CONVERSION_TIMEOUT):
    """
    Convert an SDF file (with existing 3D coordinates) directly to PDBQT.
    This preserves the 3D pose — no --gen3d is used.
    Intended for re-scoring DiffDock output poses with Vina.

    :param sdf: Path to the input SDF file (must have 3D coords)
    :param pdbqt: Path to the output PDBQT file. Defaults to replacing .sdf with .pdbqt.
    :param timeout: Maximum time in seconds for conversion
    :return: Path to the generated PDBQT file
    """
    if not os.path.exists(sdf):
        raise FileNotFoundError(f"SDF file not found: {sdf}")

    if pdbqt is None:
        pdbqt = sdf.replace(".sdf", ".pdbqt")

    obabel_cmd = f'obabel "{sdf}" -O "{pdbqt}" {SHELL_SILENCER}'

    try:
        subprocess.run(
            obabel_cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if not os.path.exists(pdbqt):
            raise RuntimeError(f"OpenBabel failed to create PDBQT file: {pdbqt}")

        with open(pdbqt, "r") as f:
            pdbqt_lines = f.readlines()

        atom_count = sum(
            1 for line in pdbqt_lines if line.startswith("ATOM") or line.startswith("HETATM")
        )
        if atom_count == 0:
            raise ValueError(f"Generated PDBQT has no ATOM records: {pdbqt}")

    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"OpenBabel SDF→PDBQT conversion timed out for: {sdf}") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"OpenBabel SDF→PDBQT conversion failed for {sdf}: {e.stderr}") from e

    return pdbqt


def protein_pdb_to_pdbqt(
    input_pdb: str,
    output_pdbqt: str | None = None,
    default_altloc: str = "A",
    allow_bad_res: bool = False,
):
    """
    Convert a PDB file to a PDBQT file.

    :param input_pdb: PDB file to be converted
    :param output_pdbqt: PDBQT file to be written
    :param default_altloc: Default alternate location
    :param allow_bad_res: Allow bad residues
    return: PDBQT file
    """
    output_pdbqt = output_pdbqt or input_pdb.replace(".pdb", ".pdbqt")
    cmd = [
        "mk_prepare_receptor.py",
        "--read_pdb",
        input_pdb,
        "--write_pdbqt",
        output_pdbqt,
    ]
    if default_altloc:
        cmd += ["--default_altloc", default_altloc]
    if allow_bad_res:
        cmd += ["--allow_bad_res"]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_pdbqt


def _flexres_str_to_meeko_args(flexres_str: str) -> list[str]:
    """
    Convert the internal ``"A:88_A:91_B:7"`` format to meeko 0.7+ ``-f`` args.

    meeko 0.7+ expects one ``-f`` flag per chain with residues comma-separated:
    ``["-f", "A:88,91", "-f", "B:7"]``.
    """
    from collections import defaultdict

    chain_residues: dict = defaultdict(list)
    for token in flexres_str.split("_"):
        chain, resnum = token.split(":")
        chain_residues[chain].append(resnum)
    args = []
    for chain, resnums in chain_residues.items():
        args += ["-f", f"{chain}:{','.join(resnums)}"]
    return args


def prepare_flex_receptor_pdbqt(
    input_pdb: str,
    output_rigid_pdbqt: str,
    output_flex_pdbqt: str,
    flexres_str: str,
    default_altloc: str = "A",
    allow_bad_res: bool = False,
) -> tuple[str, str]:
    """
    Split a receptor PDB into a rigid PDBQT and a flexible-residue PDBQT.

    Calls ``mk_prepare_receptor.py -p <basename> -f <flexres>`` (meeko 0.7+
    API).  meeko writes ``<basename>_rigid.pdbqt`` and ``<basename>_flex.pdbqt``
    automatically; these are then moved to the caller-specified paths.

    :param input_pdb: Cleaned protein PDB (post chain-isolation + renumbering).
    :param output_rigid_pdbqt: Destination for the rigid portion.
    :param output_flex_pdbqt: Destination for the flexible side chains.
    :param flexres_str: AutoDock residue spec, e.g. ``"A:1_A:3_B:7"``.
    :param default_altloc: Alternate-location selector (default ``"A"``).
    :param allow_bad_res: Pass ``--allow_bad_res`` to tolerate unusual residues.
    :return: ``(output_rigid_pdbqt, output_flex_pdbqt)`` paths.
    :raises RuntimeError: If either output file is missing or empty after the call.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    # meeko 0.7+ takes -p <basename> (no extension) and writes
    # <basename>_rigid.pdbqt / <basename>_flex.pdbqt.
    # Use a temp basename so we control where the files land, then move.
    tmp_dir = _tempfile.mkdtemp()
    basename = os.path.join(tmp_dir, "receptor")
    meeko_rigid = basename + "_rigid.pdbqt"
    meeko_flex = basename + "_flex.pdbqt"

    cmd = [
        "mk_prepare_receptor.py",
        "--read_pdb",
        input_pdb,
        "-p",
        basename,
    ] + _flexres_str_to_meeko_args(flexres_str)
    if default_altloc:
        cmd += ["--default_altloc", default_altloc]
    if allow_bad_res:
        cmd += ["--allow_bad_res"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    for path, label in [(meeko_rigid, "rigid"), (meeko_flex, "flex")]:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            _shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"prepare_flex_receptor_pdbqt: {label} output missing or empty: {path}"
            )

    try:
        _shutil.move(meeko_rigid, output_rigid_pdbqt)
        _shutil.move(meeko_flex, output_flex_pdbqt)
    except Exception:
        _shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    _shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_rigid_pdbqt, output_flex_pdbqt


def _molecule_rdkit_processing(molecule: Chem.Mol):
    """
    Process a molecule by adding hydrogens, embedding, and optimizing.
    Uses ETKDG for robust 3D embedding with MMFF→UFF fallback for broader atom type coverage.

    :param molecule: RDKit molecule
    return: Processed molecule
    """
    if molecule is None:
        raise ValueError("Cannot process None molecule")

    molecule = Chem.AddHs(molecule)

    # Try ETKDG embedding (more robust) with multiple attempts
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_result = AllChem.EmbedMolecule(molecule, params)

    # Retry with different seeds if first attempt fails
    if embed_result == -1:
        for seed in [42, 123, 999]:
            params.randomSeed = seed
            embed_result = AllChem.EmbedMolecule(molecule, params)
            if embed_result != -1:
                break

    if embed_result == -1:
        raise ValueError("Failed to generate 3D coordinates after multiple attempts")

    # Try MMFF first (better atom type coverage), then UFF, then accept unoptimized
    mmff_props = AllChem.MMFFGetMoleculeProperties(molecule)
    if mmff_props is not None:
        mmff_result = AllChem.MMFFOptimizeMolecule(molecule, mmffVariant="MMFF94")
        if mmff_result == 0:
            return molecule  # MMFF succeeded

    # Return molecule regardless of UFF result (unoptimized geometry still valid)
    return molecule


def pdb_to_sdf(input_pdb: str, output_sdf: str):
    """
    Convert a PDB file to an SDF file.

    :param input_pdb: PDB file to be converted
    :param output_sdf: SDF file to be written
    return: None
    """
    molecule = Chem.MolFromPDBFile(input_pdb)
    molecule = _molecule_rdkit_processing(molecule)
    writer = Chem.SDWriter(output_sdf)
    writer.write(molecule)


_POSE_EXTENSION = ".sdf"


def stage_user_pose(
    pose_path: str,
    ligand_sdf: str,
    ligand_pdbqt: str,
    gnina_input_mode: str = "pdbqt",
):
    """
    Stage a user-supplied ligand pose (SDF) into the standard per-combo
    artifact locations *without* re-randomising 3D coordinates.

    The default ligand-prep path in ``Guild._prepare_ligand`` runs
    ``smiles_to_sdf`` (OpenBabel ``--gen3d``) → ``sdf_to_pdb`` →
    ``ligand_pdb_to_pdbqt``, which produces an arbitrary conformer. When
    the user supplies a starting pose we instead copy the SDF verbatim
    and use ``sdf_to_pdbqt`` (coord-preserving, no ``--gen3d``) so the
    docking engine sees the actual pose.

    :param pose_path: Path to the user-supplied pose file. Must be a
        ``.sdf`` (we deliberately only support SDF for now — keeps the
        format dispatch and validation surface small; revisit if other
        formats become a frequent ask).
    :param ligand_sdf: Destination SDF path (``self.ligand_sdf``).
        Receives a byte-for-byte copy of ``pose_path``.
    :param ligand_pdbqt: Destination PDBQT path (``self.ligand_pdbqt``).
        Produced by ``sdf_to_pdbqt`` from the staged SDF. Skipped when
        ``gnina_input_mode == "sdf"`` because gnina-sdf mode reads
        ``ligand_sdf`` directly and no other consumer in ``Guild`` reads
        the per-combo PDB/PDBQT once docking has run.
    """
    ext = os.path.splitext(pose_path)[1].lower()
    if ext != _POSE_EXTENSION:
        raise ValueError(
            f"Unsupported pose file extension {ext!r} for {pose_path}; "
            f"only {_POSE_EXTENSION} is supported."
        )
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    for dest in (ligand_sdf, ligand_pdbqt):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    if os.path.abspath(pose_path) != os.path.abspath(ligand_sdf):
        shutil.copyfile(pose_path, ligand_sdf)

    if gnina_input_mode == "sdf":
        return

    sdf_to_pdbqt(sdf=ligand_sdf, pdbqt=ligand_pdbqt)
