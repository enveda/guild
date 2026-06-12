#!/usr/bin/env python3
"""
Vina rescore all DiffDock poses across sharded results.

Usage:
    python scripts/vina_rescore_all.py

Produces:
    /data/results2/vina_rescore_all.tsv   – per-combination Vina scores
"""

import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem
from vina import Vina

# ──────────────────────────── Config ──────────────────────────────────────────
RESULTS_DIR = "data/results2"
PDB_DIR = "pdbs/pdbs"
RECEPTOR_CACHE_DIR = "_receptor_pdbqt_cache"
OUTPUT_FILE = "vina_rescore_all.tsv"
PARTIAL_FILE = "vina_rescore_partial.tsv"  # incremental saves
MAX_WORKERS = 20   # leave some headroom on 24-core box
BOX_PADDING = 4.0
RANDOM_SEED = 42
SAVE_EVERY = 5000  # flush partial results every N completions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vina_rescore.log"),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Helpers ─────────────────────────────────────────

def prepare_receptor_pdbqt(protein_conf_id: str) -> str | None:
    """
    Prepare a receptor PDBQT from the raw PDB, cached to avoid redundant work.
    Returns the path to the PDBQT or None if preparation fails.
    """
    output_pdbqt = os.path.join(RECEPTOR_CACHE_DIR, f"{protein_conf_id}_raw.pdbqt")
    if os.path.isfile(output_pdbqt) and os.path.getsize(output_pdbqt) > 0:
        return output_pdbqt

    parts = protein_conf_id.split("-")
    pdb_id = parts[0]
    # Single chain or a comma-joined set ("A,B") for a multi-chain receptor.
    chain_ids = (
        [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) >= 2 else ["A"]
    )
    raw_pdb = os.path.join(PDB_DIR, f"{pdb_id}.pdb")

    if not os.path.isfile(raw_pdb):
        logger.warning(f"Raw PDB not found: {raw_pdb}")
        return None

    # Extract chain(s) — a comma-joined set ("A,B") keeps a multi-chain receptor
    chain_pdb = output_pdbqt.replace(".pdbqt", "_chain.pdb")
    kept = 0
    with open(raw_pdb) as fin, open(chain_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") and len(line) > 21 and line[21] in chain_ids:
                fout.write(line)
                kept += 1
        fout.write("END\n")

    if kept == 0:
        logger.warning(f"No ATOM records for chain(s) {chain_ids} in {raw_pdb}")
        return None

    result = subprocess.run(
        ["obabel", "-ipdb", chain_pdb, "-opdbqt", "-O", output_pdbqt, "-xr"],
        capture_output=True, text=True,
    )
    if not os.path.isfile(output_pdbqt) or os.path.getsize(output_pdbqt) == 0:
        logger.warning(f"obabel failed for {protein_conf_id}: {result.stderr}")
        return None

    return output_pdbqt


def find_best_diffdock_sdf(results_dir: str, combo_name: str) -> str | None:
    """Find the highest-confidence SDF file for a combination."""
    combo_dir = os.path.join(results_dir, combo_name)
    if not os.path.isdir(combo_dir):
        return None

    sdf_scores = {}
    for fname in os.listdir(combo_dir):
        if "_confidence" in fname and fname.endswith(".sdf"):
            try:
                score = float(fname.split("_confidence")[1].replace(".sdf", ""))
                sdf_scores[fname] = score
            except ValueError:
                continue

    if not sdf_scores:
        return None

    best_fname = max(sdf_scores, key=sdf_scores.get)
    return os.path.join(combo_dir, best_fname)


def compute_box_from_sdf(sdf_path: str, padding: float = BOX_PADDING):
    """Compute Vina box centered on ligand pose in SDF."""
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")

    conf = mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    center = tuple(positions.mean(axis=0).tolist())
    span = positions.max(axis=0) - positions.min(axis=0)
    size = tuple((span + 2.0 * padding).tolist())
    return center, size


def rescore_single(
    receptor_pdbqt: str,
    sdf_path: str,
    protein_conf_id: str,
    ligand_id: str,
) -> dict:
    """Score a single DiffDock pose with Vina."""
    combination_id = f"{protein_conf_id}_{ligand_id}"

    # Convert SDF → PDBQT (temp file in /tmp to avoid clutter)
    ligand_pdbqt = f"/tmp/vina_rescore_{os.getpid()}_{combination_id}.pdbqt"
    try:
        subprocess.run(
            f'obabel "{sdf_path}" -O "{ligand_pdbqt}" 2>/dev/null',
            shell=True, check=True, timeout=30,
        )
    except Exception as e:
        return {
            "combination": combination_id,
            "protein_config_id": protein_conf_id,
            "ligand_id": ligand_id,
            "vina_rescore_score": np.nan,
            "error": f"obabel failed: {e}",
        }

    if not os.path.isfile(ligand_pdbqt) or os.path.getsize(ligand_pdbqt) == 0:
        return {
            "combination": combination_id,
            "protein_config_id": protein_conf_id,
            "ligand_id": ligand_id,
            "vina_rescore_score": np.nan,
            "error": "Empty ligand PDBQT",
        }

    try:
        center, size = compute_box_from_sdf(sdf_path, padding=BOX_PADDING)
    except Exception as e:
        _cleanup(ligand_pdbqt)
        return {
            "combination": combination_id,
            "protein_config_id": protein_conf_id,
            "ligand_id": ligand_id,
            "vina_rescore_score": np.nan,
            "error": f"box computation failed: {e}",
        }

    try:
        v = Vina(sf_name="vina", seed=RANDOM_SEED, verbosity=False)
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(center=center, box_size=size)
        energy = v.score()
        score = float(energy[0])
    except Exception as e:
        score = np.nan
        error = str(e)
    else:
        error = ""

    _cleanup(ligand_pdbqt)

    return {
        "combination": combination_id,
        "protein_config_id": protein_conf_id,
        "ligand_id": ligand_id,
        "vina_rescore_score": score,
        "error": error,
    }


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ──────────────────────────── Worker ──────────────────────────────────────────

def process_one_combination(args):
    """Worker function for parallel execution."""
    shard_path, batch_name, protein_conf_id, ligand_id, smiles, ligand_category = args
    combination_id = f"{protein_conf_id}_{ligand_id}"

    # Receptor PDBQT (cached)
    receptor_pdbqt = os.path.join(RECEPTOR_CACHE_DIR, f"{protein_conf_id}_raw.pdbqt")
    if not os.path.isfile(receptor_pdbqt):
        return {
            "combination": combination_id,
            "protein_config_id": protein_conf_id,
            "ligand_id": ligand_id,
            "smiles": smiles,
            "ligand_category": ligand_category,
            "vina_rescore_score": np.nan,
            "error": "No receptor PDBQT",
        }

    # Find best DiffDock SDF
    results_dir = os.path.join(shard_path, "batches", batch_name, "diffdock", "results")
    sdf_path = find_best_diffdock_sdf(results_dir, combination_id)
    if sdf_path is None:
        return {
            "combination": combination_id,
            "protein_config_id": protein_conf_id,
            "ligand_id": ligand_id,
            "smiles": smiles,
            "ligand_category": ligand_category,
            "vina_rescore_score": np.nan,
            "error": "No DiffDock SDF found",
        }

    result = rescore_single(receptor_pdbqt, sdf_path, protein_conf_id, ligand_id)
    result["smiles"] = smiles
    result["ligand_category"] = ligand_category
    return result


# ──────────────────────────── Main ────────────────────────────────────────────

def collect_all_jobs():
    """Collect all (shard, batch, protein_conf_id, ligand_id) tuples to process."""
    jobs = []
    shard_dirs = sorted(
        d for d in os.listdir(RESULTS_DIR)
        if d.startswith("diffdock-shard") and os.path.isdir(os.path.join(RESULTS_DIR, d))
    )

    for shard_name in shard_dirs:
        shard_path = os.path.join(RESULTS_DIR, shard_name)
        batches_dir = os.path.join(shard_path, "batches")
        if not os.path.isdir(batches_dir):
            continue

        for batch_name in sorted(os.listdir(batches_dir)):
            batch_path = os.path.join(batches_dir, batch_name)
            combos_csv = os.path.join(batch_path, "combinations.csv")
            if not os.path.isfile(combos_csv):
                continue

            try:
                df = pd.read_csv(combos_csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue

            if df.empty:
                continue
            for _, row in df.iterrows():
                protein_conf_id = row["protein_config_id"]
                ligand_id = str(row["ligand_id"])
                smiles = row.get("smiles", "")
                ligand_category = row.get("ligand_category", "")
                jobs.append((shard_path, batch_name, protein_conf_id, ligand_id, smiles, ligand_category))

    return jobs


def load_existing_results() -> tuple[pd.DataFrame, set[str]]:
    """Load previously scored results from partial or final output files."""
    for path in (PARTIAL_FILE, OUTPUT_FILE):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            try:
                df = pd.read_csv(path, sep="\t")
                done_ids = set(df["combination"].dropna().unique())
                logger.info(f"Loaded {len(done_ids)} existing results from {path}")
                return df, done_ids
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
    return pd.DataFrame(), set()


def save_partial(results: list[dict]):
    """Save current results to the partial file."""
    df = pd.DataFrame(results)
    df.to_csv(PARTIAL_FILE, sep="\t", index=False)
    logger.info(f"Saved {len(df)} partial results to {PARTIAL_FILE}")


def main():
    os.makedirs(RECEPTOR_CACHE_DIR, exist_ok=True)

    # Step 0: Load existing results for resume
    existing_df, done_ids = load_existing_results()
    existing_results = existing_df.to_dict("records") if not existing_df.empty else []

    # Step 1: Collect all jobs
    logger.info("Collecting jobs from all shards/batches...")
    all_jobs = collect_all_jobs()
    logger.info(f"Total combinations to score: {len(all_jobs)}")

    # Deduplicate by combination_id (same combo may appear in overlapping batches)
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        combo_id = f"{job[2]}_{job[3]}"  # protein_conf_id + ligand_id
        if combo_id not in seen:
            seen.add(combo_id)
            unique_jobs.append(job)
    logger.info(f"Unique combinations: {len(unique_jobs)}")

    # Filter out already-done combinations
    if done_ids:
        remaining_jobs = [j for j in unique_jobs if f"{j[2]}_{j[3]}" not in done_ids]
        logger.info(f"Already scored: {len(done_ids)}, remaining: {len(remaining_jobs)}")
    else:
        remaining_jobs = unique_jobs

    # Step 2: Pre-build all receptor PDBQTs
    unique_protein_confs = sorted(set(j[2] for j in remaining_jobs))
    logger.info(f"Preparing {len(unique_protein_confs)} receptor PDBQTs...")
    receptor_status = {}
    for pconf in unique_protein_confs:
        result = prepare_receptor_pdbqt(pconf)
        receptor_status[pconf] = result is not None
        if result:
            logger.info(f"  ✓ {pconf}")
        else:
            logger.warning(f"  ✗ {pconf}")

    ok_receptors = sum(1 for v in receptor_status.values() if v)
    logger.info(f"Receptor PDBQTs ready: {ok_receptors}/{len(unique_protein_confs)}")

    if not remaining_jobs:
        logger.info("All combinations already scored — nothing to do.")
        # Just ensure the final output exists
        if existing_results:
            df = pd.DataFrame(existing_results)
            df.to_csv(OUTPUT_FILE, sep="\t", index=False)
            logger.info(f"Final output saved to {OUTPUT_FILE}")
        return

    # Step 3: Run rescoring in parallel
    logger.info(f"Starting Vina rescoring with {MAX_WORKERS} workers for {len(remaining_jobs)} combinations...")
    results = list(existing_results)  # start from existing
    failed = sum(1 for r in existing_results if r.get("error"))
    new_completed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_combination, job): job for job in remaining_jobs}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if result.get("error"):
                    failed += 1
            except Exception as e:
                job = futures[future]
                combo_id = f"{job[2]}_{job[3]}"
                results.append({
                    "combination": combo_id,
                    "protein_config_id": job[2],
                    "ligand_id": job[3],
                    "smiles": job[4],
                    "ligand_category": job[5],
                    "vina_rescore_score": np.nan,
                    "error": str(e),
                })
                failed += 1

            new_completed = i + 1

            if new_completed % 1000 == 0:
                elapsed = time.time() - t0
                rate = new_completed / elapsed
                eta = (len(remaining_jobs) - new_completed) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {new_completed}/{len(remaining_jobs)} "
                    f"({new_completed/len(remaining_jobs)*100:.1f}%) "
                    f"| Rate: {rate:.1f}/s | ETA: {eta/60:.0f}min "
                    f"| Failed: {failed} | Total: {len(results)}"
                )

            # Periodic save for crash resilience
            if new_completed % SAVE_EVERY == 0:
                save_partial(results)

    elapsed = time.time() - t0
    logger.info(f"Rescoring complete: {new_completed} new combinations in {elapsed/60:.1f} min")
    logger.info(f"Failed: {failed}")

    # Step 4: Save final results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    logger.info(f"Saved results to {OUTPUT_FILE}")

    # Clean up partial file
    if os.path.isfile(PARTIAL_FILE):
        os.remove(PARTIAL_FILE)
        logger.info("Removed partial file")

    # Summary stats
    scored = df["vina_rescore_score"].notna().sum()
    logger.info(f"Successfully scored: {scored}/{len(df)}")
    logger.info(f"Score range: [{df['vina_rescore_score'].min():.3f}, {df['vina_rescore_score'].max():.3f}]")
    logger.info(f"Mean score: {df['vina_rescore_score'].mean():.3f}")


if __name__ == "__main__":
    main()
