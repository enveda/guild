import os
import urllib.error
import urllib.request

import requests
from tqdm import tqdm


def pdb_fetch_protein_name(input_pdb_id):
    """
    Fetch the protein name from the RCSB PDB API
    :param input_pdb_id: PDB ID
    :return: Protein name
    """

    url = f"https://data.rcsb.org/rest/v1/core/entry/{input_pdb_id}"

    # Fetch data
    response = requests.get(url)

    if response.status_code == 200:
        metadata = response.json()
        entry_title = metadata.get("struct", {}).get("title", "N/A")
        return entry_title


def download_pdb_files(pdb_ids, download_dir="pdb_files"):
    """
    Download PDB files from the RCSB PDB database.
    :param pdb_ids: List of PDB IDs to download
    :param download_dir: Directory to save the downloaded PDB files
    """
    os.makedirs(download_dir, exist_ok=True)

    failed_pdbs = []
    for pdb_id in tqdm(pdb_ids):
        if os.path.exists(f"{download_dir}/{pdb_id}.pdb"):
            continue

        try:
            urllib.request.urlretrieve(
                f"https://files.rcsb.org/download/{pdb_id}.pdb",
                f"{download_dir}/{pdb_id}.pdb",
            )
        except urllib.error.HTTPError:
            failed_pdbs.append(pdb_id)
    print(
        f"Downloaded {len(pdb_ids)} PDB files to {download_dir}\nFailed to download {len(failed_pdbs)} PDB files"
    )
    return failed_pdbs
