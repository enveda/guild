import pandas as pd

TARGETS = [
    "7l1v-R-XGD-R", "7v3z-A-9GF-A", "4ea3-A-0NN-A", "6ot0-R-CO1-R", "6me6-A-JEY-A",
    "8gdc-R-P2E-R", "6ds0-A-H8M-A", "5o9h-A-9P2-A", "5tud-A-ERM-A", "7jvr-R-08Y-R",
]
N_PER_TARGET = 100
SEED = 42

df = pd.read_csv("vinarun_all_combinations.csv", low_memory=False)
df = df[df.protein_config_id.isin(TARGETS)].copy()

# Deterministic: sort by ligand_id first so the sample does not depend on row order.
picked = (
    df.sort_values(["protein_config_id", "ligand_id"])
      .groupby("protein_config_id", group_keys=False)
      .apply(lambda g: g.sample(n=min(N_PER_TARGET, len(g)), random_state=SEED))
      .sort_values(["protein_config_id", "ligand_id"])
)

# Repoint protein_path at this machine's mount.
picked["protein_path"] = "/workspace/data/" + picked["protein_id"] + ".pdb"

picked.to_csv("combinations.csv", index=False)
print(f"rows: {len(picked)}  targets: {picked.protein_config_id.nunique()}")
print(picked.groupby("protein_config_id").size().to_string())
print(picked.ligand_category.value_counts().to_string())
