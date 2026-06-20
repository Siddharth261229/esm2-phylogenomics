import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans

# =====================================================
# PATHS
# =====================================================

EMBEDDINGS_FILE = "output/embeddings.npy"
METADATA_FILE = "output/metadata.csv"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "paralog_candidates.csv"

MINORITY_THRESHOLD = 0.20
RANDOM_STATE = 42

# =====================================================
# LOAD DATA
# =====================================================

print("Loading embeddings and metadata...")

X = np.load(EMBEDDINGS_FILE)
meta = pd.read_csv(METADATA_FILE)

print(f"Embeddings shape: {X.shape}")
print(f"Metadata rows: {len(meta)}")

# =====================================================
# ANALYSIS
# =====================================================

results = []

families = sorted(meta["family"].unique())

print("\nAnalyzing families...\n")

for family in families:

    family_idx = meta["family"] == family

    family_meta = meta[family_idx].copy()

    family_embeddings = X[family_idx.values]

    n_sequences = len(family_meta)

    print(f"{family}: {n_sequences} sequences")

    all_flagged = set()

    # -----------------------------------------
    # Try k=2 and k=3
    # -----------------------------------------

    for k in [2, 3]:

        if n_sequences <= k:
            continue

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        clusters = kmeans.fit_predict(
            family_embeddings
        )

        cluster_sizes = pd.Series(clusters).value_counts()

        for cluster_id, size in cluster_sizes.items():

            fraction = size / n_sequences

            if fraction < MINORITY_THRESHOLD:

                flagged_ids = family_meta.iloc[
                    np.where(clusters == cluster_id)[0]
                ]["seq_id"].tolist()

                all_flagged.update(flagged_ids)

    results.append(
        {
            "family": family,
            "total_sequences": n_sequences,
            "num_putative_paralogs": len(all_flagged),
            "sequence_ids": ";".join(sorted(all_flagged))
        }
    )

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# PRINT SUMMARY
# =====================================================

print("\n========================================")
print("PUTATIVE PARALOG SUMMARY")
print("========================================\n")

for _, row in results_df.iterrows():

    print(f"Family: {row['family']}")
    print(f"Total sequences: {row['total_sequences']}")
    print(f"Putative paralogs: {row['num_putative_paralogs']}")

    if row["num_putative_paralogs"] > 0:

        ids = row["sequence_ids"].split(";")

        for seq in ids:
            print(f"  - {seq}")

    print()

print("========================================")
print(f"Saved: {OUTPUT_FILE}")
print("========================================")