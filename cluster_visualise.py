import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

EMBEDDINGS_FILE = "output/embeddings.npy"
METADATA_FILE = "output/metadata.csv"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading embeddings...")

X = np.load(EMBEDDINGS_FILE)
meta = pd.read_csv(METADATA_FILE)

print(f"Embeddings shape: {X.shape}")
print(f"Metadata rows: {len(meta)}")

# =====================================================
# UMAP
# =====================================================

print("Running UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

embedding_2d = reducer.fit_transform(X)

meta["UMAP1"] = embedding_2d[:, 0]
meta["UMAP2"] = embedding_2d[:, 1]

# =====================================================
# DARK THEME
# =====================================================

plt.style.use("dark_background")

sns.set_context(
    "talk",
    font_scale=1.1
)

# =====================================================
# PLOT 1: FAMILY
# =====================================================

print("Generating family plot...")

plt.figure(figsize=(12, 10))

sns.scatterplot(
    data=meta,
    x="UMAP1",
    y="UMAP2",
    hue="family",
    s=80,
    alpha=0.9
)

plt.title(
    "UMAP Projection of ESM2 Embeddings\nColored by Gene Family",
    fontsize=16
)

plt.tight_layout()

family_plot = PLOTS_DIR / "umap_by_family.png"

plt.savefig(
    family_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# PLOT 2: SPECIES
# =====================================================

print("Generating species plot...")

unique_species = meta["species"].nunique()

plt.figure(figsize=(14, 12))

if unique_species <= 25:

    sns.scatterplot(
        data=meta,
        x="UMAP1",
        y="UMAP2",
        hue="species",
        s=80,
        alpha=0.9
    )

else:
    # Too many species for a readable legend

    species_codes = pd.factorize(
        meta["species"]
    )[0]

    scatter = plt.scatter(
        meta["UMAP1"],
        meta["UMAP2"],
        c=species_codes,
        s=80,
        alpha=0.9
    )

    plt.colorbar(
        scatter,
        label="Species Index"
    )

plt.title(
    "UMAP Projection of ESM2 Embeddings\nColored by Species",
    fontsize=16
)

plt.tight_layout()

species_plot = PLOTS_DIR / "umap_by_species.png"

plt.savefig(
    species_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# SAVE UMAP COORDINATES
# =====================================================

meta.to_csv(
    PLOTS_DIR / "umap_coordinates.csv",
    index=False
)

# =====================================================
# DONE
# =====================================================

print("\n===================================")
print("UMAP complete")
print(f"Saved: {family_plot}")
print(f"Saved: {species_plot}")
print("===================================")