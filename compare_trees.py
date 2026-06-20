from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ete4 import Tree

# =====================================================
# PATHS
# =====================================================

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

PLOTS_DIR.mkdir(exist_ok=True)

SPECIES_TREE = RESULTS_DIR / "species.tree"

OUTPUT_CSV = RESULTS_DIR / "rf_distances.csv"
OUTPUT_PLOT = PLOTS_DIR / "rf_distances.png"

# =====================================================
# LOAD SPECIES TREE
# =====================================================

species_tree = Tree(str(SPECIES_TREE))

# =====================================================
# FIND GENE TREES
# =====================================================

tree_files = []

for tree_file in RESULTS_DIR.glob("*.tree"):

    if tree_file.name == "species.tree":
        continue

    tree_files.append(tree_file)

results = []

# =====================================================
# COMPARE TREES
# =====================================================

for tree_file in tree_files:

    family = tree_file.stem

    gene_tree = Tree(str(tree_file))

    species_copy = species_tree.copy()
    gene_copy = gene_tree.copy()

    species_taxa = set(species_copy.get_leaf_names())
    gene_taxa = set(gene_copy.get_leaf_names())

    shared_taxa = species_taxa & gene_taxa

    if len(shared_taxa) < 4:

        print(
            f"Skipping {family}: "
            f"only {len(shared_taxa)} shared taxa"
        )

        continue

    species_copy.prune(shared_taxa)
    gene_copy.prune(shared_taxa)

    comparison = gene_copy.compare(
        species_copy,
        unrooted=True
    )

    rf = comparison["rf"]

    max_rf = comparison["max_rf"]

    normalized_rf = (
        rf / max_rf if max_rf > 0 else 0
    )

    results.append(
        {
            "family": family,
            "shared_taxa": len(shared_taxa),
            "rf_distance": rf,
            "max_rf": max_rf,
            "normalized_rf": normalized_rf
        }
    )

# =====================================================
# SAVE TABLE
# =====================================================

df = pd.DataFrame(results)

df = df.sort_values(
    "normalized_rf",
    ascending=False
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# =====================================================
# PRINT TABLE
# =====================================================

print("\n==============================")
print("RF DISTANCE SUMMARY")
print("==============================\n")

print(
    df[
        [
            "family",
            "shared_taxa",
            "rf_distance",
            "normalized_rf"
        ]
    ]
)

# =====================================================
# BARPLOT
# =====================================================

plt.style.use("dark_background")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="family",
    y="normalized_rf"
)

plt.ylabel("Normalized RF Distance")
plt.xlabel("Gene Family")
plt.title(
    "Gene Tree vs Species Tree Discordance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300
)

plt.close()

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_PLOT}")