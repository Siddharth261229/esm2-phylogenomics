"""Stage 3: UMAP projection of embeddings, colored by family and species.

Usage:
    python cluster_visualise.py
    python cluster_visualise.py --embeddings output/embeddings.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

from esm2_phylo.config import EMBEDDINGS_FILE, METADATA_FILE, PLOTS_DIR
from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)


def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(X)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.embeddings.exists():
        raise SystemExit(f"Embeddings not found: {args.embeddings}. Run generate_embeddings.py first.")

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading embeddings...")
    X = np.load(args.embeddings)
    meta = pd.read_csv(args.metadata)

    if len(meta) != len(X):
        raise SystemExit(
            f"Metadata rows ({len(meta)}) != embedding rows ({len(X)}). "
            "These files are out of sync -- rerun generate_embeddings.py."
        )

    log.info("Embeddings shape: %s", X.shape)
    log.info("Running UMAP...")
    embedding_2d = run_umap(X, args.n_neighbors, args.min_dist, args.seed)
    meta["UMAP1"] = embedding_2d[:, 0]
    meta["UMAP2"] = embedding_2d[:, 1]

    plt.style.use("dark_background")
    sns.set_context("talk", font_scale=1.1)

    # ---- Plot 1: family ----
    log.info("Generating family plot...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=meta, x="UMAP1", y="UMAP2", hue="family", s=80, alpha=0.9)
    plt.title("UMAP Projection of ESM-2 Embeddings\nColored by Gene Family", fontsize=16)
    plt.tight_layout()
    family_plot = args.plots_dir / "umap_by_family.png"
    plt.savefig(family_plot, dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: species ----
    log.info("Generating species plot...")
    unique_species = meta["species"].nunique()
    plt.figure(figsize=(14, 12))

    if unique_species <= 25:
        sns.scatterplot(data=meta, x="UMAP1", y="UMAP2", hue="species", s=80, alpha=0.9)
    else:
        species_codes = pd.factorize(meta["species"])[0]
        scatter = plt.scatter(meta["UMAP1"], meta["UMAP2"], c=species_codes, s=80, alpha=0.9)
        plt.colorbar(scatter, label="Species Index")

    plt.title("UMAP Projection of ESM-2 Embeddings\nColored by Species", fontsize=16)
    plt.tight_layout()
    species_plot = args.plots_dir / "umap_by_species.png"
    plt.savefig(species_plot, dpi=300, bbox_inches="tight")
    plt.close()

    meta.to_csv(args.plots_dir / "umap_coordinates.csv", index=False)

    log.info("=" * 40)
    log.info("UMAP complete")
    log.info("Saved: %s", family_plot)
    log.info("Saved: %s", species_plot)
    log.info("=" * 40)


if __name__ == "__main__":
    main()
