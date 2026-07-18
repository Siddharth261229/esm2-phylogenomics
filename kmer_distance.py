"""Stage 6: alignment-free k-mer baseline, compared against ESM-2 embeddings.

BUG FIX from the original version: the original script read directly from
``data/*.fasta`` (the raw, unfiltered OrthoDB dumps -- thousands of
sequences per family, no length filter, no per-species dedup, and even a
stray 0-byte ``HSP70_aligned.fasta``) while the ESM-2 embeddings are
computed on the curated ``output/all_families.fasta`` (320 sequences,
80/family, deduplicated by species). The purity/V-measure comparison
table in the README was therefore comparing two different datasets, not
a fair head-to-head baseline. This version computes the k-mer baseline
on the exact same curated FASTA (and hence the exact same sequences) used
for the embeddings, so the comparison is apples-to-apples.

It also fixes a label mismatch: the original used raw filenames as family
labels (``HSP_70``, ``CYT_C``), which didn't match the standardized labels
used everywhere else (``HSP70``, ``CYTC``) -- causing two of four families
to silently fall back to matplotlib's default color in the comparison plot
because FAMILY_COLORS.get() returned None for them.

Usage:
    python kmer_distance.py
    python kmer_distance.py --k 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import v_measure_score

from esm2_phylo.config import ALL_FAMILIES_FASTA, EMBEDDINGS_FILE, FAMILY_COLORS, METADATA_FILE, PLOTS_DIR
from esm2_phylo.logging_utils import get_logger
from esm2_phylo.utils import build_kmer_vocabulary, cluster_purity, compute_kmer_vector

log = get_logger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def load_curated_sequences(fasta_file: Path):
    """Load the SAME curated dataset used for ESM-2 embeddings (fixes the
    dataset-mismatch bug described in the module docstring)."""
    seq_ids, labels, seqs = [], [], []
    for record in SeqIO.parse(fasta_file, "fasta"):
        try:
            family, _species = record.id.split("|", 1)
        except ValueError:
            log.warning("Skipping malformed header: %s", record.id)
            continue
        seq_ids.append(record.id)
        labels.append(family)
        seqs.append(str(record.seq))
    return seq_ids, labels, seqs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", type=Path, default=ALL_FAMILIES_FASTA,
                         help="Curated FASTA to use for BOTH baselines (must match embeddings)")
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.fasta.exists():
        raise SystemExit(f"Curated FASTA not found: {args.fasta}. Run dataset_prepare.py first.")
    if not args.embeddings.exists():
        raise SystemExit(f"Embeddings not found: {args.embeddings}. Run generate_embeddings.py first.")

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating k-mer vocabulary (k=%d)...", args.k)
    kmer_to_idx = build_kmer_vocabulary(AMINO_ACIDS, args.k)
    log.info("Vocabulary size: %d", len(kmer_to_idx))

    log.info("Reading curated FASTA: %s", args.fasta)
    seq_ids, labels, seqs = load_curated_sequences(args.fasta)
    labels = np.array(labels)
    log.info("Loaded %d sequences (same set used for embeddings)", len(seq_ids))

    log.info("Computing k-mer vectors...")
    X_kmer = np.vstack([
        compute_kmer_vector(s, kmer_to_idx, args.k, AMINO_ACIDS) for s in seqs
    ])
    log.info("K-mer matrix shape: %s", X_kmer.shape)

    log.info("Computing cosine distances...")
    D = squareform(pdist(X_kmer, metric="cosine"))

    n_families = len(set(labels))

    log.info("Running k-mer clustering...")
    kmer_model = AgglomerativeClustering(n_clusters=n_families, metric="precomputed", linkage="average")
    kmer_clusters = kmer_model.fit_predict(D)
    kmer_purity = cluster_purity(kmer_clusters, labels)
    kmer_vmeasure = v_measure_score(labels, kmer_clusters)
    log.info("k-mer baseline purity: %.3f | V-measure: %.3f", kmer_purity, kmer_vmeasure)

    log.info("Loading ESM embeddings...")
    X_esm = np.load(args.embeddings)
    meta = pd.read_csv(args.metadata)

    if len(meta) != len(X_esm):
        raise SystemExit(f"Metadata rows ({len(meta)}) != embedding rows ({len(X_esm)}). Rerun generate_embeddings.py.")
    if set(meta["seq_id"]) != set(seq_ids):
        log.warning(
            "seq_ids in %s and %s do not match exactly -- baseline and "
            "embedding comparison may not be on identical sequences.",
            args.metadata, args.fasta,
        )

    esm_labels = meta["family"].values

    kmeans = KMeans(n_clusters=n_families, random_state=args.seed, n_init=20)
    esm_clusters = kmeans.fit_predict(X_esm)
    esm_purity = cluster_purity(esm_clusters, esm_labels)
    esm_vmeasure = v_measure_score(esm_labels, esm_clusters)
    log.info("ESM-2 embedding purity: %.3f | V-measure: %.3f", esm_purity, esm_vmeasure)

    log.info("--- Summary (same %d sequences for both methods) ---", len(seq_ids))
    log.info("%-25s %8s %8s", "Metric", "k-mer", "ESM-2")
    log.info("%-25s %8.3f %8.3f", "Cluster Purity", kmer_purity, esm_purity)
    log.info("%-25s %8.3f %8.3f", "V-measure", kmer_vmeasure, esm_vmeasure)

    log.info("Running UMAP for both representations...")
    umap_kmer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=args.seed).fit_transform(X_kmer)
    umap_esm = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=args.seed).fit_transform(X_esm)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for family in sorted(set(labels)):
        idx = labels == family
        axes[0].scatter(umap_kmer[idx, 0], umap_kmer[idx, 1], label=family, s=40, alpha=0.8,
                         color=FAMILY_COLORS.get(family))
    axes[0].set_title(f"{args.k}-mer Frequency Baseline (curated set)", fontsize=13)
    axes[0].legend(fontsize=9)

    for family in sorted(set(esm_labels)):
        idx = esm_labels == family
        axes[1].scatter(umap_esm[idx, 0], umap_esm[idx, 1], label=family, s=40, alpha=0.8,
                         color=FAMILY_COLORS.get(family))
    axes[1].set_title("ESM-2 Embeddings", fontsize=13)
    axes[1].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    summary = (
        f"Cluster Purity  ->  k-mer: {kmer_purity:.3f}   |   ESM-2: {esm_purity:.3f}\n"
        f"V-measure       ->  k-mer: {kmer_vmeasure:.3f}   |   ESM-2: {esm_vmeasure:.3f}"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=11, family="monospace",
              bbox=dict(facecolor="#111111", edgecolor="#444444", alpha=0.9, pad=6))

    outfile = args.plots_dir / "kmer_vs_esm2.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    log.info("Saved: %s", outfile)


if __name__ == "__main__":
    main()
