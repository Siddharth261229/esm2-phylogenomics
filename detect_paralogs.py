"""Stage 5: flag candidate paralogs / embedding outliers within each family.

Upgrade over the original: the old approach (minority KMeans cluster with
k=2 and k=3) *always* flags something for every family, even one with no
real outliers, because k-means with k>1 always produces a smaller cluster.
This script keeps that heuristic (renamed 'kmeans_minority' so it's clear
what it is) but adds a second, statistically motivated signal: a robust
z-score on each sequence's embedding distance from its family centroid
(median + MAD based, so it isn't skewed by the very outliers it's trying
to detect). A sequence flagged by BOTH methods is reported as
high-confidence; flagged by only one, as low-confidence.

Usage:
    python detect_paralogs.py
    python detect_paralogs.py --minority-threshold 0.15 --z-threshold 2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from esm2_phylo.config import EMBEDDINGS_FILE, METADATA_FILE, RESULTS_DIR
from esm2_phylo.logging_utils import get_logger
from esm2_phylo.utils import flag_minority_clusters, robust_z_outlier_ids

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--minority-threshold", type=float, default=0.20)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.embeddings.exists():
        raise SystemExit(f"Embeddings not found: {args.embeddings}. Run generate_embeddings.py first.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.results_dir / "paralog_candidates.csv"
    detail_file = args.results_dir / "paralog_candidates_detail.csv"

    log.info("Loading embeddings and metadata...")
    X = np.load(args.embeddings)
    meta = pd.read_csv(args.metadata)

    if len(meta) != len(X):
        raise SystemExit(f"Metadata rows ({len(meta)}) != embedding rows ({len(X)}). Rerun generate_embeddings.py.")

    summary_rows = []
    detail_rows = []
    families = sorted(meta["family"].unique())

    log.info("Analyzing %d families...", len(families))

    for family in families:
        family_idx = meta["family"] == family
        family_meta = meta[family_idx].reset_index(drop=True)
        family_embeddings = X[family_idx.values]
        n_sequences = len(family_meta)
        seq_ids = family_meta["seq_id"].tolist()

        log.info("%s: %d sequences", family, n_sequences)

        # --- Signal 1: KMeans minority cluster (original heuristic) ---
        kmeans_flagged: set[str] = set()
        for k in [2, 3]:
            if n_sequences <= k:
                continue
            kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
            clusters = kmeans.fit_predict(family_embeddings)
            kmeans_flagged |= flag_minority_clusters(clusters, seq_ids, args.minority_threshold)

        # --- Signal 2: robust z-score distance from family centroid ---
        z_flagged = robust_z_outlier_ids(family_embeddings, seq_ids, args.z_threshold)

        high_confidence = kmeans_flagged & z_flagged
        any_flagged = kmeans_flagged | z_flagged

        for seq_id in any_flagged:
            detail_rows.append(
                {
                    "family": family,
                    "seq_id": seq_id,
                    "flagged_by_kmeans_minority": seq_id in kmeans_flagged,
                    "flagged_by_robust_z": seq_id in z_flagged,
                    "high_confidence": seq_id in high_confidence,
                }
            )

        summary_rows.append(
            {
                "family": family,
                "total_sequences": n_sequences,
                "num_kmeans_minority_flags": len(kmeans_flagged),
                "num_robust_z_flags": len(z_flagged),
                "num_high_confidence": len(high_confidence),
                "num_any_flag": len(any_flagged),
                "high_confidence_ids": ";".join(sorted(high_confidence)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_file, index=False)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(detail_file, index=False)

    log.info("=" * 40)
    log.info("PARALOG / OUTLIER SUMMARY")
    log.info("=" * 40)
    for _, row in summary_df.iterrows():
        log.info(
            "%-8s total=%-4d kmeans_flags=%-3d robust_z_flags=%-3d high_confidence=%-3d",
            row["family"], row["total_sequences"], row["num_kmeans_minority_flags"],
            row["num_robust_z_flags"], row["num_high_confidence"],
        )

    log.info("=" * 40)
    log.info("Saved: %s", output_file)
    log.info("Saved: %s", detail_file)
    log.info("=" * 40)


if __name__ == "__main__":
    main()
