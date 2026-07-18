"""Quick sanity check: species counts per family and species shared
across ALL families (useful for spotting whether a species tree / RF
comparison will have enough shared taxa to be meaningful).

Usage:
    python check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from esm2_phylo.config import METADATA_FILE
from esm2_phylo.logging_utils import get_logger
from esm2_phylo.utils import species_overlap_summary

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    args = parser.parse_args()

    if not args.metadata.exists():
        raise SystemExit(f"Metadata not found: {args.metadata}. Run generate_embeddings.py first.")

    meta = pd.read_csv(args.metadata)
    summary = species_overlap_summary(meta)

    log.info("Species per family:")
    for fam, count in summary["per_family_species_counts"].items():
        log.info("  %-8s %d unique species", fam, count)

    log.info("Species shared across ALL families: %d", summary["n_shared_species"])
    if summary["n_shared_species"]:
        log.info("  %s", summary["shared_species"][:20])


if __name__ == "__main__":
    main()
