"""Helper: build results/species.tree from NCBI taxonomy, for use with
compare_trees.py.

This requires internet access (to download/update ete4's local NCBI
taxonomy database on first run) and is NOT run as part of the sandboxed
test suite in this repo -- run it on your own machine.

Given the species names already present in output/metadata.csv (produced
by dataset_prepare.py + generate_embeddings.py), this looks up each
species' NCBI taxid and asks ete4's NCBITaxa module to build the minimal
topology tree connecting them, which approximates the true species tree
using NCBI's curated taxonomy rather than inferring one from sequence
data (avoiding circularity with the gene trees being tested against it).

Usage:
    python fetch_species_tree.py
    python fetch_species_tree.py --metadata output/metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from esm2_phylo.config import METADATA_FILE, RESULTS_DIR
from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <results-dir>/species.tree")
    args = parser.parse_args()

    try:
        from ete4 import NCBITaxa
    except ImportError as exc:
        raise SystemExit("Requires ete4: pip install ete4") from exc

    if not args.metadata.exists():
        raise SystemExit(f"Metadata not found: {args.metadata}. Run generate_embeddings.py first.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (args.results_dir / "species.tree")

    meta = pd.read_csv(args.metadata)
    species_names = sorted(meta["species"].unique())
    query_names = [s.replace("_", " ") for s in species_names]

    log.info("Looking up %d species in NCBI taxonomy...", len(species_names))
    log.info("(First run downloads/builds the local NCBI taxonomy DB -- this can take a while.)")

    ncbi = NCBITaxa()
    name_to_taxid = ncbi.get_name_translator(query_names)

    resolved = {}
    unresolved = []
    for orig_name, query_name in zip(species_names, query_names):
        taxids = name_to_taxid.get(query_name)
        if taxids:
            resolved[orig_name] = taxids[0]
        else:
            unresolved.append(orig_name)

    if unresolved:
        log.warning(
            "%d/%d species names could not be resolved against NCBI taxonomy "
            "and will be absent from the species tree: %s",
            len(unresolved), len(species_names), unresolved[:10],
        )

    if len(resolved) < 4:
        raise SystemExit(
            f"Only {len(resolved)} species resolved -- too few to build a useful tree."
        )

    taxid_to_name = {v: k for k, v in resolved.items()}
    tree = ncbi.get_topology(list(resolved.values()))

    for leaf in tree.get_leaves():
        taxid = int(leaf.name)
        if taxid in taxid_to_name:
            leaf.name = taxid_to_name[taxid]

    tree.write(outfile=str(out_path), format=9)

    log.info("=" * 40)
    log.info("Resolved %d/%d species", len(resolved), len(species_names))
    log.info("Saved species tree: %s", out_path)
    log.info("Next: python build_gene_trees.py && python compare_trees.py")
    log.info("=" * 40)


if __name__ == "__main__":
    main()
