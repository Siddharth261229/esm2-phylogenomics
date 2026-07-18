"""Stage 7b: compare each family's gene tree to a reference species tree
using Robinson-Foulds distance.

Prerequisites (this is what makes the feature previously "orphaned" --
neither file existed anywhere in the original repo):
  1. results/<FAMILY>.tree  -- generate with build_gene_trees.py
  2. results/species.tree   -- a Newick species tree whose LEAF NAMES
     match the species names used above. You need to supply this
     yourself, e.g.:
       - a subset of a public reference like TimeTree (timetree.org)
       - NCBI Taxonomy via ete3/ete4's NCBITaxa (see
         fetch_species_tree.py for a starting point -- it needs internet
         access to NCBI on your machine, not available in this sandbox)
       - your own curated tree for the species in output/metadata.csv

Usage:
    python compare_trees.py
    python compare_trees.py --species-tree my_species.tree
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ete4 import Tree

from esm2_phylo.config import RESULTS_DIR
from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)


def load_tree(path: Path) -> Tree:
    return Tree(open(path).read(), parser=0)


def prune_to_common_leaves(gene_tree: Tree, species_tree: Tree):
    gene_leaves = {leaf.name for leaf in gene_tree.leaves()}
    species_leaves = {leaf.name for leaf in species_tree.leaves()}
    common = gene_leaves & species_leaves

    if len(common) < 4:
        return None, None, common, gene_leaves, species_leaves

    g = gene_tree.copy()
    s = species_tree.copy()
    g.prune(list(common), preserve_branch_length=True)
    s.prune(list(common), preserve_branch_length=True)
    return g, s, common, gene_leaves, species_leaves


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--species-tree", type=Path, default=None,
                         help="Defaults to <results-dir>/species.tree")
    args = parser.parse_args()

    species_tree_path = args.species_tree or (args.results_dir / "species.tree")

    gene_tree_files = sorted(args.results_dir.glob("*.tree"))
    gene_tree_files = [f for f in gene_tree_files if f.name != "species.tree"]

    if not gene_tree_files:
        raise SystemExit(
            f"No gene tree files found in {args.results_dir}. "
            "Run build_gene_trees.py first to generate results/<FAMILY>.tree files."
        )

    if not species_tree_path.exists():
        raise SystemExit(
            f"Species tree not found: {species_tree_path}\n"
            "This script needs a reference species tree to compare gene trees "
            "against -- it is not something the pipeline can generate on its "
            "own without an external taxonomy source. See this script's "
            "module docstring (or fetch_species_tree.py) for how to obtain one, "
            "then place it at that path."
        )

    species_tree = load_tree(species_tree_path)
    log.info("Loaded species tree: %s (%d leaves)", species_tree_path, len(list(species_tree.leaves())))

    results = []
    for gene_tree_file in gene_tree_files:
        family = gene_tree_file.stem
        gene_tree = load_tree(gene_tree_file)

        pruned_gene, pruned_species, common, gene_leaves, species_leaves = prune_to_common_leaves(
            gene_tree, species_tree
        )

        if pruned_gene is None:
            log.warning(
                "%s: only %d overlapping leaves with the species tree "
                "(need >=4 for a meaningful RF comparison). "
                "Gene tree has %d leaves, species tree has %d leaves -- "
                "check that species names match exactly. Skipping.",
                family, len(common), len(gene_leaves), len(species_leaves),
            )
            results.append({"family": family, "status": "skipped_insufficient_overlap", "n_common_leaves": len(common)})
            continue

        rf = pruned_gene.compare(pruned_species, unrooted=True)

        results.append(
            {
                "family": family,
                "status": "ok",
                "n_common_leaves": len(common),
                "rf_distance": rf["rf"],
                "rf_max": rf["max_rf"],
                "normalized_rf": rf["rf"] / rf["max_rf"] if rf["max_rf"] else None,
            }
        )
        log.info(
            "%-8s common_leaves=%-3d RF=%s/%s (normalized=%.3f)",
            family, len(common), rf["rf"], rf["max_rf"],
            rf["rf"] / rf["max_rf"] if rf["max_rf"] else float("nan"),
        )

    out_file = args.results_dir / "tree_comparison_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    log.info("Saved: %s", out_file)


if __name__ == "__main__":
    main()
