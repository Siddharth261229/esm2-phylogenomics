"""Run the full esm2-phylogenomics pipeline end to end.

Stages (in order):
  1. dataset_prepare.py    - curate raw OrthoDB FASTA -> balanced dataset
  2. generate_embeddings.py - ESM-2 embeddings (needs torch + fair-esm)
  3. cluster_visualise.py   - UMAP plots
  4. train_mlp_classifier.py - cross-validated family classifier
  5. detect_paralogs.py     - outlier / candidate-paralog detection
  6. kmer_distance.py       - alignment-free k-mer baseline comparison
  7. build_gene_trees.py    - NJ gene trees from embeddings (optional)
  8. compare_trees.py       - RF distance vs species tree (optional, needs
                              results/species.tree supplied separately)

Usage:
    python run_pipeline.py                     # run stages 1-6
    python run_pipeline.py --with-trees         # also run 7-8 (needs species.tree)
    python run_pipeline.py --only embeddings classify
    python run_pipeline.py --skip embeddings    # e.g. if embeddings.npy already exists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)

STAGES = [
    ("dataset", "dataset_prepare.py"),
    ("embeddings", "generate_embeddings.py"),
    ("visualize", "cluster_visualise.py"),
    ("classify", "train_mlp_classifier.py"),
    ("paralogs", "detect_paralogs.py"),
    ("kmer", "kmer_distance.py"),
]

TREE_STAGES = [
    ("gene_trees", "build_gene_trees.py"),
    ("compare_trees", "compare_trees.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=None, help="Run only these stage names")
    parser.add_argument("--skip", nargs="+", default=[], help="Skip these stage names")
    parser.add_argument("--with-trees", action="store_true", help="Also run gene_trees + compare_trees")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing")
    args = parser.parse_args()

    stages = list(STAGES)
    if args.with_trees:
        stages += TREE_STAGES

    if args.only:
        stages = [(name, script) for name, script in stages if name in args.only]
    stages = [(name, script) for name, script in stages if name not in args.skip]

    if not stages:
        raise SystemExit("No stages selected to run.")

    log.info("Pipeline plan: %s", " -> ".join(name for name, _ in stages))

    if args.dry_run:
        return

    for name, script in stages:
        log.info("=" * 50)
        log.info("STAGE: %s (%s)", name, script)
        log.info("=" * 50)
        start = time.time()
        result = subprocess.run([sys.executable, script])
        elapsed = time.time() - start
        if result.returncode != 0:
            log.error("Stage '%s' failed (exit code %d) after %.1fs. Stopping pipeline.", name, result.returncode, elapsed)
            sys.exit(result.returncode)
        log.info("Stage '%s' completed in %.1fs", name, elapsed)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
