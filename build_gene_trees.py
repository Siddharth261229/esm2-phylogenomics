"""Stage 7a (NEW): build a gene tree per family directly from embedding
distances, using neighbor-joining -- no MSA or external aligner/tree
binaries required.

This finishes what was previously an orphaned feature: the repo shipped
``compare_trees.py`` (Robinson-Foulds gene-tree vs species-tree comparison)
but no script ever produced the ``results/<family>.tree`` files it needs,
and it depended on ``ete4`` which wasn't even listed in requirements.

Building gene trees from embedding cosine distances (rather than a
sequence alignment) keeps the whole pipeline consistent with the
project's alignment-free premise: if ESM-2 embeddings encode meaningful
evolutionary/functional structure, an NJ tree built from their pairwise
distances is itself a testable claim, and comparing it to a species tree
via `compare_trees.py` is a natural extension of the existing UMAP/
clustering analysis. This is a fast approximation, not a substitute for
proper ML/Bayesian phylogenetic inference (see README limitations).

Species names (not seq_ids) are used as tree leaf labels, since
``compare_trees.py`` needs matching taxa to a species tree.

Usage:
    python build_gene_trees.py
    python build_gene_trees.py --source kmer   # use k-mer distances instead
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Phylo import write as phylo_write
from scipy.spatial.distance import pdist, squareform

from esm2_phylo.config import ALL_FAMILIES_FASTA, EMBEDDINGS_FILE, METADATA_FILE, RESULTS_DIR
from esm2_phylo.logging_utils import get_logger
from esm2_phylo.utils import build_kmer_vocabulary, compute_kmer_vector

log = get_logger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_nj_tree(distance_matrix: np.ndarray, labels: list[str]):
    """Build a neighbor-joining tree from a square distance matrix."""
    # Biopython's DistanceMatrix wants a lower-triangular list of lists.
    lower_triangle = [list(distance_matrix[i, : i + 1]) for i in range(len(labels))]
    dm = DistanceMatrix(names=labels, matrix=lower_triangle)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # Biopython's NJ implementation names internal clades ("Inner1",
    # "Inner2", ...). These aren't bootstrap/support values, and several
    # Newick parsers (including ete4's default parser) choke on internal
    # node labels that aren't numeric. Strip them so the output is a
    # clean topology + branch lengths, which is also more portable.
    for clade in tree.get_nonterminals():
        clade.name = None

    return tree


def dedupe_labels_by_species(seq_ids: list[str], species: list[str], embeddings: np.ndarray):
    """compare_trees.py matches leaves by species name, so if two seq_ids
    in a family somehow share a species (shouldn't happen post
    dataset_prepare.py dedup, but guard anyway) keep only the first."""
    seen = set()
    keep_idx = []
    for i, sp in enumerate(species):
        if sp in seen:
            continue
        seen.add(sp)
        keep_idx.append(i)
    return keep_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--fasta", type=Path, default=ALL_FAMILIES_FASTA, help="Needed only for --source kmer")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--source", choices=["embeddings", "kmer"], default="embeddings")
    parser.add_argument("--k", type=int, default=3, help="k-mer size, only used with --source kmer")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata)

    if args.source == "embeddings":
        if not args.embeddings.exists():
            raise SystemExit(f"Embeddings not found: {args.embeddings}. Run generate_embeddings.py first.")
        X = np.load(args.embeddings)
        if len(meta) != len(X):
            raise SystemExit(f"Metadata rows ({len(meta)}) != embedding rows ({len(X)}). Rerun generate_embeddings.py.")
    else:
        from Bio import SeqIO
        if not args.fasta.exists():
            raise SystemExit(f"FASTA not found: {args.fasta}. Run dataset_prepare.py first.")
        kmer_to_idx = build_kmer_vocabulary(AMINO_ACIDS, args.k)
        seq_id_to_vec = {}
        for record in SeqIO.parse(args.fasta, "fasta"):
            seq_id_to_vec[record.id] = compute_kmer_vector(str(record.seq), kmer_to_idx, args.k, AMINO_ACIDS)
        X = np.vstack([seq_id_to_vec[sid] for sid in meta["seq_id"]])

    families = sorted(meta["family"].unique())
    log.info("Building gene trees for %d families (source=%s)...", len(families), args.source)

    written = []
    for family in families:
        idx = np.where((meta["family"] == family).values)[0]
        if len(idx) < 3:
            log.warning("%s: only %d sequences, need >=3 for a tree. Skipping.", family, len(idx))
            continue

        fam_meta = meta.iloc[idx].reset_index(drop=True)
        fam_X = X[idx]

        keep = dedupe_labels_by_species(fam_meta["seq_id"].tolist(), fam_meta["species"].tolist(), fam_X)
        if len(keep) < len(idx):
            log.warning("%s: dropped %d duplicate-species entries for tree leaves.", family, len(idx) - len(keep))
        fam_X = fam_X[keep]
        labels = fam_meta["species"].iloc[keep].tolist()

        D = squareform(pdist(fam_X, metric="cosine"))
        tree = build_nj_tree(D, labels)
        tree.rooted = False

        out_file = args.results_dir / f"{family}.tree"
        buf = StringIO()
        phylo_write(tree, buf, "newick")
        out_file.write_text(buf.getvalue())
        written.append(out_file)
        log.info("%-8s -> %s (%d leaves)", family, out_file.name, len(labels))

    log.info("=" * 40)
    log.info("Gene tree construction complete")
    for f in written:
        log.info("Saved: %s", f)
    if not written:
        log.warning("No gene trees were written.")
    else:
        log.info(
            "Next: provide results/species.tree (Newick, leaf names = species "
            "names matching those above) and run compare_trees.py."
        )
    log.info("=" * 40)


if __name__ == "__main__":
    main()
