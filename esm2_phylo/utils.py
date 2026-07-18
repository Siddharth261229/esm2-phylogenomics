"""Pure, unit-testable helper functions shared across pipeline stages.

These were previously inlined in individual scripts with no test coverage.
Pulling them out here means they can be exercised by tests/test_utils.py
without needing torch, ESM-2 weights, or a GPU.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd


def extract_species(description: str) -> str:
    """Extract organism_name from an OrthoDB FASTA header.

    OrthoDB headers embed a JSON-ish blob, e.g.:
        '{"organism_name":"Homo sapiens", ...}'

    Returns "Unknown_species" if no organism_name field is found, so
    callers can decide how to handle/report unparsed headers rather than
    silently losing them.
    """
    match = re.search(r'"organism_name":"([^"]+)"', description)
    if not match:
        return "Unknown_species"

    species = match.group(1)
    species = (
        species.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )
    return species


def build_kmer_vocabulary(amino_acids: str, k: int) -> dict[str, int]:
    """Build the full k-mer -> index vocabulary for a given alphabet/k."""
    kmers = ["".join(t) for t in itertools.product(amino_acids, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def compute_kmer_vector(
    sequence: str,
    kmer_to_idx: dict[str, int],
    k: int,
    amino_acids: str,
) -> np.ndarray:
    """Compute a normalized k-mer frequency vector for a protein sequence.

    Any k-mer containing a character outside ``amino_acids`` (e.g. 'X',
    'B', 'U', lowercase masked residues) is skipped rather than causing
    a KeyError, and the final vector is normalized by the count of valid
    k-mers actually counted (not sequence length), so it remains a proper
    frequency distribution even when some positions are skipped.
    """
    sequence = sequence.upper()
    vec = np.zeros(len(kmer_to_idx), dtype=np.float32)
    valid_count = 0
    aa_set = set(amino_acids)

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if any(aa not in aa_set for aa in kmer):
            continue
        vec[kmer_to_idx[kmer]] += 1
        valid_count += 1

    if valid_count > 0:
        vec /= valid_count

    return vec


def cluster_purity(clusters: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of samples whose cluster's majority label matches them."""
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    total = len(labels)
    if total == 0:
        return float("nan")

    correct = 0
    for cluster_id in np.unique(clusters):
        idx = clusters == cluster_id
        cluster_labels = labels[idx]
        majority = pd.Series(cluster_labels).value_counts().iloc[0]
        correct += majority
    return correct / total


def flag_minority_clusters(
    clusters: np.ndarray,
    seq_ids: list[str],
    minority_threshold: float,
) -> set[str]:
    """Flag sequence IDs belonging to minority KMeans clusters.

    Extracted verbatim (behaviorally) from the original detect_paralogs.py
    logic so it can be unit tested in isolation.
    """
    n = len(clusters)
    flagged: set[str] = set()
    cluster_sizes = pd.Series(clusters).value_counts()

    for cluster_id, size in cluster_sizes.items():
        fraction = size / n
        if fraction < minority_threshold:
            idx = np.where(clusters == cluster_id)[0]
            flagged.update(seq_ids[i] for i in idx)
    return flagged


def robust_z_outlier_ids(
    embeddings: np.ndarray,
    seq_ids: list[str],
    z_threshold: float = 3.0,
) -> set[str]:
    """Flag sequences whose distance to the family centroid is a robust
    z-score outlier (median absolute deviation based).

    This is a second, statistically motivated signal to combine with the
    KMeans minority-cluster heuristic, which by construction always flags
    *something* even when a family has no real outliers (with k=2/3 it
    will always produce a minority cluster of some size). MAD-based
    distance thresholding instead asks "is this point unusually far from
    the family's center relative to the family's own spread", and can
    legitimately flag zero sequences.
    """
    if len(embeddings) < 4:
        return set()

    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)

    median = np.median(dists)
    mad = np.median(np.abs(dists - median))
    if mad == 0:
        return set()

    # 1.4826 makes MAD a consistent estimator of the standard deviation
    # under normality, the standard robust-z convention.
    robust_z = 0.6745 * (dists - median) / mad
    flagged_idx = np.where(robust_z > z_threshold)[0]
    return {seq_ids[i] for i in flagged_idx}


def species_overlap_summary(meta: pd.DataFrame) -> dict[str, object]:
    """Per-family species counts and the species shared across all families.

    This is the logic from the original check.py, pulled into a reusable,
    testable function that returns data instead of only printing it.
    """
    per_family = {}
    shared: set[str] | None = None

    for fam in meta["family"].unique():
        species_set = set(meta.loc[meta["family"] == fam, "species"])
        per_family[fam] = len(species_set)
        shared = species_set if shared is None else shared & species_set

    return {
        "per_family_species_counts": per_family,
        "n_shared_species": len(shared) if shared else 0,
        "shared_species": sorted(shared) if shared else [],
    }
