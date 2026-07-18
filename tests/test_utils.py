import numpy as np
import pandas as pd
import pytest

from esm2_phylo.utils import (
    build_kmer_vocabulary,
    cluster_purity,
    compute_kmer_vector,
    extract_species,
    flag_minority_clusters,
    robust_z_outlier_ids,
    species_overlap_summary,
)


class TestExtractSpecies:
    def test_extracts_and_sanitizes(self):
        desc = '{"pub_og_id":"x","organism_name":"Homo sapiens","other":1}'
        assert extract_species(desc) == "Homo_sapiens"

    def test_strips_problem_characters(self):
        desc = '{"organism_name":"Species (with), parens/slash"}'
        result = extract_species(desc)
        assert "(" not in result and ")" not in result
        assert "," not in result and "/" not in result

    def test_missing_field_returns_unknown(self):
        assert extract_species("no organism info here") == "Unknown_species"


class TestKmer:
    def test_vocabulary_size(self):
        vocab = build_kmer_vocabulary("ACDE", k=2)
        assert len(vocab) == 16  # 4^2

    def test_vector_sums_to_one(self):
        vocab = build_kmer_vocabulary("ACDEFGHIKLMNPQRSTVWY", k=3)
        vec = compute_kmer_vector("ACDEFGHIKL", vocab, k=3, amino_acids="ACDEFGHIKLMNPQRSTVWY")
        assert vec.sum() == pytest.approx(1.0)

    def test_skips_invalid_residues_without_crashing(self):
        vocab = build_kmer_vocabulary("ACDE", k=2)
        # 'X' and 'Z' are not in the alphabet; must not raise KeyError.
        vec = compute_kmer_vector("AXCZDE", vocab, k=2, amino_acids="ACDE")
        assert vec.shape == (16,)
        assert vec.sum() == pytest.approx(1.0)

    def test_empty_sequence_returns_zero_vector(self):
        vocab = build_kmer_vocabulary("ACDE", k=3)
        vec = compute_kmer_vector("", vocab, k=3, amino_acids="ACDE")
        assert vec.sum() == 0


class TestClusterPurity:
    def test_perfect_clustering(self):
        clusters = np.array([0, 0, 1, 1])
        labels = np.array(["A", "A", "B", "B"])
        assert cluster_purity(clusters, labels) == 1.0

    def test_worst_case_two_clusters_two_labels_evenly_split(self):
        clusters = np.array([0, 0, 1, 1])
        labels = np.array(["A", "B", "A", "B"])
        assert cluster_purity(clusters, labels) == 0.5


class TestMinorityClusters:
    def test_flags_small_cluster(self):
        clusters = np.array([0] * 9 + [1])
        seq_ids = [f"s{i}" for i in range(10)]
        flagged = flag_minority_clusters(clusters, seq_ids, minority_threshold=0.2)
        assert flagged == {"s9"}

    def test_no_flags_when_balanced(self):
        clusters = np.array([0, 0, 0, 1, 1, 1])
        seq_ids = [f"s{i}" for i in range(6)]
        flagged = flag_minority_clusters(clusters, seq_ids, minority_threshold=0.2)
        assert flagged == set()


class TestRobustZOutliers:
    def test_flags_obvious_outlier(self):
        rng = np.random.default_rng(0)
        cluster = rng.normal(0, 0.1, size=(20, 5))
        outlier = np.ones((1, 5)) * 50
        embeddings = np.vstack([cluster, outlier])
        seq_ids = [f"s{i}" for i in range(21)]
        flagged = robust_z_outlier_ids(embeddings, seq_ids, z_threshold=3.0)
        assert "s20" in flagged

    def test_no_outliers_in_tight_cluster(self):
        rng = np.random.default_rng(0)
        embeddings = rng.normal(0, 0.01, size=(10, 5))
        seq_ids = [f"s{i}" for i in range(10)]
        flagged = robust_z_outlier_ids(embeddings, seq_ids, z_threshold=3.0)
        assert flagged == set()

    def test_too_few_points_returns_empty(self):
        embeddings = np.zeros((2, 5))
        assert robust_z_outlier_ids(embeddings, ["a", "b"]) == set()


class TestSpeciesOverlap:
    def test_shared_species_computed_correctly(self):
        meta = pd.DataFrame(
            {
                "family": ["A", "A", "B", "B", "C"],
                "species": ["sp1", "sp2", "sp1", "sp3", "sp1"],
            }
        )
        result = species_overlap_summary(meta)
        assert result["n_shared_species"] == 1
        assert result["shared_species"] == ["sp1"]
        assert result["per_family_species_counts"] == {"A": 2, "B": 2, "C": 1}
