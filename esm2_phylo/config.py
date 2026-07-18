"""Central configuration for the esm2-phylogenomics pipeline.

All paths and hyperparameters that used to be scattered as bare module-level
constants across individual scripts now live here. Every stage script still
accepts CLI overrides (see each script's --help), but this file defines the
sane defaults and is the single source of truth for directory layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# =====================================================
# PROJECT ROOT & DIRECTORY LAYOUT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"

ALL_FAMILIES_FASTA = OUTPUT_DIR / "all_families.fasta"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

# Input FASTA -> standardized family label.
# NOTE: these labels are the single source of truth for family naming.
# Every downstream stage (embeddings, kmer baseline, tree building) must
# use these labels, not raw filenames, or metrics silently stop being
# comparable across stages (see CHANGELOG for the bug this fixes).
FAMILY_FILES = {
    "HSP_70.fasta": "HSP70",
    "RPS3.fasta": "RPS3",
    "GAPDH.fasta": "GAPDH",
    "CYT_C.fasta": "CYTC",
}

FAMILY_COLORS = {
    "HSP70": "#e41a1c",
    "RPS3": "#377eb8",
    "GAPDH": "#4daf4a",
    "CYTC": "#984ea3",
}


@dataclass
class DatasetConfig:
    max_seqs_per_family: int = 80
    min_seq_length: int = 100
    random_seed: int = 42


@dataclass
class EmbeddingConfig:
    # Available ESM-2 checkpoints, smallest to largest. Bigger = better
    # signal but much slower on CPU. t6_8M is the previous default and
    # is fine for a CPU-only / small-VRAM machine (e.g. an integrated
    # GPU with no CUDA support, which torch will not be able to use).
    model_name: str = "esm2_t12_35M_UR50D"
    batch_size: int = 4
    max_len: int = 1022
    repr_layer: int | None = None  # None -> inferred from model name
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass
class ClassifierConfig:
    hidden_layer_sizes: tuple = (256, 128)
    max_iter: int = 500
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    standardize: bool = True


@dataclass
class ParalogConfig:
    minority_threshold: float = 0.20
    lof_contamination: float = 0.10
    random_state: int = 42


@dataclass
class KmerConfig:
    k: int = 3
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"
    max_samples: int = 2000
    random_state: int = 42


def ensure_dirs() -> None:
    """Create all standard output directories if they don't exist."""
    for d in (OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Layer index to use per ESM-2 checkpoint (final layer = number of layers).
MODEL_LAYERS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
}
