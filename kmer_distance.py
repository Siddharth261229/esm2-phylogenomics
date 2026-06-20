import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from Bio import SeqIO

from scipy.spatial.distance import (
pdist,
squareform
)

from sklearn.cluster import (
AgglomerativeClustering,
KMeans
)

import umap

import matplotlib.pyplot as plt

# =====================================================

# CONFIG

# =====================================================

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
K = 3

# =====================================================

# FAMILY COLORS

# =====================================================

FAMILY_COLORS = {
"HSP70": "#e41a1c",
"RPS3": "#377eb8",
"GAPDH": "#4daf4a",
"CYTC": "#984ea3"
}

# =====================================================

# GENERATE ALL POSSIBLE 3-MERS

# =====================================================

print("Generating k-mer vocabulary...")

kmers = [
"".join(k)
for k in itertools.product(
AA,
repeat=K
)
]

kmer_to_idx = {
kmer: i
for i, kmer in enumerate(kmers)
}

print(
f"Vocabulary size: {len(kmers)}"
)

# =====================================================

# K-MER VECTOR FUNCTION

# =====================================================

def compute_kmer_vector(sequence):
    sequence = sequence.upper()

    vec = np.zeros(len(kmers), dtype=np.float32)

    valid_count = 0

    for i in range(len(sequence) - K + 1):
        kmer = sequence[i : i + K]

        if any(aa not in AA for aa in kmer):
            continue

        idx = kmer_to_idx[kmer]
        vec[idx] += 1
        valid_count += 1

    if valid_count > 0:
        vec /= valid_count

    return vec


# =====================================================

# LOAD FASTA FILES

# =====================================================

print("\nReading FASTA files...")

vectors = []
labels = []
headers = []

for fasta_file in DATA_DIR.glob("*.fasta"):
    if "_aligned" in fasta_file.name:
        continue

    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.id
        family = fasta_file.stem

        seq = str(record.seq)
        vec = compute_kmer_vector(seq)
        vectors.append(vec)
        labels.append(family)
        headers.append(header)

X_kmer = np.vstack(vectors)

print(
f"Loaded {X_kmer.shape[0]} sequences"
)
print(
f"K-mer matrix shape: {X_kmer.shape}"
)

# If dataset is large, sample a subset for k-mer baseline to save memory
MAX_KMER_SAMPLES = 2000
n_sequences = X_kmer.shape[0]
if n_sequences > MAX_KMER_SAMPLES:
    print(f"Dataset large ({n_sequences} seqs). Sampling {MAX_KMER_SAMPLES} for k-mer baseline.")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_sequences, size=MAX_KMER_SAMPLES, replace=False)
    X_kmer_for_analysis = X_kmer[sample_idx]
    labels_kmer = np.array(labels)[sample_idx]
else:
    X_kmer_for_analysis = X_kmer
    labels_kmer = np.array(labels)

# =====================================================

# K-MER DISTANCES

# =====================================================

print("\nComputing cosine distances...")

D = squareform(pdist(X_kmer_for_analysis, metric="cosine"))

# =====================================================

# PURITY FUNCTION

# =====================================================

def cluster_purity(
clusters,
labels
):
    labels = np.array(labels)

    total = len(labels)
    correct = 0

    for cluster_id in np.unique(clusters):
        idx = clusters == cluster_id
        cluster_labels = labels[idx]
        majority = pd.Series(cluster_labels).value_counts().iloc[0]
        correct += majority

    return correct / total

# =====================================================

# K-MER CLUSTERING

# =====================================================

print(
"\nRunning k-mer clustering..."
)

kmer_model = (
AgglomerativeClustering(
n_clusters=4,
metric="precomputed",
linkage="average"
)
)

kmer_clusters = (
kmer_model.fit_predict(D)
)

kmer_purity = (
cluster_purity(
    kmer_clusters,
    labels_kmer
)
)

print(
f"\nk-mer baseline clustering purity: "
f"{kmer_purity:.3f}"
)

# =====================================================

# LOAD ESM EMBEDDINGS

# =====================================================

print(
"\nLoading ESM embeddings..."
)

X_esm = np.load(
OUTPUT_DIR /
"embeddings.npy"
)

meta = pd.read_csv(
OUTPUT_DIR /
"metadata.csv"
)

esm_labels = (
meta["family"].values
)

# =====================================================

# ESM CLUSTERING

# =====================================================

kmeans = KMeans(
n_clusters=4,
random_state=42,
n_init=20
)

esm_clusters = (
kmeans.fit_predict(X_esm)
)

esm_purity = (
cluster_purity(
esm_clusters,
esm_labels
)
)

print(
f"ESM-2 embedding clustering purity: "
f"{esm_purity:.3f}"
)

# =====================================================

# UMAP

# =====================================================

print(
"\nRunning UMAP..."
)

umap_kmer = umap.UMAP(
n_neighbors=15,
min_dist=0.1,
metric="cosine",
random_state=42
).fit_transform(X_kmer_for_analysis)

umap_esm = umap.UMAP(
n_neighbors=15,
min_dist=0.1,
metric="cosine",
random_state=42
).fit_transform(X_esm)

# =====================================================

# PLOT

# =====================================================

plt.style.use(
"dark_background"
)

fig, axes = plt.subplots(
1,
2,
figsize=(16, 7)
)

# ---------- KMER ----------

for family in sorted(
set(labels_kmer)
):
    idx = np.array(labels_kmer) == family

    axes[0].scatter(
        umap_kmer[idx, 0],
        umap_kmer[idx, 1],
        label=family,
        s=40,
        alpha=0.8,
        color=FAMILY_COLORS.get(family),
    )

axes[0].set_title(
"3-mer Frequency Baseline"
)

# ---------- ESM ----------

for family in sorted(
set(esm_labels)
):
    idx = esm_labels == family

    axes[1].scatter(
        umap_esm[idx, 0],
        umap_esm[idx, 1],
        label=family,
        s=40,
        alpha=0.8,
        color=FAMILY_COLORS.get(family),
    )

axes[1].set_title(
"ESM-2 Embeddings"
)

axes[1].legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "umap_comparison.png", dpi=150)
plt.show()

summary = (
f"k-mer purity = "
f"{kmer_purity:.3f}\n"
f"ESM-2 purity = "
f"{esm_purity:.3f}"
)

fig.text(
0.5,
0.02,
summary,
ha="center",
fontsize=12,
bbox=dict(
facecolor="black",
alpha=0.7
)
)

plt.tight_layout()

outfile = (
PLOTS_DIR /
"kmer_vs_esm2.png"
)

plt.savefig(
outfile,
dpi=300,
bbox_inches="tight"
)

plt.close()

print(
f"\nSaved: {outfile}"
)
