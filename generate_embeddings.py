from pathlib import Path

import esm
import torch
import numpy as np
import pandas as pd

from Bio import SeqIO

# =====================================================
# CONFIG
# =====================================================

FASTA_FILE = "output/all_families.fasta"

EMBEDDING_FILE = "output/embeddings.npy"
METADATA_FILE = "output/metadata.csv"

BATCH_SIZE = 4
MAX_LEN = 1022

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading ESM-2 model...")

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

model.eval()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

batch_converter = alphabet.get_batch_converter()

print(f"Using device: {device}")

# =====================================================
# READ FASTA
# =====================================================

records = []

for record in SeqIO.parse(FASTA_FILE, "fasta"):

    header = record.id

    try:
        family, species = header.split("|", 1)
    except ValueError:
        print(f"Skipping malformed header: {header}")
        continue

    seq = str(record.seq)

    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]

    records.append(
        {
            "seq_id": header,
            "family": family,
            "species": species,
            "sequence": seq,
        }
    )

print(f"\nLoaded {len(records)} sequences")

# =====================================================
# EMBEDDINGS
# =====================================================

all_embeddings = []

metadata_rows = []

total = len(records)

for start in range(0, total, BATCH_SIZE):

    batch = records[start:start + BATCH_SIZE]

    batch_data = [
        (item["seq_id"], item["sequence"])
        for item in batch
    ]

    labels, strs, tokens = batch_converter(batch_data)

    tokens = tokens.to(device)

    with torch.no_grad():

        results = model(
            tokens,
            repr_layers=[6],
            return_contacts=False
        )

    token_reps = results["representations"][6]

    for i, item in enumerate(batch):

        seq_len = len(item["sequence"])

        embedding = (
            token_reps[i, 1:seq_len + 1]
            .mean(0)
            .cpu()
            .numpy()
        )

        all_embeddings.append(embedding)

        metadata_rows.append(
            {
                "seq_id": item["seq_id"],
                "family": item["family"],
                "species": item["species"],
            }
        )

        print(
            f"[{len(all_embeddings)}/{total}] "
            f"{item['seq_id']}"
        )

# =====================================================
# SAVE
# =====================================================

embeddings_array = np.vstack(all_embeddings)

np.save(
    EMBEDDING_FILE,
    embeddings_array
)

metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_csv(
    METADATA_FILE,
    index=False
)

print("\n===================================")
print("Embedding generation complete")
print(f"Embeddings shape: {embeddings_array.shape}")
print(f"Saved: {EMBEDDING_FILE}")
print(f"Saved: {METADATA_FILE}")
print("===================================")