"""Stage 2: generate ESM-2 embeddings for every sequence in the curated
merged FASTA.

Usage:
    python generate_embeddings.py
    python generate_embeddings.py --model esm2_t30_150M_UR50D --device cpu
    python generate_embeddings.py --model esm2_t6_8M_UR50D --batch-size 8

Note on GPUs: torch's CUDA backend only supports NVIDIA GPUs. If you have
an AMD GPU (e.g. an integrated Radeon chip) and are on Windows, torch will
not be able to use it here -- it will fall back to CPU automatically. If
you're on Linux with a supported discrete AMD card you could install a
ROCm build of torch instead, but for the model sizes used in this project
(8M-150M params) CPU inference is fine.
"""

from __future__ import annotations

import argparse
import os
os.environ["TORCH_HOME"] = "D:/torch_cache"
import sys
import time
from pathlib import Path

# Must be set BEFORE torch is imported. On Windows, torch's bundled
# OpenMP runtime can conflict with the one numpy/scipy (MKL) load,
# causing a hard segfault (exit code 0xC0000005) partway through a run
# rather than a catchable Python exception. KMP_DUPLICATE_LIB_OK=TRUE
# tells the runtime to tolerate the duplicate instead of aborting;
# OMP_NUM_THREADS=1 avoids the underlying thread-contention that
# triggers it in the first place. This is a known Windows-specific
# PyTorch/MKL packaging issue, not specific to this project.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from esm2_phylo.config import ALL_FAMILIES_FASTA, EMBEDDINGS_FILE, METADATA_FILE, MODEL_LAYERS
from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            log.warning("--device cuda requested but CUDA is not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    import esm  # imported lazily so --help doesn't require the package

    if not hasattr(esm.pretrained, model_name):
        available = ", ".join(MODEL_LAYERS.keys())
        raise SystemExit(f"Unknown model '{model_name}'. Available: {available}")

    log.info("Loading ESM-2 model: %s ...", model_name)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model = model.to(device)
    return model, alphabet


def read_records(fasta_file: Path, max_len: int):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.id
        try:
            family, species = header.split("|", 1)
        except ValueError:
            log.warning("Skipping malformed header: %s", header)
            continue

        seq = str(record.seq)
        truncated = len(seq) > max_len
        if truncated:
            seq = seq[:max_len]

        records.append(
            {
                "seq_id": header,
                "family": family,
                "species": species,
                "sequence": seq,
                "truncated": truncated,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", type=Path, default=ALL_FAMILIES_FASTA)
    parser.add_argument("--out-embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--out-metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--model", default="esm2_t12_35M_UR50D", choices=list(MODEL_LAYERS.keys()))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel("DEBUG")

    if not args.fasta.exists():
        raise SystemExit(f"Input FASTA not found: {args.fasta}. Run dataset_prepare.py first.")

    device = resolve_device(args.device)
    log.info("Using device: %s", device)

    model, alphabet = load_model(args.model, device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = MODEL_LAYERS[args.model]

    records = read_records(args.fasta, args.max_len)
    n_truncated = sum(r["truncated"] for r in records)
    if n_truncated:
        log.warning("%d/%d sequences were truncated to %d residues", n_truncated, len(records), args.max_len)
    log.info("Loaded %d sequences", len(records))

    all_embeddings = []
    metadata_rows = []
    total = len(records)
    start_time = time.time()

    for start in range(0, total, args.batch_size):
        batch = records[start : start + args.batch_size]
        batch_data = [(item["seq_id"], item["sequence"]) for item in batch]
        _, _, tokens = batch_converter(batch_data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)

        token_reps = results["representations"][repr_layer]

        for i, item in enumerate(batch):
            seq_len = len(item["sequence"])
            embedding = token_reps[i, 1 : seq_len + 1].mean(0).cpu().numpy()
            all_embeddings.append(embedding)
            metadata_rows.append(
                {
                    "seq_id": item["seq_id"],
                    "family": item["family"],
                    "species": item["species"],
                    "model": args.model,
                    "embedding_dim": embedding.shape[0],
                }
            )

        done = len(all_embeddings)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        sys.stdout.write(f"\r[{done}/{total}] {rate:.1f} seq/s, ETA {eta:5.0f}s   ")
        sys.stdout.flush()

    print()  # newline after progress bar

    embeddings_array = np.vstack(all_embeddings)
    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_embeddings, embeddings_array)

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(args.out_metadata, index=False)

    log.info("=" * 40)
    log.info("Embedding generation complete")
    log.info("Model: %s (layer %d)", args.model, repr_layer)
    log.info("Embeddings shape: %s", embeddings_array.shape)
    log.info("Saved: %s", args.out_embeddings)
    log.info("Saved: %s", args.out_metadata)
    log.info("=" * 40)


if __name__ == "__main__":
    main()
