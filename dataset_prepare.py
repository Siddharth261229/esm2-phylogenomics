"""Stage 1: curate raw OrthoDB FASTA files into a balanced, deduplicated
multi-family dataset.

Usage:
    python dataset_prepare.py
    python dataset_prepare.py --max-per-family 100 --min-length 80
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from Bio import SeqIO

from esm2_phylo.config import DATA_DIR, FAMILY_FILES, OUTPUT_DIR
from esm2_phylo.logging_utils import get_logger
from esm2_phylo.utils import extract_species

log = get_logger(__name__)


def load_family_sequences(fasta_path: Path, family_label: str, min_seq_length: int, max_seqs: int, rng: random.Random):
    records = []
    seen_species = set()
    skipped_too_short = 0
    skipped_duplicate_species = 0
    unknown_species_count = 0

    for record in SeqIO.parse(fasta_path, "fasta"):
        if len(record.seq) < min_seq_length:
            skipped_too_short += 1
            continue

        species = extract_species(record.description)
        if species == "Unknown_species":
            unknown_species_count += 1

        if species in seen_species:
            skipped_duplicate_species += 1
            continue
        seen_species.add(species)

        record.id = f"{family_label}|{species}"
        record.name = ""
        record.description = ""
        records.append(record)

    if len(records) > max_seqs:
        records = rng.sample(records, max_seqs)

    if unknown_species_count:
        log.warning(
            "%s: %d records had unparsable organism_name headers "
            "(kept, but grouped/deduped under species='Unknown_species')",
            family_label, unknown_species_count,
        )
    log.debug(
        "%s: %d too short, %d duplicate-species skipped",
        family_label, skipped_too_short, skipped_duplicate_species,
    )

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-per-family", type=int, default=80)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel("DEBUG")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_records = []
    log.info("Processing families from %s", args.input_dir)

    for filename, family in FAMILY_FILES.items():
        fasta_file = args.input_dir / filename
        if not fasta_file.exists():
            log.warning("Missing file: %s -- skipping this family entirely", filename)
            continue

        records = load_family_sequences(
            fasta_file, family, args.min_length, args.max_per_family, rng
        )

        out_file = args.output_dir / f"{family}.fasta"
        SeqIO.write(records, out_file, "fasta")
        all_records.extend(records)

        log.info("%-8s -> %3d sequences", family, len(records))

    if not all_records:
        raise SystemExit(
            "No sequences were loaded. Check --input-dir points at the "
            "OrthoDB FASTA files and that filenames match esm2_phylo.config.FAMILY_FILES."
        )

    merged_file = args.output_dir / "all_families.fasta"
    SeqIO.write(all_records, merged_file, "fasta")

    log.info("=" * 40)
    log.info("Total sequences: %d", len(all_records))
    log.info("Merged FASTA: %s", merged_file)
    log.info("=" * 40)


if __name__ == "__main__":
    main()
