

from Bio import SeqIO
from pathlib import Path
import random
import re

# =====================================================
# CONFIG
# =====================================================

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SEQS_PER_FAMILY = 80
MIN_SEQ_LENGTH = 100

# Input FASTA -> Label
FAMILY_FILES = {
    "HSP_70.fasta": "HSP70",
    "RPS3.fasta": "RPS3",
    "GAPDH.fasta": "GAPDH",
    "CYT_C.fasta": "CYTC",

}

random.seed(42)

# =====================================================
# HELPERS
# =====================================================

def extract_species(description):
    """
    Extract organism_name from OrthoDB FASTA header.
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


def load_family_sequences(fasta_path, family_label):

    records = []
    seen_species = set()

    for record in SeqIO.parse(fasta_path, "fasta"):

        if len(record.seq) < MIN_SEQ_LENGTH:
            continue

        species = extract_species(record.description)

        # keep only one sequence per species
        if species in seen_species:
            continue

        seen_species.add(species)

        record.id = f"{family_label}|{species}"
        record.name = ""
        record.description = ""

        records.append(record)

    # sample if too many
    if len(records) > MAX_SEQS_PER_FAMILY:
        records = random.sample(records, MAX_SEQS_PER_FAMILY)

    return records


# =====================================================
# MAIN
# =====================================================

all_records = []

print("\nProcessing families:\n")

for filename, family in FAMILY_FILES.items():

    fasta_file = INPUT_DIR / filename

    if not fasta_file.exists():
        print(f"[WARNING] Missing file: {filename}")
        continue

    records = load_family_sequences(
        fasta_file,
        family
    )

    out_file = OUTPUT_DIR / f"{family}.fasta"

    SeqIO.write(records, out_file, "fasta")

    all_records.extend(records)

    print(
        f"{family:<8} -> {len(records):3d} sequences"
    )

merged_file = OUTPUT_DIR / "all_families.fasta"

SeqIO.write(
    all_records,
    merged_file,
    "fasta"
)

print("\n=================================")
print(f"Total sequences: {len(all_records)}")
print(f"Merged FASTA: {merged_file}")
print("=================================\n")