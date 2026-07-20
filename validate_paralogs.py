"""Validate detect_paralogs.py's flagged sequences against signals that are
independent of embedding geometry: sequence length, and raw OrthoDB header
annotations (og description, gene id).

Rationale: detect_paralogs.py flags sequences based purely on where they
sit in embedding space. That's useful, but it's also circular as evidence
on its own -- an embedding-space outlier is only interesting if it also
looks different by some independent measure. This script checks two:

1. SEQUENCE LENGTH (primary, quantitative): distinct paralogous isoforms
   (e.g. organelle-targeted vs cytosolic) often differ in length due to
   signal/transit peptides. For each family, a Mann-Whitney U test compares
   the length distribution of flagged vs non-flagged sequences. This uses
   only the curated FASTA already produced by dataset_prepare.py -- no
   extra data needed.

2. RAW HEADER ANNOTATION (secondary, qualitative, weaker): OrthoDB's
   per-record `description` field is often a generic automated annotation
   ("uncharacterized protein LOC...") rather than a reliable isoform label,
   so this is reported as supporting context to eyeball, not as a
   statistical test. Requires the raw data/*.fasta files (same ones
   dataset_prepare.py reads) to recover each curated sequence's original
   header, since dataset_prepare.py itself discards this metadata when it
   rewrites headers to `FAMILY|species`.

Usage:
    python validate_paralogs.py
    python validate_paralogs.py --results-dir runs/esm2_t12_35M_n200 --fasta runs/... (see --help)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from scipy.stats import mannwhitneyu

from esm2_phylo.config import DATA_DIR, FAMILY_FILES, RESULTS_DIR
from esm2_phylo.logging_utils import get_logger
from esm2_phylo.utils import extract_species

log = get_logger(__name__)


def load_curated_lengths(all_families_fasta: Path) -> pd.DataFrame:
    rows = []
    for record in SeqIO.parse(all_families_fasta, "fasta"):
        try:
            family, species = record.id.split("|", 1)
        except ValueError:
            continue
        rows.append({"seq_id": record.id, "family": family, "species": species, "length": len(record.seq)})
    return pd.DataFrame(rows)


def rebuild_raw_annotation_lookup(data_dir: Path) -> dict[tuple[str, str], dict]:
    """Recover each curated sequence's original description/pub_gene_id by
    re-running the same species-first-occurrence-dedup logic
    dataset_prepare.py used, keyed on (family, species). Deterministic as
    long as the raw FASTA file order hasn't changed since dataset_prepare.py
    last ran.
    """
    lookup = {}
    for filename, family in FAMILY_FILES.items():
        fasta_path = data_dir / filename
        if not fasta_path.exists():
            log.warning("Raw file not found, skipping annotation lookup for %s: %s", family, fasta_path)
            continue

        seen_species = set()
        for record in SeqIO.parse(fasta_path, "fasta"):
            species = extract_species(record.description)
            if species in seen_species:
                continue
            seen_species.add(species)

            # description field looks like: '{"pub_og_id":"...","description":"...","pub_gene_id":"..."}'
            import json
            try:
                header_json = record.description.split(" ", 1)[1]
                fields = json.loads(header_json)
            except (IndexError, ValueError):
                fields = {}

            lookup[(family, species)] = {
                "raw_description": fields.get("description", ""),
                "pub_gene_id": fields.get("pub_gene_id", ""),
            }
    return lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                         help="Directory containing paralog_candidates_detail.csv (an archived run dir, or the default results/)")
    parser.add_argument("--fasta", type=Path, default=None,
                         help="Curated all_families.fasta matching that results dir (default: output/all_families.fasta)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=None,
                         help="Output CSV path (default: <results-dir>/paralog_validation.csv)")
    args = parser.parse_args()

    detail_file = args.results_dir / "paralog_candidates_detail.csv"
    if not detail_file.exists():
        raise SystemExit(f"Not found: {detail_file}. Run detect_paralogs.py first (or point --results-dir at an archived run).")

    fasta_file = args.fasta or Path("output/all_families.fasta")
    if not fasta_file.exists():
        raise SystemExit(f"Curated FASTA not found: {fasta_file}. Pass --fasta explicitly if using an archived run.")

    out_file = args.out or (args.results_dir / "paralog_validation.csv")

    log.info("Loading flagged sequences from %s", detail_file)
    flags = pd.read_csv(detail_file)

    log.info("Loading sequence lengths from %s", fasta_file)
    lengths = load_curated_lengths(fasta_file)

    log.info("Rebuilding raw header annotation lookup from %s", args.data_dir)
    annotation_lookup = rebuild_raw_annotation_lookup(args.data_dir)

    merged = lengths.merge(flags[["seq_id", "high_confidence"]], on="seq_id", how="left")
    merged["high_confidence"] = merged["high_confidence"].fillna(False).astype(bool)
    merged["raw_description"] = merged.apply(
        lambda r: annotation_lookup.get((r["family"], r["species"]), {}).get("raw_description", ""), axis=1
    )
    merged["pub_gene_id"] = merged.apply(
        lambda r: annotation_lookup.get((r["family"], r["species"]), {}).get("pub_gene_id", ""), axis=1
    )

    log.info("=" * 60)
    log.info("LENGTH-BASED VALIDATION (Mann-Whitney U test)")
    log.info("=" * 60)

    summary_rows = []
    for family in sorted(merged["family"].unique()):
        fam = merged[merged["family"] == family]
        flagged = fam[fam["high_confidence"]]["length"]
        unflagged = fam[~fam["high_confidence"]]["length"]

        if len(flagged) < 3 or len(unflagged) < 3:
            log.info("%-8s too few flagged/unflagged sequences for a test (flagged=%d)", family, len(flagged))
            summary_rows.append({
                "family": family, "n_flagged": len(flagged), "n_unflagged": len(unflagged),
                "flagged_mean_length": flagged.mean() if len(flagged) else None,
                "unflagged_mean_length": unflagged.mean() if len(unflagged) else None,
                "mannwhitney_p": None,
            })
            continue

        stat, p = mannwhitneyu(flagged, unflagged, alternative="two-sided")
        significant = "**SIGNIFICANT (p<0.05)**" if p < 0.05 else "not significant"
        log.info(
            "%-8s flagged n=%-3d mean_len=%-6.1f | unflagged n=%-3d mean_len=%-6.1f | p=%.4f %s",
            family, len(flagged), flagged.mean(), len(unflagged), unflagged.mean(), p, significant,
        )
        summary_rows.append({
            "family": family, "n_flagged": len(flagged), "n_unflagged": len(unflagged),
            "flagged_mean_length": flagged.mean(), "unflagged_mean_length": unflagged.mean(),
            "mannwhitney_p": p,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = out_file.with_name(out_file.stem + "_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    merged.to_csv(out_file, index=False)

    log.info("=" * 60)
    log.info("Saved per-sequence detail: %s", out_file)
    log.info("Saved per-family summary: %s", summary_file)
    log.info("=" * 60)
    log.info(
        "Note: raw_description is a per-species automated annotation field "
        "from OrthoDB and is often uninformative ('uncharacterized protein "
        "LOC...'). Treat it as exploratory context, not statistical evidence "
        "-- the length test above is the quantitative signal."
    )


if __name__ == "__main__":
    main()