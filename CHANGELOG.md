# Changelog

## 0.2.0 — Upgrade pass

### Bug fixes
- **k-mer baseline dataset mismatch (significant):** `kmer_distance.py` was reading
  directly from `data/*.fasta` (the raw, unfiltered OrthoDB dumps — thousands of
  sequences per family, no length filter, no per-species dedup) while ESM-2
  embeddings were computed on the curated `output/all_families.fasta` (320
  sequences, 80/family). The k-mer-vs-ESM-2 purity/V-measure comparison was
  therefore not a fair head-to-head baseline. Fixed: both methods now run on the
  exact same curated sequences (`--fasta` defaults to `output/all_families.fasta`).
- **Family label / color mismatch:** the k-mer baseline used raw filenames as
  labels (`HSP_70`, `CYT_C`) while everything else used standardized labels
  (`HSP70`, `CYTC`), so `FAMILY_COLORS.get()` silently returned `None` for two
  of four families in the comparison plot. Fixed by using the same standardized
  labels (from `esm2_phylo.config.FAMILY_FILES`) everywhere.
- Removed the dead 0-byte `data/HSP70_aligned.fasta` file.
- Removed an unreferenced `Figure_1.png` at the repo root (not linked from
  the README; appears to be a leftover draft duplicate of `plots/kmer_vs_esm2.png`).
- Fixed the misspelled/incomplete `requirments.txt` → `requirements.txt`, added
  missing dependencies (`seaborn`, `scipy`, `ete4`, `biopython`, `pytest`) and
  version floors.
- Fixed the README/actual-filename mismatch (`cluster_and_visualize.py` →
  `cluster_visualise.py`).
- `kmer_distance.py` previously saved the same comparison figure twice under
  two different filenames/DPIs (`umap_comparison.png` and `kmer_vs_esm2.png`)
  and called `plt.show()`, which blocks/errors in non-interactive
  environments. Now saves once, no `plt.show()`. README image reference
  updated to match (`plots/kmer_vs_esm2.png`).

### New: finished the tree-comparison feature
`compare_trees.py` shipped with no script producing the `results/*.tree` gene
trees it needs, no bundled species tree, and an undeclared `ete4` dependency.
- Added **`build_gene_trees.py`**: builds a neighbor-joining gene tree per family
  directly from ESM-2 embedding (or k-mer) cosine distances — no MSA/external
  aligner needed, consistent with the project's alignment-free premise.
- Added **`fetch_species_tree.py`**: builds a reference species tree from NCBI
  taxonomy (via `ete4.NCBITaxa`) for the species present in your dataset.
- Rewrote `compare_trees.py` to fail with an actionable message when the species
  tree is missing (instead of an unhandled exception), and to handle multiple
  ete4 Newick parser formats/API changes.

### Statistical rigor
- `train_mlp_classifier.py` now reports **stratified 5-fold cross-validated
  accuracy and macro-F1 (mean ± std)** as the headline metric. The original
  single 80/20 split (64 test sequences) is kept only to render one confusion
  matrix, explicitly labeled as illustrative rather than the reported metric.
- Embeddings are standardized (zero mean/unit variance) before the MLP by
  default (`--no-standardize` to disable).
- `detect_paralogs.py` now combines the original KMeans-minority-cluster
  heuristic (which by construction always flags *something*) with a
  statistically motivated MAD/robust-z distance-from-centroid signal, and
  reports which sequences are flagged by both (high-confidence) vs. one
  (low-confidence).

### Engineering
- Introduced `esm2_phylo/` package: `config.py` (single source of truth for
  paths/constants/family labels), `logging_utils.py`, `utils.py` (pure,
  testable helper functions).
- Added `tests/test_utils.py` — 15 unit tests covering species-name parsing,
  k-mer vectorization, cluster purity, outlier detection, and species-overlap
  logic, runnable without a GPU, torch, or model weights.
- Every stage script now takes CLI arguments (`argparse`) instead of hardcoded
  paths/constants, and uses a shared logger instead of bare `print()`.
- Added `run_pipeline.py` to run all stages in order (or a subset via
  `--only`/`--skip`/`--with-trees`).
- Added `pyproject.toml` (editable install, pytest config).
- `generate_embeddings.py` now supports `--model` (choose among 5 ESM-2
  checkpoint sizes), auto CPU/CUDA device detection with a clear fallback
  message for non-CUDA GPUs (e.g. AMD integrated graphics), and a live
  throughput/ETA indicator instead of per-sequence prints.

### Known limitations (see README)
- Gene trees from `build_gene_trees.py` are fast NJ approximations from
  embedding distances, not a substitute for ML/Bayesian phylogenetic inference.
- `fetch_species_tree.py` requires internet access on the machine you run it
  on (not exercised in this sandboxed upgrade).
