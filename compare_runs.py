"""Compare two archived pipeline runs side by side (e.g. two ESM-2 model
sizes), producing a summary table and a bar chart.

This is meant to be used with the archiving convention described in
README.md's "Comparing Runs" section: after a pipeline run, move
output/embeddings.npy, output/metadata.csv, plots/*, and results/* into
a tagged folder (e.g. runs/esm2_t12_35M/) before starting the next run.

Usage:
    python compare_runs.py --run-a runs/esm2_t12_35M --run-b runs/esm2_t30_150M
    python compare_runs.py --run-a runs/esm2_t12_35M --run-b runs/esm2_t30_150M --labels "35M" "150M"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        log.warning("Missing file, skipping those metrics: %s", path)
        return None
    return json.loads(path.read_text())


def load_run_metrics(run_dir: Path) -> dict:
    classifier = load_json(run_dir / "classifier_metrics.json")
    kmer_esm = load_json(run_dir / "kmer_vs_esm2_metrics.json")

    metrics = {"run_dir": str(run_dir)}

    if classifier:
        metrics["cv_mlp_accuracy_mean"] = classifier["cv_mlp_accuracy_mean"]
        metrics["cv_mlp_accuracy_std"] = classifier["cv_mlp_accuracy_std"]
        metrics["cv_mlp_macro_f1_mean"] = classifier["cv_mlp_macro_f1_mean"]
        metrics["cv_dummy_accuracy_mean"] = classifier["cv_dummy_accuracy_mean"]
        if classifier.get("esm_model"):
            metrics["esm_model"] = classifier["esm_model"]

    if kmer_esm:
        metrics["kmer_purity"] = kmer_esm["kmer_purity"]
        metrics["kmer_v_measure"] = kmer_esm["kmer_v_measure"]
        metrics["esm_purity"] = kmer_esm["esm_purity"]
        metrics["esm_v_measure"] = kmer_esm["esm_v_measure"]
        if not metrics.get("esm_model") and kmer_esm.get("esm_model"):
            metrics["esm_model"] = kmer_esm["esm_model"]

    return metrics

def print_comparison_table(metrics_a: dict, metrics_b: dict, label_a: str, label_b: str):
    rows = [
        ("ESM-2 model", "esm_model", None),
        ("Dummy CV accuracy", "cv_dummy_accuracy_mean", "{:.3f}"),
        ("ESM-2 MLP CV accuracy", "cv_mlp_accuracy_mean", "{:.3f}"),
        ("ESM-2 MLP CV macro-F1", "cv_mlp_macro_f1_mean", "{:.3f}"),
        ("ESM-2 cluster purity", "esm_purity", "{:.3f}"),
        ("ESM-2 V-measure", "esm_v_measure", "{:.3f}"),
        ("k-mer cluster purity", "kmer_purity", "{:.3f}"),
        ("k-mer V-measure", "kmer_v_measure", "{:.3f}"),
    ]

    name_w = 26
    log.info("%-*s %15s %15s", name_w, "Metric", label_a, label_b)
    log.info("-" * (name_w + 33))
    for display_name, key, fmt in rows:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        val_a_str = fmt.format(val_a) if (fmt and val_a is not None) else str(val_a)
        val_b_str = fmt.format(val_b) if (fmt and val_b is not None) else str(val_b)
        log.info("%-*s %15s %15s", name_w, display_name, val_a_str, val_b_str)


def make_comparison_chart(metrics_a: dict, metrics_b: dict, label_a: str, label_b: str, out_path: Path):
    chart_metrics = [
        ("MLP CV\naccuracy", "cv_mlp_accuracy_mean"),
        ("MLP CV\nmacro-F1", "cv_mlp_macro_f1_mean"),
        ("ESM-2\ncluster purity", "esm_purity"),
        ("ESM-2\nV-measure", "esm_v_measure"),
        ("k-mer\ncluster purity", "kmer_purity"),
        ("k-mer\nV-measure", "kmer_v_measure"),
    ]

    labels = [name for name, key in chart_metrics if key in metrics_a and key in metrics_b]
    values_a = [metrics_a[key] for name, key in chart_metrics if key in metrics_a and key in metrics_b]
    values_b = [metrics_b[key] for name, key in chart_metrics if key in metrics_a and key in metrics_b]

    if not labels:
        log.warning("No overlapping metrics found between the two runs -- skipping chart.")
        return

    x = np.arange(len(labels))
    width = 0.35

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6))
    bars_a = ax.bar(x - width / 2, values_a, width, label=label_a, color="#377eb8")
    bars_b = ax.bar(x + width / 2, values_b, width, label=label_b, color="#e41a1c")

    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title(f"Run Comparison: {label_a} vs {label_b}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-a", type=Path, required=True, help="Directory with an archived run's plots/results")
    parser.add_argument("--run-b", type=Path, required=True, help="Directory with a second archived run's plots/results")
    parser.add_argument("--labels", nargs=2, default=None, help="Two short labels for the chart legend/table headers")
    parser.add_argument("--out", type=Path, default=Path("plots/run_comparison.png"))
    args = parser.parse_args()

    metrics_a = load_run_metrics(args.run_a)
    metrics_b = load_run_metrics(args.run_b)

    label_a, label_b = args.labels if args.labels else (
        metrics_a.get("esm_model", args.run_a.name),
        metrics_b.get("esm_model", args.run_b.name),
    )

    print_comparison_table(metrics_a, metrics_b, label_a, label_b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_comparison_chart(metrics_a, metrics_b, label_a, label_b, args.out)


if __name__ == "__main__":
    main()