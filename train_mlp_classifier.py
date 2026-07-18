"""Stage 4: classify protein family from embeddings.

Upgrades over the original single 80/20 split:
  - Stratified K-fold cross-validation (default 5-fold) is the headline
    metric, reported as mean +/- std, since a single 64-sample test set
    (the old approach) gives a noisy point estimate -- e.g. a single
    misclassified sequence moves accuracy by >1.5 points.
  - Embeddings are standardized (zero mean / unit variance per dimension)
    before the MLP, which the original script skipped.
  - A held-out split is still produced at the end purely to render one
    confusion matrix for visualization; it is NOT the reported metric.

Usage:
    python train_mlp_classifier.py
    python train_mlp_classifier.py --n-splits 10 --no-standardize
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from esm2_phylo.config import EMBEDDINGS_FILE, METADATA_FILE, PLOTS_DIR
from esm2_phylo.logging_utils import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--metadata", type=Path, default=METADATA_FILE)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()

    if not args.embeddings.exists():
        raise SystemExit(f"Embeddings not found: {args.embeddings}. Run generate_embeddings.py first.")

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    X = np.load(args.embeddings)
    meta = pd.read_csv(args.metadata)

    if len(meta) != len(X):
        raise SystemExit(f"Metadata rows ({len(meta)}) != embedding rows ({len(X)}). Rerun generate_embeddings.py.")

    y_text = meta["family"]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    log.info("Classes: %s", dict(enumerate(encoder.classes_)))

    class_counts = pd.Series(y).value_counts()
    if class_counts.min() < args.n_splits:
        log.warning(
            "Smallest class has only %d samples but n_splits=%d; reducing n_splits to %d.",
            class_counts.min(), args.n_splits, class_counts.min(),
        )
        args.n_splits = max(2, class_counts.min())

    if args.no_standardize:
        X_model = X
    else:
        X_model = StandardScaler().fit_transform(X)

    hidden = tuple(args.hidden_layers)

    # =====================================================
    # CROSS-VALIDATED HEADLINE METRIC
    # =====================================================
    log.info("=" * 40)
    log.info("Stratified %d-fold cross-validation", args.n_splits)
    log.info("=" * 40)

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    dummy_scores = cross_val_score(
        DummyClassifier(strategy="most_frequent"), X_model, y, cv=cv, scoring="accuracy"
    )
    mlp_scores = cross_val_score(
        MLPClassifier(hidden_layer_sizes=hidden, max_iter=args.max_iter, random_state=args.seed),
        X_model, y, cv=cv, scoring="accuracy",
    )
    mlp_f1_scores = cross_val_score(
        MLPClassifier(hidden_layer_sizes=hidden, max_iter=args.max_iter, random_state=args.seed),
        X_model, y, cv=cv, scoring="f1_macro",
    )

    log.info(
        "Dummy accuracy:  %.4f +/- %.4f  (folds: %s)",
        dummy_scores.mean(), dummy_scores.std(), np.round(dummy_scores, 3),
    )
    log.info(
        "MLP accuracy:    %.4f +/- %.4f  (folds: %s)",
        mlp_scores.mean(), mlp_scores.std(), np.round(mlp_scores, 3),
    )
    log.info("MLP macro-F1:    %.4f +/- %.4f", mlp_f1_scores.mean(), mlp_f1_scores.std())

    # =====================================================
    # SINGLE HOLD-OUT SPLIT (for confusion matrix + report only)
    # =====================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=args.max_iter, random_state=args.seed)
    mlp.fit(X_train, y_train)
    preds = mlp.predict(X_test)
    holdout_acc = accuracy_score(y_test, preds)

    report = classification_report(y_test, preds, target_names=encoder.classes_, output_dict=True)
    report_text = classification_report(y_test, preds, target_names=encoder.classes_)
    log.info("Hold-out split accuracy (illustrative, single split, n_test=%d): %.4f", len(y_test), holdout_acc)
    log.info("\n%s", report_text)

    cm = confusion_matrix(y_test, preds)

    plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f"MLP Confusion Matrix (single {int((1-args.test_size)*100)}/{int(args.test_size*100)} split, n={len(y_test)})", fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_file = args.plots_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=300, bbox_inches="tight")
    plt.close()

    # =====================================================
    # SAVE METRICS
    # =====================================================
    metrics = {
        "n_splits": args.n_splits,
        "cv_dummy_accuracy_mean": float(dummy_scores.mean()),
        "cv_dummy_accuracy_std": float(dummy_scores.std()),
        "cv_mlp_accuracy_mean": float(mlp_scores.mean()),
        "cv_mlp_accuracy_std": float(mlp_scores.std()),
        "cv_mlp_accuracy_folds": mlp_scores.tolist(),
        "cv_mlp_macro_f1_mean": float(mlp_f1_scores.mean()),
        "cv_mlp_macro_f1_std": float(mlp_f1_scores.std()),
        "holdout_accuracy": float(holdout_acc),
        "holdout_n_test": int(len(y_test)),
        "standardized": not args.no_standardize,
        "classification_report_holdout": report,
    }
    metrics_file = args.plots_dir / "classifier_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2))

    log.info("=" * 40)
    log.info("Summary")
    log.info("Dummy CV accuracy : %.4f +/- %.4f", dummy_scores.mean(), dummy_scores.std())
    log.info("MLP CV accuracy   : %.4f +/- %.4f", mlp_scores.mean(), mlp_scores.std())
    log.info("Saved: %s", cm_file)
    log.info("Saved: %s", metrics_file)
    log.info("=" * 40)


if __name__ == "__main__":
    main()
