import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# =====================================================
# PATHS
# =====================================================

EMBEDDINGS_FILE = "output/embeddings.npy"
METADATA_FILE = "output/metadata.csv"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")

X = np.load(EMBEDDINGS_FILE)
meta = pd.read_csv(METADATA_FILE)

print(f"Embeddings shape: {X.shape}")
print(f"Metadata rows: {len(meta)}")

# =====================================================
# LABEL ENCODING
# =====================================================

y_text = meta["family"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

print("\nClasses:")
for idx, label in enumerate(encoder.classes_):
    print(f"{idx}: {label}")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# =====================================================
# DUMMY BASELINE
# =====================================================

print("\n===================================")
print("Baseline Dummy Classifier")
print("===================================")

dummy = DummyClassifier(
    strategy="most_frequent"
)

dummy.fit(X_train, y_train)

dummy_preds = dummy.predict(X_test)

dummy_acc = accuracy_score(
    y_test,
    dummy_preds
)

print(f"Dummy Accuracy: {dummy_acc:.4f}")

# =====================================================
# MLP CLASSIFIER
# =====================================================

print("\n===================================")
print("Training MLP")
print("===================================")

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    max_iter=300,
    random_state=42,
    verbose=True
)

mlp.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

preds = mlp.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

print("\n===================================")
print("MLP Results")
print("===================================")

print(f"Accuracy: {acc:.4f}")

report = classification_report(
    y_test,
    preds,
    target_names=encoder.classes_
)

print("\nClassification Report:\n")
print(report)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    preds
)

plt.style.use("dark_background")

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title(
    "MLP Confusion Matrix",
    fontsize=14
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

outfile = PLOTS_DIR / "confusion_matrix.png"

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved: {outfile}")

# =====================================================
# SUMMARY
# =====================================================

print("\n===================================")
print("Summary")
print("===================================")
print(f"Dummy Accuracy : {dummy_acc:.4f}")
print(f"MLP Accuracy   : {acc:.4f}")
print("===================================")