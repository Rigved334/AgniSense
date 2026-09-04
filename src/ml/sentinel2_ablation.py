from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Paths
# ============================================================

ML_DATASET = Path("data/features/ml_dataset.parquet")
S2_DATASET = Path(
    "data/features/sentinel2_episode_features_batch500.parquet"
)

OUTPUT_DATASET = Path(
    "data/features/ml_dataset_sentinel2_experiment.parquet"
)


# ============================================================
# Sentinel-2 features
# ============================================================

S2_FEATURES = [
    "sentinel2_ndvi",
    "sentinel2_ndbi",
    "sentinel2_ndwi",
    "sentinel2_b02",
    "sentinel2_b03",
    "sentinel2_b04",
    "sentinel2_b08",
    "sentinel2_b11",
    "sentinel2_b12",
]


# ============================================================
# Main feature groups
# ============================================================

# These are the existing 42 predictive features.
# We remove metadata and label-related columns below.

EXCLUDED_COLUMNS = {
    "h3_cell",
    "episode_id",
    "start_date",
    "end_date",
    "latitude",
    "longitude",
    "weak_label",
    "dataset_split",
    "label_conflict",
    "strong_industrial_context",
}


# ============================================================
# Load data
# ============================================================

print("Loading ML dataset...")
ml = pd.read_parquet(ML_DATASET)

print("ML dataset shape:", ml.shape)

print("\nLoading Sentinel-2 dataset...")
s2 = pd.read_parquet(S2_DATASET)

print("Sentinel-2 dataset shape:", s2.shape)


# ============================================================
# Keep only usable Sentinel-2 observations
# ============================================================

if "sentinel2_found" in s2.columns:
    s2 = s2[s2["sentinel2_found"] == True].copy()

print("\nUsable Sentinel-2 observations:", len(s2))


# ============================================================
# Check required columns
# ============================================================

required_s2 = ["episode_id"] + S2_FEATURES

missing_s2 = [
    col for col in required_s2
    if col not in s2.columns
]

if missing_s2:
    raise ValueError(
        f"Missing Sentinel-2 columns: {missing_s2}"
    )

if "episode_id" not in ml.columns:
    raise ValueError("ML dataset does not contain episode_id")

if "weak_label" not in ml.columns:
    raise ValueError("ML dataset does not contain weak_label")


# ============================================================
# Select Sentinel-2 columns
# ============================================================

s2_small = s2[
    ["episode_id"] + S2_FEATURES
].copy()


# Remove possible duplicate episode IDs
s2_small = s2_small.drop_duplicates(
    subset="episode_id"
)


# ============================================================
# Merge
# ============================================================

experiment = ml.merge(
    s2_small,
    on="episode_id",
    how="inner",
)

print("\nMerged experiment dataset shape:")
print(experiment.shape)


# ============================================================
# Verify Sentinel-2 completeness
# ============================================================

missing_fraction = experiment[S2_FEATURES].isna().mean()

print("\nSentinel-2 missing fraction:")
print(missing_fraction)


if experiment[S2_FEATURES].isna().any().any():
    raise ValueError(
        "Missing Sentinel-2 values found after merge."
    )


# ============================================================
# Save experiment dataset
# ============================================================

experiment.to_parquet(
    OUTPUT_DATASET,
    index=False,
)

print(
    f"\nSaved experiment dataset:\n"
    f"{OUTPUT_DATASET}"
)


# ============================================================
# Feature selection
# ============================================================

all_existing_features = [
    col
    for col in ml.columns
    if col not in EXCLUDED_COLUMNS
]

# Remove Sentinel-2 columns if they somehow already exist
base_features = [
    col
    for col in all_existing_features
    if col not in S2_FEATURES
]

enhanced_features = base_features + S2_FEATURES

print("\nFeature counts:")
print("Base features:", len(base_features))
print("Sentinel-2 features:", len(S2_FEATURES))
print("Enhanced features:", len(enhanced_features))


# ============================================================
# Verify feature columns
# ============================================================

missing_base = [
    col for col in base_features
    if col not in experiment.columns
]

missing_enhanced = [
    col for col in enhanced_features
    if col not in experiment.columns
]

if missing_base:
    raise ValueError(
        f"Missing base features: {missing_base}"
    )

if missing_enhanced:
    raise ValueError(
        f"Missing enhanced features: {missing_enhanced}"
    )


# ============================================================
# Labels and spatial groups
# ============================================================

X_base = experiment[base_features]
X_enhanced = experiment[enhanced_features]

y = experiment["weak_label"]
groups = experiment["h3_cell"]


# ============================================================
# Spatial cross-validation
# ============================================================

cv = GroupKFold(n_splits=5)


def evaluate_model(X, name):
    """
    Evaluate Logistic Regression using spatial GroupKFold.
    """

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        industrial_f1 = f1_score(
                        y_test,
                        predictions,
                        labels=["industrial_fire"],
                        average="macro",
                        zero_division=0,
        )

        industrial_precision = precision_score(
            y_test,
            predictions,
            labels=["industrial_fire"],
            average="macro",
            zero_division=0,
        )

        industrial_recall = recall_score(
            y_test,
            predictions,
            labels=["industrial_fire"],
            average="macro",
            zero_division=0,
        )

        fold_results.append(
                {
                    "fold": fold,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "industrial_f1": industrial_f1,
                    "industrial_precision": industrial_precision,
                    "industrial_recall": industrial_recall,
                }
            )

        print(
            f"\n{name} | Fold {fold}"
        )

        print(
            f"Accuracy:              {accuracy:.4f}"
        )

        print(
            f"Macro F1:              {macro_f1:.4f}"
        )

        print(
            f"Macro Precision:       {macro_precision:.4f}"
        )

        print(
            f"Macro Recall:          {macro_recall:.4f}"
        )

        print(
            f"Industrial F1:         {industrial_f1:.4f}"
        )

        print(
            f"Industrial Precision:  {industrial_precision:.4f}"
        )

        print(
            f"Industrial Recall:     {industrial_recall:.4f}"
        )

    results = pd.DataFrame(fold_results)

    print(
        f"\n{'=' * 60}"
    )

    print(name)

    print(
        f"Accuracy:             "
        f"{results['accuracy'].mean():.4f} ± "
        f"{results['accuracy'].std():.4f}"
    )

    print(
        f"Macro F1:             "
        f"{results['macro_f1'].mean():.4f} ± "
        f"{results['macro_f1'].std():.4f}"
    )

    print(
        f"Macro Precision:      "
        f"{results['macro_precision'].mean():.4f} ± "
        f"{results['macro_precision'].std():.4f}"
    )

    print(
        f"Macro Recall:         "
        f"{results['macro_recall'].mean():.4f} ± "
        f"{results['macro_recall'].std():.4f}"
    )

    print(
        f"Industrial F1:        "
        f"{results['industrial_f1'].mean():.4f} ± "
        f"{results['industrial_f1'].std():.4f}"
    )

    print(
        f"Industrial Precision: "
        f"{results['industrial_precision'].mean():.4f} ± "
        f"{results['industrial_precision'].std():.4f}"
    )

    print(
        f"Industrial Recall:    "
        f"{results['industrial_recall'].mean():.4f} ± "
        f"{results['industrial_recall'].std():.4f}"
    )

    return results


# ============================================================
# Run experiments
# ============================================================

print("\n\n")
print("#" * 70)
print("EXPERIMENT 1: FIRMS + OSM + WORLD COVER")
print("#" * 70)

base_results = evaluate_model(
    X_base,
    "BASE: FIRMS + OSM + WorldCover",
)


print("\n\n")
print("#" * 70)
print("EXPERIMENT 2: FIRMS + OSM + WORLD COVER + SENTINEL-2")
print("#" * 70)

enhanced_results = evaluate_model(
    X_enhanced,
    "ENHANCED: FIRMS + OSM + WorldCover + Sentinel-2",
)


# ============================================================
# Final comparison
# ============================================================

comparison = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "industrial_f1",
            "industrial_precision",
            "industrial_recall",
        ],
        "base_mean": [
            base_results["accuracy"].mean(),
            base_results["macro_f1"].mean(),
            base_results["macro_precision"].mean(),
            base_results["macro_recall"].mean(),
            base_results["industrial_f1"].mean(),
            base_results["industrial_precision"].mean(),
            base_results["industrial_recall"].mean(),
        ],
        "enhanced_mean": [
            enhanced_results["accuracy"].mean(),
            enhanced_results["macro_f1"].mean(),
            enhanced_results["macro_precision"].mean(),
            enhanced_results["macro_recall"].mean(),
            enhanced_results["industrial_f1"].mean(),
            enhanced_results["industrial_precision"].mean(),
            enhanced_results["industrial_recall"].mean(),
        ],
    }
)

comparison["change"] = (
    comparison["enhanced_mean"]
    - comparison["base_mean"]
)

print("\n\n")
print("#" * 70)
print("FINAL SENTINEL-2 ABLATION")
print("#" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)