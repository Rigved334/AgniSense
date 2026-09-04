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

DATASET = Path(
    "data/features/ml_dataset_sentinel2_experiment.parquet"
)


# ============================================================
# Feature groups
# ============================================================

BASE_FEATURES = [
    col
    for col in pd.read_parquet(DATASET, columns=None).columns
    if col not in {
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
        "sentinel2_ndvi",
        "sentinel2_ndbi",
        "sentinel2_ndwi",
        "sentinel2_b02",
        "sentinel2_b03",
        "sentinel2_b04",
        "sentinel2_b08",
        "sentinel2_b11",
        "sentinel2_b12",
    }
]


S2_GROUPS = {
    "S2 indices": [
        "sentinel2_ndvi",
        "sentinel2_ndbi",
        "sentinel2_ndwi",
    ],

    "S2 visible": [
        "sentinel2_b02",
        "sentinel2_b03",
        "sentinel2_b04",
    ],

    "S2 NIR": [
        "sentinel2_b08",
    ],

    "S2 SWIR": [
        "sentinel2_b11",
        "sentinel2_b12",
    ],

    "S2 all bands": [
        "sentinel2_b02",
        "sentinel2_b03",
        "sentinel2_b04",
        "sentinel2_b08",
        "sentinel2_b11",
        "sentinel2_b12",
    ],

    "S2 all indices": [
        "sentinel2_ndvi",
        "sentinel2_ndbi",
        "sentinel2_ndwi",
    ],
}


# ============================================================
# Load
# ============================================================

print("Loading dataset...")

df = pd.read_parquet(DATASET)

print("Dataset shape:", df.shape)

print("\nBase features:", len(BASE_FEATURES))


# ============================================================
# Validate
# ============================================================

all_s2_features = set(
    feature
    for group in S2_GROUPS.values()
    for feature in group
)

missing = [
    feature
    for feature in all_s2_features
    if feature not in df.columns
]

if missing:
    raise ValueError(
        f"Missing Sentinel-2 features: {missing}"
    )


# ============================================================
# Data
# ============================================================

y = df["weak_label"]
groups = df["h3_cell"]


# ============================================================
# Cross-validation
# ============================================================

cv = GroupKFold(n_splits=5)


def evaluate(name, features):

    X = df[features]

    fold_results = []

    print("\n" + "=" * 70)
    print(name)
    print("Features:", len(features))
    print("=" * 70)

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

        model.fit(
            X_train,
            y_train,
        )

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
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro-F1={macro_f1:.4f}, "
            f"Industrial-F1={industrial_f1:.4f}"
        )

    results = pd.DataFrame(fold_results)

    summary = {
        "experiment": name,
        "features": len(features),
        "accuracy": results["accuracy"].mean(),
        "macro_f1": results["macro_f1"].mean(),
        "macro_precision": results["macro_precision"].mean(),
        "macro_recall": results["macro_recall"].mean(),
        "industrial_f1": results["industrial_f1"].mean(),
        "industrial_precision": results["industrial_precision"].mean(),
        "industrial_recall": results["industrial_recall"].mean(),
    }

    print("\nMean:")
    print(
        f"Accuracy:             {summary['accuracy']:.4f}"
    )
    print(
        f"Macro F1:             {summary['macro_f1']:.4f}"
    )
    print(
        f"Macro Precision:      {summary['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall:         {summary['macro_recall']:.4f}"
    )
    print(
        f"Industrial F1:        {summary['industrial_f1']:.4f}"
    )
    print(
        f"Industrial Precision: {summary['industrial_precision']:.4f}"
    )
    print(
        f"Industrial Recall:    {summary['industrial_recall']:.4f}"
    )

    return summary


# ============================================================
# Baseline
# ============================================================

experiments = []

experiments.append(
    evaluate(
        "BASE: FIRMS + OSM + WorldCover",
        BASE_FEATURES,
    )
)


# ============================================================
# Add each Sentinel-2 group independently
# ============================================================

for group_name, s2_features in S2_GROUPS.items():

    # Skip duplicate all-index experiment
    if group_name == "S2 all indices":
        continue

    features = BASE_FEATURES + s2_features

    experiments.append(
        evaluate(
            f"BASE + {group_name}",
            features,
        )
    )


# ============================================================
# Additional useful combinations
# ============================================================

experiments.append(
    evaluate(
        "BASE + NDVI + NDBI + NDWI + SWIR",
        BASE_FEATURES
        + [
            "sentinel2_ndvi",
            "sentinel2_ndbi",
            "sentinel2_ndwi",
            "sentinel2_b11",
            "sentinel2_b12",
        ],
    )
)


experiments.append(
    evaluate(
        "BASE + NDVI + NDBI + SWIR",
        BASE_FEATURES
        + [
            "sentinel2_ndvi",
            "sentinel2_ndbi",
            "sentinel2_b11",
            "sentinel2_b12",
        ],
    )
)


# ============================================================
# Final comparison
# ============================================================

results_df = pd.DataFrame(experiments)

results_df["delta_macro_f1"] = (
    results_df["macro_f1"]
    - results_df.loc[0, "macro_f1"]
)

results_df["delta_industrial_f1"] = (
    results_df["industrial_f1"]
    - results_df.loc[0, "industrial_f1"]
)


print("\n\n")
print("#" * 80)
print("SENTINEL-2 FEATURE ABLATION SUMMARY")
print("#" * 80)

display_columns = [
    "experiment",
    "features",
    "macro_f1",
    "industrial_f1",
    "industrial_precision",
    "industrial_recall",
    "delta_macro_f1",
    "delta_industrial_f1",
]

print(
    results_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# Save results
# ============================================================

OUTPUT = Path(
    "data/features/sentinel2_feature_ablation_results.csv"
)

results_df.to_csv(
    OUTPUT,
    index=False,
)

print(
    f"\nSaved results to:\n{OUTPUT}"
)