import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
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
# CONFIG
# ============================================================

INPUT_FILE = "data/features/ml_dataset.parquet"

LABEL_COLUMN = "weak_label"
SPLIT_COLUMN = "dataset_split"

RANDOM_STATE = 42


# ============================================================
# NON-FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = [
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
]


# ============================================================
# FEATURE GROUPS
# ============================================================

THERMAL_FEATURES = [
    "active_days",
    "detection_count",
    "mean_frp",
    "median_frp",
    "max_frp",
    "std_frp",
    "mean_bright_ti4",
    "max_bright_ti4",
    "mean_bright_ti5",
    "max_bright_ti5",
    "day_detection_count",
    "night_detection_count",
    "high_confidence_count",
    "nominal_confidence_count",
    "low_confidence_count",
    "duration_days",
]


LANDCOVER_FEATURES = [
    "landcover_class",
    "tree_cover_fraction_1km",
    "shrubland_fraction_1km",
    "grassland_fraction_1km",
    "cropland_fraction_1km",
    "builtup_fraction_1km",
    "bare_fraction_1km",
    "snow_ice_fraction_1km",
    "water_fraction_1km",
    "wetland_fraction_1km",
    "mangroves_fraction_1km",
    "moss_lichen_fraction_1km",
]


OSM_FEATURES = [
    "nearest_industrial_distance_m",
    "industrial_count_500m",
    "nearest_works_distance_m",
    "works_count_500m",
    "nearest_flare_distance_m",
    "flare_count_1000m",
    "nearest_storage_tank_distance_m",
    "storage_tank_count_1000m",
    "nearest_powerplant_distance_m",
    "powerplant_count_5000m",
    "nearest_quarry_distance_m",
    "quarry_count_1000m",
    "refinery_distance_m",
    "refinery_count",
]


FEATURE_GROUPS = {
    "Thermal only": THERMAL_FEATURES,
    "Land cover only": LANDCOVER_FEATURES,
    "OSM only": OSM_FEATURES,

    "Thermal + Land cover":
        THERMAL_FEATURES + LANDCOVER_FEATURES,

    "Thermal + OSM":
        THERMAL_FEATURES + OSM_FEATURES,

    "All features":
        THERMAL_FEATURES + LANDCOVER_FEATURES + OSM_FEATURES,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE ABLATION EXPERIMENT")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nDataset: {INPUT_FILE}")
print(f"Rows: {len(df):,}")


# ============================================================
# CHECK FEATURES
# ============================================================

available_features = set(df.columns)

for group_name, features in FEATURE_GROUPS.items():

    missing = [
        feature
        for feature in features
        if feature not in available_features
    ]

    if missing:
        raise ValueError(
            f"\nMissing features for '{group_name}':\n"
            + "\n".join(missing)
        )


# ============================================================
# SPLIT
# ============================================================

train_df = df[df[SPLIT_COLUMN] == "train"].copy()
val_df = df[df[SPLIT_COLUMN] == "validation"].copy()
test_df = df[df[SPLIT_COLUMN] == "test"].copy()

print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")


# ============================================================
# TRAIN/EVALUATION FUNCTION
# ============================================================

def evaluate_group(group_name, features):

    print("\n" + "=" * 70)
    print(group_name)
    print("=" * 70)

    print(f"Features: {len(features)}")

    X_train = train_df[features]
    y_train = train_df[LABEL_COLUMN]

    X_val = val_df[features]
    y_val = val_df[LABEL_COLUMN]

    X_test = test_df[features]
    y_test = test_df[LABEL_COLUMN]

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_pred = model.predict(X_val)

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_pred = model.predict(X_test)

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_weighted_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    test_macro_precision = precision_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    test_macro_recall = recall_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # INDUSTRIAL FIRE
    # --------------------------------------------------------

    industrial_label = "industrial_fire"

    industrial_precision = precision_score(
        y_test,
        test_pred,
        labels=[industrial_label],
        average="macro",
        zero_division=0,
    )

    industrial_recall = recall_score(
        y_test,
        test_pred,
        labels=[industrial_label],
        average="macro",
        zero_division=0,
    )

    industrial_f1 = f1_score(
        y_test,
        test_pred,
        labels=[industrial_label],
        average="macro",
        zero_division=0,
    )

    print("\nValidation:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Macro F1: {val_macro_f1:.4f}")

    print("\nTest:")
    print(f"Accuracy:       {test_accuracy:.4f}")
    print(f"Macro F1:       {test_macro_f1:.4f}")
    print(f"Weighted F1:    {test_weighted_f1:.4f}")
    print(f"Macro Precision:{test_macro_precision:.4f}")
    print(f"Macro Recall:   {test_macro_recall:.4f}")

    print("\nIndustrial fire:")
    print(f"Precision: {industrial_precision:.4f}")
    print(f"Recall:    {industrial_recall:.4f}")
    print(f"F1:        {industrial_f1:.4f}")

    return {
        "experiment": group_name,
        "n_features": len(features),
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_macro_precision": test_macro_precision,
        "test_macro_recall": test_macro_recall,
        "industrial_precision": industrial_precision,
        "industrial_recall": industrial_recall,
        "industrial_f1": industrial_f1,
    }


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []

for group_name, features in FEATURE_GROUPS.items():

    result = evaluate_group(
        group_name,
        features,
    )

    results.append(result)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL ABLATION RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "experiment",
            "n_features",
            "val_macro_f1",
            "test_macro_f1",
            "test_weighted_f1",
            "industrial_precision",
            "industrial_recall",
            "industrial_f1",
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = "data/features/feature_ablation_results.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"\nSaved results to:")
print(OUTPUT_FILE)