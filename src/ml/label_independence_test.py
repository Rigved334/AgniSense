import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
OUTPUT_FILE = "data/features/label_independence_results.csv"

LABEL_COLUMN = "weak_label"
GROUP_COLUMN = "h3_cell"

N_SPLITS = 5
RANDOM_STATE = 42


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


ALL_FEATURES = (
    THERMAL_FEATURES
    + LANDCOVER_FEATURES
    + OSM_FEATURES
)


# ============================================================
# FEATURES DIRECTLY USED BY WEAK-LABEL RULES
# ============================================================

# These variables are directly involved in defining the labels.
LABEL_RULE_FEATURES = [
    # Industrial context
    "industrial_count_500m",
    "works_count_500m",
    "refinery_count",
    "storage_tank_count_1000m",
    "flare_count_1000m",

    # Industrial-fire rule
    "duration_days",
    "max_frp",
    "high_confidence_count",
    "detection_count",

    # Natural-fire rules
    "tree_cover_fraction_1km",
    "shrubland_fraction_1km",
    "grassland_fraction_1km",
    "cropland_fraction_1km",
]


# ============================================================
# DEFINE EXPERIMENTS
# ============================================================

EXPERIMENTS = {

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    "All features": ALL_FEATURES,

    # --------------------------------------------------------
    # Remove direct label-rule features
    # --------------------------------------------------------

    "Remove label-rule features": [
        f
        for f in ALL_FEATURES
        if f not in LABEL_RULE_FEATURES
    ],

    # --------------------------------------------------------
    # Thermal features excluding direct rule variables
    # --------------------------------------------------------

    "Independent thermal": [
        f
        for f in THERMAL_FEATURES
        if f not in LABEL_RULE_FEATURES
    ],

    # --------------------------------------------------------
    # OSM features excluding direct rule variables
    # --------------------------------------------------------

    "Independent OSM": [
        f
        for f in OSM_FEATURES
        if f not in LABEL_RULE_FEATURES
    ],

    # --------------------------------------------------------
    # Land cover features excluding direct rule variables
    # --------------------------------------------------------

    "Independent land cover": [
        f
        for f in LANDCOVER_FEATURES
        if f not in LABEL_RULE_FEATURES
    ],

    # --------------------------------------------------------
    # Features that were NOT directly used in the rules
    # --------------------------------------------------------

    "All independent features": [
        f
        for f in ALL_FEATURES
        if f not in LABEL_RULE_FEATURES
    ],
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LABEL INDEPENDENCE TEST")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nDataset: {INPUT_FILE}")
print(f"Rows: {len(df):,}")
print(f"Total features: {len(ALL_FEATURES)}")
print(f"Spatial groups: {df[GROUP_COLUMN].nunique():,}")


# ============================================================
# VERIFY FEATURES
# ============================================================

missing = [
    f
    for f in ALL_FEATURES
    if f not in df.columns
]

if missing:
    raise ValueError(
        "Missing features:\n"
        + "\n".join(missing)
    )


# ============================================================
# DISPLAY RULE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURES REMOVED AS DIRECT LABEL-RULE VARIABLES")
print("=" * 70)

for feature in LABEL_RULE_FEATURES:
    print(f"  {feature}")


# ============================================================
# VERIFY EXPERIMENTS
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT FEATURE COUNTS")
print("=" * 70)

for name, features in EXPERIMENTS.items():

    print(
        f"{name:<35} "
        f"{len(features):>3} features"
    )

    if len(features) == 0:
        raise ValueError(
            f"Experiment '{name}' has zero features."
        )


# ============================================================
# DATA
# ============================================================

X_all = df[ALL_FEATURES]
y = df[LABEL_COLUMN]
groups = df[GROUP_COLUMN]

labels = sorted(y.unique())

print("\nClasses:")

for label in labels:
    print(f"  {label}")


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=N_SPLITS
)

results = []


for experiment_name, features in EXPERIMENTS.items():

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    print(f"Features: {len(features)}")

    X = df[features]

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        # ----------------------------------------------------
        # VERIFY SPATIAL SEPARATION
        # ----------------------------------------------------

        overlap = (
            set(groups_train)
            & set(groups_test)
        )

        if overlap:
            raise RuntimeError(
                f"Spatial leakage detected in "
                f"{experiment_name}, fold {fold}"
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
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

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        macro_precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        industrial_f1 = f1_score(
            y_test,
            y_pred,
            labels=["industrial_fire"],
            average="macro",
            zero_division=0,
        )

        industrial_precision = precision_score(
            y_test,
            y_pred,
            labels=["industrial_fire"],
            average="macro",
            zero_division=0,
        )

        industrial_recall = recall_score(
            y_test,
            y_pred,
            labels=["industrial_fire"],
            average="macro",
            zero_division=0,
        )

        fold_results.append(
            {
                "experiment": experiment_name,
                "fold": fold,
                "n_features": len(features),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "industrial_f1": industrial_f1,
                "industrial_precision": industrial_precision,
                "industrial_recall": industrial_recall,
            }
        )

        print(
            f"Fold {fold}: "
            f"Macro F1={macro_f1:.4f}, "
            f"Industrial F1={industrial_f1:.4f}"
        )

    # --------------------------------------------------------
    # EXPERIMENT SUMMARY
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    results.extend(
        fold_results
    )

    print("\nMean ± Std:")

    for metric in [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "industrial_f1",
        "industrial_precision",
        "industrial_recall",
    ]:

        mean = fold_df[metric].mean()
        std = fold_df[metric].std()

        print(
            f"{metric:<25}"
            f"{mean:.4f} ± {std:.4f}"
        )


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = (
    results_df
    .groupby(
        ["experiment", "n_features"]
    )
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),

        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),

        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),

        industrial_f1_mean=(
            "industrial_f1",
            "mean",
        ),

        industrial_f1_std=(
            "industrial_f1",
            "std",
        ),

        industrial_precision_mean=(
            "industrial_precision",
            "mean",
        ),

        industrial_precision_std=(
            "industrial_precision",
            "std",
        ),

        industrial_recall_mean=(
            "industrial_recall",
            "mean",
        ),

        industrial_recall_std=(
            "industrial_recall",
            "std",
        ),
    )
    .reset_index()
)


print("\n")
print("=" * 70)
print("FINAL LABEL-INDEPENDENCE RESULTS")
print("=" * 70)

print(
    summary.round(4).to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT_FILE)