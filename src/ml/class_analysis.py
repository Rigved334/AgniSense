import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/features/ml_dataset.parquet"

LABEL_COLUMN = "weak_label"

OUTPUT_STATS = "data/features/class_analysis_statistics.csv"
OUTPUT_IMPORTANCE = "data/features/class_analysis_feature_importance.csv"


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

FEATURES = (
    THERMAL_FEATURES
    + LANDCOVER_FEATURES
    + OSM_FEATURES
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CLASS ANALYSIS")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nRows: {len(df):,}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

class_counts = df[LABEL_COLUMN].value_counts()

class_percent = (
    df[LABEL_COLUMN]
    .value_counts(normalize=True)
    .mul(100)
)

for label in class_counts.index:

    print(
        f"{label:<35}"
        f"{class_counts[label]:>7,}"
        f" ({class_percent[label]:.2f}%)"
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE FEATURE STATISTICS")
print("=" * 70)

statistics = []

for feature in FEATURES:

    for label in sorted(df[LABEL_COLUMN].unique()):

        values = df.loc[
            df[LABEL_COLUMN] == label,
            feature
        ].dropna()

        if len(values) == 0:
            continue

        statistics.append(
            {
                "feature": feature,
                "class": label,
                "count": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "p10": values.quantile(0.10),
                "p25": values.quantile(0.25),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "min": values.min(),
                "max": values.max(),
            }
        )

stats_df = pd.DataFrame(statistics)

stats_df.to_csv(
    OUTPUT_STATS,
    index=False,
)

print(f"\nSaved:")
print(OUTPUT_STATS)


# ============================================================
# INDUSTRIAL FIRE VS PERSISTENT SOURCE
# ============================================================

industrial = df[
    df[LABEL_COLUMN] == "industrial_fire"
]

persistent = df[
    df[LABEL_COLUMN] == "persistent_industrial_source"
]

print("\n" + "=" * 70)
print("INDUSTRIAL FIRE vs PERSISTENT INDUSTRIAL SOURCE")
print("=" * 70)

print(
    f"\nIndustrial fire samples: "
    f"{len(industrial):,}"
)

print(
    f"Persistent source samples: "
    f"{len(persistent):,}"
)


comparison = []

for feature in FEATURES:

    a = industrial[feature].dropna()
    b = persistent[feature].dropna()

    if len(a) == 0 or len(b) == 0:
        continue

    comparison.append(
        {
            "feature": feature,

            "industrial_median":
                a.median(),

            "persistent_median":
                b.median(),

            "industrial_mean":
                a.mean(),

            "persistent_mean":
                b.mean(),

            "industrial_p90":
                a.quantile(0.90),

            "persistent_p90":
                b.quantile(0.90),

            "median_ratio":
                (
                    a.median() /
                    b.median()
                    if b.median() != 0
                    else np.nan
                ),

            "mean_ratio":
                (
                    a.mean() /
                    b.mean()
                    if b.mean() != 0
                    else np.nan
                ),
        }
    )

comparison_df = pd.DataFrame(comparison)

print("\nLargest median differences:")

comparison_df["absolute_median_difference"] = (
    comparison_df["industrial_median"]
    - comparison_df["persistent_median"]
).abs()

print(
    comparison_df
    .sort_values(
        "absolute_median_difference",
        ascending=False,
    )
    .head(20)
    .round(3)
    .to_string(index=False)
)


# ============================================================
# LOGISTIC REGRESSION FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION FEATURE IMPORTANCE")
print("=" * 70)

X = df[FEATURES]
y = df[LABEL_COLUMN]

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
                random_state=42,
            ),
        ),
    ]
)

model.fit(X, y)

classifier = model.named_steps["classifier"]

importance = np.abs(
    classifier.coef_
).mean(axis=0)

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": importance,
    }
).sort_values(
    "importance",
    ascending=False,
)

print(
    importance_df
    .head(25)
    .round(5)
    .to_string(index=False)
)

importance_df.to_csv(
    OUTPUT_IMPORTANCE,
    index=False,
)

print(f"\nSaved:")
print(OUTPUT_IMPORTANCE)


# ============================================================
# INDUSTRIAL FIRE FEATURE PROFILE
# ============================================================

print("\n" + "=" * 70)
print("INDUSTRIAL FIRE PROFILE")
print("=" * 70)

profile_features = [
    "duration_days",
    "active_days",
    "detection_count",
    "mean_frp",
    "median_frp",
    "max_frp",
    "high_confidence_count",
    "day_detection_count",
    "night_detection_count",
    "industrial_count_500m",
    "works_count_500m",
    "flare_count_1000m",
    "storage_tank_count_1000m",
    "powerplant_count_5000m",
    "refinery_count",
]

for feature in profile_features:

    if feature not in industrial.columns:
        continue

    values = industrial[feature].dropna()

    print(
        f"\n{feature}"
    )

    print(
        f"  median: {values.median():.3f}"
    )

    print(
        f"  mean:   {values.mean():.3f}"
    )

    print(
        f"  p90:    {values.quantile(.90):.3f}"
    )

    print(
        f"  max:    {values.max():.3f}"
    )


# ============================================================
# PERSISTENT SOURCE PROFILE
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENT INDUSTRIAL SOURCE PROFILE")
print("=" * 70)

for feature in profile_features:

    if feature not in persistent.columns:
        continue

    values = persistent[feature].dropna()

    print(
        f"\n{feature}"
    )

    print(
        f"  median: {values.median():.3f}"
    )

    print(
        f"  mean:   {values.mean():.3f}"
    )

    print(
        f"  p90:    {values.quantile(.90):.3f}"
    )

    print(
        f"  max:    {values.max():.3f}"
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)