from pathlib import Path

import pandas as pd


ML_DATASET = Path(
    "data/features/ml_dataset.parquet"
)

S2_DATASET = Path(
    "data/features/sentinel2_episode_features_batch500.parquet"
)

OUTPUT_DATASET = Path(
    "data/features/final_ml_dataset.parquet"
)


S2_FEATURES = [
    "sentinel2_ndvi",
    "sentinel2_ndbi",
    "sentinel2_ndwi",
]


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


print("Loading ML dataset...")
ml = pd.read_parquet(ML_DATASET)

print("ML dataset:", ml.shape)


print("\nLoading Sentinel-2 dataset...")
s2 = pd.read_parquet(S2_DATASET)

print("Sentinel-2 dataset:", s2.shape)


# ------------------------------------------------------------
# Keep only successful Sentinel-2 observations
# ------------------------------------------------------------

s2 = s2[
    s2["sentinel2_found"] == True
].copy()

print(
    "Usable Sentinel-2 observations:",
    len(s2)
)


# ------------------------------------------------------------
# Select only the features we decided to keep
# ------------------------------------------------------------

required = [
    "episode_id",
    *S2_FEATURES,
]

missing = [
    col
    for col in required
    if col not in s2.columns
]

if missing:
    raise ValueError(
        f"Missing Sentinel-2 columns: {missing}"
    )


s2 = s2[required].drop_duplicates(
    subset="episode_id"
)


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

df = ml.merge(
    s2,
    on="episode_id",
    how="inner",
)

print(
    "\nFinal dataset shape:",
    df.shape
)


# ------------------------------------------------------------
# Build feature list
# ------------------------------------------------------------

base_features = [
    col
    for col in ml.columns
    if col not in EXCLUDED_COLUMNS
]

base_features = [
    col
    for col in base_features
    if col not in S2_FEATURES
]


final_features = (
    base_features
    + S2_FEATURES
)


print(
    "\nBase features:",
    len(base_features)
)

print(
    "Sentinel-2 features:",
    len(S2_FEATURES)
)

print(
    "Final features:",
    len(final_features)
)


# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

missing_features = [
    col
    for col in final_features
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing final features: {missing_features}"
    )


if df[final_features].isna().any().any():

    missing_counts = (
        df[final_features]
        .isna()
        .sum()
    )

    print("\nMissing values:")
    print(
        missing_counts[
            missing_counts > 0
        ]
    )

    raise ValueError(
        "Final feature matrix contains missing values."
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_parquet(
    OUTPUT_DATASET,
    index=False,
)

print(
    f"\nSaved:\n{OUTPUT_DATASET}"
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\nFinal feature columns:")

for i, feature in enumerate(
    final_features,
    start=1,
):
    print(
        f"{i:02d}. {feature}"
    )