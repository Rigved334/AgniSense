from pathlib import Path

import joblib
import geopandas as gpd
import pandas as pd


# ============================================================
# Paths
# ============================================================

DATASET = Path(
    "data/features/final_ml_dataset.parquet"
)

MODEL_PATH = Path(
    "models/sentinel2_logistic_regression.joblib"
)

SCHEMA_PATH = Path(
    "models/sentinel2_feature_schema.txt"
)

OUTPUT_DIR = Path(
    "data/output"
)

OUTPUT_FILE = OUTPUT_DIR / (
    "classified_thermal_episodes.gpkg"
)


# ============================================================
# Load model
# ============================================================

print("Loading model...")

model = joblib.load(
    MODEL_PATH
)

with open(
    SCHEMA_PATH,
    "r",
    encoding="utf-8",
) as f:

    FEATURES = [
        line.strip()
        for line in f
        if line.strip()
    ]


print(
    f"Loaded model with "
    f"{len(FEATURES)} features."
)


# ============================================================
# Load dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_parquet(
    DATASET
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# Validate
# ============================================================

missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing:
    raise ValueError(
        f"Missing features: {missing}"
    )


if df[FEATURES].isna().any().any():

    missing_counts = (
        df[FEATURES]
        .isna()
        .sum()
    )

    print(
        "\nMissing values:"
    )

    print(
        missing_counts[
            missing_counts > 0
        ]
    )

    raise ValueError(
        "Missing feature values."
    )


# ============================================================
# Predict
# ============================================================

print("\nRunning predictions...")

X = df[FEATURES]

predictions = model.predict(X)

probabilities = model.predict_proba(X)

classes = model.classes_


df["predicted_class"] = predictions


# ------------------------------------------------------------
# Add probability columns
# ------------------------------------------------------------

for i, class_name in enumerate(classes):

    df[
        f"probability_{class_name}"
    ] = probabilities[:, i]


# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

df["prediction_confidence"] = (
    probabilities.max(axis=1)
)


# ============================================================
# Create GIS output
# ============================================================

print(
    "\nCreating GeoDataFrame..."
)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"],
    ),
    crs="EPSG:4326",
)


# ============================================================
# Select useful GIS columns
# ============================================================

GIS_COLUMNS = [
    "episode_id",
    "latitude",
    "longitude",
    "start_date",
    "end_date",

    "predicted_class",
    "prediction_confidence",

    "probability_agricultural_fire",
    "probability_industrial_fire",
    "probability_persistent_industrial_source",
    "probability_wildfire",

    "active_days",
    "detection_count",
    "mean_frp",
    "median_frp",
    "max_frp",

    "duration_days",

    "nearest_industrial_distance_m",
    "industrial_count_500m",

    "works_count_500m",
    "flare_count_1000m",
    "storage_tank_count_1000m",
    "powerplant_count_5000m",
    "refinery_count",

    "landcover_class",
    "tree_cover_fraction_1km",
    "shrubland_fraction_1km",
    "grassland_fraction_1km",
    "cropland_fraction_1km",
    "builtup_fraction_1km",

    "sentinel2_ndvi",
    "sentinel2_ndbi",
    "sentinel2_ndwi",

    "geometry",
]


# ------------------------------------------------------------
# Keep only columns that exist
# ------------------------------------------------------------

GIS_COLUMNS = [
    column
    for column in GIS_COLUMNS
    if column in gdf.columns
]


gdf = gdf[
    GIS_COLUMNS
]


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


gdf.to_file(
    OUTPUT_FILE,
    layer="classified_episodes",
    driver="GPKG",
)


print(
    f"\nSaved GIS output:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# Summary
# ============================================================

print(
    "\nPrediction distribution:"
)

print(
    gdf["predicted_class"]
    .value_counts()
)


print(
    "\nConfidence statistics:"
)

print(
    gdf["prediction_confidence"]
    .describe()
)