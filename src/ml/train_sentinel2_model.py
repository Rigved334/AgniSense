from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# Paths
# ============================================================

DATASET = Path(
    "data/features/final_ml_dataset.parquet"
)

MODEL_DIR = Path(
    "models"
)

MODEL_PATH = MODEL_DIR / "sentinel2_logistic_regression.joblib"

FEATURES_PATH = MODEL_DIR / "sentinel2_feature_schema.txt"


# ============================================================
# Load dataset
# ============================================================

print("Loading final Sentinel-2 dataset...")

df = pd.read_parquet(DATASET)

print("Dataset shape:", df.shape)


# ============================================================
# Feature definition
# ============================================================

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


FEATURES = [
    col
    for col in df.columns
    if col not in EXCLUDED_COLUMNS
]


TARGET = "weak_label"


print("\nNumber of features:", len(FEATURES))

print("\nFeatures:")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# Validate
# ============================================================

if len(FEATURES) != 45:
    raise ValueError(
        f"Expected 45 features, found {len(FEATURES)}"
    )


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
    raise ValueError(
        "Feature matrix contains missing values."
    )


# ============================================================
# Prepare training data
# ============================================================

X = df[FEATURES]

y = df[TARGET]


print("\nClass distribution:")

print(
    y.value_counts()
)


# ============================================================
# Model
# ============================================================

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
                random_state=42,
            ),
        ),
    ]
)


# ============================================================
# Train
# ============================================================

print("\nTraining model...")

model.fit(
    X,
    y,
)


print("Training complete.")


# ============================================================
# Save model
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    model,
    MODEL_PATH,
)


print(
    f"\nModel saved:\n{MODEL_PATH}"
)


# ============================================================
# Save feature schema
# ============================================================

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8",
) as f:

    for feature in FEATURES:
        f.write(feature + "\n")


print(
    f"Feature schema saved:\n{FEATURES_PATH}"
)


# ============================================================
# Training sanity check
# ============================================================

predictions = model.predict(X)

training_accuracy = (
    predictions == y
).mean()


print(
    f"\nTraining accuracy: "
    f"{training_accuracy:.4f}"
)


print(
    "\nIMPORTANT:"
)

print(
    "This training accuracy is NOT a generalization metric."
)

print(
    "Use spatial cross-validation results for model evaluation."
)