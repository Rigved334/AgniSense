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
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/features/ml_dataset.parquet"
OUTPUT_FILE = "data/features/spatial_cv_results.csv"

LABEL_COLUMN = "weak_label"
GROUP_COLUMN = "h3_cell"

N_SPLITS = 5
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("REPEATED SPATIAL CROSS-VALIDATION")
print("=" * 70)

df = pd.read_parquet(INPUT_FILE)

print(f"\nDataset: {INPUT_FILE}")
print(f"Rows: {len(df):,}")

features = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

print(f"Features: {len(features)}")
print(f"Spatial groups: {df[GROUP_COLUMN].nunique():,}")

X = df[features]
y = df[LABEL_COLUMN]
groups = df[GROUP_COLUMN]


# ============================================================
# CHECK
# ============================================================

if len(features) != 42:
    print(
        f"\nWARNING: Expected 42 features, "
        f"but found {len(features)}."
    )

if y.isna().any():
    raise ValueError("Missing labels detected.")

if groups.isna().any():
    raise ValueError("Missing H3 groups detected.")


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# SPATIAL CROSS-VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=N_SPLITS
)

labels = sorted(y.unique())

print("\nClasses:")
for label in labels:
    print(f"  {label}")


results = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1,
):

    print("\n" + "=" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 70)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print(f"Train samples: {len(train_idx):,}")
    print(f"Test samples:  {len(test_idx):,}")

    print(
        f"Train H3 cells: {groups_train.nunique():,}"
    )

    print(
        f"Test H3 cells:  {groups_test.nunique():,}"
    )

    # --------------------------------------------------------
    # VERIFY NO SPATIAL LEAKAGE
    # --------------------------------------------------------

    overlap = set(groups_train) & set(groups_test)

    print(f"H3 overlap: {len(overlap)}")

    if overlap:
        raise RuntimeError(
            "Spatial leakage detected!"
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nTest class distribution:")

    for label in labels:

        count = (y_test == label).sum()

        print(
            f"  {label:<30} {count:>5}"
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    print("\nMetrics:")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Macro F1:              {macro_f1:.4f}")
    print(f"Weighted F1:           {weighted_f1:.4f}")
    print(f"Macro Precision:       {macro_precision:.4f}")
    print(f"Macro Recall:          {macro_recall:.4f}")
    print(f"Industrial Fire F1:    {industrial_f1:.4f}")
    print(f"Industrial Precision:  {industrial_precision:.4f}")
    print(f"Industrial Recall:     {industrial_recall:.4f}")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    print("\nConfusion matrix:")

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(cm_df)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append(
        {
            "fold": fold,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_h3_cells": groups_train.nunique(),
            "test_h3_cells": groups_test.nunique(),
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


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df.round(4).to_string(
        index=False
    )
)


# ============================================================
# MEAN / STD
# ============================================================

metric_columns = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "industrial_f1",
    "industrial_precision",
    "industrial_recall",
]

print("\n")
print("=" * 70)
print("MEAN ± STD")
print("=" * 70)

for metric in metric_columns:

    mean = results_df[metric].mean()
    std = results_df[metric].std()

    print(
        f"{metric:<25} "
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"\nSaved results:")
print(OUTPUT_FILE)