from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# Paths
# ============================================================

DATASET = Path(
    "data/features/final_ml_dataset.parquet"
)

OUTPUT_DIR = Path(
    "data/output/model_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

print("Loading dataset...")

df = pd.read_parquet(DATASET)

print(
    f"Dataset shape: {df.shape}"
)


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
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


TARGET = "weak_label"

CLASSES = [
    "agricultural_fire",
    "industrial_fire",
    "persistent_industrial_source",
    "wildfire",
]


print(
    f"Number of features: {len(FEATURES)}"
)


# ============================================================
# Data
# ============================================================

X = df[FEATURES]

y = df[TARGET]

groups = df["h3_cell"]


# ============================================================
# Spatial cross-validation
# ============================================================

cv = GroupKFold(
    n_splits=5
)


all_true = []

all_predictions = []

all_probabilities = []

fold_results = []


print("\nStarting 5-fold spatial CV...")


for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups,
    ),
    start=1,
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"FOLD {fold}"
    )

    print(
        f"{'=' * 60}"
    )


    X_train = X.iloc[train_idx]

    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]

    y_test = y.iloc[test_idx]


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )


    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    all_true.extend(
        y_test.tolist()
    )

    all_predictions.extend(
        predictions.tolist()
    )

    all_probabilities.extend(
        probabilities.tolist()
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        f"Accuracy:             {accuracy:.4f}"
    )

    print(
        f"Macro F1:             {macro_f1:.4f}"
    )

    print(
        f"Macro Precision:      {macro_precision:.4f}"
    )

    print(
        f"Macro Recall:         {macro_recall:.4f}"
    )

    print(
        f"Industrial F1:        {industrial_f1:.4f}"
    )

    print(
        f"Industrial Precision: {industrial_precision:.4f}"
    )

    print(
        f"Industrial Recall:    {industrial_recall:.4f}"
    )


# ============================================================
# Fold summary
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


print(
    "\n\n"
    + "=" * 70
)

print(
    "SPATIAL CROSS-VALIDATION SUMMARY"
)

print(
    "=" * 70
)


for metric in [
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "industrial_f1",
    "industrial_precision",
    "industrial_recall",
]:

    mean = fold_df[metric].mean()

    std = fold_df[metric].std()

    print(
        f"{metric:25s}"
        f"{mean:.4f} ± {std:.4f}"
    )


# ============================================================
# Global out-of-fold predictions
# ============================================================

all_true = pd.Series(
    all_true,
    name="actual",
)

all_predictions = pd.Series(
    all_predictions,
    name="predicted",
)


# ============================================================
# Classification report
# ============================================================

print(
    "\n\n"
    + "=" * 70
)

print(
    "OUT-OF-FOLD CLASSIFICATION REPORT"
)

print(
    "=" * 70
)


report = classification_report(
    all_true,
    all_predictions,
    labels=CLASSES,
    target_names=[
        "Agricultural Fire",
        "Industrial Fire",
        "Persistent Industrial Source",
        "Wildfire",
    ],
    digits=4,
    zero_division=0,
)


print(report)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_true,
    all_predictions,
    labels=CLASSES,
)


print(
    "\n"
    + "=" * 70
)

print(
    "CONFUSION MATRIX"
)

print(
    "=" * 70
)


cm_df = pd.DataFrame(
    cm,
    index=[
        "Agricultural Fire",
        "Industrial Fire",
        "Persistent Industrial Source",
        "Wildfire",
    ],
    columns=[
        "Agricultural Fire",
        "Industrial Fire",
        "Persistent Industrial Source",
        "Wildfire",
    ],
)


print(cm_df)


# ============================================================
# Save confusion matrix
# ============================================================

cm_df.to_csv(
    OUTPUT_DIR
    / "confusion_matrix.csv"
)


# ============================================================
# Plot confusion matrix
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Agricultural",
        "Industrial",
        "Persistent",
        "Wildfire",
    ],
)


display.plot(
    ax=ax,
    values_format="d",
    xticks_rotation=30,
)


ax.set_title(
    "Out-of-Fold Spatial Confusion Matrix"
)

plt.tight_layout()


cm_path = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)


plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight",
)


plt.close()


print(
    f"\nSaved confusion matrix:"
)

print(cm_path)


# ============================================================
# Per-class error analysis
# ============================================================

analysis_df = pd.DataFrame(
    {
        "actual": all_true,
        "predicted": all_predictions,
    }
)


analysis_df["correct"] = (
    analysis_df["actual"]
    == analysis_df["predicted"]
)


print(
    "\n\n"
    + "=" * 70
)

print(
    "PER-CLASS ERROR ANALYSIS"
)

print(
    "=" * 70
)


for class_name in CLASSES:

    subset = analysis_df[
        analysis_df["actual"]
        == class_name
    ]

    errors = subset[
        ~subset["correct"]
    ]

    print(
        f"\n{class_name}"
    )

    print(
        f"Total:   {len(subset)}"
    )

    print(
        f"Correct: {subset['correct'].sum()}"
    )

    print(
        f"Errors:  {len(errors)}"
    )


    if len(errors) > 0:

        print(
            "Predicted as:"
        )

        print(
            errors[
                "predicted"
            ].value_counts()
        )


# ============================================================
# Save out-of-fold predictions
# ============================================================

oof_df = pd.DataFrame(
    {
        "actual": all_true,
        "predicted": all_predictions,
    }
)


oof_path = (
    OUTPUT_DIR
    / "out_of_fold_predictions.csv"
)


oof_df.to_csv(
    oof_path,
    index=False,
)


print(
    f"\nSaved OOF predictions:"
)

print(oof_path)


# ============================================================
# Final message
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "MODEL EVALUATION COMPLETE"
)

print(
    "=" * 70
)