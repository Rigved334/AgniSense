from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path("data/features/ml_dataset.parquet")


LABEL_COLUMN = "weak_label"
SPLIT_COLUMN = "dataset_split"

NON_FEATURE_COLUMNS = [
    "h3_cell",
    "episode_id",
    "start_date",
    "end_date",
    "latitude",
    "longitude",
    LABEL_COLUMN,
    SPLIT_COLUMN,
    "label_conflict",
    "strong_industrial_context",
]


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("\nTraining...")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    val_pred = model.predict(X_val)

    print("\nVALIDATION")
    print("-" * 70)

    print(f"Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_val, val_pred, average='macro'):.4f}")
    print(
        f"Weighted F1: {f1_score(y_val, val_pred, average='weighted'):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            val_pred,
            digits=4,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    test_pred = model.predict(X_test)

    print("\nTEST")
    print("-" * 70)

    print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_test, test_pred, average='macro'):.4f}")
    print(
        f"Weighted F1: {f1_score(y_test, test_pred, average='weighted'):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            digits=4,
            zero_division=0,
        )
    )

    print("\nConfusion matrix:")

    labels = sorted(y_test.unique())

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    print(cm_df.to_string())

    # ------------------------------------------------------------
    # Industrial fire specifically
    # ------------------------------------------------------------

    target = "industrial_fire"

    if target in labels:
        report = classification_report(
            y_test,
            test_pred,
            output_dict=True,
            zero_division=0,
        )

        metrics = report[target]

        print("\nINDUSTRIAL FIRE")
        print("-" * 70)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1-score']:.4f}")
        print(f"Support:   {int(metrics['support'])}")


def main():
    print("=" * 70)
    print("TRAINING BASELINE MODELS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    print(f"\nLoading: {INPUT}")

    df = pd.read_parquet(INPUT)

    print(f"Rows: {len(df):,}")

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    train = df[df[SPLIT_COLUMN] == "train"].copy()
    validation = df[df[SPLIT_COLUMN] == "validation"].copy()
    test = df[df[SPLIT_COLUMN] == "test"].copy()

    feature_columns = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    print(f"Features: {len(feature_columns)}")

    X_train = train[feature_columns]
    y_train = train[LABEL_COLUMN]

    X_val = validation[feature_columns]
    y_val = validation[LABEL_COLUMN]

    X_test = test[feature_columns]
    y_test = test[LABEL_COLUMN]

    print("\nTrain:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    # ------------------------------------------------------------
    # 1. Dummy classifier
    # ------------------------------------------------------------

    dummy = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent",
                ),
            ),
        ]
    )

    evaluate_model(
        "1. DUMMY CLASSIFIER",
        dummy,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------
    # 2. Logistic regression
    # ------------------------------------------------------------

    logistic = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    evaluate_model(
        "2. LOGISTIC REGRESSION",
        logistic,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------

    random_forest = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    evaluate_model(
        "3. RANDOM FOREST",
        random_forest,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()