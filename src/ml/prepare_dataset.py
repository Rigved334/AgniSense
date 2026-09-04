from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


INPUT = Path("data/features/thermal_episode_weak_labels.parquet")
OUTPUT = Path("data/features/ml_dataset.parquet")


LABEL_COLUMN = "weak_label"

# Columns that identify the observation rather than describe it.
NON_FEATURE_COLUMNS = [
    "h3_cell",
    "episode_id",
    "start_date",
    "end_date",
    "latitude",
    "longitude",
    LABEL_COLUMN,
    "label_conflict",
    "strong_industrial_context",
]

# Keep uncertain episodes in the master dataset, but they are not
# used for supervised training in this first experiment.
VALID_LABELS = {
    "wildfire",
    "agricultural_fire",
    "persistent_industrial_source",
    "industrial_fire",
}


def main():
    print("=" * 70)
    print("PREPARING ML DATASET")
    print("=" * 70)

    print(f"\nLoading: {INPUT}")
    df = pd.read_parquet(INPUT)

    print(f"Total episodes: {len(df):,}")

    # ------------------------------------------------------------
    # Keep only weakly labeled examples for supervised learning
    # ------------------------------------------------------------

    labeled = df[df[LABEL_COLUMN].isin(VALID_LABELS)].copy()

    print(f"Labeled episodes: {len(labeled):,}")

    print("\nClass distribution:")
    print(labeled[LABEL_COLUMN].value_counts().to_string())

    # ------------------------------------------------------------
    # Define predictors
    # ------------------------------------------------------------

    feature_columns = [
        c
        for c in labeled.columns
        if c not in NON_FEATURE_COLUMNS
    ]

    print(f"\nFeature count: {len(feature_columns)}")

    print("\nFeatures:")
    for column in feature_columns:
        print(f"  {column}")

    # ------------------------------------------------------------
    # Create spatial groups
    # ------------------------------------------------------------

    # H3 cell is deliberately used as the grouping variable.
    groups = labeled["h3_cell"]

    # ------------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42,
    )

    train_val_idx, test_idx = next(
        splitter.split(
            labeled,
            groups=groups,
        )
    )

    train_val = labeled.iloc[train_val_idx].copy()
    test = labeled.iloc[test_idx].copy()

    # ------------------------------------------------------------
    # Train/validation split
    # ------------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1765,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            train_val,
            groups=train_val["h3_cell"],
        )
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()

    # ------------------------------------------------------------
    # Add split information
    # ------------------------------------------------------------

    train["dataset_split"] = "train"
    validation["dataset_split"] = "validation"
    test["dataset_split"] = "test"

    result = pd.concat(
        [train, validation, test],
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print("\nRows:")
    print(result["dataset_split"].value_counts().to_string())

    print("\nClass distribution:")
    print(
        pd.crosstab(
            result["dataset_split"],
            result[LABEL_COLUMN],
        ).to_string()
    )

    print("\nUnique H3 cells:")
    print(
        result.groupby("dataset_split")["h3_cell"]
        .nunique()
        .to_string()
    )

    # Verify there is no H3 overlap.
    train_cells = set(
        result.loc[
            result["dataset_split"] == "train",
            "h3_cell",
        ]
    )

    val_cells = set(
        result.loc[
            result["dataset_split"] == "validation",
            "h3_cell",
        ]
    )

    test_cells = set(
        result.loc[
            result["dataset_split"] == "test",
            "h3_cell",
        ]
    )

    print("\nH3 overlap checks:")
    print("Train ∩ Validation:", len(train_cells & val_cells))
    print("Train ∩ Test:", len(train_cells & test_cells))
    print("Validation ∩ Test:", len(val_cells & test_cells))

    print("\nOutput:")
    print(OUTPUT)

    print("\nShape:", result.shape)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()