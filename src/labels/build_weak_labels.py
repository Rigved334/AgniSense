from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/features/thermal_episode_osm_categories.parquet")
OUTPUT = Path("data/features/thermal_episode_weak_labels.parquet")


def main():
    print("=" * 70)
    print("BUILDING WEAK LABELS")
    print("=" * 70)

    print(f"\nLoading: {INPUT}")
    df = pd.read_parquet(INPUT)

    print(f"Episodes: {len(df):,}")

    # ------------------------------------------------------------
    # 1. Strong industrial context
    # ------------------------------------------------------------

    strong_industrial_context = (
        (df["industrial_count_500m"] > 0)
        | (df["works_count_500m"] > 0)
        | (df["refinery_count"] > 0)
        | (df["storage_tank_count_1000m"] > 0)
        | (df["flare_count_1000m"] > 0)
    )

    # ------------------------------------------------------------
    # 2. Persistent industrial source
    # ------------------------------------------------------------

    persistent = (
        (df["duration_days"] >= 3)
        & strong_industrial_context
        & (
            (df["detection_count"] >= 5)
            | (df["high_confidence_count"] >= 2)
        )
    )

    # ------------------------------------------------------------
    # 3. Acute industrial fire
    # ------------------------------------------------------------
    
    acute_thermal = (
        (df["max_frp"] >= 50)
        | (
            (df["high_confidence_count"] >= 1)
            & (df["detection_count"] >= 3)
        )
    )
    
    industrial_fire = (
        strong_industrial_context
        & (df["duration_days"] <= 3)
        & acute_thermal
        & ~persistent
    )

    # ------------------------------------------------------------
    # 4. Vegetation context
    # ------------------------------------------------------------

    vegetation = (
        (df["tree_cover_fraction_1km"] >= 0.50)
        | (df["shrubland_fraction_1km"] >= 0.50)
        | (df["grassland_fraction_1km"] >= 0.50)
    )

    strong_natural_fire = (
        (df["high_confidence_count"] >= 1)
        & (
            (df["max_frp"] >= 20)
            | (df["detection_count"] >= 3)
        )
    )

    wildfire = (
        vegetation
        & strong_natural_fire
        & ~strong_industrial_context
    )

    # ------------------------------------------------------------
    # 5. Agricultural fire
    # ------------------------------------------------------------

    agricultural = (
        (df["cropland_fraction_1km"] >= 0.70)
        & strong_natural_fire
        & ~strong_industrial_context
    )

    # ------------------------------------------------------------
    # 6. Resolve conflicts
    # ------------------------------------------------------------

    # An episode satisfying both natural-fire classes is ambiguous.
    natural_conflict = wildfire & agricultural

    wildfire = wildfire & ~natural_conflict
    agricultural = agricultural & ~natural_conflict

    # ------------------------------------------------------------
    # 7. Resolve all class conflicts
    # ------------------------------------------------------------

    label_count = (
        persistent.astype(int)
        + industrial_fire.astype(int)
        + wildfire.astype(int)
        + agricultural.astype(int)
    )

    conflicts = label_count > 1

    persistent = persistent & ~conflicts
    industrial_fire = industrial_fire & ~conflicts
    wildfire = wildfire & ~conflicts
    agricultural = agricultural & ~conflicts

    # ------------------------------------------------------------
    # 8. Assign labels
    # ------------------------------------------------------------

    df["weak_label"] = "uncertain"

    df.loc[persistent, "weak_label"] = "persistent_industrial_source"
    df.loc[industrial_fire, "weak_label"] = "industrial_fire"
    df.loc[wildfire, "weak_label"] = "wildfire"
    df.loc[agricultural, "weak_label"] = "agricultural_fire"

    # ------------------------------------------------------------
    # 9. Store label provenance
    # ------------------------------------------------------------

    df["label_conflict"] = conflicts

    df["strong_industrial_context"] = strong_industrial_context

    # ------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------
    # 11. Report
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL SUMMARY")
    print("=" * 70)

    counts = df["weak_label"].value_counts()

    print(counts.to_string())

    print("\nPercentages:")

    percentages = (
        df["weak_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    print(percentages.to_string())

    print("\nLabel conflicts:", int(conflicts.sum()))

    print("\nOutput:")
    print(OUTPUT)

    print("\nShape:", df.shape)

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()