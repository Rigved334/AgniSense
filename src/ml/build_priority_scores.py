from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


INPUT = Path("data/output/classified_thermal_episodes.gpkg")
OUTPUT = Path("data/output/classified_thermal_episodes_priority.gpkg")


def minmax(series):
    series = pd.to_numeric(series, errors="coerce")

    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(0.0, index=series.index)

    return (series - min_value) / (max_value - min_value)


def main():

    print("Loading classified episodes...")

    gdf = gpd.read_file(INPUT)

    print(f"Dataset shape: {gdf.shape}")

    # ---------------------------------------------------------
    # 1. Industrial-fire probability
    # ---------------------------------------------------------

    industrial_probability = (
        pd.to_numeric(
            gdf["probability_industrial_fire"],
            errors="coerce"
        )
        .fillna(0)
        .clip(0, 1)
    )

    # ---------------------------------------------------------
    # 2. Thermal severity
    # ---------------------------------------------------------

    thermal_severity = (
        0.50 * minmax(gdf["max_frp"])
        + 0.30 * minmax(gdf["mean_frp"])
        + 0.20 * minmax(gdf["detection_count"])
    )

    # ---------------------------------------------------------
    # 3. Industrial context
    # ---------------------------------------------------------

    industrial_proximity = (
        1.0
        - minmax(gdf["nearest_industrial_distance_m"])
    )

    industrial_context = (
        0.30 * minmax(gdf["industrial_count_500m"])
        + 0.20 * minmax(gdf["storage_tank_count_1000m"])
        + 0.15 * minmax(gdf["works_count_500m"])
        + 0.15 * minmax(gdf["flare_count_1000m"])
        + 0.10 * minmax(gdf["refinery_count"])
        + 0.10 * industrial_proximity
    )

    # ---------------------------------------------------------
    # 4. Base priority score
    # ---------------------------------------------------------

    base_score = (
        0.60 * industrial_probability
        + 0.25 * thermal_severity
        + 0.15 * industrial_context
    )

    # ---------------------------------------------------------
    # 5. Class-aware industrial priority
    # ---------------------------------------------------------

    # Only events classified as industrial fires receive
    # full industrial-investigation priority.

    class_multiplier = np.where(
        gdf["predicted_class"] == "industrial_fire",
        1.0,
        0.15
    )

    priority_score = (
        base_score * class_multiplier
    ).clip(0, 1)

    gdf["industrial_probability"] = industrial_probability
    gdf["thermal_severity_score"] = thermal_severity
    gdf["industrial_context_score"] = industrial_context
    gdf["priority_score"] = priority_score

    # ---------------------------------------------------------
    # 6. Percentile-based priority levels
    # ---------------------------------------------------------

    q95 = gdf["priority_score"].quantile(0.95)
    q80 = gdf["priority_score"].quantile(0.80)
    q50 = gdf["priority_score"].quantile(0.50)

    gdf["priority_level"] = np.select(
        [
            gdf["priority_score"] >= q95,
            gdf["priority_score"] >= q80,
            gdf["priority_score"] >= q50,
        ],
        [
            "Critical",
            "High",
            "Medium",
        ],
        default="Low"
    )

    # ---------------------------------------------------------
    # 7. Rank
    # ---------------------------------------------------------

    gdf = gdf.sort_values(
        "priority_score",
        ascending=False
    ).reset_index(drop=True)

    gdf["priority_rank"] = np.arange(
        1,
        len(gdf) + 1
    )

    # ---------------------------------------------------------
    # 8. Save
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    gdf.to_file(
        OUTPUT,
        driver="GPKG"
    )

    # ---------------------------------------------------------
    # 9. Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PRIORITY SCORING COMPLETE")
    print("=" * 60)

    print("\nPriority distribution:")
    print(
        gdf["priority_level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0
        )
    )

    print("\nPriority thresholds:")
    print(f"Critical: >= {q95:.4f}")
    print(f"High:     >= {q80:.4f}")
    print(f"Medium:   >= {q50:.4f}")

    print("\nTop 20 priority events:")

    print(
        gdf[
            [
                "priority_rank",
                "priority_score",
                "priority_level",
                "predicted_class",
                "industrial_probability",
                "max_frp",
                "duration_days",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()