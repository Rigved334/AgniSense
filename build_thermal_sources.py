from pathlib import Path

import h3
import pandas as pd


INPUT = Path("data/firms/viirs_clean.parquet")

OUTPUT_DAILY = Path(
    "data/firms/thermal_daily.parquet"
)

OUTPUT_SOURCES = Path(
    "data/firms/thermal_sources.parquet"
)

OUTPUT_EPISODES = Path(
    "data/firms/thermal_episodes.parquet"
)

H3_RESOLUTION = 8

# Gap larger than this starts a new activity episode.
EPISODE_GAP_DAYS = 3


def assign_h3_cells(df: pd.DataFrame) -> pd.DataFrame:
    print("Assigning H3 cells...")

    df = df.copy()

    df["h3_cell"] = [
        h3.latlng_to_cell(
            lat,
            lon,
            H3_RESOLUTION,
        )
        for lat, lon in zip(
            df["latitude"],
            df["longitude"],
        )
    ]

    return df


def build_daily_table(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating detections by H3 cell and day...")

    df = df.copy()

    df["date"] = df["acq_datetime"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    grouped = (
        df.groupby(
            [
                "h3_cell",
                "date",
            ],
            observed=True,
        )
        .agg(
            detection_count=("frp", "size"),

            mean_frp=("frp", "mean"),
            median_frp=("frp", "median"),
            max_frp=("frp", "max"),
            std_frp=("frp", "std"),

            mean_bright_ti4=("bright_ti4", "mean"),
            max_bright_ti4=("bright_ti4", "max"),

            mean_bright_ti5=("bright_ti5", "mean"),
            max_bright_ti5=("bright_ti5", "max"),

            day_detection_count=(
                "daynight",
                lambda x: (x == "D").sum(),
            ),

            night_detection_count=(
                "daynight",
                lambda x: (x == "N").sum(),
            ),

            high_confidence_count=(
                "confidence_code",
                lambda x: (x == "h").sum(),
            ),

            nominal_confidence_count=(
                "confidence_code",
                lambda x: (x == "n").sum(),
            ),

            low_confidence_count=(
                "confidence_code",
                lambda x: (x == "l").sum(),
            ),

            satellite_count=(
                "satellite",
                "nunique",
            ),
        )
        .reset_index()
    )

    grouped["std_frp"] = grouped["std_frp"].fillna(0)

    return grouped.sort_values(
        ["h3_cell", "date"]
    ).reset_index(drop=True)


def build_sources(daily: pd.DataFrame) -> pd.DataFrame:
    print("Building thermal sources...")

    sources = (
        daily.groupby(
            "h3_cell",
            observed=True,
        )
        .agg(
            first_detected=("date", "min"),
            last_detected=("date", "max"),

            active_days=("date", "nunique"),

            detection_count=("detection_count", "sum"),

            mean_frp=("mean_frp", "mean"),
            median_frp=("median_frp", "median"),
            max_frp=("max_frp", "max"),
            std_frp=("mean_frp", "std"),

            mean_bright_ti4=(
                "mean_bright_ti4",
                "mean",
            ),

            max_bright_ti4=(
                "max_bright_ti4",
                "max",
            ),

            mean_bright_ti5=(
                "mean_bright_ti5",
                "mean",
            ),

            max_bright_ti5=(
                "max_bright_ti5",
                "max",
            ),

            day_detection_count=(
                "day_detection_count",
                "sum",
            ),

            night_detection_count=(
                "night_detection_count",
                "sum",
            ),

            high_confidence_count=(
                "high_confidence_count",
                "sum",
            ),

            nominal_confidence_count=(
                "nominal_confidence_count",
                "sum",
            ),

            low_confidence_count=(
                "low_confidence_count",
                "sum",
            ),

            satellite_count=(
                "satellite_count",
                "max",
            ),
        )
        .reset_index()
    )

    sources["calendar_span_days"] = (
        sources["last_detected"]
        - sources["first_detected"]
    ).dt.days + 1

    sources["persistence_ratio"] = (
        sources["active_days"]
        / sources["calendar_span_days"]
    )

    sources["std_frp"] = sources["std_frp"].fillna(0)

    return sources


def assign_episode_ids(daily: pd.DataFrame) -> pd.DataFrame:
    print("Creating activity episodes...")

    daily = daily.copy()

    daily = daily.sort_values(
        ["h3_cell", "date"]
    ).reset_index(drop=True)

    daily["days_since_previous"] = (
        daily.groupby("h3_cell")["date"]
        .diff()
        .dt.days
    )

    daily["new_episode"] = (
        daily["days_since_previous"].isna()
        | (
            daily["days_since_previous"]
            > EPISODE_GAP_DAYS
        )
    )

    daily["episode_number"] = (
        daily.groupby("h3_cell")["new_episode"]
        .cumsum()
    )

    daily["episode_id"] = (
        daily["h3_cell"]
        + "_"
        + daily["episode_number"]
        .astype(str)
    )

    return daily


def build_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating activity episodes...")

    episodes = (
        daily.groupby(
            [
                "h3_cell",
                "episode_id",
            ],
            observed=True,
        )
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),

            active_days=("date", "nunique"),

            detection_count=(
                "detection_count",
                "sum",
            ),

            mean_frp=("mean_frp", "mean"),
            median_frp=("median_frp", "median"),
            max_frp=("max_frp", "max"),
            std_frp=("mean_frp", "std"),

            mean_bright_ti4=(
                "mean_bright_ti4",
                "mean",
            ),

            max_bright_ti4=(
                "max_bright_ti4",
                "max",
            ),

            mean_bright_ti5=(
                "mean_bright_ti5",
                "mean",
            ),

            max_bright_ti5=(
                "max_bright_ti5",
                "max",
            ),

            day_detection_count=(
                "day_detection_count",
                "sum",
            ),

            night_detection_count=(
                "night_detection_count",
                "sum",
            ),

            high_confidence_count=(
                "high_confidence_count",
                "sum",
            ),

            nominal_confidence_count=(
                "nominal_confidence_count",
                "sum",
            ),

            low_confidence_count=(
                "low_confidence_count",
                "sum",
            ),
        )
        .reset_index()
    )

    episodes["duration_days"] = (
        episodes["end_date"]
        - episodes["start_date"]
    ).dt.days + 1

    episodes["std_frp"] = episodes["std_frp"].fillna(0)

    return episodes


def add_h3_centers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    print("Calculating H3 cell centers...")

    centers = df["h3_cell"].drop_duplicates().to_list()

    center_map = {
        cell: h3.cell_to_latlng(cell)
        for cell in centers
    }

    df = df.copy()

    df["latitude"] = df["h3_cell"].map(
        lambda cell: center_map[cell][0]
    )

    df["longitude"] = df["h3_cell"].map(
        lambda cell: center_map[cell][1]
    )

    return df


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT.resolve()}"
        )

    OUTPUT_DAILY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading FIRMS GeoParquet...")

    df = pd.read_parquet(INPUT)

    print(f"Rows loaded: {len(df):,}")

    required = {
        "latitude",
        "longitude",
        "acq_datetime",
        "frp",
        "bright_ti4",
        "bright_ti5",
        "daynight",
        "confidence_code",
        "satellite",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # H3 assignment
    # ---------------------------------------------------------

    df = assign_h3_cells(df)

    print(
        f"Unique H3 cells: "
        f"{df['h3_cell'].nunique():,}"
    )

    # ---------------------------------------------------------
    # Daily aggregation
    # ---------------------------------------------------------

    daily = build_daily_table(df)

    print(
        f"Daily cell records: "
        f"{len(daily):,}"
    )

    daily.to_parquet(
        OUTPUT_DAILY,
        index=False,
    )

    # ---------------------------------------------------------
    # Source aggregation
    # ---------------------------------------------------------

    sources = build_sources(daily)

    sources = add_h3_centers(sources)

    print(
        f"Thermal sources: "
        f"{len(sources):,}"
    )

    sources.to_parquet(
        OUTPUT_SOURCES,
        index=False,
    )

    # ---------------------------------------------------------
    # Episode generation
    # ---------------------------------------------------------

    daily = assign_episode_ids(daily)

    episodes = build_episodes(daily)

    episodes = add_h3_centers(episodes)

    print(
        f"Thermal episodes: "
        f"{len(episodes):,}"
    )

    episodes.to_parquet(
        OUTPUT_EPISODES,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n========================================")
    print("THERMAL SOURCE GENERATION COMPLETE")
    print("========================================")

    print(
        f"Raw detections:  {len(df):,}"
    )

    print(
        f"Unique H3 cells: {df['h3_cell'].nunique():,}"
    )

    print(
        f"Sources:         {len(sources):,}"
    )

    print(
        f"Episodes:        {len(episodes):,}"
    )

    print("\nOutput files:")

    print(
        f"  {OUTPUT_DAILY.resolve()}"
    )

    print(
        f"  {OUTPUT_SOURCES.resolve()}"
    )

    print(
        f"  {OUTPUT_EPISODES.resolve()}"
    )


if __name__ == "__main__":
    main()