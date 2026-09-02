from pathlib import Path

import geopandas as gpd
import pandas as pd


INPUT = Path("india_viirs_2026_jan_aug.csv")
OUTPUT = Path("data/firms/viirs_clean.parquet")


REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "frp",
    "daynight",
    "type",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required FIRMS columns: {missing}"
        )


def clean_firms(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    df = df.copy()

    # ---------------------------------------------------------
    # Numeric columns
    # ---------------------------------------------------------

    numeric_columns = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "scan",
    "track",
    "frp",
    ]

    
    # FIRMS version can contain mixed representations.
    # Store it consistently as text so Parquet/Arrow can serialize it.
    df["version"] = df["version"].astype("string")

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    valid_coordinates = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    df = df.loc[valid_coordinates].copy()

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Acquisition time
    # ---------------------------------------------------------

    time_numeric = pd.to_numeric(
        df["acq_time"],
        errors="coerce",
    )

    time_string = (
        time_numeric
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].dt.strftime("%Y-%m-%d")
        + " "
        + time_string.str[:2]
        + ":"
        + time_string.str[2:],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    # ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------

    # Preserve the original FIRMS confidence code
    df["confidence_code"] = df["confidence"].astype("string")

    confidence_map = {
        "l": "low",
        "n": "nominal",
        "h": "high",
    }

    df["confidence"] = (
        df["confidence_code"]
        .str.lower()
        .map(confidence_map)
        .fillna(df["confidence_code"])
    )

    # ---------------------------------------------------------
    # Remove records with unusable core fields
    # ---------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_datetime",
            "bright_ti4",
            "bright_ti5",
            "frp",
        ]
    ).copy()

    # ---------------------------------------------------------
    # Remove exact duplicate observations
    # ---------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "latitude",
            "longitude",
            "acq_datetime",
            "satellite",
            "instrument",
            "frp",
        ]
    ).copy()

    duplicates_removed = before - len(df)

    # ---------------------------------------------------------
    # Sort
    # ---------------------------------------------------------

    df = df.sort_values(
        "acq_datetime"
    ).reset_index(drop=True)

    return df, duplicates_removed


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT.resolve()}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading FIRMS CSV...")
    df = pd.read_csv(
    INPUT,
    low_memory=False,
    )

    print(f"Original rows: {len(df):,}")

    df, duplicates_removed = clean_firms(df)

    print(f"Clean rows: {len(df):,}")
    print(f"Exact duplicates removed: {duplicates_removed:,}")

    print("\nCreating point geometries...")

    geometry = gpd.points_from_xy(
    df["longitude"],
    df["latitude"],
    )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326",
    )

    print("\nFinal validation:")
    print(f"Rows: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")
    print(
        "Geometry types:",
        gdf.geometry.geom_type.value_counts().to_dict(),
    )

    print(
        "\nBounds:",
        gdf.total_bounds,
    )

    print(
        "\nDate range:",
        gdf["acq_datetime"].min(),
        "→",
        gdf["acq_datetime"].max(),
    )

    print("\nSaving GeoParquet...")

    gdf.to_parquet(
        OUTPUT,
        index=False,
    )

    print("\nDone.")
    print(f"Output: {OUTPUT.resolve()}")
    print(
        f"Size: "
        f"{OUTPUT.stat().st_size / (1024 ** 2):.2f} MB"
    )


if __name__ == "__main__":
    main()