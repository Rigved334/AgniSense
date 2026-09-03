from pathlib import Path
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


INPUT_FILE = Path(
    "data/features/thermal_episode_india.parquet"
)

WORLD_COVER_DIR = Path(
    "data/landcover/worldcover_2021"
)

OUTPUT_FILE = Path(
    "data/features/thermal_episode_landcover.parquet"
)

# WorldCover classes
CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "builtup",
    60: "bare",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# Approximate radius for local land-cover context.
# We use a square window around the event because this
# is much faster than constructing 586k individual buffers.
RADIUS_DEG = 0.009


def tile_name(lat, lon):
    """Return WorldCover 3° x 3° tile name."""

    lat_start = math.floor(lat / 3) * 3
    lon_start = math.floor(lon / 3) * 3

    lat_prefix = "N" if lat_start >= 0 else "S"
    lon_prefix = "E" if lon_start >= 0 else "W"

    return (
        f"{lat_prefix}{abs(lat_start):02d}"
        f"{lon_prefix}{abs(lon_start):03d}"
    )


def initialize_features(index):
    """Create empty output dataframe."""

    result = pd.DataFrame(index=index)

    result["landcover_class"] = pd.Series(
        pd.NA,
        index=index,
        dtype="Int16",
    )

    for name in CLASSES.values():
        result[f"{name}_fraction_1km"] = 0.0

    return result


def extract_point_class(src, lon, lat):
    """Extract the WorldCover class at one coordinate."""

    try:
        values = list(
            src.sample([(lon, lat)])
        )

        if not values:
            return pd.NA

        value = values[0][0]

        if value in CLASSES:
            return int(value)

        return pd.NA

    except Exception:
        return pd.NA


def extract_window_fractions(src, lon, lat):
    """
    Extract land-cover composition from approximately
    a 1 km x 1 km window around the point.
    """

    try:

        # Convert approximately 0.009 degrees around point.
        # Longitude distance varies with latitude, so use
        # a slightly conservative longitude window.
        lat_radius = RADIUS_DEG

        cos_lat = max(
            math.cos(math.radians(lat)),
            0.2,
        )

        lon_radius = RADIUS_DEG / cos_lat

        left = lon - lon_radius
        right = lon + lon_radius
        bottom = lat - lat_radius
        top = lat + lat_radius

        window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src.transform,
        )

        # Limit window size in case of edge/pathological cases.
        if window.width <= 0 or window.height <= 0:
            return {}

        data = src.read(
            1,
            window=window,
        )

        if data.size == 0:
            return {}

        data = data.ravel()

        # WorldCover uses 0 as nodata in these maps.
        data = data[data != 0]

        if data.size == 0:
            return {}

        result = {}

        for code, name in CLASSES.items():

            fraction = np.count_nonzero(
                data == code
            ) / data.size

            result[
                f"{name}_fraction_1km"
            ] = float(fraction)

        return result

    except Exception:
        return {}


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading India thermal episodes...")

    df = pd.read_parquet(INPUT_FILE)

    print(
        f"Episodes: {len(df):,}"
    )

    print("Determining WorldCover tiles...")

    df["_tile"] = [
        tile_name(lat, lon)
        for lat, lon in zip(
            df["latitude"],
            df["longitude"],
        )
    ]

    tiles = sorted(
        df["_tile"].unique()
    )

    print(
        f"Tiles required: {len(tiles)}"
    )

    # Prepare result indexed exactly like df.
    result = initialize_features(
        df.index
    )

    # --------------------------------------------------------
    # Process one WorldCover tile at a time
    # --------------------------------------------------------

    for tile_number, tile in enumerate(
        tiles,
        start=1,
    ):

        filename = (
            f"ESA_WorldCover_10m_2021_v200_"
            f"{tile}_Map.tif"
        )

        raster_path = (
            WORLD_COVER_DIR / filename
        )

        print(
            f"\n[{tile_number}/{len(tiles)}] "
            f"{tile}"
        )

        if not raster_path.exists():

            print(
                "  WARNING: raster missing. "
                "Skipping."
            )

            continue

        tile_mask = (
            df["_tile"] == tile
        )

        tile_indices = df.index[
            tile_mask
        ]

        print(
            f"  Episodes: "
            f"{len(tile_indices):,}"
        )

        with rasterio.open(
            raster_path
        ) as src:

            for count, idx in enumerate(
                tile_indices,
                start=1,
            ):

                lat = float(
                    df.at[idx, "latitude"]
                )

                lon = float(
                    df.at[idx, "longitude"]
                )

                # Point class
                point_class = extract_point_class(
                    src,
                    lon,
                    lat,
                )

                result.at[
                    idx,
                    "landcover_class"
                ] = point_class

                # Local composition
                fractions = extract_window_fractions(
                    src,
                    lon,
                    lat,
                )

                for column, value in fractions.items():

                    result.at[
                        idx,
                        column
                    ] = value

                if count % 5000 == 0:

                    print(
                        f"    Processed "
                        f"{count:,}/"
                        f"{len(tile_indices):,}"
                    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\nMerging land-cover features...")

    result = result.reset_index(drop=True)

    df = df.reset_index(drop=True)

    df = df.drop(
        columns=["_tile"]
    )

    output = pd.concat(
        [df, result],
        axis=1,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("Saving...")

    output.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("========================================")
    print("WorldCover extraction complete")
    print("========================================")
    print(
        f"Episodes: {len(output):,}"
    )
    print(
        f"Columns:  {len(output.columns)}"
    )
    print(
        f"Output:   {OUTPUT_FILE}"
    )
    print("========================================")


if __name__ == "__main__":
    main()