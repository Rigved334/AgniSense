from pathlib import Path

import geopandas as gpd
import pandas as pd


INPUT_FILE = Path(
    "data/features/thermal_episode_osm.parquet"
)

BOUNDARY_FILE = Path(
    "data/boundaries/india_boundary.geojson"
)

OUTPUT_FILE = Path(
    "data/features/thermal_episode_india.parquet"
)


def main():

    print("Reading thermal episodes...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Input episodes: {len(df):,}")

    print("Creating episode geometries...")

    episodes = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs="EPSG:4326",
    )

    print("Reading India boundary...")

    india = gpd.read_file(
        BOUNDARY_FILE
    ).to_crs("EPSG:4326")

    print("Filtering episodes...")

    filtered = gpd.sjoin(
        episodes,
        india[["geometry"]],
        how="inner",
        predicate="within",
    )

    # Remove the spatial-join index column.
    if "index_right" in filtered.columns:
        filtered = filtered.drop(
            columns=["index_right"]
        )

    # Keep the coordinates but don't save geometry.
    filtered = pd.DataFrame(
        filtered.drop(columns=["geometry"])
    )

    print(
        f"India episodes: {len(filtered):,}"
    )

    print(
        f"Removed: "
        f"{len(df) - len(filtered):,}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()