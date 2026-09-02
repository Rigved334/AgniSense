from pathlib import Path
import re

import geopandas as gpd
import pandas as pd


INPUT = Path("data/osm/final/industrial_osm.gpkg")
OUTPUT_DIR = Path("data/osm/final/clean")


LAYERS = [
    "points",
    "lines",
    "multipolygons",
]


def parse_other_tags(value):
    """
    Parse GDAL's OSM other_tags representation.

    Example:
        "power"=>"plant","plant:source"=>"coal"

    Returns:
        {"power": "plant", "plant:source": "coal"}
    """
    if pd.isna(value) or not str(value).strip():
        return {}

    text = str(value)

    matches = re.findall(
        r'"([^"]+)"=>"(.*?)"',
        text
    )

    return dict(matches)


def add_parsed_tags(gdf):
    """
    Parse other_tags and expose useful OSM tags as columns.
    """

    parsed = gdf["other_tags"].apply(parse_other_tags)

    interesting_tags = [
        "landuse",
        "man_made",
        "power",
        "industrial",
        "plant:source",
        "plant:method",
        "plant:type",
        "resource",
        "content",
        "mineral",
        "operator",
    ]

    for tag in interesting_tags:
        if tag not in gdf.columns:
            gdf[tag] = parsed.apply(lambda d: d.get(tag))

        else:
            # Keep the explicitly stored value where it exists,
            # otherwise recover it from other_tags.
            gdf[tag] = gdf[tag].fillna(
                parsed.apply(lambda d: d.get(tag))
            )

    return gdf


def read_all_layers():
    frames = []

    for layer in LAYERS:
        print(f"Reading layer: {layer}")

        gdf = gpd.read_file(
            INPUT,
            layer=layer
        )

        print(f"  Rows: {len(gdf):,}")

        if len(gdf) == 0:
            continue

        gdf = add_parsed_tags(gdf)

        frames.append(gdf)

    if not frames:
        raise RuntimeError("No OSM features were found.")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=frames[0].crs,
    )

    return combined


def save_layer(gdf, name):
    output = OUTPUT_DIR / f"{name}.parquet"

    if len(gdf) == 0:
        print(f"{name}: 0 features -- not writing.")
        return

    # Keep only valid geometries
    gdf = gdf[gdf.geometry.notna()].copy()

    gdf.to_parquet(
        output,
        index=False,
    )

    print(
        f"{name}: {len(gdf):,} features -> "
        f"{output}"
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading OSM data...\n")

    gdf = read_all_layers()

    print("\nTotal reconstructed features:")
    print(f"  {len(gdf):,}")

    # ---------------------------------------------------------
    # 1. Industrial areas
    # ---------------------------------------------------------

    industrial = gdf[
        gdf["landuse"].eq("industrial")
    ].copy()

    # ---------------------------------------------------------
    # 2. Industrial works
    # ---------------------------------------------------------

    works = gdf[
        gdf["man_made"].eq("works")
    ].copy()

    # ---------------------------------------------------------
    # 3. Flares
    # ---------------------------------------------------------

    flares = gdf[
        gdf["man_made"].eq("flare")
    ].copy()

    # ---------------------------------------------------------
    # 4. Storage tanks
    # ---------------------------------------------------------

    storage_tanks = gdf[
        gdf["man_made"].eq("storage_tank")
    ].copy()

    # ---------------------------------------------------------
    # 5. Power plants
    # ---------------------------------------------------------

    power_plants = gdf[
        gdf["power"].eq("plant")
    ].copy()

    # ---------------------------------------------------------
    # 6. Quarries
    # ---------------------------------------------------------

    quarries = gdf[
        gdf["landuse"].eq("quarry")
    ].copy()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    print("\nSaving clean datasets...\n")

    save_layer(
        industrial,
        "industrial_areas"
    )

    save_layer(
        works,
        "industrial_works"
    )

    save_layer(
        flares,
        "flares"
    )

    save_layer(
        storage_tanks,
        "storage_tanks"
    )

    save_layer(
        power_plants,
        "power_plants"
    )

    save_layer(
        quarries,
        "quarries"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()