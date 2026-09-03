from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# Paths
# ============================================================

EPISODE_FILE = Path("data/firms/thermal_episodes.parquet")
OSM_DIR = Path("data/osm/final/clean")
OUTPUT_FILE = Path("data/features/thermal_episode_osm.parquet")

# Radius in meters
RADII = {
    "industrial": 500,
    "works": 500,
    "flare": 1000,
    "storage_tank": 1000,
    "powerplant": 5000,
    "quarry": 1000,
}


# ============================================================
# Helpers
# ============================================================

def load_episode_cells():
    """Load episodes and create one GeoDataFrame row per H3 cell."""

    if not EPISODE_FILE.exists():
        raise FileNotFoundError(
            f"Episode file not found:\n{EPISODE_FILE.resolve()}"
        )

    print("Reading thermal episodes...")
    episodes = pd.read_parquet(EPISODE_FILE)

    required = {
        "episode_id",
        "h3_cell",
        "latitude",
        "longitude",
    }

    missing = required - set(episodes.columns)

    if missing:
        raise ValueError(
            f"Missing required episode columns: {sorted(missing)}"
        )

    print(f"Episodes loaded: {len(episodes):,}")

    # Only one row per H3 cell is needed for OSM enrichment.
    cells = (
        episodes[
            ["h3_cell", "latitude", "longitude"]
        ]
        .drop_duplicates("h3_cell")
        .copy()
    )

    print(f"Unique H3 cells: {len(cells):,}")

    cells = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(
            cells["longitude"],
            cells["latitude"],
        ),
        crs="EPSG:4326",
    )

    return episodes, cells


def load_osm_layer(filename):
    """Load an OSM GeoParquet layer."""

    path = OSM_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"OSM layer not found:\n{path.resolve()}"
        )

    gdf = gpd.read_parquet(path)

    if gdf.empty:
        print(f"WARNING: {filename} is empty.")
        return gdf

    if gdf.crs is None:
        raise ValueError(
            f"{filename} has no CRS."
        )

    # Make sure everything is valid.
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()

    # OSM data is currently EPSG:4326.
    gdf = gdf.to_crs("EPSG:4326")

    return gdf


def nearest_distance(cells, osm, name):
    """
    Calculate distance from every H3 cell center to
    nearest OSM feature.
    """

    column = f"nearest_{name}_distance_m"

    if osm.empty:
        cells[column] = pd.NA
        return cells

    print(f"  Calculating nearest {name}...")

    # Project to a meter-based CRS.
    left = cells[["h3_cell", "geometry"]].to_crs("EPSG:3857")
    right = osm[["geometry"]].to_crs("EPSG:3857")

    joined = gpd.sjoin_nearest(
        left,
        right,
        how="left",
        distance_col=column,
    )

    # Multiple matches can occur when two features are equally close.
    # Keep one row per H3 cell.
    joined = (
        joined
        .sort_values(column)
        .drop_duplicates("h3_cell")
    )

    distances = joined[
        ["h3_cell", column]
    ].copy()

    cells = cells.merge(
        distances,
        on="h3_cell",
        how="left",
    )

    return cells


def count_within_radius(cells, osm, name, radius):
    """
    Count OSM features intersecting a radius around each H3 cell.
    """

    column = f"{name}_count_{radius}m"

    if osm.empty:
        cells[column] = 0
        return cells

    print(
        f"  Counting {name} within {radius:,} m..."
    )

    # Project to meters.
    left = cells[["h3_cell", "geometry"]].to_crs("EPSG:3857")
    right = osm[["geometry"]].to_crs("EPSG:3857")

    # Create buffers around H3 cell centers.
    buffers = left.copy()
    buffers["geometry"] = buffers.geometry.buffer(radius)

    # INTERSECTS is intentional.
    joined = gpd.sjoin(
        buffers,
        right,
        how="left",
        predicate="intersects",
    )

    counts = (
        joined
        .dropna(subset=["index_right"])
        .groupby("h3_cell")
        .size()
        .rename(column)
        .reset_index()
    )

    cells = cells.merge(
        counts,
        on="h3_cell",
        how="left",
    )

    cells[column] = (
        cells[column]
        .fillna(0)
        .astype("int32")
    )

    return cells


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load episodes + unique H3 cells
    # --------------------------------------------------------

    episodes, cells = load_episode_cells()

    # --------------------------------------------------------
    # 2. Load OSM layers
    # --------------------------------------------------------

    print("\nLoading OSM layers...")

    osm_layers = {
        "industrial": load_osm_layer(
            "industrial_areas.parquet"
        ),
        "works": load_osm_layer(
            "industrial_works.parquet"
        ),
        "flare": load_osm_layer(
            "flares.parquet"
        ),
        "storage_tank": load_osm_layer(
            "storage_tanks.parquet"
        ),
        "powerplant": load_osm_layer(
            "power_plants.parquet"
        ),
        "quarry": load_osm_layer(
            "quarries.parquet"
        ),
    }

    for name, gdf in osm_layers.items():
        print(
            f"  {name}: {len(gdf):,} features"
        )

    # --------------------------------------------------------
    # 3. Nearest-feature distances
    # --------------------------------------------------------

    print("\nCalculating nearest-feature distances...")

    for name, osm in osm_layers.items():

        cells = nearest_distance(
            cells,
            osm,
            name,
        )

    # --------------------------------------------------------
    # 4. Feature counts within buffers
    # --------------------------------------------------------

    print("\nCalculating OSM feature counts...")

    for name, osm in osm_layers.items():

        radius = RADII[name]

        cells = count_within_radius(
            cells,
            osm,
            name,
            radius,
        )

    # --------------------------------------------------------
    # 5. Merge OSM features back onto episodes
    # --------------------------------------------------------

    print("\nMerging OSM features with episodes...")

    osm_columns = [
        "h3_cell",
        "nearest_industrial_distance_m",
        "industrial_count_500m",
        "nearest_works_distance_m",
        "works_count_500m",
        "nearest_flare_distance_m",
        "flare_count_1000m",
        "nearest_storage_tank_distance_m",
        "storage_tank_count_1000m",
        "nearest_powerplant_distance_m",
        "powerplant_count_5000m",
        "nearest_quarry_distance_m",
        "quarry_count_1000m",
    ]

    osm_features = cells[osm_columns].copy()

    result = episodes.merge(
        osm_features,
        on="h3_cell",
        how="left",
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    print("\nSaving output...")

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n========================================")
    print("OSM enrichment complete")
    print("========================================")
    print(f"Episodes: {len(result):,}")
    print(f"Columns:  {len(result.columns)}")
    print(f"Output:   {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()