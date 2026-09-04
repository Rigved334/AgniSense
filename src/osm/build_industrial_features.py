from pathlib import Path

import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

OSM_DIR = Path("data/osm/final/clean")

INPUT_EPISODES = Path(
    "data/features/thermal_episode_landcover.parquet"
)

OUTPUT = Path(
    "data/features/thermal_episode_osm_categories.parquet"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METRIC_CRS = "EPSG:3857"

# Distances in metres.
SEARCH_RADII = {
    "refinery": 5_000,
    "oil_facility": 5_000,
    "chemical_facility": 5_000,
    "cement_facility": 5_000,
    "steel_facility": 5_000,
    "mine": 2_000,
    "fuel_storage": 2_000,
    "oil_storage": 2_000,
    "gas_storage": 2_000,
    "lpg_storage": 2_000,
    "coal_powerplant": 5_000,
    "gas_powerplant": 5_000,
    "oil_powerplant": 5_000,
    "flare": 2_000,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )


def combine_tags(df):
    """
    Combine useful OSM tag columns into one searchable string.
    """

    columns = [
        "name",
        "industrial",
        "plant:source",
        "plant:method",
        "plant:type",
        "resource",
        "content",
        "mineral",
        "operator",
        "other_tags",
    ]

    existing = [
        c for c in columns
        if c in df.columns
    ]

    combined = pd.Series(
        "",
        index=df.index,
        dtype="string",
    )

    for column in existing:
        combined = (
            combined
            + " "
            + clean_text(df[column])
        )

    return combined


def load_layer(filename):
    path = OSM_DIR / filename

    print(f"Loading {path}")

    gdf = gpd.read_parquet(path)

    if gdf.crs is None:
        raise ValueError(
            f"{filename} has no CRS."
        )

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf


# ---------------------------------------------------------------------
# Facility classification
# ---------------------------------------------------------------------

def classify_industrial_areas():
    gdf = load_layer(
        "industrial_areas.parquet"
    )

    text = combine_tags(gdf)

    categories = {}

    categories["refinery"] = (
        text.str.contains(
            r"\brefinery\b|oil refinery|petroleum refinery",
            regex=True,
            na=False,
        )
    )

    categories["oil_facility"] = (
        text.str.contains(
            r"\boil\b|petroleum|petrochemical",
            regex=True,
            na=False,
        )
    )

    categories["chemical_facility"] = (
        text.str.contains(
            r"\bchemical\b|chemical_plant",
            regex=True,
            na=False,
        )
    )

    categories["cement_facility"] = (
        text.str.contains(
            r"\bcement\b",
            regex=True,
            na=False,
        )
    )

    categories["steel_facility"] = (
        text.str.contains(
            r"\bsteel\b|steel_mill",
            regex=True,
            na=False,
        )
    )

    categories["mine"] = (
        text.str.contains(
            r"\bmine\b|mining|\bcoal\b|\biron_ore\b|\blimestone\b",
            regex=True,
            na=False,
        )
    )

    result = {}

    for category, mask in categories.items():

        selected = gdf.loc[mask].copy()

        if len(selected) == 0:
            print(
                f"{category}: 0 features"
            )
            continue

        result[category] = selected

        print(
            f"{category}: {len(selected)} features"
        )

    return result


def classify_industrial_works():

    gdf = load_layer(
        "industrial_works.parquet"
    )

    text = combine_tags(gdf)

    categories = {
        "refinery": r"\brefinery\b|oil refinery|petroleum refinery",
        "oil_facility": r"\boil\b|petroleum|petrochemical",
        "chemical_facility": r"\bchemical\b",
        "cement_facility": r"\bcement\b",
        "steel_facility": r"\bsteel\b|steel_mill",
        "mine": r"\bmine\b|mining|\bcoal\b",
    }

    result = {}

    for category, pattern in categories.items():

        mask = text.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        selected = gdf.loc[mask].copy()

        if len(selected) > 0:
            result[category] = selected

        print(
            f"works/{category}: {len(selected)}"
        )

    return result


def classify_powerplants():

    gdf = load_layer(
        "power_plants.parquet"
    )

    source = clean_text(
        gdf["plant:source"]
    )

    result = {}

    for category, pattern in {
        "coal_powerplant": r"\bcoal\b",
        "gas_powerplant": r"\bgas\b",
        "oil_powerplant": r"\boil\b|diesel",
    }.items():

        mask = source.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        selected = gdf.loc[mask].copy()

        result[category] = selected

        print(
            f"{category}: {len(selected)}"
        )

    return result


def classify_storage():

    gdf = load_layer(
        "storage_tanks.parquet"
    )

    content = clean_text(
        gdf["content"]
    )

    result = {}

    patterns = {
        "fuel_storage": r"\bfuel\b",
        "oil_storage": r"\boil\b|petroleum",
        "gas_storage": r"\bgas\b",
        "lpg_storage": r"\blpg\b",
    }

    for category, pattern in patterns.items():

        mask = content.str.contains(
            pattern,
            regex=True,
            na=False,
        )

        selected = gdf.loc[mask].copy()

        result[category] = selected

        print(
            f"{category}: {len(selected)}"
        )

    return result


def classify_quarries():

    gdf = load_layer(
        "quarries.parquet"
    )

    text = combine_tags(gdf)

    mask = text.str.contains(
        r"\bmine\b|mining|\bcoal\b|\biron_ore\b|"
        r"\blimestone\b|\bbauxite\b|\bmanganese\b|"
        r"\bcopper\b|\buranium\b",
        regex=True,
        na=False,
    )

    selected = gdf.loc[mask].copy()

    print(
        f"mining/quarry facilities: {len(selected)}"
    )

    return {
        "mine": selected
    }


def classify_flares():

    gdf = load_layer(
        "flares.parquet"
    )

    print(
        f"flare: {len(gdf)}"
    )

    return {
        "flare": gdf
    }


# ---------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------

def calculate_features(
    episodes,
    facilities,
):

    print(
        f"\nCalculating features for "
        f"{len(episodes):,} episodes..."
    )

    result = pd.DataFrame(
        index=episodes.index
    )

    # Work in metric CRS.
    episode_points = gpd.GeoDataFrame(
        episodes[
            ["latitude", "longitude"]
        ].copy(),
        geometry=gpd.points_from_xy(
            episodes["longitude"],
            episodes["latitude"],
        ),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    for category, facility_gdf in facilities.items():

        if len(facility_gdf) == 0:
            continue

        radius = SEARCH_RADII[category]

        print(
            f"  {category}: "
            f"{len(facility_gdf):,} facilities, "
            f"radius={radius:,} m"
        )

        facility_metric = (
            facility_gdf
            .to_crs(METRIC_CRS)
        )

        # -------------------------------------------------------------
        # Nearest distance
        # -------------------------------------------------------------

        nearest = gpd.sjoin_nearest(
            episode_points,
            facility_metric[
                ["geometry"]
            ],
            how="left",
            distance_col="_distance",
        )

        # Because multiple facilities can have the same nearest
        # distance, keep the minimum distance for each episode.
        distances = (
            nearest
            .groupby(nearest.index)["_distance"]
            .min()
        )

        result[
            f"{category}_distance_m"
        ] = distances.reindex(
            episodes.index
        )

        # -------------------------------------------------------------
        # Count within radius
        # -------------------------------------------------------------

        # Buffer episode points.
        buffered = episode_points.copy()

        buffered["geometry"] = (
            buffered.geometry
            .buffer(radius)
        )

        joined = gpd.sjoin(
        buffered[["geometry"]],
        facility_metric[["geometry"]],
        how="left",
        predicate="intersects",
        )

        # Count only actual facility matches.
        matched = joined[
            joined["index_right"].notna()
        ]

        counts = (
            matched
            .groupby(matched.index)
            .size()
        )

        result[
            f"{category}_count"
        ] = (
            counts
            .reindex(
                episodes.index,
                fill_value=0,
            )
            .astype("int32")
        )

        return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("BUILDING OSM INDUSTRIAL CATEGORY FEATURES")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load episodes
    # -------------------------------------------------------------

    print(
        f"\nLoading episodes: "
        f"{INPUT_EPISODES}"
    )

    episodes = pd.read_parquet(
        INPUT_EPISODES
    )

    print(
        f"Episodes: {len(episodes):,}"
    )

    # -------------------------------------------------------------
    # Classify facilities
    # -------------------------------------------------------------

    facilities = {}

    for classifier in [
        classify_industrial_areas,
        classify_industrial_works,
        classify_powerplants,
        classify_storage,
        classify_quarries,
        classify_flares,
    ]:

        classified = classifier()

        for category, gdf in classified.items():

            if category not in facilities:

                facilities[category] = gdf

            else:

                facilities[category] = pd.concat(
                    [
                        facilities[category],
                        gdf,
                    ],
                    ignore_index=True,
                )

    # -------------------------------------------------------------
    # Calculate spatial features
    # -------------------------------------------------------------

    features = calculate_features(
        episodes,
        facilities,
    )

    print("\nFeature summary:")
    print(features.describe().T)

    # -------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------

    output = episodes.copy()

    for column in features.columns:
        output[column] = features[column]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Output:",
        OUTPUT
    )

    print(
        "Shape:",
        output.shape
    )


if __name__ == "__main__":
    main()