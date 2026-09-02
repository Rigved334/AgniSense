from pathlib import Path
import geopandas as gpd
import pandas as pd


GPKG = Path("data/osm/final/industrial_osm.gpkg")

LAYERS = [
    "points",
    "lines",
    "multipolygons",
]


def count_values(series: pd.Series, name: str) -> None:
    print(f"\n{'=' * 70}")
    print(name)
    print("=" * 70)

    values = series.dropna().astype(str).str.strip()

    if values.empty:
        print("No values.")
        return

    print(values.value_counts().head(30).to_string())


def main() -> None:
    for layer in LAYERS:
        print(f"\n\n{'#' * 80}")
        print(f"LAYER: {layer}")
        print("#" * 80)

        gdf = gpd.read_file(GPKG, layer=layer)

        print(f"Rows: {len(gdf):,}")
        print(f"Columns: {list(gdf.columns)}")

        for column in [
            "landuse",
            "man_made",
            "power",
            "type",
            "industrial",
        ]:
            if column in gdf.columns:
                count_values(gdf[column], column)

        if "other_tags" in gdf.columns:
            print(f"\n{'=' * 70}")
            print("Sample other_tags")
            print("=" * 70)

            tags = (
                gdf["other_tags"]
                .dropna()
                .astype(str)
            )

            for value in tags.head(30):
                print(value)


if __name__ == "__main__":
    main()