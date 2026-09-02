from pathlib import Path
import geopandas as gpd


DATA_DIR = Path("data/osm/final/clean")

FILES = [
    "industrial_areas.parquet",
    "industrial_works.parquet",
    "flares.parquet",
    "storage_tanks.parquet",
    "power_plants.parquet",
    "quarries.parquet",
]


def main():
    print("\nCLEAN OSM DATASET\n")

    for filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"{filename}: MISSING")
            continue

        gdf = gpd.read_parquet(path)

        print(f"{filename}")
        print(f"  Features: {len(gdf):,}")
        print(f"  CRS: {gdf.crs}")
        print(
            f"  Geometry: "
            f"{gdf.geometry.geom_type.value_counts().to_dict()}"
        )
        print()


if __name__ == "__main__":
    main()