from pathlib import Path
import pandas as pd

osm_dir = Path("data/osm/final/clean")

files = [
    "industrial_areas.parquet",
    "industrial_works.parquet",
    "power_plants.parquet",
    "quarries.parquet",
    "storage_tanks.parquet",
    "flares.parquet",
]

columns_to_check = [
    "industrial",
    "plant:source",
    "plant:method",
    "plant:type",
    "resource",
    "content",
    "mineral",
]

for filename in files:

    path = osm_dir / filename

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    df = pd.read_parquet(path)

    for column in columns_to_check:

        if column not in df.columns:
            continue

        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if values.empty:
            continue

        print(f"\n--- {column} ---")
        print(
            values
            .value_counts()
            .head(30)
            .to_string()
        )