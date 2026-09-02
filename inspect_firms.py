from pathlib import Path

import pandas as pd


FIRMS_FILE = Path("india_viirs_2026_jan_aug.csv")


def main() -> None:
    if not FIRMS_FILE.exists():
        raise FileNotFoundError(
            f"FIRMS file not found:\n{FIRMS_FILE.resolve()}"
        )

    df = pd.read_csv(FIRMS_FILE)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nDate range:")
    print(df["acq_date"].min(), "→", df["acq_date"].max())

    print("\nSatellites:")
    print(df["satellite"].value_counts(dropna=False))

    print("\nConfidence:")
    print(df["confidence"].value_counts(dropna=False))

    print("\nDay/Night:")
    print(df["daynight"].value_counts(dropna=False))

    print("\nFRP statistics:")
    print(df["frp"].describe())

    print("\nBrightness temperature - bright_ti4:")
    print(df["bright_ti4"].describe())

    print("\nBrightness temperature - bright_ti5:")
    print(df["bright_ti5"].describe())

    print("\nCoordinate bounds:")
    print("Latitude:", df["latitude"].min(), "→", df["latitude"].max())
    print("Longitude:", df["longitude"].min(), "→", df["longitude"].max())


if __name__ == "__main__":
    main()