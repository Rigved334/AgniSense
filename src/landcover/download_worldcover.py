from pathlib import Path
import math

import pandas as pd
import requests


EPISODE_FILE = Path(
    "data/features/thermal_episode_osm.parquet"
)

OUTPUT_DIR = Path(
    "data/landcover/worldcover_2021"
)

BASE_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "v200/2021/map"
)


def tile_name(lat, lon):
    """Return the 3° x 3° WorldCover tile containing a point."""

    lat_start = math.floor(lat / 3) * 3
    lon_start = math.floor(lon / 3) * 3

    lat_prefix = "N" if lat_start >= 0 else "S"
    lon_prefix = "E" if lon_start >= 0 else "W"

    return (
        f"{lat_prefix}{abs(lat_start):02d}"
        f"{lon_prefix}{abs(lon_start):03d}"
    )


def download_file(session, url, output):
    """Download a file using streaming HTTP."""

    with session.get(
        url,
        stream=True,
        timeout=120,
    ) as response:

        response.raise_for_status()

        with open(output, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading thermal episodes...")

    df = pd.read_parquet(
        EPISODE_FILE,
        columns=["latitude", "longitude"],
    )

    print(f"Episodes: {len(df):,}")

    print("Determining required tiles...")

    tiles = set()

    for lat, lon in zip(
        df["latitude"],
        df["longitude"],
    ):
        tiles.add(tile_name(lat, lon))

    tiles = sorted(tiles)

    print(f"Unique tiles required: {len(tiles)}")
    print()

    session = requests.Session()

    for i, tile in enumerate(tiles, start=1):

        filename = (
            f"ESA_WorldCover_10m_2021_v200_"
            f"{tile}_Map.tif"
        )

        output = OUTPUT_DIR / filename

        if output.exists():
            print(
                f"[{i}/{len(tiles)}] "
                f"Already exists: {filename}"
            )
            continue

        url = f"{BASE_URL}/{filename}"

        print(
            f"[{i}/{len(tiles)}] "
            f"Downloading: {filename}"
        )

        try:

            download_file(
                session,
                url,
                output,
            )

            print("  Download complete")

        except Exception as e:

            print(
                f"  FAILED: {type(e).__name__}: {e}"
            )

            if output.exists():
                output.unlink()

    print()
    print("WorldCover download complete.")


if __name__ == "__main__":
    main()