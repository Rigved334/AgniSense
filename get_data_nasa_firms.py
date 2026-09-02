import pandas as pd
from datetime import date, timedelta
import time

# ---- CONFIG ----
MAP_KEY = "d6d5faecc74d8525b4494ee2492d9a0c"
INDIA_BBOX = "68.7,8.4,97.25,37.6"
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 8, 31)
DAY_RANGE = 5                          # FIRMS Area API hard limit is 1-5
OUTPUT_FILE = "india_viirs_2026_jan_aug.csv"

# Exact SP/NRT cutoff dates pulled from your data_availability response —
# don't recompute this from "days old", it drifts as FIRMS processes new data.
SOURCE_CUTOFFS = {
    "NOAA20": {"sp_max": date(2026, 5, 31), "nrt_min": date(2026, 6, 1)},
    "SNPP":   {"sp_max": date(2026, 4, 27), "nrt_min": date(2026, 4, 28)},
}

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{range}/{start}"


def get_source(sensor: str, chunk_start: date) -> str:
    cutoffs = SOURCE_CUTOFFS[sensor]
    return f"VIIRS_{sensor}_SP" if chunk_start <= cutoffs["sp_max"] else f"VIIRS_{sensor}_NRT"


def fetch_chunk(sensor: str, chunk_start: date) -> pd.DataFrame:
    source = get_source(sensor, chunk_start)
    url = BASE_URL.format(key=MAP_KEY, source=source, bbox=INDIA_BBOX,
                           range=DAY_RANGE, start=chunk_start.isoformat())
    try:
        df = pd.read_csv(url)
        if df.empty or "latitude" not in df.columns:
            print(f"  [{source}] {chunk_start}: no data / bad response")
            return pd.DataFrame()
        df["source_sensor"] = sensor
        df["source_dataset"] = source
        return df
    except Exception as e:
        print(f"  [{source}] {chunk_start}: FAILED — {e}")
        return pd.DataFrame()


def download_all() -> pd.DataFrame:
    all_chunks = []
    current = START_DATE
    while current <= END_DATE:
        for sensor in ["NOAA20", "SNPP"]:
            print(f"Fetching {sensor} from {current}...")
            chunk_df = fetch_chunk(sensor, current)
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
            time.sleep(1)
        current += timedelta(days=DAY_RANGE)

    if not all_chunks:
        raise RuntimeError("No data retrieved — check MAP_KEY, bbox, or date range.")

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time", "source_sensor"]
    )
    return combined


if __name__ == "__main__":
    df = download_all()
    print(f"\nTotal records: {len(df)}")
    print(df["acq_date"].min(), "to", df["acq_date"].max())
    print(df["source_dataset"].value_counts())
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")