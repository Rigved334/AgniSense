import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from rasterio.io import MemoryFile


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/features/thermal_episode_weak_labels.parquet"

OUTPUT_FILE = (
    "data/features/sentinel2_episode_features_batch500.parquet"
)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "process/v1"
)

SAMPLES_PER_CLASS = 125

# Prefer pre-event observations.
PRE_EVENT_DAYS = list(range(-1, -16, -1))

# Only use post-event data if no pre-event image is usable.
POST_EVENT_DAYS = [1, 2, 3]

BBOX_HALF_SIZE = 0.005

WIDTH = 128
HEIGHT = 128

MIN_VALID_FRACTION = 0.20

RANDOM_STATE = 42

# Save progress after every successful episode.
SAVE_EVERY = 1

# Small delay between requests.
REQUEST_DELAY_SECONDS = 0.25

# Number of HTTP retries for temporary failures.
MAX_RETRIES = 3


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET "
        "not found in .env"
    )


# ============================================================
# AUTHENTICATION
# ============================================================

def get_access_token():

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()["access_token"]


# ============================================================
# SENTINEL-2 REQUEST
# ============================================================

def request_sentinel2(
    token,
    latitude,
    longitude,
    date,
):
    """
    Request Sentinel-2 data.

    Returns:
        (tiff_bytes, token)

    If CDSE returns HTTP 401, obtain a fresh token
    and retry the request automatically.
    """

    half = BBOX_HALF_SIZE

    bbox = [
        longitude - half,
        latitude - half,
        longitude + half,
        latitude + half,
    ]

    evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: [
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "B11",
                    "B12",
                    "SCL"
                ]
            }],
            output: {
                bands: 7,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [
            sample.B02,
            sample.B03,
            sample.B04,
            sample.B08,
            sample.B11,
            sample.B12,
            sample.SCL
        ];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date}T00:00:00Z",
                            "to": f"{date}T23:59:59Z",
                        }
                    }
                }
            ]
        },

        "output": {
            "width": WIDTH,
            "height": HEIGHT,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    }
                }
            ]
        },

        "evalscript": evalscript
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.post(
                PROCESS_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )

            # ------------------------------------------------
            # TOKEN EXPIRED
            # ------------------------------------------------

            if response.status_code == 401:

                print(
                    "401 Unauthorized — "
                    "refreshing CDSE token..."
                )

                token = get_access_token()

                print(
                    "Token refreshed successfully."
                )

                # Retry immediately with fresh token.
                continue

            response.raise_for_status()

            return response.content, token

        except requests.RequestException as e:

            if attempt >= MAX_RETRIES:
                raise

            print(
                f"Request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{type(e).__name__}: {e}"
            )

            time.sleep(
                attempt * 2
            )

    raise RuntimeError(
        "Sentinel-2 request failed after retries."
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_features(tiff_bytes):

    with MemoryFile(tiff_bytes) as memfile:

        with memfile.open() as src:
            data = src.read()

    b02 = data[0]
    b03 = data[1]
    b04 = data[2]
    b08 = data[3]
    b11 = data[4]
    b12 = data[5]
    scl = data[6]

    finite = (
        np.isfinite(b02)
        & np.isfinite(b03)
        & np.isfinite(b04)
        & np.isfinite(b08)
        & np.isfinite(b11)
        & np.isfinite(b12)
    )

    # Clear/usable SCL classes:
    #
    # 4 = vegetation
    # 5 = bare soil
    # 6 = water
    # 7 = unclassified
    #
    # Cloud/shadow/snow/etc. are excluded.

    clear = np.isin(
        scl,
        [4, 5, 6, 7]
    )

    valid = finite & clear

    valid_fraction = float(
        valid.mean()
    )

    if valid_fraction < MIN_VALID_FRACTION:

        return {
            "usable": False,
            "valid_fraction": valid_fraction,
            "ndvi": np.nan,
            "ndbi": np.nan,
            "ndwi": np.nan,
            "b02": np.nan,
            "b03": np.nan,
            "b04": np.nan,
            "b08": np.nan,
            "b11": np.nan,
            "b12": np.nan,
        }

    def safe_index(
        numerator,
        denominator,
    ):

        result = np.full(
            numerator.shape,
            np.nan,
            dtype=np.float32,
        )

        mask = (
            valid
            & np.isfinite(denominator)
            & (denominator != 0)
        )

        result[mask] = (
            numerator[mask]
            / denominator[mask]
        )

        return result

    ndvi = safe_index(
        b08 - b04,
        b08 + b04,
    )

    ndbi = safe_index(
        b11 - b08,
        b11 + b08,
    )

    ndwi = safe_index(
        b03 - b08,
        b03 + b08,
    )

    def masked_mean(array):

        values = array[valid]

        if len(values) == 0:
            return np.nan

        return float(
            np.nanmean(values)
        )

    return {
        "usable": True,
        "valid_fraction": valid_fraction,
        "ndvi": masked_mean(ndvi),
        "ndbi": masked_mean(ndbi),
        "ndwi": masked_mean(ndwi),
        "b02": masked_mean(b02),
        "b03": masked_mean(b03),
        "b04": masked_mean(b04),
        "b08": masked_mean(b08),
        "b11": masked_mean(b11),
        "b12": masked_mean(b12),
    }


# ============================================================
# SAMPLE SELECTION
# ============================================================

def select_sample(df):

    target_per_class = 125

    classes = [
        "industrial_fire",
        "persistent_industrial_source",
        "wildfire",
        "agricultural_fire",
    ]

    samples = []

    for label in classes:

        subset = df[
            df["weak_label"] == label
        ].copy()

        if len(subset) <= target_per_class:
            # Use every available episode.
            sampled = subset.copy()

            print(
                f"{label}: using all "
                f"{len(sampled)} available episodes"
            )

        else:
            sampled = subset.sample(
                n=target_per_class,
                random_state=RANDOM_STATE,
            )

            print(
                f"{label}: sampled "
                f"{len(sampled)} episodes "
                f"from {len(subset)}"
            )

        samples.append(sampled)

    return pd.concat(
        samples,
        ignore_index=True,
    )


# ============================================================
# CREATE FAILED RESULT
# ============================================================

def make_failed_result(row):

    return {
        "episode_id":
            row["episode_id"],

        "weak_label":
            row["weak_label"],

        "latitude":
            float(row["latitude"]),

        "longitude":
            float(row["longitude"]),

        "event_date":
            pd.Timestamp(
                row["start_date"]
            ).normalize(),

        "sentinel2_date":
            pd.NaT,

        "days_from_event":
            np.nan,

        "observation_type":
            "unavailable",

        "sentinel2_valid_fraction":
            0.0,

        "sentinel2_ndvi":
            np.nan,

        "sentinel2_ndbi":
            np.nan,

        "sentinel2_ndwi":
            np.nan,

        "sentinel2_b02":
            np.nan,

        "sentinel2_b03":
            np.nan,

        "sentinel2_b04":
            np.nan,

        "sentinel2_b08":
            np.nan,

        "sentinel2_b11":
            np.nan,

        "sentinel2_b12":
            np.nan,

        "sentinel2_found":
            False,
    }


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_results(results):

    result_df = pd.DataFrame(
        results
    )

    result_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SENTINEL-2 PRODUCTION BATCH")
print("=" * 70)

print(
    f"\nSamples per class: "
    f"{SAMPLES_PER_CLASS}"
)

print(
    f"Total target episodes: "
    f"{SAMPLES_PER_CLASS * 4}"
)

print(
    "\nPreferred temporal window:"
)

print(
    "  Pre-event: "
    f"{PRE_EVENT_DAYS[-1]} to "
    f"{PRE_EVENT_DAYS[0]} days"
)

print(
    "  Post-event fallback: "
    f"{POST_EVENT_DAYS}"
)


# ============================================================
# LOAD EPISODES
# ============================================================

df = pd.read_parquet(
    INPUT_FILE
)

df = df[
    df["weak_label"] != "uncertain"
].copy()

print(
    f"\nAvailable labelled episodes: "
    f"{len(df):,}"
)


# ============================================================
# SELECT STRATIFIED SAMPLE
# ============================================================

sample = select_sample(
    df
)

print("\nSelected samples:")

print(
    sample["weak_label"]
    .value_counts()
)


# ============================================================
# RESUME EXISTING RESULTS
# ============================================================

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_parquet(
        OUTPUT_FILE
    )

    processed_ids = set(
        existing["episode_id"]
    )

    results = existing.to_dict(
        orient="records"
    )

    print(
        f"\nExisting progress found:"
        f" {len(processed_ids)} episodes"
    )

else:

    processed_ids = set()

    results = []

    print(
        "\nNo existing progress found."
    )


# ============================================================
# AUTHENTICATION
# ============================================================

print(
    "\nAuthenticating with CDSE..."
)

token = get_access_token()

print(
    "Authentication successful."
)


# ============================================================
# PROCESS
# ============================================================

remaining = sample[
    ~sample["episode_id"].isin(
        processed_ids
    )
].copy()

print(
    f"\nRemaining episodes: "
    f"{len(remaining)}"
)


for index, row in remaining.iterrows():

    episode_id = row[
        "episode_id"
    ]

    latitude = float(
        row["latitude"]
    )

    longitude = float(
        row["longitude"]
    )

    event_date = pd.Timestamp(
        row["start_date"]
    ).normalize()

    label = row[
        "weak_label"
    ]

    print("\n" + "-" * 70)

    print(
        f"Episode "
        f"{len(processed_ids) + 1}/"
        f"{len(sample)}"
    )

    print(
        f"ID: {episode_id}"
    )

    print(
        f"Class: {label}"
    )

    print(
        f"Location: "
        f"{latitude:.5f}, "
        f"{longitude:.5f}"
    )

    print(
        f"Event date: "
        f"{event_date.date()}"
    )

    found = False

    # ========================================================
    # PHASE 1: PRE-EVENT
    # ========================================================

    for offset in PRE_EVENT_DAYS:

        candidate_date = (
            event_date
            + pd.Timedelta(
                days=offset
            )
        )

        date_string = (
            candidate_date.strftime(
                "%Y-%m-%d"
            )
        )

        print(
            f"  Pre-event "
            f"{date_string} "
            f"({offset:+d})...",
            end=" "
        )

        try:

            tiff_bytes, token = request_sentinel2(
                                token=token,
                                latitude=latitude,
                                longitude=longitude,
                                date=date_string,
            )

            features = extract_features(
                tiff_bytes
            )

            if features["usable"]:

                print(
                    f"usable "
                    f"({features['valid_fraction']:.1%})"
                )

                results.append(
                    {
                        "episode_id":
                            episode_id,

                        "weak_label":
                            label,

                        "latitude":
                            latitude,

                        "longitude":
                            longitude,

                        "event_date":
                            event_date,

                        "sentinel2_date":
                            candidate_date,

                        "days_from_event":
                            offset,

                        "observation_type":
                            "pre_event",

                        "sentinel2_valid_fraction":
                            features[
                                "valid_fraction"
                            ],

                        "sentinel2_ndvi":
                            features[
                                "ndvi"
                            ],

                        "sentinel2_ndbi":
                            features[
                                "ndbi"
                            ],

                        "sentinel2_ndwi":
                            features[
                                "ndwi"
                            ],

                        "sentinel2_b02":
                            features[
                                "b02"
                            ],

                        "sentinel2_b03":
                            features[
                                "b03"
                            ],

                        "sentinel2_b04":
                            features[
                                "b04"
                            ],

                        "sentinel2_b08":
                            features[
                                "b08"
                            ],

                        "sentinel2_b11":
                            features[
                                "b11"
                            ],

                        "sentinel2_b12":
                            features[
                                "b12"
                            ],

                        "sentinel2_found":
                            True,
                    }
                )

                found = True

                break

            else:

                print(
                    f"not usable "
                    f"({features['valid_fraction']:.1%})"
                )

        except Exception as e:

            print(
                f"error: "
                f"{type(e).__name__}: {e}"
            )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    # ========================================================
    # PHASE 2: POST-EVENT FALLBACK
    # ========================================================

    if not found:

        print(
            "  No usable pre-event "
            "observation."
        )

        for offset in POST_EVENT_DAYS:

            candidate_date = (
                event_date
                + pd.Timedelta(
                    days=offset
                )
            )

            date_string = (
                candidate_date.strftime(
                    "%Y-%m-%d"
                )
            )

            print(
                f"  Post-event fallback "
                f"{date_string} "
                f"({offset:+d})...",
                end=" "
            )

            try:

                tiff_bytes = request_sentinel2(
                    token=token,
                    latitude=latitude,
                    longitude=longitude,
                    date=date_string,
                )

                features = extract_features(
                    tiff_bytes
                )

                if features["usable"]:

                    print(
                        f"usable "
                        f"({features['valid_fraction']:.1%})"
                    )

                    results.append(
                        {
                            "episode_id":
                                episode_id,

                            "weak_label":
                                label,

                            "latitude":
                                latitude,

                            "longitude":
                                longitude,

                            "event_date":
                                event_date,

                            "sentinel2_date":
                                candidate_date,

                            "days_from_event":
                                offset,

                            "observation_type":
                                "post_event",

                            "sentinel2_valid_fraction":
                                features[
                                    "valid_fraction"
                                ],

                            "sentinel2_ndvi":
                                features[
                                    "ndvi"
                                ],

                            "sentinel2_ndbi":
                                features[
                                    "ndbi"
                                ],

                            "sentinel2_ndwi":
                                features[
                                    "ndwi"
                                ],

                            "sentinel2_b02":
                                features[
                                    "b02"
                                ],

                            "sentinel2_b03":
                                features[
                                    "b03"
                                ],

                            "sentinel2_b04":
                                features[
                                    "b04"
                                ],

                            "sentinel2_b08":
                                features[
                                    "b08"
                                ],

                            "sentinel2_b11":
                                features[
                                    "b11"
                                ],

                            "sentinel2_b12":
                                features[
                                    "b12"
                                ],

                            "sentinel2_found":
                                True,
                        }
                    )

                    found = True

                    break

                else:

                    print(
                        f"not usable "
                        f"({features['valid_fraction']:.1%})"
                    )

            except Exception as e:

                print(
                    f"error: "
                    f"{type(e).__name__}: {e}"
                )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    # ========================================================
    # FAILURE
    # ========================================================

    if not found:

        print(
            "  No usable Sentinel-2 "
            "observation found."
        )

        results.append(
            make_failed_result(row)
        )

    # --------------------------------------------------------
    # Mark processed and save
    # --------------------------------------------------------

    processed_ids.add(
        episode_id
    )

    if (
        len(results) % SAVE_EVERY == 0
    ):

        save_results(
            results
        )

        print(
            f"  Progress saved: "
            f"{len(results)}/{len(sample)}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

result_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"\nTotal episodes: "
    f"{len(result_df)}"
)

print(
    f"Usable Sentinel-2: "
    f"{result_df['sentinel2_found'].sum()}"
)

print(
    f"Unavailable/unusable: "
    f"{(~result_df['sentinel2_found']).sum()}"
)

print(
    f"Success rate: "
    f"{result_df['sentinel2_found'].mean():.1%}"
)


# ============================================================
# SUCCESS BY CLASS
# ============================================================

print(
    "\nSuccess by class:"
)

class_success = (
    result_df
    .groupby("weak_label")
    ["sentinel2_found"]
    .agg(
        [
            "count",
            "sum",
            "mean",
        ]
    )
)

class_success.columns = [
    "total",
    "usable",
    "success_rate",
]

print(
    class_success
    .round(3)
    .to_string()
)


# ============================================================
# OBSERVATION TYPE
# ============================================================

print(
    "\nObservation type:"
)

print(
    result_df[
        "observation_type"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# TEMPORAL OFFSETS
# ============================================================

usable = result_df[
    result_df["sentinel2_found"]
].copy()

if len(usable) > 0:

    print(
        "\nSelected temporal offsets:"
    )

    print(
        usable[
            "days_from_event"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nSpectral summary:"
    )

    print(
        usable[
            [
                "sentinel2_valid_fraction",
                "sentinel2_ndvi",
                "sentinel2_ndbi",
                "sentinel2_ndwi",
                "sentinel2_b02",
                "sentinel2_b03",
                "sentinel2_b04",
                "sentinel2_b08",
                "sentinel2_b11",
                "sentinel2_b12",
            ]
        ]
        .describe()
        .round(4)
    )


# ============================================================
# FINAL SAVE
# ============================================================

save_results(
    results
)

print(
    f"\nSaved:"
)

print(
    OUTPUT_FILE
)