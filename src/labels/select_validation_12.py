import pandas as pd
import numpy as np


INPUT_FILE = "data/features/industrial_fire_human_validation.csv"
OUTPUT_FILE = "data/features/industrial_fire_human_validation_12.csv"

TARGET = 12
MIN_DISTANCE_KM = 50


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Available candidates: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

df["infrastructure_score"] = (
    (df["industrial_count_500m"] > 0).astype(int)
    + (df["works_count_500m"] > 0).astype(int)
    + (df["refinery_count"] > 0).astype(int)
    + (df["storage_tank_count_1000m"] > 0).astype(int)
    + (df["flare_count_1000m"] > 0).astype(int)
)

df["high_frp"] = df["max_frp"] >= 50

df["high_confidence"] = df["high_confidence_count"] >= 1

df["specialized_infrastructure"] = (
    (df["works_count_500m"] > 0)
    | (df["refinery_count"] > 0)
    | (df["storage_tank_count_1000m"] > 0)
    | (df["flare_count_1000m"] > 0)
)


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance_km(lat1, lon1, lat2, lon2):

    lat_scale = 111.0

    lon_scale = 111.0 * np.cos(
        np.radians((lat1 + lat2) / 2)
    )

    dy = (lat2 - lat1) * lat_scale
    dx = (lon2 - lon1) * lon_scale

    return np.sqrt(dx ** 2 + dy ** 2)


# ============================================================
# CHECK WHETHER LOCATION IS FAR ENOUGH
# ============================================================

def far_enough(row, selected):

    for chosen in selected:

        distance = distance_km(
            row["latitude"],
            row["longitude"],
            chosen["latitude"],
            chosen["longitude"],
        )

        if distance < MIN_DISTANCE_KM:
            return False

    return True


# ============================================================
# SELECT CANDIDATES
# ============================================================

selected = []


def select_from(pool, number):

    global selected

    pool = pool.copy()

    # Strongest candidates first
    pool = pool.sort_values(
        [
            "max_frp",
            "high_confidence_count",
            "infrastructure_score",
            "detection_count",
        ],
        ascending=False,
    )

    for _, row in pool.iterrows():

        if len(selected) >= TARGET:
            break

        if far_enough(row, selected):
            selected.append(row)


# ============================================================
# 1. HIGH FRP
# ============================================================

select_from(
    df[df["high_frp"]],
    4,
)


# ============================================================
# 2. HIGH CONFIDENCE
# ============================================================

select_from(
    df[df["high_confidence"]],
    3,
)


# ============================================================
# 3. SPECIALIZED INFRASTRUCTURE
# ============================================================

select_from(
    df[df["specialized_infrastructure"]],
    3,
)


# ============================================================
# 4. GEOGRAPHICALLY DIVERSE REMAINING CASES
# ============================================================

remaining = df.copy()

selected_ids = {
    row["episode_id"]
    for row in selected
}

remaining = remaining[
    ~remaining["episode_id"].isin(selected_ids)
]

remaining = remaining.sort_values(
    "max_frp",
    ascending=False,
)

select_from(
    remaining,
    2,
)


# ============================================================
# FILL IF WE HAVE FEWER THAN 12
# ============================================================

if len(selected) < TARGET:

    selected_ids = {
        row["episode_id"]
        for row in selected
    }

    remaining = df[
        ~df["episode_id"].isin(selected_ids)
    ]

    for _, row in remaining.sort_values(
        "max_frp",
        ascending=False,
    ).iterrows():

        if len(selected) >= TARGET:
            break

        selected.append(row)


# ============================================================
# DATAFRAME
# ============================================================

result = pd.DataFrame(selected)


# ============================================================
# ADD ANNOTATION COLUMNS
# ============================================================

for column in [
    "human_label",
    "human_confidence",
    "industrial_facility_visible",
    "thermal_source_visible",
    "source_type",
    "evidence",
    "notes",
]:

    if column not in result.columns:
        result[column] = ""


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    "max_frp",
    ascending=False,
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print("\n========================================")
print("12-Candidate Validation Set")
print("========================================")

print(f"Selected: {len(result)}")
print(f"Saved:    {OUTPUT_FILE}\n")

print(
    result[
        [
            "cluster",
            "cluster_size",
            "episode_id",
            "latitude",
            "longitude",
            "max_frp",
            "detection_count",
            "high_confidence_count",
            "industrial_count_500m",
            "works_count_500m",
            "refinery_count",
            "storage_tank_count_1000m",
            "flare_count_1000m",
        ]
    ].to_string(index=False)
)