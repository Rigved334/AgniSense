import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/features/thermal_episode_weak_labels.parquet"

OUTPUT_FILE = "data/features/industrial_fire_human_validation.csv"

CLUSTER_EPS_KM = 5

# Number of physical clusters to select
TARGET_CLUSTERS = 30

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

df = df[df["weak_label"] == "industrial_fire"].copy()

print(f"Industrial-fire candidates: {len(df)}")


# ============================================================
# SPATIAL CLUSTERING
# ============================================================

coords = df[["latitude", "longitude"]].to_numpy()

mean_lat = np.radians(df["latitude"].mean())

# Approximate conversion to kilometres
X = coords * np.array([
    111.0,
    111.0 * np.cos(mean_lat)
])

clusterer = DBSCAN(
    eps=CLUSTER_EPS_KM,
    min_samples=1
)

df["cluster"] = clusterer.fit_predict(X)

df["cluster_size"] = (
    df.groupby("cluster")["episode_id"]
    .transform("size")
)

print(f"Spatial clusters: {df['cluster'].nunique()}")


# ============================================================
# DERIVE SAMPLING FEATURES
# ============================================================

df["infrastructure_score"] = (
    (df["industrial_count_500m"] > 0).astype(int)
    + (df["works_count_500m"] > 0).astype(int)
    + (df["refinery_count"] > 0).astype(int)
    + (df["storage_tank_count_1000m"] > 0).astype(int)
    + (df["flare_count_1000m"] > 0).astype(int)
)

df["high_frp"] = df["max_frp"] >= 50

df["high_confidence"] = (
    df["high_confidence_count"] >= 1
)

df["specialized_infrastructure"] = (
    (df["works_count_500m"] > 0)
    | (df["refinery_count"] > 0)
    | (df["storage_tank_count_1000m"] > 0)
    | (df["flare_count_1000m"] > 0)
)


# ============================================================
# SELECT REPRESENTATIVE EPISODE PER CLUSTER
# ============================================================
#
# We don't want multiple episodes from the same physical
# location dominating the initial human-validation set.
#
# Select the strongest episode from each cluster first.
# ============================================================

representatives = (
    df.sort_values(
        [
            "cluster",
            "max_frp",
            "high_confidence_count",
            "detection_count"
        ],
        ascending=[True, False, False, False]
    )
    .groupby("cluster", as_index=False)
    .first()
)

print(
    f"Representative physical locations: "
    f"{len(representatives)}"
)


# ============================================================
# STRATIFICATION SCORES
# ============================================================

# Priority score for selecting diverse/interesting clusters.

representatives["sampling_score"] = (
    representatives["high_frp"].astype(int) * 4
    + representatives["high_confidence"].astype(int) * 3
    + representatives["specialized_infrastructure"].astype(int) * 3
    + np.log1p(representatives["detection_count"]) * 0.5
    + np.log1p(representatives["infrastructure_score"]) * 1.0
)


# ============================================================
# GEOGRAPHIC DIVERSITY
# ============================================================
#
# Instead of simply taking the top 30 scores, greedily select
# candidates while discouraging nearby locations.
# ============================================================

candidates = representatives.sort_values(
    "sampling_score",
    ascending=False
).copy()

selected = []

# Minimum separation between selected validation locations.
MIN_DISTANCE_KM = 50


def distance_km(lat1, lon1, lat2, lon2):
    """
    Approximate distance between two points in kilometres.
    """
    lat_scale = 111.0
    lon_scale = 111.0 * np.cos(
        np.radians((lat1 + lat2) / 2)
    )

    dy = (lat2 - lat1) * lat_scale
    dx = (lon2 - lon1) * lon_scale

    return np.sqrt(dx ** 2 + dy ** 2)


for _, row in candidates.iterrows():

    if len(selected) >= TARGET_CLUSTERS:
        break

    # Always accept the first candidate.
    if not selected:
        selected.append(row)
        continue

    too_close = False

    for chosen in selected:

        distance = distance_km(
            chosen["latitude"],
            chosen["longitude"],
            row["latitude"],
            row["longitude"]
        )

        if distance < MIN_DISTANCE_KM:
            too_close = True
            break

    if not too_close:
        selected.append(row)


selected = pd.DataFrame(selected)


# ============================================================
# IF GEOGRAPHIC FILTER LEFT US WITH TOO FEW CANDIDATES
# ============================================================

if len(selected) < TARGET_CLUSTERS:

    selected_clusters = set(selected["cluster"])

    remaining = candidates[
        ~candidates["cluster"].isin(selected_clusters)
    ]

    needed = TARGET_CLUSTERS - len(selected)

    selected = pd.concat(
        [
            selected,
            remaining.head(needed)
        ],
        ignore_index=True
    )


# ============================================================
# ADD HUMAN-VALIDATION COLUMNS
# ============================================================

selected["human_label"] = ""
selected["human_confidence"] = ""
selected["evidence"] = ""
selected["notes"] = ""


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [
    "cluster",
    "cluster_size",
    "episode_id",
    "start_date",
    "end_date",
    "latitude",
    "longitude",

    "max_frp",
    "mean_frp",
    "detection_count",
    "high_confidence_count",

    "industrial_count_500m",
    "works_count_500m",
    "refinery_count",
    "storage_tank_count_1000m",
    "flare_count_1000m",
    "powerplant_count_5000m",

    "tree_cover_fraction_1km",
    "cropland_fraction_1km",
    "builtup_fraction_1km",
    "landcover_class",

    "human_label",
    "human_confidence",
    "evidence",
    "notes"
]


selected = selected[output_columns]


# ============================================================
# SORT
# ============================================================

selected = selected.sort_values(
    ["max_frp", "cluster_size"],
    ascending=[False, False]
)


# ============================================================
# SAVE
# ============================================================

selected.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("Human Validation Sample")
print("========================================")

print(f"Selected clusters : {len(selected)}")
print(f"Output file       : {OUTPUT_FILE}")

print("\nSelected candidates:")
print(
    selected[
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

print("\nSampling complete.")