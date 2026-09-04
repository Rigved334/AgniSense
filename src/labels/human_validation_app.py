import os
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/features/industrial_fire_human_validation_12.csv"
OUTPUT_FILE = "data/features/industrial_fire_human_validation_12.csv"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial Fire Human Validation",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):
    st.error(
        f"Validation file not found:\n\n{INPUT_FILE}"
    )
    st.stop()


df = pd.read_csv(INPUT_FILE)


# Human annotation columns must be strings.
# Empty CSV columns can otherwise be inferred as float64 by pandas.
annotation_columns = [
    "human_label",
    "human_confidence",
    "industrial_facility_visible",
    "thermal_source_visible",
    "source_type",
    "evidence",
    "notes",
]

for column in annotation_columns:
    if column not in df.columns:
        df[column] = ""

    df[column] = df[column].fillna("").astype("string")


# ============================================================
# SESSION STATE
# ============================================================

if "current_index" not in st.session_state:
    st.session_state.current_index = 0


# ============================================================
# HEADER
# ============================================================

st.title("🔥 Industrial Fire Human Validation")

st.markdown(
    """
This dashboard is used to independently validate the weakly-labelled
industrial-fire candidates.

**Do not use the existing weak label as evidence when making the
human judgment.**
"""
)


# ============================================================
# PROGRESS
# ============================================================

total = len(df)

completed = (
    df["human_label"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

st.progress(
    completed / total if total else 0
)

st.write(
    f"**Progress:** {completed} / {total} candidates validated"
)


# ============================================================
# NAVIGATION
# ============================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("⬅ Previous", use_container_width=True):
        st.session_state.current_index = max(
            0,
            st.session_state.current_index - 1
        )
        st.rerun()

with col2:
    selected_number = st.number_input(
        "Candidate number",
        min_value=1,
        max_value=total,
        value=st.session_state.current_index + 1,
        step=1,
    )

    if selected_number - 1 != st.session_state.current_index:
        st.session_state.current_index = selected_number - 1
        st.rerun()

with col3:
    if st.button("Next ➡", use_container_width=True):
        st.session_state.current_index = min(
            total - 1,
            st.session_state.current_index + 1
        )
        st.rerun()


# ============================================================
# CURRENT CANDIDATE
# ============================================================

idx = st.session_state.current_index

row = df.iloc[idx]


st.divider()

st.subheader(
    f"Candidate {idx + 1} / {total}"
)


# ============================================================
# LOCATION
# ============================================================

st.markdown("### 📍 Location")

location_col1, location_col2, location_col3 = st.columns(3)

with location_col1:
    st.metric(
        "Latitude",
        f"{row['latitude']:.6f}"
    )

with location_col2:
    st.metric(
        "Longitude",
        f"{row['longitude']:.6f}"
    )

with location_col3:
    st.metric(
        "Cluster",
        int(row["cluster"])
    )


# ============================================================
# MAP
# ============================================================

map_df = pd.DataFrame(
    {
        "lat": [row["latitude"]],
        "lon": [row["longitude"]],
    }
)

st.map(
    map_df,
    latitude="lat",
    longitude="lon",
    zoom=10,
)


# ============================================================
# FIRMS / THERMAL INFORMATION
# ============================================================

st.markdown("### 🔥 FIRMS / Thermal Evidence")

thermal_col1, thermal_col2, thermal_col3, thermal_col4 = st.columns(4)

with thermal_col1:
    st.metric(
        "Max FRP",
        f"{row['max_frp']:.2f}"
    )

with thermal_col2:
    st.metric(
        "Detections",
        int(row["detection_count"])
    )

with thermal_col3:
    st.metric(
        "High-confidence detections",
        int(row["high_confidence_count"])
    )

with thermal_col4:
    st.metric(
        "Cluster episodes",
        int(row["cluster_size"])
    )


st.write(
    f"**Episode:** `{row['episode_id']}`"
)

st.write(
    f"**Date:** {row['start_date']} → {row['end_date']}"
)


# ============================================================
# OSM INFORMATION
# ============================================================

st.markdown("### 🏭 OSM Infrastructure")

osm_col1, osm_col2, osm_col3 = st.columns(3)

with osm_col1:
    st.metric(
        "Industrial features ≤500 m",
        int(row["industrial_count_500m"])
    )

with osm_col2:
    st.metric(
        "Works ≤500 m",
        int(row["works_count_500m"])
    )

with osm_col3:
    st.metric(
        "Refineries",
        int(row["refinery_count"])
    )


osm_col4, osm_col5, osm_col6 = st.columns(3)

with osm_col4:
    st.metric(
        "Storage tanks ≤1 km",
        int(row["storage_tank_count_1000m"])
    )

with osm_col5:
    st.metric(
        "Flares ≤1 km",
        int(row["flare_count_1000m"])
    )

with osm_col6:
    st.metric(
        "Power plants ≤5 km",
        int(row["powerplant_count_5000m"])
    )


# ============================================================
# LAND COVER
# ============================================================

st.markdown("### 🌍 Land Cover")

lc_col1, lc_col2, lc_col3, lc_col4 = st.columns(4)

with lc_col1:
    st.metric(
        "Tree cover",
        f"{row['tree_cover_fraction_1km']:.3f}"
    )

with lc_col2:
    st.metric(
        "Cropland",
        f"{row['cropland_fraction_1km']:.3f}"
    )

with lc_col3:
    st.metric(
        "Built-up",
        f"{row['builtup_fraction_1km']:.3f}"
    )

with lc_col4:
    st.metric(
        "WorldCover class",
        int(row["landcover_class"])
    )


# ============================================================
# HUMAN ANNOTATION
# ============================================================

st.divider()

st.subheader("🧑‍💻 Human Assessment")


human_label_options = [
    "",
    "confirmed_industrial_fire",
    "probable_industrial_fire",
    "not_industrial_fire",
    "uncertain",
]

confidence_options = [
    "",
    "high",
    "medium",
    "low",
]

visibility_options = [
    "",
    "yes",
    "no",
    "uncertain",
]

source_type_options = [
    "",
    "industrial_process",
    "storage_facility",
    "refinery",
    "power_plant",
    "waste_burning",
    "agricultural",
    "vegetation",
    "unknown",
]


label_index = (
    human_label_options.index(row["human_label"])
    if row["human_label"] in human_label_options
    else 0
)

confidence_index = (
    confidence_options.index(row["human_confidence"])
    if row["human_confidence"] in confidence_options
    else 0
)

facility_index = (
    visibility_options.index(row["industrial_facility_visible"])
    if row["industrial_facility_visible"] in visibility_options
    else 0
)

thermal_index = (
    visibility_options.index(row["thermal_source_visible"])
    if row["thermal_source_visible"] in visibility_options
    else 0
)

source_index = (
    source_type_options.index(row["source_type"])
    if row["source_type"] in source_type_options
    else 0
)


human_label = st.selectbox(
    "Human label",
    human_label_options,
    index=label_index,
)

human_confidence = st.selectbox(
    "Confidence",
    confidence_options,
    index=confidence_index,
)

industrial_facility_visible = st.selectbox(
    "Industrial facility visible / identifiable?",
    visibility_options,
    index=facility_index,
)

thermal_source_visible = st.selectbox(
    "Thermal source identifiable?",
    visibility_options,
    index=thermal_index,
)

source_type = st.selectbox(
    "Likely source type",
    source_type_options,
    index=source_index,
)


evidence = st.text_area(
    "Evidence",
    value="" if pd.isna(row["evidence"]) else str(row["evidence"]),
    placeholder=(
        "Example: FIRMS coordinate falls inside a large industrial "
        "facility; multiple thermal detections occur near the same "
        "facility."
    ),
    height=100,
)

notes = st.text_area(
    "Notes",
    value="" if pd.isna(row["notes"]) else str(row["notes"]),
    placeholder="Additional observations...",
    height=100,
)


# ============================================================
# SAVE
# ============================================================

if st.button(
    "💾 Save Assessment",
    type="primary",
    use_container_width=True,
):

    df.loc[idx, "human_label"] = human_label
    df.loc[idx, "human_confidence"] = human_confidence
    df.loc[idx, "industrial_facility_visible"] = (
        industrial_facility_visible
    )
    df.loc[idx, "thermal_source_visible"] = (
        thermal_source_visible
    )
    df.loc[idx, "source_type"] = source_type
    df.loc[idx, "evidence"] = evidence
    df.loc[idx, "notes"] = notes

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    st.success(
        f"Assessment saved for candidate {idx + 1}."
    )

    st.rerun()


# ============================================================
# VALIDATION SUMMARY
# ============================================================

st.divider()

st.subheader("Validation Summary")

summary = (
    df["human_label"]
    .replace("", pd.NA)
    .value_counts(dropna=False)
)

st.dataframe(
    summary.rename("count"),
    use_container_width=True,
)