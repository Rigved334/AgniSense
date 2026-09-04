from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st


# ============================================================
# Configuration
# ============================================================

DATASET = Path(
    "data/output/classified_thermal_episodes_priority.gpkg"
)

LAYER = "classified_thermal_episodes_priority"


CLASS_LABELS = {
    "industrial_fire": "Industrial Fire",
    "persistent_industrial_source": "Persistent Industrial Source",
    "wildfire": "Wildfire",
    "agricultural_fire": "Agricultural Fire",
}


CLASS_COLORS = {
    "industrial_fire": [255, 60, 60],
    "persistent_industrial_source": [255, 165, 0],
    "wildfire": [50, 120, 255],
    "agricultural_fire": [50, 200, 100],
}


PRIORITY_COLORS = {
    "Critical": [255, 0, 0],
    "High": [255, 140, 0],
    "Medium": [255, 210, 0],
    "Low": [120, 120, 120],
}


PRIORITY_ORDER = [
    "Critical",
    "High",
    "Medium",
    "Low",
]


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Industrial Fire Detection",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# Title
# ============================================================

st.title(
    "🔥 AI-Based Industrial Fire Detection"
)

st.caption(
    "NASA FIRMS + OSM + ESA WorldCover + Sentinel-2"
)

st.warning(
    "Prototype: classification results are based on weak labels "
    "and a 443-episode Sentinel-2-enriched evaluation sample. "
    "They should not be interpreted as independently validated "
    "ground-truth performance. Investigation priority is a "
    "prototype ranking heuristic, not a calibrated risk score."
)


# ============================================================
# Load data
# ============================================================

@st.cache_data
def load_data():

    gdf = gpd.read_file(
        DATASET,
        layer=LAYER,
    )

    gdf["start_date"] = pd.to_datetime(
        gdf["start_date"]
    )

    gdf["end_date"] = pd.to_datetime(
        gdf["end_date"]
    )

    return gdf


try:

    gdf = load_data()

except Exception as e:

    st.error(
        f"Could not load GeoPackage:\n\n{e}"
    )

    st.stop()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Filters")


# ------------------------------------------------------------
# Priority
# ------------------------------------------------------------

available_priorities = [
    p
    for p in PRIORITY_ORDER
    if p in gdf["priority_level"].astype(str).unique()
]


selected_priorities = st.sidebar.multiselect(
    "Investigation priority",
    options=available_priorities,
    default=available_priorities,
)


# ------------------------------------------------------------
# Class
# ------------------------------------------------------------

available_classes = [
    c
    for c in CLASS_LABELS
    if c in gdf["predicted_class"].unique()
]


selected_classes = st.sidebar.multiselect(
    "Prediction class",
    options=available_classes,
    default=available_classes,
    format_func=lambda x: CLASS_LABELS[x],
)


# ------------------------------------------------------------
# Confidence
# ------------------------------------------------------------

min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)


# ------------------------------------------------------------
# FRP
# ------------------------------------------------------------

max_frp_value = float(
    gdf["max_frp"].max()
)


min_frp = st.sidebar.slider(
    "Minimum maximum FRP",
    min_value=0.0,
    max_value=max_frp_value,
    value=0.0,
)


# ------------------------------------------------------------
# Date
# ------------------------------------------------------------

min_date = gdf["start_date"].min().date()
max_date = gdf["start_date"].max().date()


date_range = st.sidebar.date_input(
    "Start date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)


# ============================================================
# Apply filters
# ============================================================

filtered = gdf[
    gdf["priority_level"]
    .astype(str)
    .isin(selected_priorities)
].copy()


filtered = filtered[
    filtered["predicted_class"]
    .isin(selected_classes)
]


filtered = filtered[
    filtered["prediction_confidence"]
    >= min_confidence
]


filtered = filtered[
    filtered["max_frp"]
    >= min_frp
]


if isinstance(date_range, tuple) and len(date_range) == 2:

    start_filter = pd.Timestamp(
        date_range[0]
    )

    end_filter = pd.Timestamp(
        date_range[1]
    )

    filtered = filtered[
        (filtered["start_date"] >= start_filter)
        &
        (filtered["start_date"] <= end_filter)
    ]


# Always keep highest-priority events first.

filtered = filtered.sort_values(
    "priority_score",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# Priority Summary
# ============================================================

st.subheader("Investigation Priority")


priority_counts = (
    filtered["priority_level"]
    .astype(str)
    .value_counts()
)


p1, p2, p3, p4 = st.columns(4)


p1.metric(
    "🔴 Critical",
    int(priority_counts.get("Critical", 0)),
)

p2.metric(
    "🟠 High",
    int(priority_counts.get("High", 0)),
)

p3.metric(
    "🟡 Medium",
    int(priority_counts.get("Medium", 0)),
)

p4.metric(
    "⚪ Low",
    int(priority_counts.get("Low", 0)),
)


# ============================================================
# Event Summary
# ============================================================

st.subheader("Event Summary")


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Total events",
    len(filtered),
)


c2.metric(
    "Industrial fires",
    int(
        (
            filtered["predicted_class"]
            == "industrial_fire"
        ).sum()
    ),
)


c3.metric(
    "Persistent sources",
    int(
        (
            filtered["predicted_class"]
            == "persistent_industrial_source"
        ).sum()
    ),
)


c4.metric(
    "Wildfires",
    int(
        (
            filtered["predicted_class"]
            == "wildfire"
        ).sum()
    ),
)


c5.metric(
    "Agricultural fires",
    int(
        (
            filtered["predicted_class"]
            == "agricultural_fire"
        ).sum()
    ),
)


# ============================================================
# Priority Ranked Events
# ============================================================

st.subheader("🚨 Priority-Ranked Events")


if len(filtered) == 0:

    st.info(
        "No events match the selected filters."
    )

else:

    priority_table = filtered[
        [
            "priority_rank",
            "priority_level",
            "predicted_class",
            "industrial_probability",
            "prediction_confidence",
            "max_frp",
            "duration_days",
            "latitude",
            "longitude",
            "start_date",
        ]
    ].copy()


    priority_table["predicted_class"] = (
        priority_table["predicted_class"]
        .map(CLASS_LABELS)
    )


    priority_table["industrial_probability"] = (
        priority_table["industrial_probability"]
        .map(lambda x: f"{x:.1%}")
    )


    priority_table["prediction_confidence"] = (
        priority_table["prediction_confidence"]
        .map(lambda x: f"{x:.1%}")
    )


    priority_table["start_date"] = (
        priority_table["start_date"]
        .dt.date
    )


    priority_table = priority_table.rename(
        columns={
            "priority_rank": "Rank",
            "priority_level": "Priority",
            "predicted_class": "Class",
            "industrial_probability":
                "Industrial Probability",
            "prediction_confidence":
                "Confidence",
            "max_frp": "Max FRP",
            "duration_days": "Duration (days)",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "start_date": "Date",
        }
    )


    st.dataframe(
        priority_table.head(50),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# Map
# ============================================================

st.subheader("🗺️ Thermal Event Map")


if len(filtered) == 0:

    st.info(
        "No events match the selected filters."
    )

else:

    map_data = filtered[
        [
            "episode_id",
            "latitude",
            "longitude",
            "predicted_class",
            "priority_level",
            "priority_score",
            "prediction_confidence",
            "max_frp",
            "duration_days",
        ]
    ].copy()


    map_data["latitude"] = pd.to_numeric(
        map_data["latitude"],
        errors="coerce",
    )


    map_data["longitude"] = pd.to_numeric(
        map_data["longitude"],
        errors="coerce",
    )


    # Use priority as the map color.

    map_data["color"] = (
        map_data["priority_level"]
        .map(PRIORITY_COLORS)
    )


    map_data["class_display"] = (
        map_data["predicted_class"]
        .map(CLASS_LABELS)
    )


    map_data["priority_score_display"] = (
        map_data["priority_score"]
        .map(lambda x: f"{x:.3f}")
    )


    map_data["confidence_display"] = (
        map_data["prediction_confidence"]
        .map(lambda x: f"{x:.1%}")
    )


    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=[
            "longitude",
            "latitude",
        ],
        get_fill_color="color",
        get_radius=5000,
        radius_min_pixels=5,
        radius_max_pixels=18,
        pickable=True,
        auto_highlight=True,
    )


    view_state = pdk.ViewState(
        latitude=float(
            filtered["latitude"].mean()
        ),
        longitude=float(
            filtered["longitude"].mean()
        ),
        zoom=4,
    )


    tooltip = {
        "html": """
        <b>Event</b>: {episode_id}<br/>
        <b>Class</b>: {class_display}<br/>
        <b>Priority</b>: {priority_level}<br/>
        <b>Priority Score</b>: {priority_score_display}<br/>
        <b>Industrial Probability</b>: {confidence_display}<br/>
        <b>Max FRP</b>: {max_frp}<br/>
        <b>Duration</b>: {duration_days} day(s)
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black",
        },
    }


    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )


    st.pydeck_chart(
        deck,
        use_container_width=True,
    )


    st.caption(
        "Map colors: 🔴 Critical  |  🟠 High  |  "
        "🟡 Medium  |  ⚪ Low"
    )


# ============================================================
# Event Investigation
# ============================================================

st.subheader("🔎 Event Investigation")


if len(filtered) > 0:

    event_options = filtered[
        "episode_id"
    ].tolist()


    def format_event_option(episode_id):

        row = filtered[
            filtered["episode_id"]
            == episode_id
        ].iloc[0]


        rank = int(
            row["priority_rank"]
        )


        priority = str(
            row["priority_level"]
        ).upper()


        class_name = CLASS_LABELS.get(
            row["predicted_class"],
            row["predicted_class"],
        )


        confidence = (
            row["prediction_confidence"]
        )


        frp = row["max_frp"]


        date = row["start_date"].date()


        return (
            f"#{rank} | {priority} | "
            f"{class_name} | "
            f"{confidence:.1%} confidence | "
            f"FRP {frp:.2f} | "
            f"{date}"
        )


    selected_event = st.selectbox(
        "Select an event to investigate",
        options=event_options,
        format_func=format_event_option,
    )


    event = filtered[
        filtered["episode_id"]
        == selected_event
    ].iloc[0]


    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    st.markdown(
        "### Investigation Priority"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Priority",
        str(event["priority_level"]),
    )


    c2.metric(
        "Priority score",
        f"{event['priority_score']:.3f}",
    )


    c3.metric(
        "Industrial probability",
        f"{event['industrial_probability']:.2%}",
    )


    c4.metric(
        "Priority rank",
        f"#{int(event['priority_rank'])}",
    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    st.markdown(
        "### Classification"
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Predicted class",
        CLASS_LABELS.get(
            event["predicted_class"],
            event["predicted_class"],
        ),
    )


    c2.metric(
        "Confidence",
        f"{event['prediction_confidence']:.2%}",
    )


    c3.metric(
        "Maximum FRP",
        f"{event['max_frp']:.2f}",
    )


    # --------------------------------------------------------
    # Thermal
    # --------------------------------------------------------

    st.markdown(
        "### Thermal Characteristics"
    )


    thermal_data = {

        "Detection count":
            event["detection_count"],

        "Active days":
            event["active_days"],

        "Duration":
            event["duration_days"],

        "Mean FRP":
            event["mean_frp"],

        "Median FRP":
            event["median_frp"],

        "Maximum FRP":
            event["max_frp"],

        "Thermal severity score":
            event["thermal_severity_score"],
    }


    st.dataframe(
        pd.DataFrame(
            thermal_data.items(),
            columns=[
                "Feature",
                "Value",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Industrial context
    # --------------------------------------------------------

    st.markdown(
        "### Industrial Context"
    )


    industrial_data = {

        "Nearest industrial feature (m)":
            event[
                "nearest_industrial_distance_m"
            ],

        "Industrial features within 500m":
            event[
                "industrial_count_500m"
            ],

        "Works within 500m":
            event[
                "works_count_500m"
            ],

        "Flares within 1km":
            event[
                "flare_count_1000m"
            ],

        "Storage tanks within 1km":
            event[
                "storage_tank_count_1000m"
            ],

        "Power plants within 5km":
            event[
                "powerplant_count_5000m"
            ],

        "Refineries":
            event[
                "refinery_count"
            ],

        "Industrial context score":
            event[
                "industrial_context_score"
            ],
    }


    st.dataframe(
        pd.DataFrame(
            industrial_data.items(),
            columns=[
                "Feature",
                "Value",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Environmental + Sentinel-2
    # --------------------------------------------------------

    st.markdown(
        "### Environmental & Sentinel-2 Context"
    )


    environmental_data = {

        "Land-cover class":
            event["landcover_class"],

        "Tree cover fraction":
            event["tree_cover_fraction_1km"],

        "Shrubland fraction":
            event["shrubland_fraction_1km"],

        "Grassland fraction":
            event["grassland_fraction_1km"],

        "Cropland fraction":
            event["cropland_fraction_1km"],

        "Built-up fraction":
            event["builtup_fraction_1km"],

        "Sentinel-2 NDVI":
            event["sentinel2_ndvi"],

        "Sentinel-2 NDBI":
            event["sentinel2_ndbi"],

        "Sentinel-2 NDWI":
            event["sentinel2_ndwi"],
    }


    st.dataframe(
        pd.DataFrame(
            environmental_data.items(),
            columns=[
                "Feature",
                "Value",
            ],
        ),
        hide_index=True,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Classification probabilities
    # --------------------------------------------------------

    st.markdown(
        "### Classification Probabilities"
    )


    probability_columns = [
        column
        for column in event.index
        if column.startswith(
            "probability_"
        )
    ]


    if probability_columns:

        probabilities = {

            CLASS_LABELS.get(
                column.replace(
                    "probability_",
                    "",
                ),
                column,
            ):
                event[column]

            for column in probability_columns
        }


        probability_df = pd.DataFrame(
            {
                "Class":
                    probabilities.keys(),

                "Probability":
                    probabilities.values(),
            }
        )


        probability_df = (
            probability_df
            .sort_values(
                "Probability",
                ascending=False,
            )
        )


        probability_df["Probability"] = (
            probability_df["Probability"]
            .map(
                lambda x: f"{x:.2%}"
            )
        )


        st.dataframe(
            probability_df,
            hide_index=True,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    st.markdown(
        "### Location"
    )


    st.write(
        f"**Latitude:** "
        f"{event['latitude']:.6f}"
    )


    st.write(
        f"**Longitude:** "
        f"{event['longitude']:.6f}"
    )


    st.write(
        f"**Start:** "
        f"{event['start_date'].date()}"
    )


    st.write(
        f"**End:** "
        f"{event['end_date'].date()}"
    )