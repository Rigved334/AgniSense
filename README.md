# 🔥 AgniSense

### AI-Based Detection and Classification of Industrial Fires and Persistent Thermal Sources

AgniSense is a geospatial AI-based decision-support system for detecting and classifying thermal anomalies using satellite-derived observations and industrial infrastructure data.

The system combines **NASA FIRMS**, **OpenStreetMap (OSM)**, **ESA WorldCover**, and **Sentinel-2** data to distinguish between:

- 🔥 Industrial fires
- 🏭 Persistent industrial thermal sources
- 🌲 Wildfires
- 🌾 Agricultural fires

The system also assigns an **investigation priority** to detected events and provides an interactive GIS dashboard for analysts.

> **Project type:** Prototype / Decision-Support System  
> **Application domain:** Disaster Management, Remote Sensing, Geospatial AI  
> **Target use case:** Industrial fire detection and thermal-source monitoring

---

## 🎯 Problem

Satellite thermal anomaly products such as NASA FIRMS can identify locations exhibiting elevated thermal radiation, but a thermal anomaly alone does not explain **what caused it**.

A detected hotspot could correspond to:

- An industrial fire
- A refinery or flare
- A persistent industrial heat source
- A forest fire
- Agricultural burning
- Other thermal activity

For disaster-management and infrastructure monitoring, identifying the **context and likely source type** of a thermal anomaly is therefore critical.

AgniSense addresses this problem through **multi-source geospatial feature fusion**.

---

# 🧠 Solution Overview

AgniSense follows this pipeline:

``
                 NASA FIRMS
              Thermal Anomalies
                     │
                     ▼
              Thermal Episodes
                     │
                     ▼
                H3 Aggregation
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
       OSM        WorldCover   Sentinel-2
   Industrial      Land Cover   Pre-event
    Context        Context      Spectral
        │            │            │
        └────────────┼────────────┘
                     ▼
              Feature Fusion
                     │
                     ▼
              ML Classification
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Industrial     Persistent    Natural /
      Fire          Source       Agricultural
        │
        ▼
 Investigation Priority
        │
        ▼
     GIS Dashboard
