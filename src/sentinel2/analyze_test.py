import numpy as np
import rasterio


INPUT_FILE = "data/features/sentinel2_test.tif"


def masked_stats(values, mask):
    values = values[mask]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def main():

    with rasterio.open(INPUT_FILE) as src:

        data = src.read().astype(np.float32)

    B02 = data[0]
    B03 = data[1]
    B04 = data[2]
    B08 = data[3]
    B11 = data[4]
    B12 = data[5]
    SCL = data[6]

    # Sentinel-2 SCL classes that we consider usable.
    #
    # 4 = Vegetation
    # 5 = Not-vegetated
    # 6 = Water
    #
    # Exclude:
    # 0 = No data
    # 1 = Saturated/defective
    # 2 = Dark area
    # 3 = Cloud shadow
    # 7 = Unclassified
    # 8 = Cloud medium probability
    # 9 = Cloud high probability
    # 10 = Thin cirrus
    # 11 = Snow/ice
    valid = np.isin(SCL, [4, 5, 6])

    print("TOTAL PIXELS:", B02.size)
    print("VALID PIXELS:", valid.sum())
    print(
        "VALID FRACTION:",
        f"{valid.mean():.2%}"
    )

    # Avoid division by zero.
    eps = 1e-10

    NDVI = (B08 - B04) / (B08 + B04 + eps)

    NDBI = (B11 - B08) / (B11 + B08 + eps)

    NDWI = (B03 - B08) / (B03 + B08 + eps)

    print("\nSPECTRAL FEATURES")
    print("=" * 70)

    features = {
        "B02": B02,
        "B03": B03,
        "B04": B04,
        "B08": B08,
        "B11": B11,
        "B12": B12,
        "NDVI": NDVI,
        "NDBI": NDBI,
        "NDWI": NDWI,
    }

    for name, values in features.items():

        stats = masked_stats(
            values,
            valid & np.isfinite(values)
        )

        print(
            f"{name:6s} "
            f"count={stats['count']:6d} "
            f"mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} "
            f"std={stats['std']:.4f}"
        )

    print("\nSCL DISTRIBUTION")
    print("=" * 70)

    classes, counts = np.unique(
        SCL.astype(np.int16),
        return_counts=True
    )

    for cls, count in zip(classes, counts):

        print(
            f"SCL {cls:2d}: "
            f"{count:6d} pixels "
            f"({count / SCL.size:.2%})"
        )


if __name__ == "__main__":
    main()