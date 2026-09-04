import os
from pathlib import Path

import numpy as np
import requests
import rasterio
from dotenv import load_dotenv


load_dotenv()


CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/"
    "process/v1"
)


LATITUDE = 34.188899
LONGITUDE = 73.828170

OUTPUT_FILE = Path(
    "data/features/sentinel2_test.tif"
)


def get_access_token():

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["access_token"]


def build_request():

    # Approximately 1 km x 1 km around the FIRMS location.
    half_size = 0.005

    bbox = [
        LONGITUDE - half_size,
        LATITUDE - half_size,
        LONGITUDE + half_size,
        LATITUDE + half_size,
    ]

    evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [
                "B02",
                "B03",
                "B04",
                "B08",
                "B11",
                "B12",
                "SCL"
            ],
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

    request_body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": (
                        "http://www.opengis.net/"
                        "def/crs/OGC/1.3/CRS84"
                    )
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": "2026-02-28T00:00:00Z",
                            "to": "2026-02-28T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": 128,
            "height": 128,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": evalscript,
    }

    return request_body


def main():

    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is missing."
        )

    print("Getting access token...")

    token = get_access_token()

    print("Token obtained.")

    request_body = build_request()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("Requesting Sentinel-2 pixels...")
    print(f"Location: {LATITUDE}, {LONGITUDE}")

    response = requests.post(
        PROCESS_URL,
        headers=headers,
        json=request_body,
        timeout=120,
    )

    print("Status:", response.status_code)

    if not response.ok:
        print(response.text[:2000])
        response.raise_for_status()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_FILE, "wb") as file:
        file.write(response.content)

    print()
    print("Saved:", OUTPUT_FILE)
    print("Size:", len(response.content), "bytes")

    # Inspect the returned raster.
    with rasterio.open(OUTPUT_FILE) as src:

        print()
        print("RASTER INFORMATION")
        print("=" * 60)

        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

        data = src.read()

        print()
        print("ARRAY")
        print("=" * 60)

        print("Shape:", data.shape)
        print("dtype:", data.dtype)

        band_names = [
            "B02",
            "B03",
            "B04",
            "B08",
            "B11",
            "B12",
            "SCL",
        ]

        print()

        for i, name in enumerate(band_names):

            band = data[i]

            valid = np.isfinite(band)

            if not valid.any():
                print(name, ": no valid pixels")
                continue

            values = band[valid]

            print(
                f"{name}: "
                f"min={values.min():.4f}, "
                f"mean={values.mean():.4f}, "
                f"max={values.max():.4f}"
            )


if __name__ == "__main__":
    main()