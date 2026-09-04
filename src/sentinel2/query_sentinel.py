import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CDSE_CLIENT_ID")
CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

CATALOGUE_URL = (
    "https://catalogue.dataspace.copernicus.eu/"
    "odata/v1/Products"
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


def search_sentinel2(
    latitude,
    longitude,
    start_date,
    end_date,
    max_cloud=30,
):
    token = get_access_token()

    # Small point around the FIRMS location.
    # Intersects() is used by the CDSE OData catalogue.
    area = (
        f"geography'SRID=4326;"
        f"POINT ({longitude} {latitude})'"
    )

    filters = [
        "Collection/Name eq 'SENTINEL-2'",

        # Sentinel-2 Level-2A
        (
            "Attributes/OData.CSC.StringAttribute/"
            "any(att:att/Name eq 'productType' "
            "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
        ),

        # Cloud cover
        (
            "Attributes/OData.CSC.DoubleAttribute/"
            f"any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
        ),

        # Spatial intersection
        f"OData.CSC.Intersects(area={area})",

        # Date range
        (
            f"ContentDate/Start ge "
            f"{start_date}T00:00:00.000Z"
        ),
        (
            f"ContentDate/Start le "
            f"{end_date}T23:59:59.999Z"
        ),
    ]

    params = {
        "$filter": " and ".join(filters),
        "$orderby": "ContentDate/Start asc",
        "$top": 20,
    }

    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(
        CATALOGUE_URL,
        params=params,
        headers=headers,
        timeout=60,
    )

    print("Status:", response.status_code)

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json()["value"]


if __name__ == "__main__":

    results = search_sentinel2(
        latitude=34.188899,
        longitude=73.828170,
        start_date="2026-02-20",
        end_date="2026-03-20",
        max_cloud=30,
    )

    print(f"\nFound {len(results)} Sentinel-2 L2A products\n")

    for product in results:
        print("Name:", product.get("Name"))
        print("ID:", product.get("Id"))
        print("Date:", product.get("ContentDate", {}).get("Start"))
        print("S3Path:", product.get("S3Path"))
    
        print("Attributes:")
        for attr in product.get("Attributes", []):
            print(
                " ",
                attr.get("Name"),
                "=",
                attr.get("Value")
            )
    
        print("-" * 80)