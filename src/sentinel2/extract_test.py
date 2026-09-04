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

PRODUCT_ID = "7c9701ff-21d2-4698-a2d0-183c3f13174e"

DOWNLOAD_URL = (
    "https://download.dataspace.copernicus.eu/"
    f"odata/v1/Products({PRODUCT_ID})/$value"
)


def get_token():

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


def main():

    token = get_token()

    response = requests.get(
        DOWNLOAD_URL,
        headers={
            "Authorization": f"Bearer {token}"
        },
        stream=True,
        timeout=60,
    )

    print("Status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Content-Length:", response.headers.get("Content-Length"))

    if not response.ok:
        print(response.text[:1000])
        response.raise_for_status()

    print("\nDownload endpoint is accessible.")
    print("We will NOT download the product yet.")


if __name__ == "__main__":
    main()