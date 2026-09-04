import os
import requests
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("CDSE_CLIENT_ID")
client_secret = os.getenv("CDSE_CLIENT_SECRET")

if not client_id or not client_secret:
    raise RuntimeError("CDSE_CLIENT_ID or CDSE_CLIENT_SECRET is missing")

token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

data = {
    "client_id": client_id,
    "client_secret": client_secret,
    "grant_type": "client_credentials",
}

response = requests.post(token_url, data=data)

print("Status:", response.status_code)

if response.ok:
    token = response.json()["access_token"]
    print("Authentication successful!")
    print("Token received:", token[:20] + "...")
else:
    print("Authentication failed.")
    print(response.text)