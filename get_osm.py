from pathlib import Path
import hashlib
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://download.geofabrik.de/asia/india"

# Start with the country extract.
# You can switch this to the sub-region URLs below if Geofabrik is busy.
FILES = {
    "india-latest.osm.pbf": f"{BASE_URL}-latest.osm.pbf",
}

OUTPUT_DIR = Path("osm_data/raw")
CHUNK_SIZE = 1024 * 1024  # 1 MB

MAX_RETRIES = 8
BACKOFF_FACTOR = 2


def create_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "SIH2026-Industrial-Fire-Research/1.0"
    })

    return session


def download_file(session: requests.Session, url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_file = destination.with_suffix(destination.suffix + ".part")

    existing_size = temp_file.stat().st_size if temp_file.exists() else 0

    headers = {}

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"Resuming download from {existing_size / (1024**2):.2f} MB")

    print(f"\nURL: {url}")
    print(f"Output: {destination.resolve()}")

    try:
        with session.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, 300),
        ) as response:

            # Server ignored Range request.
            if existing_size > 0 and response.status_code == 200:
                print("Server does not support resume. Restarting download.")
                existing_size = 0
                temp_file.unlink(missing_ok=True)

            response.raise_for_status()

            content_length = response.headers.get("Content-Length")

            if content_length is not None:
                total_size = int(content_length) + existing_size
            else:
                total_size = None

            mode = "ab" if existing_size > 0 else "wb"

            downloaded = existing_size

            with open(temp_file, mode) as f:
                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):
                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded / total_size * 100

                        print(
                            f"\rProgress: {percent:6.2f}% "
                            f"({downloaded / (1024**2):,.1f} / "
                            f"{total_size / (1024**2):,.1f} MB)",
                            end="",
                        )
                    else:
                        print(
                            f"\rDownloaded: "
                            f"{downloaded / (1024**2):,.1f} MB",
                            end="",
                        )

        print()

        temp_file.replace(destination)

        print("Download completed.")
        print(
            f"Final size: "
            f"{destination.stat().st_size / (1024**3):.2f} GB"
        )

    except requests.RequestException as e:
        print(f"\nDownload failed: {e}")
        print(f"Partial file preserved at: {temp_file}")
        raise


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    print("\nCalculating MD5 checksum...")

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            md5.update(chunk)

    actual_md5 = md5.hexdigest()

    print(f"Expected: {expected_md5}")
    print(f"Actual:   {actual_md5}")

    return actual_md5.lower() == expected_md5.lower()


def main():
    session = create_session()

    for filename, url in FILES.items():
        destination = OUTPUT_DIR / filename

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                download_file(session, url, destination)
                break

            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    print("Maximum retry count reached.")
                    raise

                wait = min(BACKOFF_FACTOR ** attempt, 120)

                print(
                    f"Retrying in {wait} seconds "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                )

                time.sleep(wait)


if __name__ == "__main__":
    main()