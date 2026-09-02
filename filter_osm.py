from pathlib import Path
import subprocess
import shutil
import sys


INPUT = Path(r"data\osm\raw\india-latest.osm.pbf")
OUTPUT = Path(r"data\osm\filtered\industrial-relevant.osm.pbf")

TAGS = [
    "n/landuse=industrial",
    "w/landuse=industrial",
    "r/landuse=industrial",

    "n/man_made=works",
    "w/man_made=works",
    "r/man_made=works",

    "n/man_made=flare",
    "w/man_made=flare",
    "r/man_made=flare",

    "n/man_made=storage_tank",
    "w/man_made=storage_tank",
    "r/man_made=storage_tank",

    "n/power=plant",
    "w/power=plant",
    "r/power=plant",

    "n/landuse=quarry",
    "w/landuse=quarry",
    "r/landuse=quarry",
]


def main():
    if shutil.which("osmium") is None:
        raise RuntimeError(
            "osmium was not found. Make sure the fire-gis environment is active."
        )

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT.resolve()}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    command = [
    "osmium",
    "tags-filter",
    str(INPUT),
    *TAGS,
    "-o",
    str(OUTPUT),
    "--overwrite",
    ]

    print("Input:")
    print(f"  {INPUT.resolve()}")

    print("\nOutput:")
    print(f"  {OUTPUT.resolve()}")

    print("\nTags being extracted:")
    for tag in TAGS:
        print(f"  {tag}")

    print("\nStarting OSM filtering...\n")

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        print(
            f"\nOsmium failed with exit code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    if not OUTPUT.exists():
        raise RuntimeError(
            "Osmium finished without creating the expected output file."
        )

    size_gb = OUTPUT.stat().st_size / (1024 ** 3)

    print("\nFiltering completed successfully.")
    print(f"Output size: {size_gb:.2f} GB")
    print(f"Output file: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()