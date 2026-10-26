"""Download the UCI Adult (Census Income) dataset into data/raw/.

Usage (from the project root):
    python data/download_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
RAW_DIR = Path(__file__).resolve().parent / "raw"
FILES = ["adult.data", "adult.test", "adult.names"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / f).exists() for f in FILES):
        print(f"Data already present in {RAW_DIR}")
        return

    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name in FILES:
        (RAW_DIR / name).write_bytes(archive.read(name))
        print(f"  saved {RAW_DIR / name}")


if __name__ == "__main__":
    main()
