"""
scripts/download_data.py
========================
Automated dataset download helper for the Intel Image Classification dataset
via the Kaggle API.

Prerequisites
-------------
1. Install the Kaggle Python package:
       pip install kaggle

2. Create a Kaggle API token:
   - Log in at https://www.kaggle.com
   - Go to Account → API → Create New API Token
   - Save the downloaded `kaggle.json` to:
       Windows:  C:\\Users\\<username>\\.kaggle\\kaggle.json
       Linux/macOS: ~/.kaggle/kaggle.json
   - On Windows, you may also set the environment variable:
       KAGGLE_CONFIG_DIR=C:\\Users\\<username>\\.kaggle

Usage
-----
    python scripts/download_data.py

The script downloads and extracts the dataset into the `data/` directory
at the project root.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path


def check_kaggle_installed() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False


def check_kaggle_credentials() -> bool:
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json"
        if os.environ.get("KAGGLE_CONFIG_DIR") else None
    ]
    return any(p and p.exists() for p in kaggle_json_paths if p)


def download_and_extract():
    print("=" * 60)
    print("  Intel Image Classification Dataset Downloader")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Dependency checks
    # -----------------------------------------------------------------------
    if not check_kaggle_installed():
        print("\n[ERROR] The `kaggle` package is not installed.")
        print("  Run:  pip install kaggle")
        sys.exit(1)

    if not check_kaggle_credentials():
        print("\n[ERROR] Kaggle API credentials not found.")
        print("  Please create a Kaggle API token and save it as:")
        print("    Windows:     C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("    Linux/macOS: ~/.kaggle/kaggle.json")
        print("  See: https://www.kaggle.com/docs/api")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "intel-image-classification.zip")

    print(f"\n[Download] Downloading dataset to '{data_dir}/'...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "puneet6060/intel-image-classification",
            "-p", data_dir
        ],
        capture_output=False
    )

    if result.returncode != 0:
        print("\n[ERROR] Kaggle download failed. Check your credentials and internet connection.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Extract
    # -----------------------------------------------------------------------
    print(f"\n[Extract] Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    print(f"[Extract] Done. Dataset available in '{data_dir}/'.")

    # -----------------------------------------------------------------------
    # Validate structure
    # -----------------------------------------------------------------------
    expected_dirs = [
        "data/seg_train/seg_train",
        "data/seg_test/seg_test",
        "data/seg_pred/seg_pred"
    ]
    print("\n[Validate] Checking directory structure...")
    all_ok = True
    for d in expected_dirs:
        exists = os.path.isdir(d)
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  {d}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n[OK] Dataset downloaded and validated successfully.")
        print("  You can now run:  python src/train.py")
    else:
        print("\n[WARNING] Some directories are missing. The Kaggle archive structure may")
        print("  have changed. Please check the 'data/' directory and update")
        print("  configs/config.yaml → data.train_dir / test_dir / pred_dir accordingly.")


if __name__ == "__main__":
    download_and_extract()
