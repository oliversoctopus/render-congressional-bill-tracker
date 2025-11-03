"""
Download Bulk Bill Status Data from GovInfo

This script downloads pre-packaged BILLSTATUS data from the Library of Congress
GovInfo bulk data service. This is MUCH faster than API calls (30-60 minutes vs 7 days).

NO API KEY REQUIRED for bulk downloads.
(API key is still needed for live website queries - this doesn't affect that)

Usage:
    python download_bulk_data.py

Output:
    - Downloads ZIP files to data/bulk_downloads/{congress}/
    - Extracts XML files
    - Progress is saved, so you can resume if interrupted

Expected Runtime: 30-60 minutes
Expected Storage: ~500-800 MB compressed, ~2-3 GB extracted
"""

import requests
import os
import zipfile
from pathlib import Path
import json
from datetime import datetime
import time

# Configuration
CONGRESSES = [113, 114, 115, 116, 117, 118]
BILL_TYPES = ['hr', 's', 'hjres', 'sjres']
BASE_URL = "https://www.govinfo.gov/bulkdata/BILLSTATUS"

# Output directories
DOWNLOAD_DIR = Path("bulk_downloads")
PROGRESS_FILE = DOWNLOAD_DIR / "download_progress.json"


def load_progress():
    """Load download progress from checkpoint file"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save download progress to checkpoint file"""
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def download_file(url, output_path, chunk_size=8192):
    """
    Download a file with progress reporting.

    Returns: (success, error_message)
    """
    try:
        # Make request
        response = requests.get(url, stream=True, timeout=60)

        if response.status_code != 200:
            return (False, f"HTTP {response.status_code}")

        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024)

        # Download with progress
        downloaded = 0
        start_time = time.time()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress update every 5 MB
                    if downloaded % (5 * 1024 * 1024) < chunk_size:
                        progress_pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        elapsed = time.time() - start_time
                        speed_mbps = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0

                        print(f"    {progress_pct:.1f}% ({downloaded/(1024*1024):.1f}/{total_size_mb:.1f} MB) "
                              f"@ {speed_mbps:.1f} MB/s", end='\r')

        # Final update
        elapsed = time.time() - start_time
        speed_mbps = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        print(f"    100.0% ({total_size_mb:.1f}/{total_size_mb:.1f} MB) "
              f"@ {speed_mbps:.1f} MB/s - Complete!              ")

        return (True, None)

    except requests.exceptions.Timeout:
        return (False, "Timeout")
    except requests.exceptions.ConnectionError:
        return (False, "Connection error")
    except Exception as e:
        return (False, str(e))


def extract_zip(zip_path, extract_to):
    """
    Extract ZIP file to directory.

    Returns: (success, num_files_extracted, error_message)
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()

            # Extract all
            zip_ref.extractall(extract_to)

            return (True, len(file_list), None)

    except zipfile.BadZipFile:
        return (False, 0, "Corrupted ZIP file")
    except Exception as e:
        return (False, 0, str(e))


def download_congress_data(congress, bill_type, progress):
    """
    Download and extract data for one congress and bill type.

    Returns: (success, error_message)
    """
    # Check if already downloaded
    key = f"{congress}_{bill_type}"
    if progress.get(key, {}).get('extracted', False):
        print(f"  ✓ {bill_type.upper():6s} - Already downloaded and extracted (cached)")
        return (True, "cached")

    # Prepare paths
    congress_dir = DOWNLOAD_DIR / str(congress)
    congress_dir.mkdir(parents=True, exist_ok=True)

    zip_filename = f"BILLSTATUS-{congress}-{bill_type}.zip"
    zip_path = congress_dir / zip_filename
    extract_dir = congress_dir / bill_type

    # Build URL
    url = f"{BASE_URL}/{congress}/{bill_type}/{zip_filename}"

    print(f"  → {bill_type.upper():6s} - Downloading...")

    # Download if not already downloaded
    if not zip_path.exists() or not progress.get(key, {}).get('downloaded', False):
        success, error = download_file(url, zip_path)

        if not success:
            print(f"  ✗ {bill_type.upper():6s} - Download failed: {error}")
            return (False, error)

        # Mark as downloaded
        if key not in progress:
            progress[key] = {}
        progress[key]['downloaded'] = True
        progress[key]['download_time'] = datetime.now().isoformat()
        save_progress(progress)
    else:
        print(f"    ZIP already downloaded, skipping download")

    # Extract
    print(f"  → {bill_type.upper():6s} - Extracting...")
    success, num_files, error = extract_zip(zip_path, extract_dir)

    if not success:
        print(f"  ✗ {bill_type.upper():6s} - Extraction failed: {error}")
        return (False, error)

    print(f"  ✓ {bill_type.upper():6s} - Extracted {num_files} XML files")

    # Mark as extracted
    progress[key]['extracted'] = True
    progress[key]['num_files'] = num_files
    progress[key]['extract_time'] = datetime.now().isoformat()
    save_progress(progress)

    return (True, None)


def main():
    """Main download orchestration"""
    print("="*70)
    print("BULK BILL STATUS DATA DOWNLOAD")
    print("="*70)
    print(f"Source: GovInfo (Library of Congress)")
    print(f"Congresses: {', '.join(map(str, CONGRESSES))}")
    print(f"Bill Types: {', '.join([t.upper() for t in BILL_TYPES])}")
    print(f"Total Downloads: {len(CONGRESSES) * len(BILL_TYPES)} ZIP files")
    print(f"Output: {DOWNLOAD_DIR.absolute()}")
    print("="*70)
    print()

    # Load progress
    progress = load_progress()
    if progress:
        completed = sum(1 for v in progress.values() if v.get('extracted', False))
        print(f"Resuming from previous run: {completed} files already completed")
        print()

    # Statistics
    total_files = len(CONGRESSES) * len(BILL_TYPES)
    completed_files = 0
    failed_files = 0
    cached_files = 0
    start_time = datetime.now()

    # Download each congress
    for congress in CONGRESSES:
        print(f"\n{'='*70}")
        print(f"{congress}th Congress")
        print(f"{'='*70}")

        for bill_type in BILL_TYPES:
            success, error = download_congress_data(congress, bill_type, progress)

            if success:
                if error == "cached":
                    cached_files += 1
                else:
                    completed_files += 1
            else:
                failed_files += 1

            # Brief pause between downloads to be nice to server
            time.sleep(0.5)

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60  # minutes

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total files: {total_files}")
    print(f"Newly downloaded: {completed_files}")
    print(f"Cached (skipped): {cached_files}")
    print(f"Failed: {failed_files}")
    print(f"Total time: {elapsed:.1f} minutes")
    print(f"\nNext step: Run parse_bulk_data.py to convert XML to CSV")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Progress has been saved. Re-run this script to resume.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print("Progress has been saved. Re-run this script to resume.")
