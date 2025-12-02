#!/usr/bin/env python3
"""
Download required datasets for Arctic-XBeach examples.
Run this script after installation to fetch case study data.

Usage:
    python download_data.py
    python download_data.py --target path/to/arctic-xbeach
"""

import os
import urllib.request
import argparse
from pathlib import Path

# Data files to download
DATASETS = {
    "barter_island": {
        "base_url": "https://deltares-usa-software.s3.us-east-1.amazonaws.com/arctic_xbeach/database_barter_island",
        "files": ["era5.csv", "storms.csv"],
        "target_subdir": "examples/case_studies/barter_island/database"
    }
}

def download_file(url, target_path):
    """Download a file with progress indication."""
    print(f"  Downloading: {url}")
    print(f"  Target: {target_path}")
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"  ✓ Complete ({os.path.getsize(target_path) / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Arctic-XBeach datasets")
    parser.add_argument("--target", type=str, default=".", 
                        help="Path to arctic-xbeach repository root")
    args = parser.parse_args()
    
    repo_root = Path(args.target).resolve()
    print(f"\nArctic-XBeach Data Downloader")
    print(f"Repository root: {repo_root}\n")
    
    success_count = 0
    total_count = 0
    
    for case_name, case_info in DATASETS.items():
        print(f"[{case_name}]")
        target_dir = repo_root / case_info["target_subdir"]
        
        for filename in case_info["files"]:
            total_count += 1
            url = f"{case_info['base_url']}/{filename}"
            target_path = target_dir / filename
            
            if download_file(url, target_path):
                success_count += 1
        print()
    
    print(f"Downloaded {success_count}/{total_count} files successfully.")
    
    if success_count == total_count:
        print("\n✓ All data files ready!")
    else:
        print("\n⚠ Some downloads failed. Check your internet connection and try again.")

if __name__ == "__main__":
    main()
