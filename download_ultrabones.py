#!/usr/bin/env python3
"""
download_ultrabones.py - Download and extract specimens from the UltraBones100k Hugging Face dataset.

Usage:
------
    # Download and extract specimen 1 (default) for quick validation
    python download_ultrabones.py --specimens 1 --dest_dir ./data/UltraBones100k

    # Download all 14 specimens (Warning: ~40 GB total)
    python download_ultrabones.py --specimens all --dest_dir ./data/UltraBones100k
"""

import argparse
import sys
import os
import zipfile
from pathlib import Path

# Repo and specimen config
REPO_ID = "luohwu/UltraBones100k"
TOTAL_SPECIMENS = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the UltraBones100k dataset from Hugging Face."
    )
    parser.add_argument(
        "--specimens",
        type=str,
        default="1",
        help="Comma-separated list of specimen numbers (1-14) to download, or 'all'. Default is '1'.",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="./data/UltraBones100k",
        help="Directory where specimens will be extracted.",
    )
    parser.add_argument(
        "--keep_zips",
        action="store_true",
        help="Keep downloaded ZIP files after extraction. Default is to delete them to save space.",
    )
    return parser.parse_args()


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path.name} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get total size of files to extract for basic progress reporting
        uncompress_size = sum((file.file_size for file in zip_ref.infolist()))
        extracted_size = 0
        
        for file in zip_ref.infolist():
            zip_ref.extract(file, extract_to)
            extracted_size += file.file_size
            # Simple progress percentage
            pct = (extracted_size / uncompress_size) * 100
            if pct % 10 < 1:  # print approximately every 10%
                sys.stdout.write(f"\r  Extraction progress: {pct:.1f}%")
                sys.stdout.flush()
    print("\nExtraction complete.")


def main() -> None:
    args = parse_args()

    # Ensure huggingface_hub is installed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[-] Error: 'huggingface_hub' is required.")
        print("    Please install it using: pip install huggingface_hub")
        sys.exit(1)

    dest_path = Path(args.dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Determine specimens to download
    if args.specimens.lower() == "all":
        specimen_indices = list(range(1, TOTAL_SPECIMENS + 1))
    else:
        try:
            specimen_indices = [int(x.strip()) for x in args.specimens.split(",") if x.strip()]
        except ValueError:
            print(f"[-] Invalid specimen format: '{args.specimens}'. Must be comma-separated integers.")
            sys.exit(1)

    # Validate indices
    for idx in specimen_indices:
        if not (1 <= idx <= TOTAL_SPECIMENS):
            print(f"[-] Specimen index {idx} out of range (1-{TOTAL_SPECIMENS}).")
            sys.exit(1)

    print(f"[+] Dest Directory: {dest_path.resolve()}")
    print(f"[+] Selected specimens: {specimen_indices}")

    for idx in specimen_indices:
        zip_filename = f"specimen{idx:02d}.zip"
        print(f"\n[+] Downloading {zip_filename} from Hugging Face ({REPO_ID}) ...")
        
        try:
            downloaded_zip = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_filename,
                repo_type="dataset",
            )
            downloaded_path = Path(downloaded_zip)
            print(f"[+] Downloaded successfully: {downloaded_path}")
            
            # Extract
            extract_zip(downloaded_path, dest_path)
            
            # Clean up ZIP to save space if not requested to keep
            if not args.keep_zips:
                print(f"[+] Cleaning up ZIP file: {downloaded_path}")
                try:
                    os.remove(downloaded_path)
                except OSError as e:
                    print(f"[-] Warning: Failed to delete zip file: {e}")
            
        except Exception as e:
            print(f"[-] Error processing {zip_filename}: {e}")
            print("    Please ensure you have network connectivity and 'huggingface_hub' is up-to-date.")
            sys.exit(1)

    print("\n[+] Done! All selected specimens downloaded and extracted.")


if __name__ == "__main__":
    main()
