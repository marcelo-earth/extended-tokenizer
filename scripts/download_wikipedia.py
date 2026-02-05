#!/usr/bin/env python3
"""Download Wikipedia dump for tokenizer training."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extended_tokenizer.data.wikipedia import WikipediaDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia dump for tokenizer training"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/wikipedia",
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Custom URL to download (defaults to latest English Wikipedia dump)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Wikipedia Dump Downloader")
    print("=" * 60)
    print()
    print(f"Output directory: {args.output}")
    print()
    print("NOTE: The English Wikipedia dump is approximately 20GB compressed.")
    print("Make sure you have enough disk space and a stable connection.")
    print()

    downloader = WikipediaDownloader(output_dir=args.output)

    try:
        output_path = downloader.download(
            url=args.url,
            show_progress=not args.no_progress,
        )
        print()
        print(f"Download complete: {output_path}")
        print()
        print("Next step: Run the preprocessing script:")
        print(f"  python scripts/preprocess.py --input {output_path}")

    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
