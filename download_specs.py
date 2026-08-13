"""
Download sample 3GPP specification PDFs from ETSI (free, public).
These are the official published versions of 3GPP specifications.

Usage:
    python download_specs.py
"""
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR


# ETSI hosts official 3GPP specs as free PDFs
# Format: https://www.etsi.org/deliver/etsi_ts/SERIES/SPEC/VERSION/FILENAME
SPECS_TO_DOWNLOAD = [
    {
        "name": "TS 23.501 - 5G System Architecture",
        "url": "https://www.etsi.org/deliver/etsi_ts/123500_123599/123501/17.06.00_60/ts_123501v170600p.pdf",
        "filename": "ts_123501v170600p.pdf",
    },
    {
        "name": "TS 23.502 - 5G Procedures",
        "url": "https://www.etsi.org/deliver/etsi_ts/123500_123599/123502/17.06.00_60/ts_123502v170600p.pdf",
        "filename": "ts_123502v170600p.pdf",
    },
    {
        "name": "TS 23.503 - Policy and Charging Control",
        "url": "https://www.etsi.org/deliver/etsi_ts/123500_123599/123503/17.04.00_60/ts_123503v170400p.pdf",
        "filename": "ts_123503v170400p.pdf",
    },
    {
        "name": "TS 38.300 - NR Overall Description",
        "url": "https://www.etsi.org/deliver/etsi_ts/138300_138399/138300/17.03.00_60/ts_138300v170300p.pdf",
        "filename": "ts_138300v170300p.pdf",
    },
    {
        "name": "TS 33.501 - 5G Security Architecture",
        "url": "https://www.etsi.org/deliver/etsi_ts/133500_133599/133501/17.07.00_60/ts_133501v170700p.pdf",
        "filename": "ts_133501v170700p.pdf",
    },
    {
        "name": "TS 29.500 - 5G Service Based Architecture",
        "url": "https://www.etsi.org/deliver/etsi_ts/129500_129599/129500/17.12.00_60/ts_129500v171200p.pdf",
        "filename": "ts_129500v171200p.pdf",
    },
    {
        "name": "TS 24.501 - 5G NAS Protocol",
        "url": "https://www.etsi.org/deliver/etsi_ts/124500_124599/124501/17.08.00_60/ts_124501v170800p.pdf",
        "filename": "ts_124501v170800p.pdf",
    },
]


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc[:40]) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"    ❌ Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    print("=" * 60)
    print("📥 3GPP Specification Downloader")
    print("=" * 60)
    print(f"\nDownload directory: {DATA_DIR}")
    print(f"Specs to download: {len(SPECS_TO_DOWNLOAD)}")
    print("\nSource: ETSI (official, free, public)")
    print("-" * 60)
    
    # Create directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for spec in SPECS_TO_DOWNLOAD:
        dest_path = DATA_DIR / spec["filename"]
        
        print(f"\n📄 {spec['name']}")
        
        # Skip if already downloaded
        if dest_path.exists():
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ Already exists ({size_mb:.1f} MB)")
            skipped += 1
            continue
        
        # Download
        print(f"   URL: {spec['url']}")
        success = download_file(spec["url"], dest_path, spec["name"])
        
        if success:
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"   ✓ Downloaded ({size_mb:.1f} MB)")
            downloaded += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"   Downloaded: {downloaded}")
    print(f"   Skipped (existing): {skipped}")
    print(f"   Failed: {failed}")
    print(f"\n   PDF location: {DATA_DIR}")
    
    if downloaded + skipped > 0:
        print(f"\n✅ Ready! Run 'python ingest.py' to build the vector store.")
    
    if failed > 0:
        print(f"\n⚠️  Some downloads failed. This may be due to:")
        print(f"   - Network issues (try again)")
        print(f"   - ETSI server temporarily unavailable")
        print(f"   - URL version changed (check etsi.org manually)")
        print(f"\n   You can also download specs manually from:")
        print(f"   https://www.etsi.org/standards#page=1&search=3GPP")
        print(f"   https://www.3gpp.org/ftp/Specs/")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
