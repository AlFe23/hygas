#!/usr/bin/env python3
"""Download explicitly selected EMIT scene NetCDF/GeoTIFF products, without retrievals."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.downloads.emit import EmitDownloadError, download_emit_products, search_emit_products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, help="Exact CMR short name, e.g. EMITL2ARFL")
    parser.add_argument("--version", required=True, help="Explicit collection/file version, e.g. 001")
    parser.add_argument("--scene", required=True, help="YYYYMMDDTHHMMSS_ORBIT_SCENE, without version")
    parser.add_argument("--products", required=True, nargs="+", help="Exact file codes, e.g. RAD OBS or MASK")
    parser.add_argument("--output", required=True, type=Path, help="Destination for data and manifests")
    parser.add_argument("--dry-run", action="store_true", help="Public search only; no login, files or directories created")
    parser.add_argument("--auth", choices=["environment", "interactive", "netrc"], default="environment",
                        help="earthaccess strategy, persist=False; default uses environment credentials or the saved HyGAS token")
    args = parser.parse_args(argv)
    try:
        assets = search_emit_products(collection=args.collection, version=args.version,
            scene=args.scene, products=args.products)
        print(json.dumps({"assets": [asdict(a) for a in assets], "dry_run": args.dry_run}, indent=2))
        if not args.dry_run:
            manifest = download_emit_products(assets, args.output, strategy=args.auth)
            print(f"Download complete. Manifest: {manifest}")
        return 0
    except EmitDownloadError as exc:
        print(f"EMIT acquisition failed: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Do not echo upstream exception strings: URLs/response bodies can hold secrets.
        print("EMIT acquisition failed: check exact collection/version/scene/products, credentials, network and destination. "
              "No existing file was overwritten; inspect .partial if present.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
