#!/usr/bin/env python3
"""
Download the full Planet Tanager Core Imagery *open* STAC catalogue:
- crawls the static STAC recursively
- saves STAC metadata JSONs (root catalog + collections + each item.json)
- downloads *all* assets for *all* items (all file types)

Usage:
  pip install pystac requests
  python download_tanager_full_catalog.py \
    --catalog "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json" \
    --out "./tanager_full" \
    --workers 8 \
    --skip-existing

Optional auth (usually NOT needed for open data):
  export PLANET_BEARER="..."   # or PLANET_API_KEY="..."
  python ... --bearer-env PLANET_BEARER
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import requests
import pystac
from pystac.utils import make_absolute_href


@dataclass(frozen=True)
class AuthConfig:
    api_key: Optional[str] = None         # HTTP Basic: <api_key>:
    bearer_token: Optional[str] = None    # Authorization: Bearer <token>


def _safe_name_from_url(url: str) -> str:
    """
    Extract a stable filename from a URL (keeps extension).
    Falls back to a generic name if path has no filename.
    """
    p = urlparse(url)
    name = Path(unquote(p.path)).name
    return name if name else "asset.bin"


def _session(auth: AuthConfig) -> requests.Session:
    s = requests.Session()
    # Small retry/backoff for transient network issues
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)

    if auth.bearer_token:
        s.headers.update({"Authorization": f"Bearer {auth.bearer_token}"})
    return s


def _download_one(
    sess: requests.Session,
    url: str,
    dst: Path,
    *,
    timeout_s: int,
    retries: int,
    resume: bool,
    skip_existing: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and dst.exists() and dst.stat().st_size > 0:
        return

    part = dst.with_suffix(dst.suffix + ".part")

    headers = {}
    if resume and part.exists() and part.stat().st_size > 0:
        headers["Range"] = f"bytes={part.stat().st_size}-"

    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            with sess.get(url, stream=True, timeout=timeout_s, headers=headers, allow_redirects=True) as r:
                # If we tried Range but server doesn't support it, it might return 200; restart download.
                if r.status_code == 416:
                    # Requested range not satisfiable: treat as complete or restart
                    part.unlink(missing_ok=True)
                    headers.pop("Range", None)
                    continue

                r.raise_for_status()

                mode = "ab" if ("Range" in headers and r.status_code == 206) else "wb"
                if mode == "wb" and part.exists():
                    part.unlink(missing_ok=True)

                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            part.rename(dst)
            return

        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            wait = min(30, 2 ** attempt)
            print(f"[WARN] ({attempt}/{retries}) failed: {url} -> {e} ; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(f"Failed downloading {url}: {last_err}")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Root STAC catalog.json URL or local path")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download workers (network I/O). Use 0 to auto-set to CPU thread count.",
    )
    ap.add_argument("--timeout", type=int, default=180, help="Request timeout seconds")
    ap.add_argument("--retries", type=int, default=6, help="Retries per file")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume via HTTP Range")
    ap.add_argument("--skip-existing", action="store_true", help="Skip already-downloaded files")

    # Optional auth (normally not needed for open data)
    ap.add_argument("--api-key", default="", help="Planet API key (Basic auth). Prefer env var.")
    ap.add_argument("--bearer", default="", help="Bearer token. Prefer env var.")
    ap.add_argument("--api-key-env", default="PLANET_API_KEY", help="Env var name for API key")
    ap.add_argument("--bearer-env", default="PLANET_BEARER", help="Env var name for bearer token")

    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key.strip() or os.getenv(args.api_key_env, "").strip()
    bearer = args.bearer.strip() or os.getenv(args.bearer_env, "").strip()
    auth = AuthConfig(api_key=api_key or None, bearer_token=bearer or None)

    # Load STAC root
    root_href = args.catalog
    root = pystac.Catalog.from_file(root_href)
    root_self = root.get_self_href() or root_href

    # Save root catalog metadata
    stac_meta_dir = out_dir / "_stac"
    _write_json(stac_meta_dir / "root_catalog.json", root.to_dict())

    # Save all collections metadata (if any)
    try:
        collections = list(root.get_collections(recursive=True))
    except TypeError:
        # older pystac versions may not accept recursive here; fallback:
        collections = []
        for c in root.get_children():
            if isinstance(c, pystac.Collection):
                collections.append(c)

    for col in collections:
        col_id = col.id or "collection"
        _write_json(stac_meta_dir / "collections" / f"{col_id}.json", col.to_dict())

    # Enumerate items recursively
    items = list(root.get_items(recursive=True))

    print(f"[INFO] Items discovered: {len(items)}")

    # Build download tasks (all assets of all items)
    manifest_path = out_dir / "manifest.jsonl"
    tasks = []

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for it in items:
            collection_id = getattr(it, "collection_id", None) or it.collection or "no_collection"
            item_dir = out_dir / "data" / collection_id / it.id
            item_dir.mkdir(parents=True, exist_ok=True)

            # Save item metadata
            _write_json(item_dir / "item.json", it.to_dict())

            # Base for resolving relative hrefs
            base = it.get_self_href()
            if not base:
                parent = it.get_parent()
                base = (parent.get_self_href() if parent else None) or root_self

            for asset_key, asset in it.assets.items():
                href = make_absolute_href(asset.href, base)

                filename = _safe_name_from_url(href)
                # Prefix with asset key to avoid collisions when different assets share same basename
                local = item_dir / f"{asset_key}__{filename}"

                mf.write(
                    json.dumps(
                        {
                            "collection_id": collection_id,
                            "item_id": it.id,
                            "asset_key": asset_key,
                            "href": href,
                            "path": str(local.relative_to(out_dir)),
                        }
                    )
                    + "\n"
                )

                tasks.append((href, local))

    print(f"[INFO] Assets queued for download: {len(tasks)}")
    if not tasks:
        print("[INFO] Nothing to download.")
        return 0

    sess = _session(auth)

    # If API key is used, requests supports BasicAuth via auth=(user, pass)
    basic_auth = (auth.api_key, "") if auth.api_key else None

    failures = 0
    done = 0
    t0 = time.time()

    def _worker(url: str, dst: Path) -> None:
        # inject basic auth per request if needed
        if basic_auth:
            # clone session headers; simplest is to call requests.get directly with auth
            # but we want session pooling; use session.request with auth param
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
                return

            part = dst.with_suffix(dst.suffix + ".part")
            headers = {}
            if (not args.no_resume) and part.exists() and part.stat().st_size > 0:
                headers["Range"] = f"bytes={part.stat().st_size}-"

            last_err = None
            for attempt in range(1, args.retries + 1):
                try:
                    with sess.get(
                        url,
                        stream=True,
                        timeout=args.timeout,
                        headers=headers,
                        allow_redirects=True,
                        auth=basic_auth,
                    ) as r:
                        if r.status_code == 416:
                            part.unlink(missing_ok=True)
                            headers.pop("Range", None)
                            continue
                        r.raise_for_status()
                        mode = "ab" if ("Range" in headers and r.status_code == 206) else "wb"
                        if mode == "wb" and part.exists():
                            part.unlink(missing_ok=True)
                        with open(part, mode) as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    part.rename(dst)
                    return
                except Exception as e:
                    last_err = e
                    if attempt == args.retries:
                        break
                    time.sleep(min(30, 2 ** attempt))
            raise RuntimeError(f"Failed downloading {url}: {last_err}")

        # no basic auth
        _download_one(
            sess,
            url,
            dst,
            timeout_s=args.timeout,
            retries=args.retries,
            resume=not args.no_resume,
            skip_existing=args.skip_existing,
        )

    worker_count = args.workers if args.workers > 0 else (os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as ex:
        futs = [ex.submit(_worker, url, dst) for url, dst in tasks]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                failures += 1
                print(f"[ERROR] {e}", file=sys.stderr)
            finally:
                done += 1
                if done % 25 == 0 or done == len(futs):
                    dt = time.time() - t0
                    rate = done / dt if dt > 0 else 0
                    print(f"[INFO] Progress: {done}/{len(futs)}  failures={failures}  ({rate:.2f} files/s)")

    print(f"[DONE] Completed. failures={failures}")
    print(f"[DONE] Output folder: {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
