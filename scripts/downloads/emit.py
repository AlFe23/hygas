"""Discover exact-scene EMIT NetCDF/GeoTIFF assets and download with earthaccess.

No implicit authentication, version upgrades, retrievals or QA application.
Credentials are handled by earthaccess, never stored in HyGAS manifests.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import uuid
from urllib.parse import unquote, urlsplit

from .credentials import DEFAULT_TOKEN_FILE, read_token


class EmitDownloadError(RuntimeError):
    """An acquisition error safe to display without exposing upstream secrets."""


@dataclass(frozen=True)
class EmitAsset:
    collection: str
    version: str
    scene: str
    product: str
    filename: str
    url: str
    granule_id: str
    granule_revision: str
    granule_name: str
    size_bytes: int | None = None
    checksum_algorithm: str | None = None
    checksum_value: str | None = None
    cmr_size: float | None = None
    cmr_size_unit: str | None = None


_SCENE = re.compile(r"\d{8}T\d{6}_\d{7}_\d{3}")
_FILE = re.compile(r"EMIT_L[12][AB]_([A-Z][A-Z0-9]*)_(\d{3})_(\d{8}T\d{6}_\d{7}_\d{3})\.(?:nc|tif)")


def _validate_query(collection, version, scene, products):
    if not re.fullmatch(r"EMIT[A-Z0-9]+", collection):
        raise ValueError("collection must be an explicit EMIT CMR short name")
    if not re.fullmatch(r"\d{3}", version):
        raise ValueError("version must be explicit and three digits, e.g. 001")
    if not _SCENE.fullmatch(scene):
        raise ValueError("scene must be YYYYMMDDTHHMMSS_ORBIT_SCENE, without version")
    datetime.strptime(scene.split("_")[0], "%Y%m%dT%H%M%S")
    products = tuple(products)
    if not products or len(set(products)) != len(products):
        raise ValueError("provide at least one product, without duplicates")
    if any(not re.fullmatch(r"[A-Z][A-Z0-9]*", p) for p in products):
        raise ValueError("products must be explicit codes, e.g. RAD OBS or MASK")
    return products


def _client(client=None):
    if client is not None:
        return client
    try:
        return importlib.import_module("earthaccess")
    except ImportError:
        raise EmitDownloadError("Install the optional dependency: python -m pip install -r requirements-earthdata.txt") from None


@contextmanager
def _private_call():
    # Upstream HTTP exceptions/logs may include signed URLs or response bodies.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def _safe_url(url):
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.username or parts.password or parts.query or parts.fragment
            or not (parts.hostname or "").endswith((".nasa.gov", ".usgs.gov"))):
        raise EmitDownloadError("CMR returned a non-canonical or untrusted download URL")
    return url


def _distribution_info(umm, filename):
    entries = umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
    matches = [x for x in entries if x.get("Name") == filename]
    if len(matches) > 1:
        raise EmitDownloadError("Ambiguous file metadata in CMR")
    if not matches:
        return None, None, None, None, None
    item = matches[0]
    raw_size = float(item["Size"]) if "Size" in item else None
    unit = item.get("SizeUnit")
    if raw_size is not None and (not math.isfinite(raw_size) or raw_size < 0):
        raise EmitDownloadError("Invalid CMR size metadata")
    # CMR MB sizes can be rounded and are not exact byte counts.
    size_value = item.get("SizeInBytes")
    if size_value is None and str(unit).upper() == "BYTES":
        size_value = item.get("Size")
    size = int(size_value) if size_value is not None else None
    if size is not None and (size <= 0 or float(size_value) != size):
        raise EmitDownloadError("Invalid exact CMR byte count")
    checksum = item.get("Checksum", {})
    algorithm = checksum.get("Algorithm", "").lower().replace("-", "")
    if algorithm not in ("md5", "sha1", "sha256", "sha512"):
        return size, None, None, raw_size, unit
    value = checksum.get("Value", "").lower()
    length = hashlib.new(algorithm).digest_size * 2
    if not re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value):
        raise EmitDownloadError("Malformed CMR checksum")
    return size, algorithm, value, raw_size, unit


def search_emit_products(*, collection, version, scene, products, client=None):
    """Return one exact asset per requested code; public search requires no login.

    The collection is explicit: legacy MASK is in EMITL2ARFL.001, whereas newer
    standalone MASK collections have independent versions. No automatic pairing
    between versions or collections is inferred.
    """
    products = _validate_query(collection, version, scene, products)
    client = _client(client)
    try:
        with _private_call():
            results = client.search_data(short_name=collection, version=version,
                granule_name=f"EMIT_*_{version}_{scene}", provider="LPCLOUD", count=-1)
    except Exception:
        raise EmitDownloadError("NASA CMR search failed; check network/service availability") from None
    found = {p: [] for p in products}
    for granule in results:
        umm = granule.get("umm", {})
        meta = granule.get("meta", {})
        ref = umm.get("CollectionReference", {})
        if ref.get("ShortName") != collection or ref.get("Version") != version:
            raise EmitDownloadError("CMR result does not match the requested collection/version")
        try:
            links = granule.data_links(access="external")
        except Exception:
            raise EmitDownloadError("Unable to obtain external links from CMR result") from None
        for url in dict.fromkeys(links):
            filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
            match = _FILE.fullmatch(filename)
            if not match or match.group(1) not in found:
                continue
            product, file_version, file_scene = match.groups()
            if file_version != version or file_scene != scene:
                raise EmitDownloadError("CMR asset identity does not match the requested scene/version")
            _safe_url(url)
            size, algorithm, value, raw_size, unit = _distribution_info(umm, filename)
            found[product].append(EmitAsset(collection, version, scene, product, filename, url,
                str(meta.get("concept-id", "")), str(meta.get("revision-id", "")),
                str(umm.get("GranuleUR", "")), size, algorithm, value, raw_size, unit))
    missing = [p for p, assets in found.items() if not assets]
    ambiguous = [p for p, assets in found.items() if len(assets) > 1]
    if missing or ambiguous:
        raise EmitDownloadError(f"Exact product selection failed: missing={missing}, ambiguous={ambiguous}; no fallback performed")
    return [found[p][0] for p in products]


def authenticate_earthdata(*, strategy="environment", client=None):
    """Explicit login; environment credentials take precedence over a saved token.

    A configured HyGAS token file is used only when environment credentials are
    absent. No fallback to prompting or netrc occurs in environment mode.
    """
    if strategy not in ("environment", "interactive", "netrc"):
        raise ValueError("unsupported authentication strategy")
    saved_token = None
    if strategy == "environment" and not (os.environ.get("EARTHDATA_TOKEN") or (
            os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))):
        if not DEFAULT_TOKEN_FILE.is_file():
            raise EmitDownloadError("Configure a persistent token with python scripts/configure_earthdata.py, or export EARTHDATA_TOKEN")
        try:
            saved_token = read_token(DEFAULT_TOKEN_FILE)
        except (OSError, ValueError):
            raise EmitDownloadError("Cannot read saved Earthdata token; run python scripts/configure_earthdata.py --replace") from None
    client = _client(client)
    previous_token = os.environ.get("EARTHDATA_TOKEN")
    if saved_token is not None:
        os.environ["EARTHDATA_TOKEN"] = saved_token
    try:
        if strategy == "interactive":
            # Prompts must remain visible; no logging of returned auth objects.
            auth = client.login(strategy=strategy, persist=False)
        else:
            with _private_call():
                auth = client.login(strategy=strategy, persist=False)
        if not auth.authenticated:
            raise ValueError("not authenticated")
    except Exception:
        raise EmitDownloadError("Earthdata authentication failed; check credentials, expiration and required authorizations") from None
    finally:
        if saved_token is not None:
            if previous_token is None:
                os.environ.pop("EARTHDATA_TOKEN", None)
            else:
                os.environ["EARTHDATA_TOKEN"] = previous_token
    return client


def _hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_file(path, asset):
    with path.open("rb") as f:
        header = f.read(8)
    if path.suffix == ".tif":
        if header[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            raise EmitDownloadError("Downloaded file is not a TIFF (possibly an HTML login/error page)")
        try:
            import rasterio
            with rasterio.open(path) as raster:
                if raster.driver != "GTiff" or raster.count < 1 or min(raster.shape) < 1 or raster.crs is None:
                    raise EmitDownloadError("EMIT GeoTIFF has invalid raster/georeferencing metadata")
        except ImportError:
            raise EmitDownloadError("rasterio is required to validate EMIT GeoTIFF downloads") from None
        except OSError:
            raise EmitDownloadError("GeoTIFF structure is unreadable or truncated; file not accepted") from None
    else:
        if header != b"\x89HDF\r\n\x1a\n":
            raise EmitDownloadError("Downloaded file is not an EMIT HDF5/NetCDF file (possibly an HTML login/error page)")
        try:
            import h5py
            with h5py.File(path, "r") as f:
                version = f.attrs.get("product_version")
                if isinstance(version, bytes):
                    version = version.decode("utf8")
                if version is not None and version != "V" + asset.version:
                    raise EmitDownloadError("NetCDF product_version differs from requested version")
        except ImportError:
            raise EmitDownloadError("h5py is required to validate EMIT NetCDF downloads") from None
        except OSError:
            raise EmitDownloadError("NetCDF/HDF5 structure is unreadable or truncated; file not accepted") from None
    size = path.stat().st_size
    if asset.size_bytes is not None and size != asset.size_bytes:
        raise EmitDownloadError("File size differs from exact CMR byte count")
    digest = _hash(path)
    if asset.checksum_algorithm:
        actual = digest if asset.checksum_algorithm == "sha256" else _hash(path, asset.checksum_algorithm)
        if actual != asset.checksum_value:
            raise EmitDownloadError("File checksum differs from CMR; file not accepted")
    return size, digest


def _json(path, value):
    # Exclusive writes: never replace an existing user file or earlier manifest.
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def download_emit_products(assets, output_dir, *, strategy="environment", client=None):
    """Download only selected assets into private staging, validate, then publish.

    Existing assets are reused only with a matching HyGAS receipt or a verified
    CMR checksum. Partial/failed downloads are retained in .partial for inspection;
    resume is not implemented. No files are silently overwritten or deleted.
    """
    assets = list(assets)
    if not assets:
        raise ValueError("no assets selected")
    if len({a.filename for a in assets}) != len(assets):
        raise ValueError("duplicate output filenames")
    for asset in assets:
        _safe_url(asset.url)
        match = _FILE.fullmatch(asset.filename)
        if not match or match.groups() != (asset.product, asset.version, asset.scene):
            raise ValueError("invalid asset identity/filename")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = output / ".hygas-emit"
    if receipts.is_symlink() or (output / ".partial").is_symlink():
        raise EmitDownloadError("Refusing symlinked download bookkeeping directories")
    receipts.mkdir(exist_ok=True)
    lock = receipts / "download.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise EmitDownloadError("Download lock exists; check for another active process before removing a stale lock") from None
    os.close(lock_fd)
    try:
        return _download_locked(assets, output, receipts, strategy, client)
    finally:
        lock.unlink()  # Only the empty lock created by this call, never user data.


def _download_locked(assets, output, receipts, strategy, client):
    completed = []
    pending = []
    for asset in assets:
        destination = output / asset.filename
        receipt_path = receipts / (asset.filename + ".json")
        if destination.is_symlink() or receipt_path.is_symlink():
            raise EmitDownloadError("Refusing to use symlinked destinations/receipts")
        if destination.exists():
            size, digest = _validate_file(destination, asset)
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf8"))
                if receipt.get("sha256") != digest or receipt.get("asset") != asdict(asset):
                    raise EmitDownloadError("Existing file/receipt differs from this request; not overwritten")
            elif not asset.checksum_algorithm:
                raise EmitDownloadError("Existing file has no matching receipt or CMR checksum; not overwritten or assumed complete")
            record = dict(asset=asdict(asset), bytes=size, sha256=digest, status="verified_existing")
            if not receipt_path.exists():
                _json(receipt_path, record)
            completed.append(record)
        else:
            if receipt_path.exists():
                raise EmitDownloadError("Receipt exists but data file is missing; inspect before retrying")
            pending.append(asset)
    run = uuid.uuid4().hex
    manifest = output / f"emit-download-{run}.json"
    document = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        earthaccess_version=importlib.metadata.version("earthaccess") if client is None else "injected-client",
        downloader_source_sha256=_hash(Path(__file__)),
        state="planned", selection=[asdict(a) for a in assets], completed=completed,
        authentication_strategy=strategy, credentials_recorded=False)
    # Plan remains immutable even on failure; final receipt records completion.
    _json(manifest, document)
    if pending:
        client = authenticate_earthdata(strategy=strategy, client=client)
        for asset in pending:
            staging = output / ".partial" / run / asset.product
            staging.mkdir(parents=True, exist_ok=False)
            try:
                with _private_call():
                    paths = client.download([asset.url], local_path=staging, provider="LPCLOUD",
                                            threads=1, show_progress=False)
            except Exception:
                raise EmitDownloadError("NASA download failed; check token/authorization/network. Partial files retained in .partial") from None
            staged = staging / asset.filename
            if not paths or len(paths) != 1 or Path(paths[0]).resolve() != staged.resolve() or not staged.is_file():
                raise EmitDownloadError("Downloader did not return the expected file; staging retained")
            size, digest = _validate_file(staged, asset)
            destination = output / asset.filename
            try:
                # Same-filesystem hard link is atomic and cannot overwrite a target.
                os.link(staged, destination)
            except OSError:
                raise EmitDownloadError("Cannot publish without overwriting; validated file retained in .partial") from None
            record = dict(asset=asdict(asset), bytes=size, sha256=digest, status="downloaded_verified")
            _json(receipts / (asset.filename + ".json"), record)
            completed.append(record)
            staged.unlink()  # Validated content is retained at destination.
            staging.rmdir()
    final = output / f"emit-download-{run}-complete.json"
    _json(final, {**document, "state": "complete", "completed": completed,
                  "finished_utc": datetime.now(timezone.utc).isoformat()})
    return final
