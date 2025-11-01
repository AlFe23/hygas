#!/usr/bin/env python3
"""
GHGSat Catalog CSV to GeoJSON converter.

This utility turns the tabular export returned by GHGSat's catalog API into two
GeoJSON files: one containing the original centroid points and a second one
with geodesic buffers around each centroid (default radius: 5 km). Latitude and
longitude columns are detected heuristically, so the script can cope with minor
column naming differences across catalog snapshots.

Run from the repository root so relative paths resolve as expected:

    PYTHONPATH=. python scripts/ghgsat_catalog_to_geojson.py \
        --csv 100261_CatalogRequest_Oct2025.csv \
        --points ghgsat_centroids.geojson \
        --buffers ghgsat_centroids_buffer_5km.geojson \
        --radius-m 5000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


DEFAULT_RADIUS_M = 5000.0

# # Example usage (run from repo root):
# PYTHONPATH=. python scripts/ghgsat_catalog_to_geojson.py \
# PYTHONPATH=. python scripts/ghgsat_catalog_to_geojson.py \
#   --csv /mnt/d/Lavoro/Assegno_Ricerca_Sapienza/CLEAR_UP/CH4_detection/GHGsat/observations_catalogue/observations_catalogue/4th_request/100261_CatalogRequest_Oct2025.csv \
#   --points ghgsat_centroids.geojson \
#   --buffers ghgsat_centroids_buffer_5km.geojson \
#   --radius-m 5000



def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find latitude/longitude columns using a mix of name heuristics and numeric ranges."""
    lowered = {c.lower().strip(): c for c in df.columns}
    lat_keys = (
        "lat",
        "latitude",
        "centroid_lat",
        "y",
        "lat_dd",
        "center_lat",
        "centroid latitude",
        "centroid-y",
        "centroid_y",
    )
    lon_keys = (
        "lon",
        "lng",
        "longitude",
        "centroid_lon",
        "x",
        "lon_dd",
        "center_lon",
        "centroid longitude",
        "centroid-x",
        "centroid_x",
    )

    def pick(keys: Sequence[str], fallbacks: Iterable[str]) -> Optional[str]:
        for key in keys:
            if key in lowered:
                return lowered[key]
        for col in df.columns:
            lowered_col = col.lower()
            if any(token in lowered_col for token in fallbacks):
                return col
        return None

    lat_col = pick(lat_keys, ("lat",))
    lon_col = pick(lon_keys, ("lon", "lng"))

    if not lat_col or not lon_col:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        lat_guess = next(
            (
                col
                for col in numeric_cols
                if df[col].dropna().between(-90, 90).mean() > 0.95
            ),
            None,
        )
        lon_guess = next(
            (
                col
                for col in numeric_cols
                if df[col].dropna().between(-180, 180).mean() > 0.95 and col != lat_guess
            ),
            None,
        )
        if lat_guess and lon_guess:
            lat_col, lon_col = lat_guess, lon_guess

    if not lat_col or not lon_col:
        raise RuntimeError(
            f"Could not detect latitude/longitude columns. Columns: {list(df.columns)}"
        )

    return lat_col, lon_col


def feature_properties(row: pd.Series, lat_col: str, lon_col: str) -> Dict[str, object]:
    """Strip latitude/longitude columns from the row and drop NaN values."""
    props: Dict[str, object] = {}
    for key, value in row.items():
        if key in (lat_col, lon_col):
            continue
        if isinstance(value, float) and math.isnan(value):
            value = None
        props[key] = value
    return props


def make_point_feature(lon: float, lat: float, props: Dict[str, object]) -> Dict[str, object]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def geodesic_ring_pyproj(lon: float, lat: float, radius_m: float, n: int = 72) -> List[List[float]]:
    import pyproj

    geod = pyproj.Geod(ellps="WGS84")
    step = max(1, int(360 / n))
    lons, lats = [], []
    for az in range(0, 360, step):
        lon2, lat2, _ = geod.fwd(lon, lat, az, radius_m)
        lons.append(lon2)
        lats.append(lat2)
    lons.append(lons[0])
    lats.append(lats[0])
    return [[lo, la] for lo, la in zip(lons, lats)]


def geodesic_ring_approx(lon: float, lat: float, radius_m: float, n: int = 72) -> List[List[float]]:
    ring: List[List[float]] = []
    lat_radius = 111320.0
    lon_radius = 111320.0 * math.cos(math.radians(lat))
    for idx in range(n):
        ang = 2 * math.pi * idx / n
        dx = radius_m * math.cos(ang)
        dy = radius_m * math.sin(ang)
        ring.append([lon + dx / lon_radius, lat + dy / lat_radius])
    ring.append(ring[0])
    return ring


def make_buffer_feature(
    point_feature: Dict[str, object], radius_m: float, n_vertices: int = 72
) -> Dict[str, object]:
    lon, lat = point_feature["geometry"]["coordinates"]
    try:
        ring = geodesic_ring_pyproj(lon, lat, radius_m, n_vertices)
    except Exception:
        ring = geodesic_ring_approx(lon, lat, radius_m, n_vertices)
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": point_feature["properties"],
    }


def to_feature_collection(name: str, features: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "type": "FeatureCollection",
        "name": name,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features,
    }


def convert_catalog(
    csv_path: Path,
    points_path: Path,
    buffers_path: Path,
    radius_m: float = DEFAULT_RADIUS_M,
    n_vertices: int = 72,
) -> tuple[str, str]:
    df = pd.read_csv(csv_path)
    lat_col, lon_col = detect_lat_lon_columns(df)

    point_features: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        try:
            lon = float(row[lon_col])
            lat = float(row[lat_col])
        except Exception:
            continue
        if math.isnan(lon) or math.isnan(lat):
            continue
        props = feature_properties(row, lat_col, lon_col)
        point_features.append(make_point_feature(lon, lat, props))

    points_fc = to_feature_collection("ghgsat_centroids", point_features)
    buffer_features = [
        make_buffer_feature(feat, radius_m=radius_m, n_vertices=n_vertices) for feat in point_features
    ]
    buffers_fc = to_feature_collection(
        f"ghgsat_centroids_buffer_{int(radius_m)}m", buffer_features
    )

    points_path.parent.mkdir(parents=True, exist_ok=True)
    buffers_path.parent.mkdir(parents=True, exist_ok=True)

    points_path.write_text(json.dumps(points_fc), encoding="utf-8")
    buffers_path.write_text(json.dumps(buffers_fc), encoding="utf-8")

    return lat_col, lon_col


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="100261_CatalogRequest_Oct2025.csv", help="Input CSV path.")
    parser.add_argument("--points", default="ghgsat_centroids.geojson", help="Output GeoJSON for centroids.")
    parser.add_argument(
        "--buffers",
        default="ghgsat_centroids_buffer_5km.geojson",
        help="Output GeoJSON for buffer polygons.",
    )
    parser.add_argument(
        "--radius-m",
        type=float,
        default=DEFAULT_RADIUS_M,
        help="Buffer radius in meters (default: %(default)s).",
    )
    parser.add_argument(
        "--vertices",
        type=int,
        default=72,
        help="Number of samples for polygon buffers (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    points_path = Path(args.points)
    buffers_path = Path(args.buffers)

    lat_col, lon_col = convert_catalog(
        csv_path=csv_path,
        points_path=points_path,
        buffers_path=buffers_path,
        radius_m=args.radius_m,
        n_vertices=args.vertices,
    )

    print(
        "OK\n"
        f"Detected lat column: {lat_col}\n"
        f"Detected lon column: {lon_col}\n"
        f"Wrote points: {points_path}\n"
        f"Wrote buffers: {buffers_path}"
    )


if __name__ == "__main__":
    main()
