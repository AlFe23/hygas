"""Read EMIT L1B NetCDF radiance in its native, unrotated detector grid.

``read_emit_scene`` exposes windows, native indices, FWHM, declared invalid
values, location and OBS geometry. Optional GLTs retain their original 1-based
indices into the *full* native scene, even when radiance is windowed. No noise
model, homogeneity mask, orthorectification or cloud/saturation QA is inferred.
``load_emit_scene`` preserves the dictionary interface of the SNR notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


RADIANCE_UNIT = "uW cm-2 sr-1 nm-1"
_PRODUCT = re.compile(
    r"^EMIT_L1B_(RAD|OBS)_(\d{3})_(\d{8}T\d{6})_(\d+)_(\d+)\.nc$", re.IGNORECASE
)
DEFAULT_GEOMETRY_LABELS = {
    "Path length (sensor-to-ground in meters)": "Path length",
    "To-sensor azimuth (0 to 360 degrees CW from N)": "Sensor azimuth",
    "To-sensor zenith (0 to 90 degrees from zenith)": "Sensor zenith",
    "To-sun azimuth (0 to 360 degrees CW from N)": "Sun azimuth",
    "To-sun zenith (0 to 90 degrees from zenith)": "Sun zenith",
    "Solar phase (degrees between to-sensor and to-sun vectors in principal plane)": "Solar phase",
    "Slope (local surface slope as derived from DEM in degrees)": "Slope",
    "Aspect (local surface aspect 0 to 360 degrees clockwise from N)": "Aspect",
    "Cosine(i) (apparent local illumination factor based on DEM slope and aspect and to sun vector)": "Cos(i)",
    "UTC Time (decimal hours for mid-line pixels)": "UTC time",
    "Earth-sun distance (AU)": "Earth-sun distance",
}


@dataclass
class EmitScene:
    """Selected native radiance and its explicit spatial/spectral identities.

    cube and valid_radiance: (bands, rows, columns). Validity means finite and
    within *declared* fill/range rules, not homogeneous, cloud-free or unsaturated.
    Location and observation layers: (rows, columns). Missing optional sources
    remain empty/None, never synthetic arrays of valid values.
    """

    cube: np.ndarray
    wavelengths: np.ndarray
    fwhm: np.ndarray
    band_indices: np.ndarray
    row_indices: np.ndarray
    column_indices: np.ndarray
    valid_radiance: np.ndarray
    location: Dict[str, np.ndarray]
    observation: Dict[str, np.ndarray]
    glt_x: Optional[np.ndarray]
    glt_y: Optional[np.ndarray]
    glt_valid: Optional[np.ndarray]
    glt_in_window: Optional[np.ndarray]
    scene_id: str
    metadata: Dict[str, object]

    @property
    def valid_pixels(self) -> np.ndarray:
        """All *selected* radiance bands valid; this is not official scene QA."""
        return np.all(self.valid_radiance, axis=0)

    @property
    def geometry_lines(self) -> List[str]:
        lines = [f"Scene: {self.scene_id}"]
        if self.observation:
            lines.append("Geometry source: EMIT L1B OBS NetCDF (selected native window)")
        for name, layer in self.observation.items():
            summary = _summarize_band(layer, DEFAULT_GEOMETRY_LABELS.get(name, name))
            if summary:
                lines.append(summary)
        return lines


def _attribute(value):
    """Convert ordinary NetCDF attributes to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return [_attribute(v) for v in value] if value.ndim else _attribute(value.item())
    if isinstance(value, np.generic):
        return _attribute(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return value


def _decode_strings(values: Iterable) -> List[str]:
    return [str(_attribute(v)) for v in values]


def _summarize_band(data: np.ndarray, label: str) -> Optional[str]:
    sample = data[np.isfinite(data)]
    if not sample.size:
        return None
    return (f"{label}: mean={np.mean(sample):.3f}, median={np.median(sample):.3f}, "
            f"min={np.min(sample):.3f}, max={np.max(sample):.3f}")


def _identify_inputs(paths: Sequence[str]) -> Tuple[str, Optional[str]]:
    """Require one canonical RAD and at most one matching OBS, in any order."""
    products = {"RAD": [], "OBS": []}
    identities = []
    for path in paths:
        path = str(path)
        match = _PRODUCT.fullmatch(Path(path).name)
        if match is None:
            raise ValueError(f"Not a canonical EMIT L1B RAD/OBS NetCDF filename: {path}")
        products[match[1].upper()].append(path)
        identities.append(tuple(part.upper() for part in match.groups()[1:]))
    if len(products["RAD"]) != 1 or len(products["OBS"]) > 1:
        raise ValueError("EMIT requires exactly one RAD and at most one OBS; duplicate inputs are ambiguous.")
    if len(set(identities)) != 1:
        raise ValueError("EMIT RAD and OBS must have the same version, acquisition, orbit and scene identifiers.")
    return products["RAD"][0], products["OBS"][0] if products["OBS"] else None


def _dataset(root: h5py.File, name: str) -> h5py.Dataset:
    if name not in root or not isinstance(root[name], h5py.Dataset):
        raise ValueError(f"Missing required EMIT dataset {name!r} in {root.filename}")
    return root[name]


def _window(value: Optional[slice], size: int, label: str) -> slice:
    if value is None:
        return slice(0, size)
    if not isinstance(value, slice) or value.step not in (None, 1):
        raise ValueError(f"{label} must be a contiguous slice with step 1.")
    try:
        start = 0 if value.start is None else operator.index(value.start)
        stop = size if value.stop is None else operator.index(value.stop)
    except TypeError as exc:
        raise ValueError(f"{label} bounds must be integers.") from exc
    if not 0 <= start < stop <= size:
        raise ValueError(f"{label} slice {value} is empty or outside [0, {size}).")
    return slice(start, stop)


def _bands(selection, size: int) -> np.ndarray:
    if selection is None:
        return np.arange(size)
    if isinstance(selection, slice):
        if selection.step is not None and selection.step <= 0:
            raise ValueError("Band slice step must be positive.")
        start = 0 if selection.start is None else selection.start
        stop = size if selection.stop is None else selection.stop
        if not 0 <= start < stop <= size:
            raise ValueError("Band slice is empty or outside the native spectral axis.")
        ids = np.arange(size)[selection]
    else:
        ids = np.asarray(selection)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("Band indices must be a one-dimensional integer sequence.")
    if not ids.size or np.any(ids < 0) or np.any(ids >= size) or np.unique(ids).size != ids.size:
        raise ValueError("Band indices must be nonempty, unique and within the native spectral axis.")
    return ids.astype(np.int64)


def _numeric(dataset: h5py.Dataset, selection) -> np.ndarray:
    """Apply declared missing/range rules in packed space, then CF linear scale."""
    data = np.asarray(dataset[selection], dtype=np.float64)
    invalid = ~np.isfinite(data)
    for key in ("_FillValue", "missing_value"):
        for value in np.asarray(dataset.attrs.get(key, [])).reshape(-1):
            invalid |= data == float(value)
    if "valid_range" in dataset.attrs:
        limits = np.asarray(dataset.attrs["valid_range"]).reshape(-1)
        if limits.size != 2 or limits[0] > limits[1]:
            raise ValueError(f"Invalid valid_range attribute for {dataset.name}")
        invalid |= (data < limits[0]) | (data > limits[1])
    for key, comparison in (("valid_min", np.less), ("valid_max", np.greater)):
        if key in dataset.attrs:
            invalid |= comparison(data, float(np.asarray(dataset.attrs[key]).item()))
    scale = float(np.asarray(dataset.attrs.get("scale_factor", 1.0)).item())
    offset = float(np.asarray(dataset.attrs.get("add_offset", 0.0)).item())
    if not np.isfinite(scale) or not np.isfinite(offset) or scale == 0:
        raise ValueError(f"Invalid CF radiometric scale/offset for {dataset.name}")
    data = data * scale + offset
    data[invalid] = np.nan
    return data


def _cube(dataset: h5py.Dataset, rows: slice, columns: slice, bands: np.ndarray) -> np.ndarray:
    order = np.argsort(bands)
    sorted_bands = bands[order]
    spectral = (slice(int(sorted_bands[0]), int(sorted_bands[-1]) + 1)
                if np.all(np.diff(sorted_bands) == 1) else sorted_bands.tolist())
    selected = _numeric(dataset, (rows, columns, spectral))
    selected = selected[..., np.argsort(order)]
    return np.transpose(selected, (2, 0, 1)).astype(np.float32)


def _check_units(dataset: h5py.Dataset) -> str:
    source = str(_attribute(dataset.attrs.get("units", "")))
    normalized = source.lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    if normalized not in {"uw/cm^2/sr/nm", "uwcm-2sr-1nm-1"}:
        raise ValueError(f"Unsupported or missing EMIT radiance units: {source!r}")
    return source


def _check_pair(rad: h5py.File, obs: h5py.File, spatial_shape: Tuple[int, int]) -> h5py.Dataset:
    dataset = _dataset(obs, "obs")
    if dataset.ndim != 3 or dataset.shape[:2] != spatial_shape:
        raise ValueError("EMIT RAD/OBS native spatial dimensions do not match.")
    for key in ("flight_line", "product_version", "software_build_version",
                "time_coverage_start", "time_coverage_end"):
        if key in rad.attrs and key in obs.attrs and _attribute(rad.attrs[key]) != _attribute(obs.attrs[key]):
            raise ValueError(f"EMIT RAD/OBS metadata mismatch: {key}")
    return dataset


def read_emit_scene(
    paths: Sequence[str], *, rows: Optional[slice] = None, columns: Optional[slice] = None,
    band_indices: Optional[Sequence[int] | slice] = None, load_location: bool = True,
    load_observation: bool = True, load_glt: bool = False,
) -> EmitScene:
    """Read only the requested native radiance window; indices are zero-based.

    Stop bounds are exclusive. Spatial slices must be contiguous and in bounds.
    Band indices may be in any unique order, which is preserved. The GLT, when
    requested, is the full orthorectified-grid lookup with original 1-based native
    indices: glt_in_window identifies the portion addressable by this radiance crop.
    No automatic OBS discovery, official QA, unit guess or missing-column filling.
    """
    rad_path, obs_path = _identify_inputs(paths)
    observation: Dict[str, np.ndarray] = {}
    location: Dict[str, np.ndarray] = {}
    gx = gy = gv = gw = None
    with h5py.File(rad_path, "r") as rad:
        dataset = _dataset(rad, "radiance")
        if dataset.ndim != 3:
            raise ValueError("EMIT radiance must have shape (downtrack, crosstrack, bands).")
        nr, nc, nb = dataset.shape
        row_slice = _window(rows, nr, "rows")
        col_slice = _window(columns, nc, "columns")
        bands = _bands(band_indices, nb)
        unit_source = _check_units(dataset)
        calibration = [_dataset(rad, "sensor_band_parameters/" + name) for name in ("wavelengths", "fwhm")]
        for vector in calibration:
            if str(_attribute(vector.attrs.get("units", ""))).lower().strip() != "nm":
                raise ValueError(f"EMIT spectral calibration must explicitly use nm: {vector.name}")
        wavelengths, fwhm = [_numeric(vector, slice(None)) for vector in calibration]
        if wavelengths.shape != (nb,) or fwhm.shape != (nb,):
            raise ValueError("EMIT wavelength/FWHM dimensions must match the native radiance band count.")
        if not (np.all(np.isfinite(wavelengths[bands]) & (wavelengths[bands] > 0))
                and np.all(np.isfinite(fwhm[bands]) & (fwhm[bands] > 0))):
            raise ValueError("Selected EMIT bands have invalid wavelength/FWHM calibration.")
        obs_attributes = None
        if obs_path:
            with h5py.File(obs_path, "r") as obs:
                obs_dataset = _check_pair(rad, obs, (nr, nc))
                names_raw = _dataset(obs, "sensor_band_parameters/observation_bands")[:]
                if names_raw.ndim != 1 or len(names_raw) != obs_dataset.shape[2]:
                    raise ValueError("EMIT observation band labels do not match OBS dimensions.")
                names = _decode_strings(names_raw)
                if len(names) != len(set(names)):
                    raise ValueError("EMIT observation band labels must be unique.")
                if load_observation:
                    layers = _numeric(obs_dataset, (row_slice, col_slice, slice(None)))
                    observation = {name: layers[..., i] for i, name in enumerate(names)}
                obs_attributes = {k: _attribute(v) for k, v in obs.attrs.items()}
        cube = _cube(dataset, row_slice, col_slice, bands)
        if load_location:
            for name in ("lat", "lon", "elev"):
                layer = _dataset(rad, "location/" + name)
                if layer.shape != (nr, nc):
                    raise ValueError(f"EMIT location/{name} dimensions do not match radiance.")
                location[name] = _numeric(layer, (row_slice, col_slice))
        if load_glt:
            x, y = _dataset(rad, "location/glt_x"), _dataset(rad, "location/glt_y")
            if x.ndim != 2 or x.shape != y.shape or x.dtype.kind not in "iu" or y.dtype.kind not in "iu":
                raise ValueError("EMIT GLT must contain matching two-dimensional integer x/y lookups.")
            gx, gy = x[:], y[:]
            gv = (gx >= 1) & (gx <= nc) & (gy >= 1) & (gy <= nr)
            for layer, values in ((x, gx), (y, gy)):
                for key in ("_FillValue", "missing_value"):
                    for fill in np.asarray(layer.attrs.get(key, [])).reshape(-1):
                        gv &= values != fill
            gw = (gv & (gx > col_slice.start) & (gx <= col_slice.stop)
                  & (gy > row_slice.start) & (gy <= row_slice.stop))
        attributes = {k: _attribute(v) for k, v in rad.attrs.items()}
        if "product_version" in attributes:
            expected = "V" + _PRODUCT.fullmatch(Path(rad_path).name)[2]
            if attributes["product_version"] != expected:
                raise ValueError("EMIT RAD product_version disagrees with its filename.")
        metadata = dict(
            source_paths={"radiance": rad_path, "observation": obs_path},
            product_attributes=attributes, observation_attributes=obs_attributes,
            native_shape_brc=[nb, nr, nc],
            native_window=dict(rows=[row_slice.start, row_slice.stop], columns=[col_slice.start, col_slice.stop]),
            radiance_unit=RADIANCE_UNIT, radiance_unit_source=unit_source, unit_multiplier=1.0,
            radiance_attributes={k: _attribute(v) for k, v in dataset.attrs.items()
                                 if k in ("units", "_FillValue", "missing_value", "valid_range", "valid_min", "valid_max", "scale_factor", "add_offset")},
            spatial_axes="native downtrack rows, native crosstrack columns; no rotation/resampling",
            index_base=0, glt_index_base=1, glt_addresses="full native scene, not crop-relative",
            quality=dict(official_masks_loaded=[], validity_is_official_qa=False,
                         validity_rule="finite selected radiance, declared fill/missing/range rules only"),
            location_loaded=load_location, observation_loaded=bool(observation), glt_loaded=load_glt,
        )
    return EmitScene(
        cube=cube, wavelengths=wavelengths[bands], fwhm=fwhm[bands], band_indices=bands,
        row_indices=np.arange(row_slice.start, row_slice.stop),
        column_indices=np.arange(col_slice.start, col_slice.stop), valid_radiance=np.isfinite(cube),
        location=location, observation=observation, glt_x=gx, glt_y=gy, glt_valid=gv, glt_in_window=gw,
        scene_id=Path(rad_path).stem, metadata=metadata,
    )


def load_emit_scene(paths: Sequence[str]) -> Dict[str, object]:
    """Backward-compatible payload for scripts.snr_experiment.SceneData.

    Retrieval/reference code should use read_emit_scene to retain native indices,
    validity and calibration metadata rather than this reduced notebook adapter.
    """
    scene = read_emit_scene(paths, load_location=False, load_glt=False)
    metadata = {"radiance": scene.metadata["source_paths"]["radiance"]}
    obs_path = scene.metadata["source_paths"]["observation"]
    if obs_path:
        metadata["observation"] = obs_path
    return dict(cube=scene.cube, wavelengths=scene.wavelengths, scene_id=scene.scene_id,
                metadata=metadata, geometry_lines=scene.geometry_lines,
                reference_wavelengths={"vnir_nm": None, "swir_nm": None})


def read_emit_water_vapour(mask_path: str, scene: EmitScene) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read only H2O from an exactly paired L2A MASK, not its exclusion flags.

    Public V001 MASK carries the atmospheric retrieval as ``H2O (g cm-2)``.
    Require the same version/scene, native shape and selected lat/lon grid.
    Fill/range validity is applied, but cloud/Aggregate/AOD masks are not.
    """
    rad_match = _PRODUCT.fullmatch(Path(scene.metadata["source_paths"]["radiance"]).name)
    expected = "EMIT_L2A_MASK_" + "_".join(rad_match.groups()[1:]) + ".nc"
    if Path(mask_path).name.upper() != expected.upper():
        raise ValueError("Atmospheric MASK must match the RAD version, acquisition, orbit and scene.")
    window = scene.metadata["native_window"]
    rows, columns = slice(*window["rows"]), slice(*window["columns"])
    native_shape = tuple(scene.metadata["native_shape_brc"][1:])
    with h5py.File(mask_path, "r") as root:
        dataset = _dataset(root, "mask")
        if dataset.ndim != 3 or dataset.shape[:2] != native_shape:
            raise ValueError("Atmospheric MASK and RAD native dimensions differ.")
        attrs = {k: _attribute(v) for k, v in root.attrs.items()}
        rad_attrs = scene.metadata["product_attributes"]
        for key in ("product_version", "flight_line", "time_coverage_start", "time_coverage_end"):
            if key in attrs and key in rad_attrs and attrs[key] != rad_attrs[key]:
                raise ValueError(f"Atmospheric MASK/RAD metadata mismatch: {key}")
        names = _decode_strings(_dataset(root, "sensor_band_parameters/mask_bands")[:])
        if len(names) != dataset.shape[2] or names.count("H2O (g cm-2)") != 1:
            raise ValueError("Expected one explicitly labelled H2O (g cm-2) band in MASK.")
        for name in ("lat", "lon"):
            layer = _dataset(root, "location/" + name)
            if layer.shape != native_shape or name not in scene.location:
                raise ValueError("Atmospheric pairing requires native RAD/MASK geolocation.")
            values = _numeric(layer, (rows, columns))
            if not np.array_equal(values, scene.location[name], equal_nan=True):
                raise ValueError(f"Atmospheric MASK/RAD {name} grids differ; no resampling inferred.")
        index = names.index("H2O (g cm-2)")
        water = _numeric(dataset, (rows, columns, index))
    return water, dict(path=str(mask_path), dataset="mask", band_index=index,
                      label=names[index], units="g cm-2", product_attributes=attrs,
                      grid_check="exact selected native lat/lon", QA_exclusions_applied=False)


def apply_glt(image: np.ndarray, scene: EmitScene, *, nodata=np.nan) -> np.ndarray:
    """Nearest lookup on the original full GLT; never treat crop IDs as native IDs."""
    if scene.glt_x is None or scene.glt_y is None or scene.glt_in_window is None:
        raise ValueError("GLT must be loaded explicitly before orthorectification.")
    if image.shape[:2] != scene.cube.shape[1:]:
        raise ValueError("Native map dimensions must match the selected scene window.")
    out = np.full(scene.glt_x.shape + image.shape[2:], nodata,
                  dtype=np.result_type(image.dtype, np.float32))
    valid = scene.glt_in_window
    x = scene.glt_x[valid].astype(np.int64) - 1 - scene.column_indices[0]
    y = scene.glt_y[valid].astype(np.int64) - 1 - scene.row_indices[0]
    out[valid] = image[y, x]
    return out


def save_ortho_map(path: str | Path, image: np.ndarray, scene: EmitScene, *, units: str) -> None:
    """Export a GLT-mapped raster with the RAD's original CRS/geotransform."""
    from osgeo import gdal, osr

    attrs = scene.metadata["product_attributes"]
    transform, crs = attrs.get("geotransform"), attrs.get("spatial_ref")
    if transform is None or len(transform) != 6 or not np.all(np.isfinite(transform)) or not crs:
        raise ValueError("Orthorectification requires explicit RAD geotransform and spatial_ref.")
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(str(crs)) != 0:
        raise ValueError("Unrecognized EMIT spatial_ref.")
    ortho = apply_glt(image, scene)
    if ortho.ndim != 2:
        raise ValueError("save_ortho_map expects a single native map.")
    raster = gdal.GetDriverByName("GTiff").Create(
        str(path), ortho.shape[1], ortho.shape[0], 1, gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE"],
    )
    if raster is None:
        raise OSError(f"Cannot create EMIT GeoTIFF: {path}")
    raster.SetGeoTransform(tuple(transform))
    raster.SetProjection(srs.ExportToWkt())
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(float("nan"))
    band.SetUnitType(units)
    band.WriteArray(ortho)
    raster.SetMetadataItem("GRID", "EMIT full GLT; nearest native lookup, 1-based source indices")
    raster.FlushCache()
    raster = None
