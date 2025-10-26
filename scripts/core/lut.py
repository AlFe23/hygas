"""
Shared utilities for LUT access, atmospheric parameter handling, and ancillary
data preparation. This module centralizes the logic that both PRISMA and EnMAP
pipelines rely on when building target spectra from MODTRAN-based look-up tables.
"""

import numpy as np
import xarray as xr
import h5py
import scipy.ndimage


def mean_elev_from_dem(dem_file: str, bbox: tuple) -> float:
    """
    Return mean ground altitude in km over bbox. Falls back to 0.0 km over water/NoData.
    dem_file: NetCDF with variables lat, lon, elev [meters].
    bbox: (min_lon, max_lon, min_lat, max_lat)
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    ds = xr.open_dataset(dem_file)
    try:
        # Slice safely even if lat is descending
        lat_slice = slice(min_lat, max_lat) if ds["lat"][0] <= ds["lat"][-1] else slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon) if ds["lon"][0] <= ds["lon"][-1] else slice(max_lon, min_lon)

        elevation_subset = ds.sel(lon=lon_slice, lat=lat_slice)
        if "elev" not in elevation_subset:
            raise KeyError("Variable 'elev' not found in DEM.")

        arr = elevation_subset["elev"]  # meters
        # Robust mean ignoring NaNs
        m_val = arr.where(np.isfinite(arr)).mean(skipna=True).values
        if m_val is None or not np.isfinite(m_val):
            print("Mean Elevation within Bounding Box in Km: NaN → using sea level (0 km).")
            return 0.0
        mean_km = float(m_val) / 1000.0
        print("Mean Elevation within Bounding Box in Km:", mean_km)
        return mean_km
    finally:
        ds.close()


def normalize_ground_km(ground_km: float | np.ndarray, fallback: float = 0.0) -> float:
    """Return a finite ground altitude in km. Over water/NoData → fallback (default 0)."""
    g = float(np.nan_to_num(ground_km, nan=fallback, posinf=fallback, neginf=fallback))
    # keep bounds of your LUT grid, e.g., 0–3 km
    return float(np.clip(g, 0.0, 3.0))


def normalize_wv_gcm2(wv: float | np.ndarray, fallback: float = 0.0) -> float:
    """Finite water vapor in g/cm^2, clipped to LUT domain [0,6]."""
    w = float(np.nan_to_num(wv, nan=fallback, posinf=fallback, neginf=fallback))
    return float(np.clip(w, 0.0, 6.0))


###################################################################################
# FUNZIONI PER LA LETTURA DELLA LUT


def check_param(value, min_val, max_val, name):
    """
    Ensures that a parameter is within the specified range. If it exceeds the range, it is clamped.
    """
    if value < min_val:
        print(f"Warning: {name} value ({value}) is below the minimum ({min_val}). Setting to {min_val}.")
        return min_val
    elif value > max_val:
        print(f"Warning: {name} value ({value}) exceeds the maximum ({max_val}). Setting to {max_val}.")
        return max_val
    return value


@np.vectorize
def get_5deg_zenith_angle_index(zenith_value):
    check_param(zenith_value, 0, 80, "Zenith Angle")
    return zenith_value / 5


@np.vectorize
def get_5deg_sensor_height_index(sensor_value):  # [1, 2, 4, 10, 20, 120]
    # Only check lower bound here, atmosphere ends at 120 km so clamping there is okay.
    check_param(sensor_value, 1, np.inf, "Sensor Height")
    # There's not really a pattern here, so just linearly interpolate between values -- piecewise linear
    if sensor_value < 1.0:
        return np.float64(0.0)
    elif sensor_value < 2.0:
        idx = sensor_value - 1.0
        return idx
    elif sensor_value < 4:
        return sensor_value / 2
    elif sensor_value < 10:
        return (sensor_value / 6) + (4.0 / 3.0)
    elif sensor_value < 20:
        return (sensor_value / 10) + 2
    elif sensor_value < 120:
        return (sensor_value / 100) + 3.8
    else:
        return 5


@np.vectorize
def get_5deg_ground_altitude_index(ground_value):  # [0, 0.5, 1.0, 2.0, 3.0]
    check_param(ground_value, 0, 3, "Ground Altitude")
    if ground_value < 1:
        return 2 * ground_value
    else:
        return 1 + ground_value


@np.vectorize
def get_5deg_water_vapor_index(water_value):  # [0,1,2,3,4,5,6]
    check_param(water_value, 0, 6, "Water Vapor")
    return water_value


@np.vectorize
def get_5deg_methane_index(methane_value):
    # the parameter clamps should rarely be called because there are default concentrations, but the --concentraitons parameter exposes these
    check_param(methane_value, 0, 64000, "Methane Concentration")
    if methane_value <= 0:
        return 0
    elif methane_value < 1000:
        return methane_value / 1000
    return np.log2(methane_value / 500)


@np.vectorize
def get_carbon_dioxide_index(coo_value):
    check_param(coo_value, 0, 1280000, "Carbon Dioxode Concentration")
    if coo_value <= 0:
        return 0
    elif coo_value < 20000:
        return coo_value / 20000
    return np.log2(coo_value / 10000)


def get_5deg_lookup_index(zenith=0, sensor=120, ground=0, water=0, conc=0, gas="ch4"):
    if "ch4" in gas:
        idx = np.asarray(
            [
                [get_5deg_zenith_angle_index(zenith)],
                [get_5deg_sensor_height_index(sensor)],
                [get_5deg_ground_altitude_index(ground)],
                [get_5deg_water_vapor_index(water)],
                [get_5deg_methane_index(conc)],
            ]
        )
    elif "co2" in gas:
        idx = np.asarray(
            [
                [get_5deg_zenith_angle_index(zenith)],
                [get_5deg_sensor_height_index(sensor)],
                [get_5deg_ground_altitude_index(ground)],
                [get_5deg_water_vapor_index(water)],
                [get_carbon_dioxide_index(conc)],
            ]
        )
    else:
        raise ValueError("Unknown gas provided.")
    return idx


def spline_5deg_lookup(grid_data, zenith=0, sensor=120, ground=0, water=0, conc=0, gas="ch4", order=1):
    coords = get_5deg_lookup_index(zenith=zenith, sensor=sensor, ground=ground, water=water, conc=conc, gas=gas)
    if order == 1:
        coords_fractional_part, coords_whole_part = np.modf(coords)
        coords_near_slice = tuple(
            (
                slice(int(c[0]), int(c[0] + 2))
                if isinstance(c, np.ndarray)
                else slice(int(c), int(c + 2))
                for c in coords_whole_part
            )
        )
        near_grid_data = grid_data[coords_near_slice]
        new_coord = np.concatenate(
            (coords_fractional_part * np.ones((1, near_grid_data.shape[-1])), np.arange(near_grid_data.shape[-1])[None, :]),
            axis=0,
        )
        lookup = scipy.ndimage.map_coordinates(near_grid_data, coordinates=new_coord, order=1, mode="nearest")
    elif order == 3:
        lookup = np.asarray(
            [
                scipy.ndimage.map_coordinates(im, coordinates=coords_fractional_part, order=order, mode="nearest")
                for im in np.moveaxis(near_grid_data, 5, 0)
            ]
        )
    else:
        raise ValueError("Unsupported interpolation order: {order}")
    return lookup.squeeze()


def load_ch4_dataset(lut_file_path):
    # Ensure the function uses the passed file path instead of a hardcoded one
    datafile = h5py.File(lut_file_path, "r", rdcc_nbytes=4194304)
    return datafile["modtran_data"], datafile["modtran_param"], datafile["wave"], "ch4"


def generate_library(gas_concentration_vals, lut_file, zenith=0, sensor=120, ground=0, water=0, order=1, dataset_fcn=load_ch4_dataset):
    # Use the passed `dataset_fcn` function, allowing for flexibility in data loading.
    grid, params, wave, gas = dataset_fcn(lut_file)
    rads = np.empty((len(gas_concentration_vals), grid.shape[-1]))
    for i, ppmm in enumerate(gas_concentration_vals):
        rads[i, :] = spline_5deg_lookup(grid, zenith=zenith, sensor=sensor, ground=ground, water=water, conc=ppmm, gas=gas, order=order)
    return rads, np.array(wave)

