import os
import numpy as np
import math
from osgeo import gdal, ogr
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point
from shapely.ops import unary_union

##############################################################
#  SENSOR CONFIGURATION
##############################################################

SENSOR_CONFIG = {
    "PRISMA": {"a": 0.37, "b": 0.70},
    "TANAGER": {"a": 0.37, "b": 0.70},  # use PRISMA coefficients unless overridden
    "ENMAP":  {"a": 0.37, "b": 0.69},
    "EMIT":   {"a": 0.45, "b": 0.67},
}

##############################################################
#  LOADING & UNIT CONVERSION
##############################################################

def load_geotiff(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray()
    return arr, ds.GetGeoTransform(), ds.GetProjection(), ds

def discard_neg(arr):
    return np.where(arr > 0, arr, 0)

def ppm_m_to_ppb(arr):
    return arr * 0.125

##############################################################
#  IME & UNCERTAINTY
##############################################################

def calculate_ime(ppb_arr, gsd):
    k = (gsd**2) * (16.04/28.97) * 10000 * 1e-9
    ppb = discard_neg(ppb_arr)
    return np.sum(ppb) * k

def calculate_sigma_ime(sigma_ppb_arr, gsd):
    k = (gsd**2) * (16.04/28.97) * 10000 * 1e-9
    sigma_clean = np.where(np.isfinite(sigma_ppb_arr) & (sigma_ppb_arr > 0), sigma_ppb_arr, 0)
    return k * np.sqrt(np.sum(sigma_clean ** 2))


def compute_plume_stats(ppb_arr):
    """Return summary stats (min/max/mean/median) for positive ppb pixels."""
    valid = ppb_arr[np.isfinite(ppb_arr) & (ppb_arr > 0)]
    if valid.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "pixel_count": 0,
        }
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "pixel_count": int(valid.size),
    }

##############################################################
#  CLIPPING
##############################################################

def _vsimem_path_exists(path):
    """Return True if a /vsimem file exists."""
    try:
        return gdal.VSIStatL(path) is not None
    except RuntimeError:
        return False


def clip_raster_to_polygon(raster_path, polygon, out_path):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if drv is None:
        raise RuntimeError("ESRI Shapefile driver is not available.")

    tmp = "/vsimem/tmp_poly.shp"
    if _vsimem_path_exists(tmp):
        drv.DeleteDataSource(tmp)

    ds_tmp = drv.CreateDataSource(tmp)
    layer = ds_tmp.CreateLayer("poly", geom_type=ogr.wkbPolygon)
    feat = ogr.Feature(layer.GetLayerDefn())

    if isinstance(polygon, ogr.Geometry):
        ogr_geom = polygon
    else:
        ogr_geom = ogr.CreateGeometryFromWkb(polygon.wkb)

    feat.SetGeometry(ogr_geom)
    layer.CreateFeature(feat)
    ds_tmp.FlushCache()

    gdal.Warp(
        out_path,
        raster_path,
        cutlineDSName=tmp,
        cropToCutline=True,
        dstNodata=-9999,
    )
    arr, _, _, _ = load_geotiff(out_path)

    # Clean up the temporary vector datasource.
    ds_tmp = None
    drv.DeleteDataSource(tmp)

    return arr


# def _infer_local_utm_epsg(gdf):
#     """Infer a projected UTM EPSG code from a GeoDataFrame centroid."""
#     gdf_wgs84 = gdf if gdf.crs.to_epsg() == 4326 else gdf.to_crs("EPSG:4326")
#     centroid = gdf_wgs84.geometry.unary_union.centroid
#     zone = int((centroid.x + 180) / 6) + 1
#     base = 32600 if centroid.y >= 0 else 32700
#     return base + zone




def _infer_local_utm_epsg(gdf):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined.")

    gdf_wgs84 = gdf if gdf.crs.to_epsg() == 4326 else gdf.to_crs("EPSG:4326")
    geoms = [geom for geom in gdf_wgs84.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError("No valid geometries to infer UTM zone.")

    try:
        centroid = unary_union(geoms).centroid  # avoids GeoSeries.union_all
    except TypeError:
        minx, miny, maxx, maxy = gdf_wgs84.total_bounds
        centroid = Point((minx + maxx) / 2, (miny + maxy) / 2)

    zone = int((centroid.x + 180) / 6) + 1
    base = 32600 if centroid.y >= 0 else 32700
    return base + zone


def project_gdf_to_local_utm(gdf, target_epsg=None):
    """Project the plume polygons to a meter-based CRS (default: local UTM)."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS defined.")
    epsg = target_epsg or _infer_local_utm_epsg(gdf)
    return gdf.to_crs(f"EPSG:{epsg}"), epsg


def compute_area_and_length(projected_gdf):
    """Compute plume area (m^2) and characteristic length from projected data."""
    area = projected_gdf.geometry.area.sum()
    return area, math.sqrt(area)

##############################################################
#  WIND RESPONSE & FLUX PROPAGATION
##############################################################

def compute_u_eff(u10, sensor_type):
    sc = SENSOR_CONFIG[sensor_type]
    return sc["a"] * u10 + sc["b"], sc["a"], sc["b"]

def compute_flux(ime_kg, u_eff, L_m):
    return (u_eff * ime_kg * 3.6) / L_m

def propagate_flux_uncertainty(q, ime_kg, sigma_ime, u10, sigma_u10, u_eff, a):
    sigma_q_ime = 0 if sigma_ime is None else q * (sigma_ime / ime_kg)
    sigma_ueff = abs(a) * sigma_u10
    sigma_q_wind = q * (sigma_ueff / u_eff)
    sigma_q = math.sqrt(sigma_q_ime**2 + sigma_q_wind**2)
    return sigma_q, sigma_q_ime, sigma_q_wind

##############################################################
#  REPORT
##############################################################

def write_report(out_path, plume_id, area, L, u10, u_eff,
                 ime, sigma_ime, q, sigma_q, sigma_q_ime, sigma_q_wind,
                 plume_stats=None):
    with open(out_path, "w") as rep:
        rep.write(f"Plume index: {plume_id}\n")
        rep.write(f"Area: {area:.2f} m2\n")
        rep.write(f"L: {L:.2f} m\n\n")
        rep.write(f"U10: {u10:.2f} m/s\n")
        rep.write(f"U_eff: {u_eff:.2f} m/s\n\n")
        rep.write(f"IME: {ime:.6f} kg\n")
        if sigma_ime is not None:
            rep.write(f"σ_IME: ±{sigma_ime:.6f} kg\n\n")
        if plume_stats:
            rep.write("Plume ppb statistics (positive pixels only):\n")
            rep.write(f" - Count: {plume_stats['pixel_count']}\n")
            rep.write(f" - Min: {plume_stats['min']:.2f} ppb\n")
            rep.write(f" - Max: {plume_stats['max']:.2f} ppb\n")
            rep.write(f" - Mean: {plume_stats['mean']:.2f} ppb\n")
            rep.write(f" - Median: {plume_stats['median']:.2f} ppb\n\n")
        rep.write(f"Q: {q:.6f} t/h\n")
        rep.write(f"σ_Q total: ±{sigma_q:.6f} t/h\n")
        rep.write(f" - σ_Q from IME:  {sigma_q_ime:.6f} t/h\n")
        rep.write(f" - σ_Q from wind: {sigma_q_wind:.6f} t/h\n")

##############################################################
#  MAIN DRIVER — NOW ONLY A WRAPPER
##############################################################

def process_plume_image(input_file,
                        shapefile_path,
                        u10,
                        sensor_type="EMIT",
                        gsd=60,
                        uncertainty_file=None,
                        sigma_u10=None,
                        output_dir="output"):

    os.makedirs(output_dir, exist_ok=True)
    sensor_type = sensor_type.upper()

    # Load main maps
    conc_data, _, projection_wkt, conc_ds = load_geotiff(input_file)
    conc_ppb = ppm_m_to_ppb(discard_neg(conc_data))
    raster_crs = CRS.from_wkt(projection_wkt)
    conc_ds = None

    if uncertainty_file:
        unc_data, _, _, _ = load_geotiff(uncertainty_file)
        unc_ppb = ppm_m_to_ppb(discard_neg(unc_data))
    else:
        unc_ppb = None

    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError("Shapefile must have a defined CRS.")
    gdf_raster = gdf.to_crs(raster_crs)
    projected_gdf, _ = project_gdf_to_local_utm(gdf)
    if sigma_u10 is None:
        sigma_u10 = 1.5

    for (idx, row_raster), (_, row_projected) in zip(gdf_raster.iterrows(), projected_gdf.iterrows()):
        poly = row_raster.geometry

        clipped_file = os.path.join(output_dir, f"plume_{idx}.tif")
        clipped = clip_raster_to_polygon(input_file, poly, clipped_file)
        clipped_ppb = ppm_m_to_ppb(discard_neg(clipped))
        plume_stats = compute_plume_stats(clipped_ppb)

        ime = calculate_ime(clipped_ppb, gsd)

        if unc_ppb is not None:
            unc_clip_file = os.path.join(output_dir, f"plume_unc_{idx}.tif")
            unc_clip = clip_raster_to_polygon(uncertainty_file, poly, unc_clip_file)
            unc_clip_ppb = ppm_m_to_ppb(discard_neg(unc_clip))
            sigma_ime = calculate_sigma_ime(unc_clip_ppb, gsd)
        else:
            sigma_ime = None

        area = row_projected.geometry.area
        L = math.sqrt(area)

        u_eff, a, b = compute_u_eff(u10, sensor_type)
        q = compute_flux(ime, u_eff, L)
        sigma_q, sigma_q_ime, sigma_q_wind = propagate_flux_uncertainty(
            q, ime, sigma_ime, u10, sigma_u10, u_eff, a
        )

        report_path = os.path.join(output_dir, f"plume_{idx}_report.txt")
        write_report(report_path, idx, area, L, u10, u_eff,
                     ime, sigma_ime, q, sigma_q, sigma_q_ime, sigma_q_wind,
                     plume_stats=plume_stats)

    print(f"Processing completed. Results stored in: {output_dir}")
