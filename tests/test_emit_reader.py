"""Small synthetic NetCDF/HDF5 fixtures; no external scene or GDAL required."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from scripts.satellites import emit_utils as emit


class EmitReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hygas-emit-reader-")
        self.addCleanup(self.temp.cleanup)
        # OBS lives under a parent containing RAD: this reproduces the original bug.
        self.directory = Path(self.temp.name)/"EMITL1BRAD_001"
        self.directory.mkdir()
        self.rad = self.directory/"EMIT_L1B_RAD_001_20240624T091109_2417606_010.nc"
        self.obs = self.directory/self.rad.name.replace("_RAD_", "_OBS_")
        self.raw = np.arange(6*5*4,dtype=np.float32).reshape(6,5,4)/10
        self.raw[1,2,0] = -9999
        self.raw[2,3,1] = np.nan
        self.raw[3,1,2] = np.inf
        self.raw[4,2,3] = -.25  # Legitimate negative radiance is not a fill value.
        for path in (self.rad,self.obs):
            with h5py.File(path,"w") as f:
                f.attrs["product_version"]="V001"
                f.attrs["software_build_version"]="010618"
                f.attrs["flight_line"]="emit20240624t091109_o17606_s003"
                f.attrs["time_coverage_start"]="2024-06-24T09:11:09+0000"
                if path==self.rad:
                    d=f.create_dataset("radiance",data=self.raw)
                    d.attrs["_FillValue"]=-9999.;d.attrs["units"]="uW/cm^2/SR/nm"
                    for key, values in (("wavelengths",[1600.,1900.,2200.,2300.]),("fwhm",[8.,8.,9.,9.])):
                        d=f.create_dataset("sensor_band_parameters/"+key,data=values)
                        d.attrs["units"]="nm"
                    for i,name in enumerate(("lat","lon","elev")):
                        d=f.create_dataset("location/"+name,data=np.arange(30).reshape(6,5)+i*50.)
                        d.attrs["_FillValue"]=-9999.
                    gx=np.array([[1,2,5,0,-9999,6],[3,3,3,3,3,3]],dtype=np.int32)
                    gy=np.array([[1,2,6,0,-9999,6],[2,3,4,5,6,7]],dtype=np.int32)
                    for name,data in (("glt_x",gx),("glt_y",gy)):
                        d=f.create_dataset("location/"+name,data=data);d.attrs["_FillValue"]=-9999
                else:
                    layers=np.zeros((6,5,2),np.float32)
                    layers[...,0]=30.;layers[...,1]=10.;layers[0,0,0]=-9999.
                    d=f.create_dataset("obs",data=layers);d.attrs["_FillValue"]=-9999.
                    f.create_dataset("sensor_band_parameters/observation_bands",data=np.array(
                        ["To-sun zenith (0 to 90 degrees from zenith)","Slope (local surface slope as derived from DEM in degrees)"],
                        dtype=h5py.string_dtype()))

    def read(self,**kwargs):
        return emit.read_emit_scene([str(self.rad),str(self.obs)],**kwargs)

    def test_pairing_uses_basename_and_is_order_independent(self):
        for paths in ([self.rad,self.obs],[self.obs,self.rad]):
            self.assertEqual(emit._identify_inputs(paths),(str(self.rad),str(self.obs)))

    def test_pairing_rejects_duplicates_missing_rad_or_unrecognized_files(self):
        for paths in ([],[self.obs],[self.rad,self.rad],[self.rad,self.obs,self.obs],[self.rad,"other.nc"]):
            with self.subTest(paths=paths),self.assertRaises(ValueError):emit._identify_inputs(paths)

    def test_pairing_rejects_other_scene_or_version(self):
        for name in (self.obs.name.replace("20240624","20240824"),self.obs.name.replace("_001_","_002_")):
            with self.subTest(name=name),self.assertRaises(ValueError):
                emit._identify_inputs([self.rad,self.directory/name])

    def test_native_orientation_and_fill_validity(self):
        scene=self.read()
        expected=self.raw.copy();expected[(expected==-9999)|~np.isfinite(expected)]=np.nan
        np.testing.assert_array_equal(scene.cube,expected.transpose(2,0,1))
        np.testing.assert_array_equal(scene.valid_radiance,np.isfinite(scene.cube))
        self.assertEqual(scene.cube.dtype,np.float32)
        self.assertEqual(scene.cube[3,4,2],-.25)
        self.assertFalse(scene.valid_pixels[1,2])
        self.assertFalse(scene.metadata["quality"]["validity_is_official_qa"])
        self.assertEqual(scene.metadata["quality"]["official_masks_loaded"],[])

    def test_window_preserves_native_ids_and_requested_band_order(self):
        scene=self.read(rows=slice(1,5),columns=slice(1,4),band_indices=[3,0,2])
        np.testing.assert_array_equal(scene.row_indices,np.arange(1,5))
        np.testing.assert_array_equal(scene.column_indices,np.arange(1,4))
        np.testing.assert_array_equal(scene.band_indices,[3,0,2])
        np.testing.assert_array_equal(scene.wavelengths,[2300,1600,2200])
        np.testing.assert_array_equal(scene.fwhm,[9,8,9])
        full=self.read()
        np.testing.assert_array_equal(scene.cube,full.cube[[3,0,2],1:5,1:4])
        np.testing.assert_array_equal(scene.location["lat"],full.location["lat"][1:5,1:4])

    def test_read_is_windowed_at_hdf5_dataset(self):
        original=h5py.Dataset.__getitem__;requests=[]
        def record(dataset,selection):
            if dataset.name=="/radiance":requests.append(selection)
            return original(dataset,selection)
        with patch.object(h5py.Dataset,"__getitem__",record):
            self.read(rows=slice(1,3),columns=slice(2,4),band_indices=[3,0])
        self.assertEqual(len(requests),1)
        self.assertEqual(requests[0],(slice(1,3),slice(2,4),[0,3]))

    def test_invalid_spatial_windows_fail_instead_of_silent_truncation(self):
        for kwargs in (dict(rows=slice(-1,4)),dict(rows=slice(0,7)),dict(rows=slice(3,3)),
                       dict(rows=slice(0,5,2)),dict(columns=slice(1,7)),dict(columns=[1,2])):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.read(**kwargs)

    def test_invalid_band_requests_fail(self):
        for ids in ([],[4],[-1],[1,1],[1.5],[True],slice(0,8),slice(None,None,-1)):
            with self.subTest(ids=ids),self.assertRaises(ValueError):self.read(band_indices=ids)

    def test_single_band_and_strided_spectral_slice(self):
        self.assertEqual(self.read(band_indices=[2]).cube.shape,(1,6,5))
        np.testing.assert_array_equal(self.read(band_indices=slice(0,4,2)).band_indices,[0,2])

    def test_missing_units_or_unknown_units_rejected(self):
        with h5py.File(self.rad,"r+") as f:f["radiance"].attrs["units"]="W/m2/sr/um"
        with self.assertRaisesRegex(ValueError,"units"):self.read()
        with h5py.File(self.rad,"r+") as f:del f["radiance"].attrs["units"]
        with self.assertRaisesRegex(ValueError,"units"):self.read()

    def test_declared_range_and_cf_scaling_before_validity(self):
        with h5py.File(self.rad,"r+") as f:
            d=f["radiance"];d.attrs["valid_range"]=[0.,10.]
            d.attrs["scale_factor"]=2.;d.attrs["add_offset"]=1.
        scene=self.read()
        self.assertEqual(scene.cube[1,0,0],np.float32(self.raw[0,0,1]*2+1))
        self.assertTrue(np.isnan(scene.cube[3,4,2]))
        self.assertTrue(np.isnan(scene.cube[3,5,4]))
        self.assertTrue(np.isnan(scene.cube[0,1,2]))

    def test_finite_missing_value_attribute_is_not_valid(self):
        with h5py.File(self.rad,"r+") as f:
            f["radiance"].attrs["missing_value"]=np.array([1.2,1.3],dtype=np.float32)
        scene=self.read()
        self.assertFalse(scene.valid_radiance[0,0,3])

    def test_obs_mismatches_are_errors_not_geometry_summary_strings(self):
        for field in ("software_build_version","flight_line","time_coverage_start"):
            with h5py.File(self.obs,"r+") as f:
                old=f.attrs[field];f.attrs[field]="different"
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):self.read()
            with h5py.File(self.obs,"r+") as f:f.attrs[field]=old

    def test_wrong_obs_dimensions_and_labels_rejected(self):
        with h5py.File(self.obs,"r+") as f:
            del f["obs"];f.create_dataset("obs",data=np.zeros((5,5,2)))
        with self.assertRaisesRegex(ValueError,"spatial"):self.read()
        with h5py.File(self.obs,"r+") as f:
            del f["obs"];f.create_dataset("obs",data=np.zeros((6,5,3)))
        with self.assertRaisesRegex(ValueError,"labels"):self.read()

    def test_selected_spectral_metadata_must_be_valid(self):
        with h5py.File(self.rad,"r+") as f:f["sensor_band_parameters/fwhm"][1]=np.nan
        with self.assertRaisesRegex(ValueError,"wavelength/FWHM"):self.read()
        self.assertEqual(self.read(band_indices=[3]).cube.shape[0],1)

    def test_filename_and_product_version_must_agree(self):
        with h5py.File(self.rad,"r+") as f:f.attrs["product_version"]="V002"
        with self.assertRaisesRegex(ValueError,"product_version"):
            emit.read_emit_scene([self.rad],load_location=False)

    def test_spectral_calibration_units_are_explicit(self):
        with h5py.File(self.rad,"r+") as f:f["sensor_band_parameters/wavelengths"].attrs["units"]="um"
        with self.assertRaisesRegex(ValueError,"nm"):self.read()

    def test_missing_required_dataset_not_replaced_by_substring_match(self):
        with h5py.File(self.rad,"r+") as f:
            del f["radiance"];f.create_dataset("other_radiance",data=self.raw)
        with self.assertRaisesRegex(ValueError,"required EMIT dataset"):self.read()

    def test_glt_is_1_based_full_scene_and_crop_membership_is_separate(self):
        scene=self.read(rows=slice(1,5),columns=slice(1,4),load_glt=True)
        self.assertEqual(scene.glt_x[0,0],1)
        self.assertEqual(scene.glt_y[0,0],1)
        self.assertTrue(scene.glt_valid[0,0])
        self.assertFalse(scene.glt_in_window[0,0])
        self.assertTrue(scene.glt_in_window[0,1])
        np.testing.assert_array_equal(scene.glt_valid[0],[True,True,True,False,False,False])
        np.testing.assert_array_equal(scene.glt_in_window[1],[True,True,True,True,False,False])
        with h5py.File(self.rad) as f:np.testing.assert_array_equal(scene.glt_x,f["location/glt_x"][:])

    def test_optional_sources_are_explicitly_absent(self):
        scene=emit.read_emit_scene([self.rad],load_location=False)
        self.assertEqual(scene.observation,{})
        self.assertEqual(scene.location,{})
        self.assertIsNone(scene.glt_x)
        self.assertIsNone(scene.glt_valid)
        self.assertFalse(scene.metadata["observation_loaded"])

    def test_obs_fill_excluded_from_geometry_and_location_fill_masked(self):
        with h5py.File(self.rad,"r+") as f:f["location/lat"][0,0]=-9999.
        scene=self.read()
        self.assertTrue(np.isnan(scene.location["lat"][0,0]))
        self.assertTrue(np.isnan(next(iter(scene.observation.values()))[0,0]))
        self.assertTrue(any("Sun zenith: mean=30.000" in s for s in scene.geometry_lines))

    def test_legacy_payload_remains_scene_data_compatible(self):
        payload=emit.load_emit_scene([self.rad,self.obs])
        self.assertEqual(set(payload),{"cube","wavelengths","scene_id","metadata","geometry_lines","reference_wavelengths"})
        self.assertEqual(payload["cube"].shape,(4,6,5))
        self.assertEqual(payload["metadata"]["observation"],str(self.obs))


if __name__=="__main__":unittest.main()
