"""Bounded synthetic tests for EMIT wiring, native indices and noise semantics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal, osr

from scripts.core import jpl_matched_filter, matched_filter, noise, targets
from scripts.main import build_parser, main
from scripts.pipelines import emit_pipeline as pipeline
from scripts.satellites import emit_utils


class EmitPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hygas-emit-pipeline-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rad = self.root/"EMIT_L1B_RAD_001_20240112T185917_2401212_016.nc"
        self.obs = self.root/self.rad.name.replace("_RAD_", "_OBS_")
        self.atm = self.root/self.rad.name.replace("_L1B_RAD_", "_L2A_MASK_")
        self.wave = np.array([1000,1900,1950,2000,2050,2100,2200,2250,2300,2350,2400,2450.])
        self.rng = np.random.default_rng(142)
        self.raw = 3 + self.rng.normal(0,.03,(64,6,12))
        self.raw[2,1,5] = -9999  # Invalid for both gas windows.
        lat, lon = np.indices((64,6)).astype(float)
        lat, lon = 30 + lat*.001, 10 + lon*.001
        srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
        for path in (self.rad, self.obs, self.atm):
            with h5py.File(path,"w") as f:
                f.attrs["product_version"]="V001"
                f.attrs["software_build_version"]="010618"
                f.attrs["flight_line"]="same_flight"
                f.attrs["time_coverage_start"]="2024-01-12T18:59:17+0000"
                if path == self.rad:
                    f.attrs["geotransform"]=[10,.001,0,30,0,-.001]
                    f.attrs["spatial_ref"]=srs.ExportToWkt()
                    d=f.create_dataset("radiance",data=self.raw.astype(np.float32))
                    d.attrs["units"]="uW/cm^2/SR/nm"; d.attrs["_FillValue"]=-9999.
                    for name,values in (("wavelengths",self.wave),("fwhm",np.full(12,8.))):
                        d=f.create_dataset("sensor_band_parameters/"+name,data=values); d.attrs["units"]="nm"
                    for name,values in (("glt_x",np.tile(np.arange(1,7),(64,1))),
                                        ("glt_y",np.tile(np.arange(1,65)[:,None],(1,6)))):
                        f.create_dataset("location/"+name,data=values.astype(np.int32))
                elif path == self.obs:
                    f.create_dataset("obs",data=np.full((64,6,1),30.))
                    f.create_dataset("sensor_band_parameters/observation_bands", data=np.array(
                        ["To-sun zenith (0 to 90 degrees from zenith)"],dtype=h5py.string_dtype()))
                else:
                    # Flags are deliberately all 1: must NOT be applied as exclusions.
                    values=np.ones((64,6,8)); values[...,6]=2.
                    d=f.create_dataset("mask",data=values); d.attrs["_FillValue"]=-9999.
                    f.create_dataset("sensor_band_parameters/mask_bands",data=np.array(
                        ["Cloud flag","Cirrus flag","Water flag","Spacecraft Flag","Dilated Cloud Flag",
                         "AOD550","H2O (g cm-2)","Aggregate Flag"],dtype=h5py.string_dtype()))
                if path != self.obs:
                    for name,values in (("lat",lat),("lon",lon),("elev",np.full((64,6),100.))):
                        f.create_dataset("location/"+name,data=values)
        self.refpath=self.root/"reference.npz"
        self.reference=noise.ColumnwiseSNRReference(
            self.wave, np.full((12,6),3.),np.full((12,6),200.),np.arange(6),
            metadata=dict(sensor="emit",radiance_unit=emit_utils.RADIANCE_UNIT,
                          column_interpolation=False,native_column_indices=list(range(6)),
                          status="experimental",product_version="V001",software_build_version="010618"),
            band_indices=np.arange(12),
        )
        self.reference.save(self.refpath)
        self.lut=self.root/"dummy_lut.h5"
        with h5py.File(self.lut,"w") as f:f.create_dataset("fixture",data=[1])
        self.noisefile=self.root/"noise.txt"
        self.noisefile.write_text("# wavelength a b c\n900 .01 1 .001\n2500 .02 1 .001\n")

    def library(self, concentrations, *args, **kwargs):
        wave=np.arange(1800.,2551.)
        # A nonconstant, negative unit absorption in both gas windows.
        scale=1e-6 if kwargs["gas"]=="ch4" else 5e-8
        absorption=scale*(1+.5*np.sin(wave/30))
        rads=3*np.exp(-np.asarray(concentrations)[:,None]*absorption)
        return rads,wave

    def run_pipeline(self, **kwargs):
        options=dict(snr_reference_path=str(self.refpath),export_ortho=False)
        options.update(kwargs)
        output=self.root/("result_"+str(len(list(self.root.glob("result_*")))))
        with patch.object(pipeline.lut,"generate_library",side_effect=self.library):
            report=pipeline.detection_emit(str(self.rad),str(self.obs),str(self.atm),
                                           str(self.lut),str(output),**options)
        native=next(output.glob("*_native.npz"))
        with np.load(native) as z:arrays={k:z[k] for k in z.files}
        return report,arrays,output

    def test_all_four_mf_modes_and_both_gases(self):
        for gas in ("ch4","co2"):
            for mode in pipeline.MF_MODES:
                with self.subTest(gas=gas,mode=mode):
                    report,a,_=self.run_pipeline(gas=gas,mf_mode=mode,jpl_options=dict(fixed_alpha=.1))
                    self.assertEqual(report["support"]["radiance_valid"],383)
                    self.assertEqual(report["QA_exclusions_applied"],[])
                    self.assertTrue(np.isnan(a["MF"][2,1]))
                    self.assertTrue(np.isnan(a["MF_uncertainty"][2,1]))
                    self.assertEqual(report["uncertainty_method"],"hygas")
                    self.assertGreater(np.isfinite(a["MF"]).sum(),370)
                    self.assertGreater(np.isfinite(a["MF_uncertainty"]).sum(),370)
                    self.assertEqual(report["lut_parameters"]["water_gcm2_used"],2.)

    def test_hygas_uncertainty_exactly_calls_shared_formula(self):
        report,a,_=self.run_pipeline()
        scene=emit_utils.read_emit_scene([self.rad,self.obs],band_indices=a["band_indices"])
        ref,_=pipeline._reference_for_scene(self.refpath,scene)
        sigma=noise.compute_sigma_map_from_reference(ref,scene.cube)
        expected=noise.propagate_rmn_uncertainty_per_pixel(scene.cube,sigma,a["unit_absorption"])
        expected[~scene.valid_pixels]=np.nan
        np.testing.assert_array_equal(a["MF_uncertainty"],expected.astype(np.float32))

    def test_column_crop_selects_physical_reference_without_interpolation(self):
        self.reference.mean_radiance[:]=np.arange(1,7)[None,:]
        self.reference.save(self.refpath)
        with patch.object(noise.ColumnwiseSNRReference,"ensure_column_count",
                          autospec=True,side_effect=lambda obj,n:obj if obj.mean_radiance.shape[1]==n else self.fail("resizing")):
            _,a,_=self.run_pipeline(rows=slice(10,42),columns=slice(2,5))
        np.testing.assert_array_equal(a["column_indices"],[2,3,4])
        np.testing.assert_array_equal(a["row_indices"],np.arange(10,42))
        scene=emit_utils.read_emit_scene([self.rad],columns=slice(2,5))
        ref,_=pipeline._reference_for_scene(self.refpath,scene)
        np.testing.assert_array_equal(ref.mean_radiance[0],[3,4,5])

    def test_reference_holes_disable_uncertainty_not_enhancement(self):
        self.reference.mean_radiance[8,2]=-1
        self.reference.valid_columns=np.array([0,1,2,3,4])
        self.reference.save(self.refpath)
        report,a,_=self.run_pipeline(mf_mode="jpl",jpl_options=dict(fixed_alpha=.1))
        self.assertTrue(np.all(np.isfinite(a["MF"][:,2])))
        self.assertTrue(np.all(np.isfinite(a["MF"][:,5])))
        self.assertTrue(np.all(np.isnan(a["MF_uncertainty"][:,2])))
        self.assertTrue(np.all(np.isnan(a["MF_uncertainty"][:,5])))
        self.assertFalse(a["reference_supported"][:,5].any())

    def test_parametric_noise_independent_of_mf(self):
        for mode in ("full-column","jpl"):
            report,a,_=self.run_pipeline(mf_mode=mode,noise_backend="jpl-parametric",
                snr_reference_path=None,noise_parameters_path=str(self.noisefile),jpl_options=dict(fixed_alpha=.1))
            self.assertEqual(report["noise_backend"],"jpl-parametric")
            self.assertGreater(np.isfinite(a["MF_uncertainty"]).sum(),370)

    def test_jpl_propagation_is_explicit_and_matches_current_shared_core(self):
        _,a,_=self.run_pipeline(mf_mode="jpl",uncertainty_method="jpl",jpl_options=dict(fixed_alpha=.1))
        scene=emit_utils.read_emit_scene([self.rad,self.obs],band_indices=a["band_indices"])
        ref,_=pipeline._reference_for_scene(self.refpath,scene)
        sigma=noise.compute_sigma_map_from_reference(ref,scene.cube)
        mf,unc,sens=jpl_matched_filter.run_jpl_mf(scene.cube.transpose(1,2,0),a["unit_absorption"],
            scene.valid_pixels,noise_cube=sigma.transpose(1,2,0),fixed_alpha=.1,nodata_value=np.nan)
        np.testing.assert_array_equal(a["MF"],mf)
        np.testing.assert_array_equal(a["MF_uncertainty"],unc)
        np.testing.assert_array_equal(a["MF_sensitivity"],sens)

    def test_invalid_noise_or_formula_configurations_are_rejected(self):
        for kwargs in (dict(uncertainty_method="jpl"),dict(noise_backend="unknown"),
                       dict(snr_reference_path=None),dict(noise_backend="jpl-parametric"),
                       dict(noise_parameters_path=str(self.noisefile))):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.run_pipeline(**kwargs)

    def test_ambiguous_reference_units_or_axis_rejected(self):
        scene=emit_utils.read_emit_scene([self.rad])
        for key,value in (("sensor","prisma"),("radiance_unit","W m-2"),
                          ("column_interpolation",True),("native_column_indices",[0,1])):
            old=self.reference.metadata[key];self.reference.metadata[key]=value;self.reference.save(self.refpath)
            with self.subTest(key=key),self.assertRaises(ValueError):pipeline._reference_for_scene(self.refpath,scene)
            self.reference.metadata[key]=old

    def test_reference_spectral_identity_not_nearest_band(self):
        scene=emit_utils.read_emit_scene([self.rad])
        self.reference.band_nm[4]+=3
        self.reference.save(self.refpath)
        with self.assertRaisesRegex(ValueError,"wavelengths"):pipeline._reference_for_scene(self.refpath,scene)

    def test_atmosphere_identity_grid_labels_and_fill(self):
        scene=emit_utils.read_emit_scene([self.rad,self.obs])
        with h5py.File(self.atm,"r+") as f:f["mask"][0,0,6]=-9999
        water,meta=emit_utils.read_emit_water_vapour(self.atm,scene)
        self.assertTrue(np.isnan(water[0,0]));self.assertFalse(meta["QA_exclusions_applied"])
        with self.assertRaisesRegex(ValueError,"version"):emit_utils.read_emit_water_vapour(str(self.atm).replace("_001_","_002_"),scene)
        with h5py.File(self.atm,"r+") as f:f["location/lat"][0,0]+=1
        with self.assertRaisesRegex(ValueError,"grids differ"):emit_utils.read_emit_water_vapour(self.atm,scene)

    def test_missing_h2o_and_invalid_geometry_fail_without_default(self):
        with h5py.File(self.atm,"r+") as f:f["mask"][:,:,6]=-9999
        with self.assertRaisesRegex(ValueError,"H2O"):self.run_pipeline()
        with h5py.File(self.atm,"r+") as f:f["mask"][:,:,6]=2
        with h5py.File(self.obs,"r+") as f:f["obs"][:]=90
        with self.assertRaisesRegex(ValueError,"zenith"):self.run_pipeline()

    def test_glt_crop_is_not_shifted_or_dilated(self):
        scene=emit_utils.read_emit_scene([self.rad],rows=slice(10,20),columns=slice(2,5),load_glt=True)
        image=np.arange(30,dtype=np.float32).reshape(10,3)
        ortho=emit_utils.apply_glt(image,scene)
        np.testing.assert_array_equal(ortho[10:20,2:5],image)
        self.assertEqual(np.isfinite(ortho).sum(),30)
        self.assertTrue(np.isnan(ortho[0,0]))

    def test_orthorectified_exports_and_report_hashes(self):
        report,a,out=self.run_pipeline(export_ortho=True,rows=slice(10,40),columns=slice(1,5))
        raster=gdal.Open(str(next(out.glob("*_MF_ortho.tif"))))
        values=raster.ReadAsArray()
        np.testing.assert_array_equal(values[10:40,1:5],a["MF"])
        self.assertEqual(raster.GetGeoTransform(),(10,.001,0,30,0,-.001))
        self.assertIn("4326",raster.GetProjection())
        self.assertTrue(np.isnan(raster.GetRasterBand(1).GetNoDataValue()))
        raster=None
        loaded=json.loads(next(out.glob("*_processing.json")).read_text())
        for data in loaded["outputs"].values():
            self.assertEqual(pipeline._identity(data["path"])["sha256"],data["sha256"])

    def test_existing_outputs_are_never_overwritten(self):
        _,_,out=self.run_pipeline()
        before={p.name:pipeline._identity(p)["sha256"] for p in out.iterdir()}
        with patch.object(pipeline.lut,"generate_library",side_effect=self.library),self.assertRaises(FileExistsError):
            pipeline.detection_emit(str(self.rad),str(self.obs),str(self.atm),str(self.lut),str(out),
                                    snr_reference_path=str(self.refpath),export_ortho=False)
        self.assertEqual(before,{p.name:pipeline._identity(p)["sha256"] for p in out.iterdir()})

    def test_full_column_numerical_parity_with_shared_helper(self):
        _,a,_=self.run_pipeline(mf_mode="full-column")
        scene=emit_utils.read_emit_scene([self.rad],band_indices=a["band_indices"])
        for col in (0,1,5):
            valid=scene.valid_pixels[:,col]
            rad=scene.cube[:,valid,col].T[:,None,:]
            mu,cov=matched_filter.calculate_column_statistics(rad)
            expected=matched_filter.calculate_matched_filter_columnwise(rad,np.zeros(rad.shape[:2],int),
                mu,cov,a["unit_absorption"][:,None],1)[:,0]
            np.testing.assert_array_equal(a["MF"][valid,col],expected.astype(np.float32))

    def test_native_column_selection_preserves_gaps_and_rejects_compression(self):
        self.reference.valid_columns=np.array([0,2,5])
        ref=self.reference.subset_native_columns([5,1,2],native_column_count=6)
        np.testing.assert_array_equal(ref.valid_columns,[0,2])
        self.assertTrue(np.isnan(ref.snr[:,1]).all())
        with self.assertRaises(ValueError):self.reference.subset_native_columns([0,1],native_column_count=7)
        for indices in ([1,1],[-1],[6],[1.5],[]):
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                self.reference.subset_native_columns(indices,native_column_count=6)

    def test_cli_emit_dispatch_and_validation(self):
        args=build_parser().parse_args(["--satellite","emit","--lut","lut","--gas","co2"])
        self.assertEqual(args.emit_uncertainty_method,"hygas")
        with patch.object(pipeline,"detection_emit") as run:
            main(["--satellite","emit","--lut","lut","--emit-rad","rad","--emit-obs","obs",
                  "--emit-atmosphere","mask","--output","out","--emit-columns","10","20"])
            self.assertEqual(run.call_args.kwargs["columns"],slice(10,20))
        with self.assertRaises(SystemExit):main(["--satellite","emit","--lut","lut"])

    def test_batch_requires_unique_complete_triplets_before_execution(self):
        with patch.object(pipeline,"detection_emit",return_value={}) as run:
            reports=pipeline.process_directory_emit(self.root,self.root/"batch",lut_file=str(self.lut))
            self.assertEqual(len(reports),1);self.assertEqual(run.call_count,1)
        duplicate=self.root/"duplicate";duplicate.mkdir()
        (duplicate/self.rad.name).write_bytes(b"duplicate")
        with patch.object(pipeline,"detection_emit") as run,self.assertRaisesRegex(ValueError,"Duplicate"):
            pipeline.process_directory_emit(self.root,self.root/"batch",lut_file=str(self.lut))
        run.assert_not_called()


if __name__=="__main__":unittest.main()
