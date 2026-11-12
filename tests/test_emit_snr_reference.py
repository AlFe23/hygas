"""Reference-generation checks with native gaps and a portable public baseline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from scripts import SNR_emit_reference as generator
from scripts.core.noise import ColumnwiseSNRReference
from scripts.pipelines.emit_pipeline import _reference_for_scene
from scripts.satellites.emit_utils import read_emit_scene


REPO = Path(__file__).resolve().parents[1]


class EmitSNRReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hygas-emit-reference-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rad = self.root / "EMIT_L1B_RAD_001_20240624T091109_2417606_010.nc"
        rng = np.random.default_rng(341)
        self.wave = np.linspace(1005, 2490, 12)
        values = (3 + rng.normal(0, .02, (64, 6, 12))).astype(np.float32)
        values[10:15, 1] = -9999  # Five missing native rows: six pairs are lost.
        values[:, 5] = -9999     # Unsupported detector column, never compressed.
        values[::2, 4] = -9999   # Many valid pixels but zero adjacent valid pairs.
        values[:, 2, 0] = -1 + rng.normal(0, .01, 64)  # Real signed weak signal.
        with h5py.File(self.rad, "w") as f:
            f.attrs["product_version"] = "V001"
            f.attrs["software_build_version"] = "010618"
            d = f.create_dataset("radiance", data=values)
            d.attrs["units"] = "uW/cm^2/SR/nm"
            d.attrs["_FillValue"] = -9999.
            for name, array in (("wavelengths", self.wave), ("fwhm", np.full(12, 8.))):
                d = f.create_dataset("sensor_band_parameters/" + name, data=array)
                d.attrs["units"] = "nm"

    def scene(self, **kwargs):
        return read_emit_scene([self.rad], load_location=False, **kwargs)

    def test_native_support_preserves_gaps_and_signed_cells(self):
        ref, support, pca = generator.measure_reference(self.scene())
        np.testing.assert_array_equal(ref.valid_columns, [0, 1, 2, 3])
        np.testing.assert_array_equal(support["valid_pixel_counts"], [64, 59, 64, 64, 32, 0])
        np.testing.assert_array_equal(support["adjacent_pair_counts"], [63, 57, 63, 63, 0, 0])
        self.assertTrue(np.isnan(ref.snr[:, 4:]).all())
        self.assertLess(ref.mean_radiance[0, 2], 0)
        self.assertLess(ref.snr[0, 2], 0)
        self.assertFalse(support["positive_mean_and_snr"][0, 2])
        self.assertEqual(ref.metadata["native_column_indices"], list(range(6)))
        self.assertFalse(ref.metadata["column_interpolation"])
        self.assertFalse(ref.metadata["official_QA_applied"])
        self.assertEqual(pca["fit_pixel_count"], 283)

    def test_row_selection_and_native_bands_are_recorded(self):
        ref, support, _ = generator.measure_reference(self.scene(rows=slice(20, 60), band_indices=[2, 4, 6, 8, 10]), k_pca=2)
        np.testing.assert_array_equal(ref.band_indices, [2, 4, 6, 8, 10])
        np.testing.assert_array_equal(support["native_row_indices"], np.arange(20, 60))
        self.assertEqual(ref.metadata["native_row_range"], [20, 60])

    def test_refuses_column_crop_and_degenerate_pca(self):
        with self.assertRaisesRegex(ValueError, "full native"):
            generator.measure_reference(self.scene(columns=slice(1, 5)))
        for k in (0, -1, 12):
            with self.subTest(k=k), self.assertRaises(ValueError):
                generator.measure_reference(self.scene(), k_pca=k)
        with self.assertRaises(ValueError):
            generator.measure_reference(self.scene(), min_column_pixels=2)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            generator.measure_reference(self.scene(), min_column_pixels=1000)

    def test_generation_roundtrip_report_hashes_and_overwrite_refusal(self):
        output = self.root / "generated"
        refpath = generator.build_reference(self.rad, output)
        ref = ColumnwiseSNRReference.load(refpath)
        native = self.scene(band_indices=[8, 9, 10])
        selected, support = _reference_for_scene(refpath, native)
        self.assertEqual(selected.snr.shape, (3, 6))
        self.assertFalse(support[:, 4:].any())
        self.assertEqual(ref.metadata["source_inputs"][0]["sha256"], hashlib.sha256(self.rad.read_bytes()).hexdigest())
        report = json.loads((output / "generation_report.json").read_text())
        for entry in report["outputs"]:
            self.assertEqual(hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest(), entry["sha256"])
        with self.assertRaises(FileExistsError):
            generator.build_reference(self.rad, output)
        compared = generator.build_reference(self.rad, self.root / "compared", compare_path=refpath)
        report = json.loads((compared.parent / "generation_report.json").read_text())
        self.assertTrue(report["comparison"]["passed"])

    def test_comparison_detects_changes_and_native_column_identity(self):
        ref, _, _ = generator.measure_reference(self.scene())
        self.assertTrue(generator.compare_reference(ref, ref)["passed"])
        changed = ColumnwiseSNRReference(ref.band_nm, ref.mean_radiance.copy(), ref.snr.copy(), ref.valid_columns.copy(), band_indices=ref.band_indices.copy())
        changed.snr[3, 0] *= 2
        with self.assertRaisesRegex(ValueError, "snr"):
            generator.compare_reference(ref, changed)
        changed.snr = ref.snr.copy()
        changed.valid_columns = np.array([0, 1, 2])
        with self.assertRaisesRegex(ValueError, "identities"):
            generator.compare_reference(ref, changed)

    def test_direct_cli_works_outside_repository(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts/SNR_emit_reference.py"), "--input", str(self.rad),
             "--output-dir", str(self.root / "cli")], cwd=self.root,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "cli/snr_reference_columnwise_experimental.npz").is_file())


class DistributedReferenceTests(unittest.TestCase):
    def test_public_catalog_hashes_dimensions_and_sensor_identity(self):
        catalog = json.loads((REPO / "reference_snr/scenes.json").read_text())
        self.assertEqual({s["sensor"] for s in catalog["references"]}, {"prisma", "enmap", "tanager", "emit"})
        for entry in catalog["references"]:
            path = REPO / entry["reference_path"]
            with self.subTest(sensor=entry["sensor"]):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["reference_sha256"])
                ref = ColumnwiseSNRReference.load(path)
                self.assertEqual(list(ref.snr.shape), entry["shape_bands_columns"])
                self.assertEqual(ref.metadata["sensor"], entry["sensor"])

    def test_emit_public_support_and_both_gas_windows(self):
        root = REPO / "reference_snr/emit/20240624T091109_2417606_010"
        ref = ColumnwiseSNRReference.load(root / "snr_reference_columnwise_experimental.npz")
        np.testing.assert_array_equal(ref.valid_columns, np.arange(1242))
        np.testing.assert_array_equal(ref.band_indices, np.arange(84, 285))
        with np.load(root / "support_and_QA.npz", allow_pickle=False) as support:
            self.assertTrue(np.all(support["counts"] == 2176))
            self.assertTrue(np.all(support["adjacent_pairs"] == 2175))
            self.assertFalse(support["QA_applied"].item())
        for lo, hi, holes in ((2100, 2450, 0), (1900, 2100, 10), (1920, 2100, 0)):
            ids = (ref.band_nm >= lo) & (ref.band_nm <= hi)
            valid = (ref.mean_radiance[ids] > 0) & (ref.snr[ids] > 0)
            with self.subTest(window=(lo, hi)):
                self.assertGreater(ids.sum(), 1)
                self.assertEqual(int((~valid).sum()), holes)


if __name__ == "__main__":
    unittest.main()
