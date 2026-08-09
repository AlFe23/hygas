"""Unit tests for the gas-aware Tanager CO2 matched-filter path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.core import lut
from scripts.main import build_parser
from scripts.pipelines.tanager_pipeline import resolve_gas_settings


class CarbonDioxideLutTests(unittest.TestCase):
    def test_co2_index_matches_lut_axis(self):
        concentrations = lut.default_concentrations("co2")
        np.testing.assert_allclose(lut.get_carbon_dioxide_index(concentrations), np.arange(8))

    def test_co2_loader_labels_dataset_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "co2_lut.h5"
            with h5py.File(path, "w") as f:
                f.create_dataset("modtran_data", data=np.ones((2, 2, 2, 2, 2, 4)))
                f.create_dataset("modtran_param", data=np.ones((2, 2, 2, 2, 2, 5)))
                f.create_dataset("wave", data=np.arange(4, dtype=float))

            grid, _, wave, gas = lut.load_co2_dataset(path)
            self.assertEqual(gas, "co2")
            self.assertEqual(grid.shape[-1], wave.shape[0])
            grid.file.close()


class TanagerCarbonDioxideConfigTests(unittest.TestCase):
    def test_co2_defaults_match_iker_window_and_lut_levels(self):
        gas, min_wavelength, max_wavelength, concentrations = resolve_gas_settings("co2", None, None)
        self.assertEqual(gas, "co2")
        self.assertEqual((min_wavelength, max_wavelength), (1900.0, 2100.0))
        np.testing.assert_allclose(concentrations, lut.default_concentrations("co2"))

    def test_ch4_defaults_are_unchanged(self):
        gas, min_wavelength, max_wavelength, concentrations = resolve_gas_settings("ch4", None, None)
        self.assertEqual(gas, "ch4")
        self.assertEqual((min_wavelength, max_wavelength), (2100.0, 2450.0))
        np.testing.assert_allclose(concentrations, lut.default_concentrations("ch4"))

    def test_cli_exposes_tanager_co2(self):
        args = build_parser().parse_args(["--satellite", "tanager", "--lut", "co2_lut.h5", "--gas", "co2"])
        self.assertEqual(args.gas, "co2")
        self.assertIsNone(args.min_wavelength)
        self.assertIsNone(args.max_wavelength)


if __name__ == "__main__":
    unittest.main()
