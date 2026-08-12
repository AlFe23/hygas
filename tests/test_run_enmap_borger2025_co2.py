"""Tests for EnMAP Borger-2025 CO2 batch archive selection."""

from __future__ import annotations

import argparse
import json
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

from scripts.run_enmap_borger2025_co2 import (
    acquisition_key,
    has_complete_outputs,
    run,
    select_archives,
)


class ArchiveSelectionTests(unittest.TestCase):
    def test_selects_newest_duplicate_revision_and_excludes_unassigned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case = root / "2023-08-24_Matla_South_Africa"
            unassigned = root / "unassigned_adjacent_scenes"
            case.mkdir()
            unassigned.mkdir()
            older = case / "ENMAP01-____L1B-DT0000038106_20230824T085645Z_003_V010506_20260810T143737Z.ZIP"
            newer = case / "ENMAP01-____L1B-DT0000038106_20230824T085645Z_003_V010506_20260810T143956Z.ZIP"
            adjacent = unassigned / "ENMAP01-____L1B-DT0000038106_20230824T085641Z_002_V010506_20260810T143704Z.ZIP"
            for path in (older, newer, adjacent):
                path.touch()

            selections = select_archives(root)

            self.assertEqual(len(selections), 1)
            self.assertEqual(selections[0].archive, newer)
            self.assertEqual(selections[0].skipped_revisions, (older,))

    def test_acquisition_key_ignores_product_revision(self):
        archive = Path("ENMAP01-____L1B-DT0000044547_20231005T084552Z_004_V010506_20260810T145018Z.ZIP")
        self.assertEqual(acquisition_key(archive), "L1B-DT0000044547_20231005T084552Z_004")

    def test_complete_outputs_require_both_primary_geotiffs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "scene_MF.tif").touch()
            self.assertFalse(has_complete_outputs(output_dir))
            (output_dir / "scene_MF_uncertainty.tif").touch()
            self.assertTrue(has_complete_outputs(output_dir))

    def test_completed_rerun_replaces_a_previous_failed_manifest_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "ENMAP01-____L1B-DT0000038106_20230824T085645Z_003_V010506_20260810T143956Z.ZIP"
            archive.touch()
            key = acquisition_key(archive)
            (root / "co2_batch_manifest.json").write_text(
                json.dumps([{"acquisition_key": key, "status": "failed"}]),
                encoding="utf-8",
            )
            lut_file = root / "lut.h5"
            snr_file = root / "snr.npz"
            lut_file.touch()
            snr_file.touch()
            args = argparse.Namespace(
                root=root,
                lut=lut_file,
                snr_reference=snr_file,
                include_unassigned=False,
                dry_run=False,
                scratch_dir=None,
                rerun_completed=False,
            )

            with patch("scripts.run_enmap_borger2025_co2.safe_extract"), patch(
                "scripts.run_enmap_borger2025_co2.locate_scene_files",
                return_value=(root / "vnir.tif", root / "swir.tif", root / "metadata.xml"),
            ), patch("scripts.run_enmap_borger2025_co2.enmap_pipeline.detection_enmap"):
                self.assertEqual(run(args), 0)

            manifest = json.loads((root / "co2_batch_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["acquisition_key"], key)
            self.assertEqual(manifest[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
