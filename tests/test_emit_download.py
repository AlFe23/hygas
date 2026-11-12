"""Offline acquisition tests: no NASA credentials, network or science mutations."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.downloads import emit
from scripts import download_emit


SCENE = "20240624T091109_2417606_010"
NAME = f"EMIT_L2A_MASK_001_{SCENE}.nc"
URL = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/granule/{NAME}"


class Granule(dict):
    def data_links(self, access):
        assert access == "external"
        return self["test_links"]


def granule(urls=None):
    return Granule(meta={"concept-id": "G123-LPCLOUD", "revision-id": 1},
        umm={"GranuleUR": f"EMIT_L2A_RFL_001_{SCENE}",
             "CollectionReference": {"ShortName": "EMITL2ARFL", "Version": "001"},
             "DataGranule": {"ArchiveAndDistributionInformation": [
                 {"Name": NAME, "Size": 79.2668342590332, "SizeUnit": "MB"}]}},
        test_links=urls if urls is not None else [URL])


def asset():
    return emit.EmitAsset("EMITL2ARFL", "001", SCENE, "MASK", NAME, URL, "G123-LPCLOUD", "1", "granule")


class Client:
    def __init__(self):
        self.results = [granule()]
        self.search_calls = []
        self.login_calls = []
        self.download_calls = []
        self.fail = False
        self.html = False
        self.version = "V001"

    def search_data(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.results

    def login(self, **kwargs):
        self.login_calls.append(kwargs)
        return types.SimpleNamespace(authenticated=True)

    def download(self, urls, **kwargs):
        self.download_calls.append((urls, kwargs))
        path = Path(kwargs["local_path"]) / NAME
        if self.fail:
            path.write_bytes(b"partial")
            raise RuntimeError("SECRET should not reach logs")
        if self.html:
            path.write_bytes(b"<html>login required</html>")
        else:
            with h5py.File(path, "w") as f:
                f.attrs["product_version"] = self.version
                f.create_dataset("mask", data=[0, 1, 0])
        return [str(path)]


class EmitDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hygas-emit-download-")
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "download"
        self.client = Client()
        self.environ = patch.dict("os.environ", {"EARTHDATA_TOKEN": "test-not-a-real-token"}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        token_file = patch.object(emit, "DEFAULT_TOKEN_FILE", Path(self.temp.name) / "earthdata-token")
        token_file.start()
        self.addCleanup(token_file.stop)

    def search(self, **overrides):
        args = dict(collection="EMITL2ARFL", version="001", scene=SCENE,
                    products=["MASK"], client=self.client)
        args.update(overrides)
        return emit.search_emit_products(**args)

    def download(self, assets=None):
        return emit.download_emit_products(assets or [asset()], self.output, client=self.client)

    def test_exact_public_search_does_not_login_or_download(self):
        result = self.search()
        self.assertEqual([x.filename for x in result], [NAME])
        self.assertEqual(self.client.search_calls[0]["granule_name"], f"EMIT_*_001_{SCENE}")
        self.assertEqual(self.client.search_calls[0]["version"], "001")
        self.assertEqual(self.client.login_calls, [])
        self.assertEqual(self.client.download_calls, [])
        self.assertIsNone(result[0].size_bytes)  # Rounded MB must not be exact bytes.
        self.assertGreater(result[0].cmr_size, 79)
        self.assertFalse(self.output.exists())

    def test_selection_excludes_reflectance_browse_and_metadata(self):
        self.client.results = [granule([URL, URL.replace("_MASK_", "_RFL_"), URL + ".dmrpp", URL.replace(".nc", ".png")])]
        self.assertEqual(len(self.search()), 1)

    def test_ch4_geotiff_search_keeps_explicit_version_and_product(self):
        name = f"EMIT_L2B_CH4ENH_002_{SCENE}.tif"
        url = URL.replace("EMITL2ARFL.001", "EMITL2BCH4ENH.002").replace(NAME, name)
        record = granule([url, url.replace(".tif", ".png"), url.replace("_CH4ENH_", "_CH4UNCERT_")])
        record["umm"]["CollectionReference"] = {"ShortName": "EMITL2BCH4ENH", "Version": "002"}
        self.client.results = [record]
        selected = self.search(collection="EMITL2BCH4ENH", version="002", products=["CH4ENH"])
        self.assertEqual(len(selected), 1)
        self.assertEqual((selected[0].filename, selected[0].product, selected[0].version), (name, "CH4ENH", "002"))

    def test_geotiff_download_validation_receipt_and_reuse(self):
        selected = replace(asset(), collection="EMITL2BCH4ENH", version="002", product="CH4ENH",
                           filename=f"EMIT_L2B_CH4ENH_002_{SCENE}.tif")
        selected = replace(selected, url=URL.replace(NAME, selected.filename))

        def download(urls, **kwargs):
            path = Path(kwargs["local_path"]) / selected.filename
            with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                               dtype="float32", crs="EPSG:4326", transform=from_origin(16, 42, 0.001, 0.001)) as dst:
                dst.write(np.ones((1, 2, 2), dtype="float32"))
            return [str(path)]

        with patch.object(self.client, "download", side_effect=download) as transfer:
            manifest = self.download([selected])
            result = json.loads(manifest.read_text())
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["completed"][0]["asset"]["product"], "CH4ENH")
            self.download([selected])
            self.assertEqual(transfer.call_count, 1)

    def test_geotiff_html_and_truncated_structure_are_rejected(self):
        selected = replace(asset(), filename=f"EMIT_L2B_CH4ENH_002_{SCENE}.tif", version="002", product="CH4ENH")
        self.output.mkdir()
        path = self.output / selected.filename
        for content in (b"<html>login required</html>", b"II*\x00\x00\x00\x00\x00"):
            path.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(emit.EmitDownloadError):
                emit._validate_file(path, selected)

    def test_cmr_exact_bytes_and_checksum_metadata(self):
        entry = self.client.results[0]["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]
        entry.update(SizeInBytes=83117300, Checksum={"Algorithm": "SHA-512", "Value": "a" * 128})
        selected = self.search()[0]
        self.assertEqual(selected.size_bytes, 83117300)
        self.assertEqual(selected.checksum_algorithm, "sha512")
        self.assertEqual(selected.checksum_value, "a" * 128)

    def test_malformed_cmr_checksum_and_byte_count_are_rejected(self):
        entry = self.client.results[0]["umm"]["DataGranule"]["ArchiveAndDistributionInformation"][0]
        entry["SizeInBytes"] = -1
        with self.assertRaises(emit.EmitDownloadError):
            self.search()
        entry["SizeInBytes"] = 83117300
        entry["Checksum"] = {"Algorithm": "SHA-512", "Value": "not-a-digest"}
        with self.assertRaises(emit.EmitDownloadError):
            self.search()

    def test_standalone_mask_collection_uses_its_own_explicit_version(self):
        record = granule([URL.replace("_001_", "_003_")])
        record["umm"]["CollectionReference"] = {"ShortName": "EMITL2AMASK", "Version": "003"}
        self.client.results = [record]
        selected = self.search(collection="EMITL2AMASK", version="003")[0]
        self.assertEqual(selected.version, "003")
        self.assertEqual(selected.collection, "EMITL2AMASK")

    def test_duplicate_identical_link_in_one_granule_is_harmless(self):
        self.client.results = [granule([URL, URL])]
        self.assertEqual(len(self.search()), 1)

    def test_missing_or_ambiguous_results_fail_without_fallback(self):
        for result in ([], [granule(), granule()]):
            with self.subTest(result=result):
                self.client.results = result
                with self.assertRaises(emit.EmitDownloadError):
                    self.search()

    def test_version_mismatch_is_rejected(self):
        self.client.results[0]["umm"]["CollectionReference"]["Version"] = "002"
        with self.assertRaises(emit.EmitDownloadError):
            self.search()

    def test_file_identity_mismatch_is_rejected(self):
        self.client.results = [granule([URL.replace("_001_", "_002_")])]
        with self.assertRaises(emit.EmitDownloadError):
            self.search()

    def test_unsafe_signed_or_non_https_links_are_rejected(self):
        for url in (URL + "?token=SECRET", URL.replace("https:", "http:"), URL.replace("data.lpdaac.earthdatacloud.nasa.gov", "evil.example")):
            with self.subTest(url=url):
                self.client.results = [granule([url])]
                with self.assertRaises(emit.EmitDownloadError):
                    self.search()

    def test_query_validation(self):
        for kwargs in (dict(version="latest"), dict(scene="20240624"), dict(products=[]),
                       dict(products=["MASK", "MASK"]), dict(collection="OTHER")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.search(**kwargs)

    def test_environment_login_is_explicit_and_never_persisted(self):
        emit.authenticate_earthdata(client=self.client)
        self.assertEqual(self.client.login_calls, [dict(strategy="environment", persist=False)])

    def test_missing_token_fails_without_prompt_or_login(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(emit.EmitDownloadError):
            emit.authenticate_earthdata(client=self.client)
        self.assertEqual(self.client.login_calls, [])

    def test_download_creates_receipt_manifest_and_reuses_verified_file(self):
        manifest = self.download()
        self.assertTrue((self.output / NAME).is_file())
        content = json.loads(manifest.read_text())
        self.assertEqual(content["state"], "complete")
        self.assertEqual(content["completed"][0]["status"], "downloaded_verified")
        self.assertNotIn("test-not-a-real-token", manifest.read_text())
        self.assertFalse((self.output / ".hygas-emit/download.lock").exists())
        self.download()
        self.assertEqual(len(self.client.download_calls), 1)
        self.assertEqual(len(self.client.login_calls), 1)

    def test_changed_existing_file_is_never_overwritten(self):
        self.download()
        with h5py.File(self.output / NAME, "a") as f:
            f.attrs["tampered"] = 1
        digest = emit._hash(self.output / NAME)
        with self.assertRaises(emit.EmitDownloadError):
            self.download()
        self.assertEqual(emit._hash(self.output / NAME), digest)

    def test_unreceipted_existing_file_is_not_assumed_complete(self):
        self.output.mkdir()
        with h5py.File(self.output / NAME, "w") as f:
            f.create_dataset("mask", data=[1])
        with self.assertRaises(emit.EmitDownloadError):
            self.download()
        self.assertEqual(self.client.download_calls, [])

    def test_failed_download_retains_partial_and_redacts_upstream_error(self):
        self.client.fail = True
        with self.assertRaises(emit.EmitDownloadError) as caught:
            self.download()
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertFalse((self.output / NAME).exists())
        self.assertTrue(list((self.output / ".partial").rglob(NAME)))
        self.assertFalse((self.output / ".hygas-emit/download.lock").exists())

    def test_html_and_wrong_hdf_version_are_rejected(self):
        self.client.html = True
        with self.assertRaises(emit.EmitDownloadError):
            self.download()
        self.client.html = False
        self.client.version = "V002"
        with self.assertRaises(emit.EmitDownloadError):
            self.download()
        self.assertFalse((self.output / NAME).exists())

    def test_header_only_truncated_hdf_is_rejected(self):
        self.output.mkdir()
        (self.output / NAME).write_bytes(b"\x89HDF\r\n\x1a\n")
        with self.assertRaises(emit.EmitDownloadError):
            self.download()

    def test_cmr_checksum_mismatch_is_rejected(self):
        with self.assertRaises(emit.EmitDownloadError):
            self.download([replace(asset(), checksum_algorithm="sha256", checksum_value="0" * 64)])
        self.assertFalse((self.output / NAME).exists())

    def test_exact_cmr_size_mismatch_is_rejected(self):
        with self.assertRaises(emit.EmitDownloadError):
            self.download([replace(asset(), size_bytes=999999)])
        self.assertFalse((self.output / NAME).exists())

    def test_exact_cmr_checksum_allows_verified_existing_file(self):
        self.output.mkdir()
        path = self.output / NAME
        with h5py.File(path, "w") as f:
            f.create_dataset("mask", data=[1])
        selected = replace(asset(), checksum_algorithm="sha256", checksum_value=emit._hash(path))
        self.download([selected])
        self.assertEqual(self.client.download_calls, [])
        self.assertEqual(self.client.login_calls, [])

    def test_only_empty_owned_lock_is_removed(self):
        (self.output / ".hygas-emit").mkdir(parents=True)
        lock = self.output / ".hygas-emit/download.lock"
        lock.write_text("another process")
        with self.assertRaises(emit.EmitDownloadError):
            self.download()
        self.assertEqual(lock.read_text(), "another process")

    def test_invalid_or_duplicate_asset_names_fail(self):
        for assets in ([replace(asset(), filename="../bad.nc")], [asset(), asset()]):
            with self.subTest(assets=assets), self.assertRaises(ValueError):
                self.download(assets)

    def test_dry_run_cli_never_downloads_or_writes(self):
        with patch.object(download_emit, "search_emit_products", return_value=[asset()]), \
             patch.object(download_emit, "download_emit_products") as download, redirect_stdout(io.StringIO()):
            code = download_emit.main(["--collection", "EMITL2ARFL", "--version", "001", "--scene", SCENE,
                                       "--products", "MASK", "--output", str(self.output), "--dry-run"])
        self.assertEqual(code, 0)
        download.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unexpected_cli_exception_never_echoes_secret(self):
        capture = io.StringIO()
        with patch.object(download_emit, "search_emit_products", side_effect=RuntimeError("SECRET")), redirect_stderr(capture):
            code = download_emit.main(["--collection", "EMITL2ARFL", "--version", "001", "--scene", SCENE,
                                       "--products", "MASK", "--output", str(self.output)])
        self.assertEqual(code, 2)
        self.assertNotIn("SECRET", capture.getvalue())


if __name__ == "__main__":
    unittest.main()
