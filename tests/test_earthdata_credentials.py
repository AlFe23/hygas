"""Persistent credentials use private files, environment precedence and no echo."""
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import configure_earthdata
from scripts.downloads import credentials, emit


class CredentialsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hygas-credentials-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config" / "earthdata-token"
        for item in (patch.dict(os.environ, {}, clear=True),
                     patch.object(emit, "DEFAULT_TOKEN_FILE", self.path),
                     patch.object(configure_earthdata, "DEFAULT_TOKEN_FILE", self.path)):
            item.start()
            self.addCleanup(item.stop)

    def test_token_file_is_private_and_readable_in_a_new_environment(self):
        credentials.save_token("test-token", self.path)
        self.assertEqual(credentials.read_token(self.path), "test-token")
        self.assertNotIn("EARTHDATA_TOKEN", os.environ)
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)

    def test_existing_token_requires_explicit_replacement(self):
        credentials.save_token("original", self.path)
        with self.assertRaises(FileExistsError):
            credentials.save_token("new", self.path)
        self.assertEqual(credentials.read_token(self.path), "original")
        credentials.save_token("new", self.path, replace=True)
        self.assertEqual(credentials.read_token(self.path), "new")

    def test_invalid_token_does_not_create_a_file(self):
        for token in ("", " \n", "two tokens", "EARTHDATA_TOKEN=two lines"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                credentials.save_token(token, self.path)
        self.assertFalse(self.path.exists())

    def test_saved_token_is_used_for_login_then_removed_from_environment(self):
        credentials.save_token("saved-test-token", self.path)

        def login(**kwargs):
            self.assertEqual(os.environ["EARTHDATA_TOKEN"], "saved-test-token")
            self.assertEqual(kwargs, dict(strategy="environment", persist=False))
            return types.SimpleNamespace(authenticated=True)

        client = types.SimpleNamespace(login=login)
        self.assertIs(emit.authenticate_earthdata(client=client), client)
        self.assertNotIn("EARTHDATA_TOKEN", os.environ)

    def test_environment_token_has_precedence_over_saved_file(self):
        credentials.save_token("saved-test-token", self.path)
        os.environ["EARTHDATA_TOKEN"] = "environment-test-token"
        client = types.SimpleNamespace(login=lambda **kw: types.SimpleNamespace(authenticated=True))
        with patch.object(emit, "read_token") as read:
            emit.authenticate_earthdata(client=client)
            read.assert_not_called()
        self.assertEqual(os.environ["EARTHDATA_TOKEN"], "environment-test-token")

    def test_authentication_failure_restores_environment_and_redacts_secrets(self):
        credentials.save_token("saved-test-token", self.path)

        def login(**kwargs):
            raise RuntimeError("signed URL with saved-test-token")

        with self.assertRaises(emit.EmitDownloadError) as caught:
            emit.authenticate_earthdata(client=types.SimpleNamespace(login=login))
        self.assertNotIn("saved-test-token", str(caught.exception))
        self.assertNotIn("EARTHDATA_TOKEN", os.environ)

    def test_setup_and_check_never_echo_token(self):
        output = io.StringIO()
        with patch.object(configure_earthdata.getpass, "getpass", return_value="setup-test-token"), redirect_stdout(output):
            self.assertEqual(configure_earthdata.main([]), 0)
            self.assertEqual(configure_earthdata.main(["--check"]), 0)
        self.assertEqual(credentials.read_token(self.path), "setup-test-token")
        self.assertNotIn("setup-test-token", output.getvalue())


if __name__ == "__main__":
    unittest.main()
