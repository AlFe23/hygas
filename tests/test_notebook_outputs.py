"""Offline tests of staged-versus-local notebook maintenance using temporary Git repos."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import notebook_outputs as tool


class NotebookOutputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hygas-notebook-policy-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "--initial-branch=develop")
        self.git("config", "user.name", "Notebook policy test")
        self.git("config", "user.email", "notebook-test@example.invalid")
        self.nb = dict(nbformat=4, nbformat_minor=5,
                       metadata={"language_info": {"name": "python"}, "custom": [1, 2]},
                       cells=[dict(cell_type="markdown", id="intro", source=["# Results — CH4\n"],
                                   metadata={"preserve": True}, attachments={"plot.png": {"image/png": "example"}}),
                              dict(cell_type="code", source=["print('CH4')\n"],
                                   metadata={"collapsed": True, "execution": {"timestamp": "preserve"}},
                                   outputs=[], execution_count=None)])
        self.path = self.root / "notebooks/example.ipynb"
        self.write(self.path, self.nb)
        self.git("add", "--", "notebooks/example.ipynb")
        self.git("commit", "-m", "Initial clean notebook")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.PIPE)

    def write(self, path, notebook):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def execute_locally(self):
        nb = copy.deepcopy(self.nb)
        nb["cells"][1].update(outputs=[dict(output_type="stream", name="stdout", text=["CH4\n"])], execution_count=7)
        self.write(self.path, nb)
        return nb

    def run_tool(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = tool.main(list(args), repo=self.root)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_check_uses_index_and_preserves_local_results(self):
        self.execute_locally()
        original = self.path.read_bytes()
        code, out, _ = self.run_tool("check")
        self.assertEqual(code, 0)
        self.assertIn("Git index", out)
        self.assertEqual(self.run_tool("check", "--working-tree")[0], 1)
        self.assertEqual(self.path.read_bytes(), original)

    def test_clean_requires_manual_restaging(self):
        self.execute_locally()
        self.git("add", "--", "notebooks/example.ipynb")
        staged = self.git("show", ":notebooks/example.ipynb")
        self.assertEqual(self.run_tool("check")[0], 1)
        self.assertEqual(self.run_tool("clean")[0], 0)
        self.assertEqual(self.git("show", ":notebooks/example.ipynb"), staged)
        self.assertEqual(self.run_tool("check")[0], 1)
        self.assertEqual(self.run_tool("check", "--working-tree")[0], 0)
        self.git("add", "--", "notebooks/example.ipynb")
        self.assertEqual(self.run_tool("check")[0], 0)

    def test_clean_preserves_every_other_field_and_is_idempotent(self):
        original = self.execute_locally()
        self.assertEqual(self.run_tool("clean")[0], 0)
        expected = copy.deepcopy(original)
        expected["cells"][1].update(outputs=[], execution_count=None)
        self.assertEqual(json.loads(self.path.read_bytes()), expected)
        self.assertNotIn("id", json.loads(self.path.read_bytes())["cells"][1])
        once = self.path.read_bytes()
        mtime = self.path.stat().st_mtime_ns
        self.assertEqual(self.run_tool("clean")[0], 0)
        self.assertEqual(self.path.read_bytes(), once)
        self.assertEqual(self.path.stat().st_mtime_ns, mtime)

    def test_untracked_notebooks_are_untouched_unless_explicitly_selected(self):
        dirty = self.execute_locally()
        untracked = self.root / "notebooks/private experiment.ipynb"
        self.write(untracked, dirty)
        original = untracked.read_bytes()
        self.assertEqual(self.run_tool("clean")[0], 0)
        self.assertEqual(untracked.read_bytes(), original)
        self.assertEqual(self.run_tool("clean", "notebooks/private experiment.ipynb")[0], 0)
        self.assertEqual(json.loads(untracked.read_bytes())["cells"][1]["outputs"], [])
        self.assertIn("notebooks/private experiment.ipynb", os.fsdecode(self.git("ls-files", "--others")))
        self.assertEqual(self.run_tool("check", "notebooks/private experiment.ipynb")[0], 2)

    def test_staged_addition_and_deletion_follow_the_index(self):
        extra = self.root / "notebooks/added.ipynb"
        self.write(extra, self.execute_locally())
        self.git("add", "--", "notebooks/added.ipynb")
        self.assertEqual(self.run_tool("check")[0], 1)
        self.git("rm", "--cached", "--", "notebooks/added.ipynb")
        self.assertEqual(self.run_tool("check")[0], 0)

    def test_count_without_outputs_is_rejected_and_cleared(self):
        nb = copy.deepcopy(self.nb)
        nb["cells"][1]["execution_count"] = 4
        self.write(self.path, nb)
        self.assertEqual(self.run_tool("check", "--working-tree")[0], 1)
        self.assertEqual(self.run_tool("clean")[0], 0)
        self.assertIsNone(json.loads(self.path.read_bytes())["cells"][1]["execution_count"])

    def test_uppercase_notebook_extension_is_checked(self):
        upper = self.root / "notebooks/UPPER.IPYNB"
        self.write(upper, self.execute_locally())
        self.git("add", "--", "notebooks/UPPER.IPYNB")
        self.assertEqual(self.run_tool("check")[0], 1)
        self.assertEqual(self.run_tool("clean", "notebooks/UPPER.IPYNB")[0], 0)
        self.assertEqual(json.loads(upper.read_bytes())["cells"][1]["outputs"], [])

    def test_malformed_json_prevents_any_cleanup(self):
        self.execute_locally()
        original = self.path.read_bytes()
        invalid = self.root / "notebooks/z_invalid.ipynb"
        invalid.write_text("{invalid JSON")
        self.git("add", "--", "notebooks/z_invalid.ipynb")
        self.assertEqual(self.run_tool("check")[0], 2)
        self.assertEqual(self.run_tool("clean")[0], 2)
        self.assertEqual(self.path.read_bytes(), original)

    def test_explicit_selection_does_not_clean_other_notebooks(self):
        dirty = self.execute_locally()
        other = self.root / "notebooks/other/result.ipynb"
        self.write(other, dirty)
        self.git("add", "--", "notebooks/other/result.ipynb")
        original = self.path.read_bytes()
        self.assertEqual(self.run_tool("clean", "notebooks/other")[0], 0)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(json.loads(other.read_bytes())["cells"][1]["outputs"], [])

    def test_publication_branches_and_detached_head_are_read_only(self):
        self.execute_locally()
        original = self.path.read_bytes()
        for branch in ("mdpi_methane_paper", "paper/example", "publication/example", "release/example"):
            with self.subTest(branch=branch):
                self.git("switch", "-c", branch)
                self.assertEqual(self.run_tool("clean")[0], 2)
                self.assertEqual(self.path.read_bytes(), original)
        self.git("checkout", "--detach")
        self.assertEqual(self.run_tool("clean")[0], 2)
        self.assertEqual(self.run_tool("check", "--working-tree")[0], 1)
        self.assertEqual(self.path.read_bytes(), original)

    def test_paths_outside_repository_are_rejected(self):
        self.assertEqual(self.run_tool("clean", "../outside.ipynb")[0], 2)

    def test_notebook_symlinks_are_rejected(self):
        link = self.root / "notebooks/link.ipynb"
        try:
            link.symlink_to(self.path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")
        self.assertEqual(self.run_tool("clean", "notebooks/link.ipynb")[0], 2)
        self.git("add", "--", "notebooks/link.ipynb")
        self.assertEqual(self.run_tool("check")[0], 2)

    def test_replacement_rejects_concurrent_edits(self):
        original = self.path.read_bytes()
        self.path.write_bytes(original + b"\n")
        changed = self.path.read_bytes()
        with self.assertRaises(RuntimeError):
            tool.replace_notebook(self.path, original, self.nb)
        self.assertEqual(self.path.read_bytes(), changed)

    def test_standalone_cli_uses_current_repository(self):
        result = subprocess.run([sys.executable, str(Path(tool.__file__).resolve()), "check"],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 notebooks in the Git index", result.stdout)


if __name__ == "__main__":
    unittest.main()
