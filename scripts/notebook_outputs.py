#!/usr/bin/env python3
"""Check staged notebooks or explicitly clear working-copy execution outputs.

Python standard library only. No notebook execution, Git staging or commits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PUBLICATION_BRANCHES = {"mdpi_methane_paper"}
PUBLICATION_PREFIXES = ("paper/", "publication/", "release/")


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.PIPE)


def indexed_notebooks(root: Path) -> dict[str, tuple[str, str]]:
    """Index paths -> (mode, blob ID), including staged additions/deletions."""
    result = {}
    for entry in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        info, raw_path = entry.split(b"\t", 1)
        path = os.fsdecode(raw_path)
        if not path.lower().endswith(".ipynb"):
            continue
        mode, oid, stage = info.decode("ascii").split()
        if stage != "0":
            raise ValueError(f"Resolve the notebook merge conflict first: {path}")
        result[path] = (mode, oid)
    return result


def select_paths(root: Path, indexed: dict, requested: list[str], *, working: bool) -> list[str]:
    """Defaults to tracked notebooks; explicit new files are allowed in working mode."""
    if not requested:
        return sorted(indexed)
    selected = set()
    for spec in requested:
        candidate = Path(spec).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_symlink():
            raise ValueError(f"Notebook symlinks are not supported: {spec}")
        try:
            relative = candidate.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Path must stay inside the repository: {spec}") from exc
        matches = {p for p in indexed if relative == "." or p == relative or p.startswith(relative + "/")}
        if working and candidate.is_file() and candidate.suffix.lower() == ".ipynb":
            matches.add(relative)
        if not matches:
            raise ValueError(f"No selected notebooks for {spec!r}; stage new files before an index check.")
        selected.update(matches)
    return sorted(selected)


def load_notebook(raw: bytes, name: str) -> dict:
    notebook = json.loads(raw)
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"Expected a Jupyter notebook in nbformat 4: {name}")
    for cell in notebook["cells"]:
        if not isinstance(cell, dict) or "cell_type" not in cell:
            raise ValueError(f"Malformed notebook cell: {name}")
        if cell["cell_type"] == "code" and (not isinstance(cell.get("outputs"), list) or "execution_count" not in cell):
            raise ValueError(f"Code cells require outputs and execution_count fields: {name}")
    return notebook


def execution_fields(notebook: dict) -> tuple[int, int]:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    return (sum(bool(c["outputs"]) for c in code),
            sum(c["execution_count"] is not None for c in code))


def working_bytes(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Notebook must be a regular file inside the repository: {name}")
    return path.read_bytes()


def require_development_branch(root: Path) -> None:
    branch = git(root, "branch", "--show-current").decode().strip()
    if not branch or branch in PUBLICATION_BRANCHES or branch.startswith(PUBLICATION_PREFIXES):
        raise ValueError("Cleaning is disabled on publication branches and detached HEADs; use a development branch.")


def replace_notebook(path: Path, original: bytes, notebook: dict) -> None:
    """Replace only a changed working file, preserving its mode and all other fields."""
    if path.read_bytes() != original:
        raise RuntimeError(f"Notebook changed during cleanup: {path}")
    cleaned = (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".notebook-clean-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(cleaned)
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None, *, repo: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check index blobs by default; do not modify files.")
    check.add_argument("--working-tree", action="store_true", help="Inspect working files instead of staged/index versions.")
    check.add_argument("paths", nargs="*", help="Optional repository-relative notebook files/directories.")
    clean = commands.add_parser("clean", help="Clear working files only; manually git add them afterwards.")
    clean.add_argument("paths", nargs="*", help="Optional files/directories; default tracked notebooks only.")
    args = parser.parse_args(argv)
    try:
        root = Path(git(repo or Path.cwd(), "rev-parse", "--show-toplevel").decode().strip()).resolve()
        working = args.command == "clean" or args.working_tree
        if args.command == "clean":
            require_development_branch(root)
        indexed = indexed_notebooks(root)
        paths = select_paths(root, indexed, args.paths, working=working)
        prepared = []
        # Read/validate every selected file before changing the first one.
        for name in paths:
            if name in indexed and indexed[name][0] not in ("100644", "100755"):
                raise ValueError(f"Notebook must be a regular Git file: {name}")
            raw = working_bytes(root, name) if working else git(root, "cat-file", "blob", indexed[name][1])
            nb = load_notebook(raw, name)
            prepared.append((name, raw, nb, execution_fields(nb)))
        problems = [(name, counts) for name, _, _, counts in prepared if any(counts)]
        if args.command == "check":
            for name, (outputs, counts) in problems:
                print(f"{name}: {outputs} cells with outputs; {counts} non-null execution counts")
            scope = "working copy" if working else "Git index"
            if problems:
                print(f"FAIL: {len(problems)} of {len(paths)} notebooks in the {scope} contain execution results.")
                print("Clean the affected working files, then git add them again before committing.")
                return 1
            print(f"OK: {len(paths)} notebooks in the {scope}; outputs empty and execution counts null.")
            return 0
        for name, raw, nb, counts in prepared:
            if not any(counts):
                continue
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            replace_notebook(root / name, raw, nb)
            print(f"Cleaned: {name}")
        print(f"Cleaned {len(problems)} of {len(paths)} working notebooks; Git index unchanged. Review and git add explicitly.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        message = exc.stderr.decode(errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
