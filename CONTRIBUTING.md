# Contributing to HyGAS

Start new development branches from an updated `develop`. Keep unrelated case
studies and experiments out of a retrieval or maintenance commit.

## Notebook outputs on development branches

Notebook files committed to `develop`, `main` and feature/maintenance branches
must have empty code-cell `outputs` arrays and `execution_count: null`.
Keep code, Markdown explanations, attachments, cell IDs and existing metadata.
Scientific figures embedded intentionally in Markdown are retained.

You can execute notebooks and keep their results in your **local working copy**.
Before committing, explicitly clean the notebooks you intend to publish. From
the repository root, using Python 3.11+ and Git:

```bash
# One notebook; only its working file is modified.
python scripts/notebook_outputs.py clean notebooks/example.ipynb

# Review and stage the cleaned version yourself.
git diff -- notebooks/example.ipynb
git add -- notebooks/example.ipynb

# Check all notebook versions currently prepared for the commit.
python scripts/notebook_outputs.py check
```

The tool needs only Python's standard library. It does not run notebook cells,
stage files, create commits or download data. Cleaning removes saved outputs
from the working file; export or copy any results you want to retain first.

With no paths, `clean` processes all tracked notebooks. An explicitly named
new/untracked notebook can also be cleaned. Untracked and ignored experiments
are not discovered automatically. Directory arguments select tracked notebooks
under that directory. Already clean files stay byte-identical. Invalid notebooks
are reported before any selected file is changed.

`check` reads the **Git index**, including staged additions/deletions. Local
results that have not been staged do not cause a failure. If you clean a file
after staging it, run `git add` again: cleaning does not replace the staged copy.
For an optional inspection of local working files:

```bash
python scripts/notebook_outputs.py check --working-tree
```

Exit codes: `0` means clean/success, `1` means saved execution results were found
by `check`, and `2` means an invalid selection, malformed notebook or Git error.
The tool supports nbformat 4 notebooks, preserves all other JSON fields and
refuses notebook symlinks. Paths are relative to the repository root.

## Automated check

[Notebook outputs](.github/workflows/notebook-outputs.yml) runs on pushes to
`develop`/`main` and pull requests **targeting** either branch, regardless of the
source branch name. It checks every indexed notebook, including existing ones,
and runs the maintenance tool's offline tests. It requires no scientific Python
environment, GDAL, satellite inputs or NASA credentials.

The workflow reports failures; it does not edit files or push corrective commits.
To enforce a passing result before merging, configure branch protection or a
ruleset for `develop` and `main` requiring the **Notebook outputs** check and
restrict direct pushes as appropriate for the team. A failed check on a direct
push reports a problem after that push; it does not undo it. These repository
settings are separate from committing the workflow.

Run the same checks locally:

```bash
python -m unittest discover -s tests -p 'test_notebook_outputs.py'
python scripts/notebook_outputs.py check
```

## Frozen publication versions

Executed notebooks with reviewed results belong in publication snapshots.
The existing `mdpi_methane_paper` branch and `mdpi-methane-paper-v1.0` release tag
preserve the published methane work. Keep those references on their recorded
commits; make later scientific developments on new branches.

For a future publication:

1. Start a dedicated branch named `paper/<name>` from the selected code commit.
2. Execute the notebooks against the documented inputs and environment. Review
   results, figures and warnings; retain the intended outputs in that snapshot.
3. Commit the reviewed notebooks with data identifiers, configuration and
   dependency/provenance information sufficient to interpret the results.
4. Create a versioned tag and GitHub release. Cite that tag/commit for the paper.
   Self-contained HTML/PDF exports can be attached to make results easy to read.

The output workflow does not run on pushes to publication branches or tags, or
on PRs targeting publication branches. A PR from a publication branch **into
`develop` or `main` is checked**, because the destination follows the clean-output
policy. Transfer reusable code back through a development branch with cleaned
notebooks, preserving the publication snapshot.

As an additional local guard, `clean` refuses `mdpi_methane_paper`, branches
starting with `paper/`, `publication/` or `release/`, and detached HEADs (including
checked-out release tags). This convention is not a GitHub permission rule:
maintainers must still preserve/protect the published refs themselves.
