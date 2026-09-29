# Source release scope

The release was assembled from an explicit source-file allowlist. Original Git
history, remotes, manuscript files, local notes, credentials, cluster scripts,
generated responses, experiment outputs, and model/tokenizer binaries were
excluded.

Personal credential aliases and hardcoded cloud configuration were replaced
with public project labels or environment settings.
Standalone script imports were corrected where needed for the documented
commands. Third-party baseline and evaluator source is not included.

The controller, gate equations, distribution reconstruction, and optimization
objectives are preserved. The portable count-recovery helper exposes the
existing job-script conversion. Tests that previously required local benchmark
files now use temporary synthetic records.

`docs/source_export.json` lists adjusted source files. `MANIFEST.sha256` covers
the distributable files. Run `python scripts/audit_release.py` to check syntax,
credentials, local paths, binary artifacts, and manifest integrity. The optional
`--anonymous` mode additionally rejects email addresses and configured remotes.
To add identifiers specific to your environment, use repeatable `--deny`
arguments; the scanner prints finding locations, not matched values. After
intentionally editing a release, use `--write-manifest` to refresh its checksums.

Generated source archives contain no `.git` directory, caches, or file-owner
metadata. The public branch starts from a clean release commit so removed files
are not present in its reachable history.
