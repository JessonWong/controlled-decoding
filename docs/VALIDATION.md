# Release validation

Validation used CPU operations, temporary synthetic data, and mocked API
clients. Hugging Face offline flags were enabled. No target API generation,
paid judging, model download, or GPU experiment was run.

## Completed checks

* Retained regression suite: **299 passed**, with **62 subtests passed** and no
  skips. Removed baseline, evaluator, and historical targeted-audit tests are
  not included in this count.
* A synthetic pipeline recovered exact counts from an FP16 floor cache,
  rejected an existing output directory, materialized a uniform Dirichlet
  cache, trained a small BiasNet on CPU, and reloaded the saved checkpoint.
  The saved prior mode and strength were checked.
* The project built into a wheel and installed successfully with existing
  dependencies (`pip install --no-deps --no-build-isolation --target ... .`).
  This checks package construction, not a fresh resolution of every dependency.
* The retained command examples and package entrypoints were smoke-tested with
  `--help`. The three documented `prefix_risk` module entrypoints also passed
  `--help` from the installed wheel outside the checkout.
* Python sources were parsed using the Python 3.10 grammar. The release scan
  checks credential patterns, private identifiers, personal paths, unexpected
  binary artifacts, and symlinks; anonymous mode also checks identity markers
  and remotes. The final manifest records release checksums.

The test run emitted existing PyTorch AMP deprecation warnings. They did not
cause test failures. Provider availability and numerical reproduction of
manuscript results are outside these release checks.

## Validation environment

The available Python environment was reused for these checks, with test/build
tools added in a separate temporary directory. These are observed versions,
not an exhaustive lockfile or a requirement to use the same CUDA build:

| Component | Version |
| --- | --- |
| Python | 3.13.7 |
| PyTorch | 2.11.0+cu126 |
| Transformers | 5.7.0 |
| Datasets | 5.0.1 |
| NumPy | 2.4.3 |
| Accelerate | 1.14.0 |
| Google Gen AI | 2.17.0 |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.9 |
| pytest | 9.1.1 |

To rerun the regression suite from a source checkout:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
python scripts/audit_release.py
```
