# External software and data

## JULI-derived code

`modeling_biasnet.py` and `training/train_biasnet.py` are modified from the
BiasNet implementation in [JULI v1.0.0](https://github.com/JessonWong/JULI/tree/04e32b840a5d3a39140034aedf0e122c37ac3819),
commit `04e32b840a5d3a39140034aedf0e122c37ac3819`, "Jailbreak Large
Language Models by Self-Introspection." JULI and these modified files are
licensed under the Apache License 2.0. The required attribution and modification
notice is retained in [NOTICE](NOTICE) and in the source-file headers.

The controlled-decoding version adds support for reconstructed sampled
distributions, count-aware inputs, prefix conditioning, expanded training
metadata, and the current checkpoint and projection modes.

## Non-vendored dependencies and artifacts

This repository does not vendor baseline implementations, evaluator code or
judge prompts, benchmark corpora, model or tokenizer snapshots, trained
weights, or generated responses. Python packages installed through
`pyproject.toml`, hosted APIs, external datasets, and downloaded model artifacts
remain subject to their providers' licenses and terms.
