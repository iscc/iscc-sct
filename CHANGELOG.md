# Changelog

## [0.2.0] - Unreleased

- Migrated project tooling from Poetry to [uv](https://docs.astral.sh/uv/) (uv_build backend)
- Generate `requirements.txt` from `uv.lock` via `poe export-requirements`
- Include LICENSE file in built distributions via `license-files`
- CI: pin uv version, enforce `uv sync --locked`, verify `requirements.txt` matches `uv.lock`
- CI: derive ONNX model cache directory from `iscc_sct.utils.MODEL_PATH`
- Updated Hugging Face Space to Gradio 5.26.0 (matches locked version)
- Fixed `format_yml` glob pattern that only worked on Windows
- Updated dependencies (Gradio 6, pytest 9, pytest-cov 7, coverage 7.14)
- Fixed CLI subprocess coverage measurement for pytest-cov 7 via coverage `patch = ["subprocess"]`
- Adapted Gradio demo to Gradio 6: `theme` and `css` are now passed to `launch()`
- Added [prek](https://github.com/j178/prek)-based pre-commit hooks (file hygiene checks, ruff
  format/lint, mdformat)

## [0.1.4] - 2025-04-24

- Added `bytes_offsets` option to generate UTF-8 byte positions instead of character positions
- Updated dependencies

## [0.1.3] - 2025-04-02

- Update license, dependencies, and project metadata.
- Update dependencies to the latest versions and workflow configurations (Poetry and GitHub
  Actions).
- Update CLI tests to use dynamic SCT command execution.
- Add UTF-32BE chunk retrieval test for semantic text code.
- Fix Python 3.13 support by removing Python 3.9 compatibility and updating version constraints.
- Enhance documentation in demo.py.

## [0.1.2] - 2024-08-19

- Encode granular features with base64
- Refactor result format to generic ISCC data model
- Add optional gradio GUI demo

## [0.1.1] - 2024-06-25

- Handle text decoding errors gracefully
- Handle feature bit-lengths independently
- Improve model load time
- Improve memory use with batched embedding

## [0.1.0] - 2024-06-25

- Initial pre-release
