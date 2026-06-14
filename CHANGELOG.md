# Changelog

## [0.2.0] - 2026-06-14

- Optimized the ONNX embedding model (`iscc-sct-v0.2.0.onnx`): the transformer graph is now fused
    offline via `onnxruntime.transformers.optimizer` (attention, embedding layer norm, skip layer
    norm, bias GELU). CPU inference is 1.25x-1.6x faster depending on hardware; generated ISCC codes
    are unchanged (validated bit-identical at 256 bits across AVX2, AVX512-VNNI and CUDA on 1900+
    chunks). The model weights are identical to v0.1.0 — only the graph structure changed
- Fixed super-linear `split_text` runtime on texts without regular paragraph breaks (typical for
    PDF-extracted text): such inputs now chunk via a guarded Python token sizer that skips
    tokenizing huge splitter probes (114s → 7s for a 600KB book, #24). Chunk boundaries are
    unchanged and now frozen by test vectors in `tests/chunking_vectors.json`
- **BREAKING**: `onnxruntime` is no longer a base dependency — install `iscc-sct[cpu]` or
    `iscc-sct[gpu]` instead. Previously the `[gpu]` extra was a silent no-op because the
    unconditional CPU package shadowed `onnxruntime-gpu` (#23)
- **BREAKING**: dropped Python 3.10 support and added Python 3.14 — `requires-python` is now
    `>=3.11`. `onnxruntime` stopped shipping 3.10 wheels (last was 1.23.2) and added 3.14 wheels, so
    the supported range tracks the runtime
- **BREAKING**: renamed the CLI command from `sct` to `iscc-sct` (package name = CLI command is the
    standard across all iscc projects); enables `uvx "iscc-sct[cpu]" <file>` one-liners
- Added an `iscc-sct doctor` command that diagnoses the ONNX runtime (missing, or a CPU package
    shadowing the GPU build), recommends the right `cpu`/`gpu` extra, and installs it on
    confirmation
- Import `onnxruntime` lazily on first model use, so `import iscc_sct` and the `iscc-sct` CLI load
    without a runtime installed; the instructive `ImportError` now fires on first code generation
- Warn at runtime when `onnxruntime-gpu` is installed but shadowed by the CPU package
- Call `onnxruntime.preload_dlls()` before CUDA session creation so pip-provided NVIDIA libraries
    are found without a system-wide CUDA install
- Migrated project tooling from Poetry to [uv](https://docs.astral.sh/uv/) (uv_build backend)
- Generate `requirements.txt` from `uv.lock` via `poe export-requirements`
- Include LICENSE file in built distributions via `license-files`
- CI: pin uv version, enforce `uv sync --locked`, verify `requirements.txt` matches `uv.lock`
- CI: derive ONNX model cache directory from `iscc_sct.utils.MODEL_PATH`
- CI: test matrix now covers Python 3.11–3.14 (dropped 3.10, added 3.14)
- CI: added a release workflow that re-runs the full test matrix and publishes to PyPI on a
    published GitHub Release, gated by a tag/version guard
- Updated Hugging Face Space to Gradio 5.26.0 (matches locked version)
- Fixed `format_yml` glob pattern that only worked on Windows
- Updated dependencies (Gradio 6, pytest 9, pytest-cov 7, coverage 7.14)
- Fixed CLI subprocess coverage measurement for pytest-cov 7 via coverage `patch = ["subprocess"]`
- Adapted Gradio demo to Gradio 6: `theme` and `css` are now passed to `launch()`
- Added [prek](https://github.com/j178/prek)-based pre-commit hooks (file hygiene checks, ruff
    format/lint, mdformat)
- Added a Zensical documentation site deployed to GitHub Pages at
    [sct.iscc.codes](https://sct.iscc.codes), including a *For Coding Agents* reference page and
    `llms.txt`/`llms-full.txt` for machine consumption

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
