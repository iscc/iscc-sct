# Changelog

## [0.2.2] - Unreleased

- Changed the default inference batch size from a fixed 100 to auto (`batch_size=0`), which resolves
    to 1 chunk per batch on CPU and 100 on CUDA (#25). The vendored tokenizer pads each batch to its
    longest chunk and attention cost is quadratic in sequence length, so on CPU large batches mostly
    pay for padding. Measured on a 2238-chunk book: ~1.4x faster and 1529 MB → 1110 MB steady RSS
    single-process; on a 12-worker pool 1.45x throughput and 18.5 GB → 13.4 GB total PSS. Generated
    ISCC codes are unchanged (verified bit-identical at 256 bits across ~50 configurations, and
    identical between the CPU and CUDA providers). A GPU scales the opposite way, which is why the
    default is provider-aware rather than one global value: on a GTX 1080, throughput rises from 384
    chunks/s at batch 1 to 668 at batch 100 and then plateaus (batch 200 is no faster and nearly
    doubles VRAM), so the CUDA path keeps batch 100. Host RSS on the GPU path is unaffected by batch
    size — those activations live in VRAM
- Added the `batch_size` option (`ISCC_SCT_BATCH_SIZE`) to override the batch size per call or
    globally, and `intra_op_threads` (`ISCC_SCT_INTRA_OP_THREADS`) to set the ONNX Runtime intra-op
    thread count. Both default to 0, meaning auto. `intra_op_threads` configures the process-wide
    inference session, so it only takes effect through the global options before the session is
    created — worker pools with one process per core should set it to 1 to avoid thread
    oversubscription (measured 1.7x higher total throughput on a 12-worker pool)
- The cached text splitters are now keyed only on the chunking options (`max_tokens`, `overlap`,
    `trim`). Previously every distinct combination of unrelated per-call options (e.g. `bits`, or
    the batch size added above) retained an additional tokenizer-backed `TextSplitter` instance

## [0.2.1] - 2026-06-16

- Disabled truncation on the tokenizer used for chunk sizing (new `chunking_tokenizer()`, separate
    from the embedding `tokenizer()`). The vendored tokenizer truncates to 128 tokens, which made
    the `tokenizers` >=0.23 chunk sizer emit one overflow encoding per 128 tokens, so sizing a large
    probe string cost O(length) — the dominant cause of the issue #24 super-linear chunking. Sizing
    the full text yields identical token counts and therefore identical chunk boundaries (verified
    against `tests/chunking_vectors.json`) and unchanged ISCC codes, while further reducing the
    guarded worst-case chunking time for PDF-extracted text. The token sizer is simplified
    accordingly (no overflow summing; `count_nonpad_ids` removed)
- Updated `semantic-text-splitter` to `>=0.32.0`, which adds the upstream "avoid sizing whole
    distant split sections" fix (benbrandt/text-splitter#1184) and releases the GIL during native
    chunking. The upstream fix probes lower-level semantic boundaries, so it speeds up texts with
    distant but present separators; it does **not** cover spans with no intermediate separator at
    all (the issue #24 PDF shape and spaceless CJK), where native chunking stays super-linear, so
    the chunking guard is retained

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
