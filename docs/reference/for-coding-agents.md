---
icon: lucide/bot
description: Dense, prescriptive reference for AI coding agents working on or integrating iscc-sct - architecture map, constraints, side effects, task recipes, change playbook, and common mistakes.
---

# For Coding Agents

A compressed reference for AI agents working on **iscc-sct** (ISCC Semantic Text-Code). Read this
before editing the codebase or integrating the library. Terminology matches the source exactly.

`iscc-sct` turns text into a cross-lingual, similarity-preserving binary ISCC-UNIT (MainType
SEMANTIC, SubType TEXT). It is an experimental proof of concept, **not** part of ISO 24138:2024.

## Architecture map

### File layout

| Path                                  | Contains                                                                                                                                                                                                                                              |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iscc_sct/main.py`                    | `create()` - high-level API; returns a `Metadata` object in Object-Format.                                                                                                                                                                            |
| `iscc_sct/code_semantic_text.py`      | Core algorithm. `gen_text_code_semantic()` (returns a plain dict, Index-Format), splitting, embedding, pooling, binarization. Module-level `@cache` singletons: `tokenizer()`, `chunking_tokenizer()`, `splitter()`, `splitter_guarded()`, `model()`. |
| `iscc_sct/models.py`                  | Pydantic schema: `Feature`, `FeatureSet`, `Metadata` + format converters.                                                                                                                                                                             |
| `iscc_sct/options.py`                 | `SctOptions` (pydantic-settings), `sct_opts` singleton, `.override()`.                                                                                                                                                                                |
| `iscc_sct/utils.py`                   | Codecs (base32/base64url), distances, model download + blake3 integrity, `char_to_byte_offsets`, `MODEL_PATH`, `timer`.                                                                                                                               |
| `iscc_sct/cli.py`                     | `iscc-sct` console entry point (`main()`): glob, charset detection, `gui` subcommand.                                                                                                                                                                 |
| `iscc_sct/demo.py`, `iscc_sct/app.py` | Gradio demo (Hugging Face Space). Omitted from coverage.                                                                                                                                                                                              |
| `iscc_sct/dev.py`                     | Dev-only poe task helpers. Omitted from coverage.                                                                                                                                                                                                     |
| `iscc_sct/tokenizer.json`             | Vendored tokenizer. Byte-exact; excluded from whitespace/EOL hooks.                                                                                                                                                                                   |
| `tests/chunking_vectors.json`         | Frozen chunk-boundary test vectors. Never hand-edit (see Change playbook).                                                                                                                                                                            |

### Pipeline

```text
text -> split_text() -> embed_chunks() -> mean_pooling() -> binarize() -> ISCC header + base32
```

`gen_text_code_semantic(text, **options)` orchestrates this:

1. Reject empty text (`ValueError`), then `sct_opts.override(options)`.
1. `split_text()` -> `[(offset, chunk), ...]` at semantic boundaries (max 127 tokens, 48 overlap).
1. `embed_chunks()` tokenizes (batch 100), runs the ONNX model, `attention_pooling()` per chunk.
1. `mean_pooling()` averages chunk vectors into one L2-normalized document vector.
1. `binarize()` (`vec >= 0` -> bits), truncate to `bits // 8`, prefix the 2-byte ISCC header,
    `encode_base32()`, prepend `"ISCC:"`.

### Import flow

```text
iscc_sct/__init__.py  (star-imports, defines __version__)
  -> options -> utils -> code_semantic_text -> models -> main
main            -> models, code_semantic_text, options
code_semantic_text -> onnxruntime (lazy import), semantic_text_splitter, tokenizers, numpy,
                       iscc_sct as sct  (uses sct.* at call time, not import time)
utils           -> models
cli             -> main, charset_normalizer
```

### Public API

`iscc_sct/__init__.py` re-exports every module's `__all__`. The public surface
(`import iscc_sct as sct`):

| Symbol                                                                          | Source             | Purpose                                          |
| ------------------------------------------------------------------------------- | ------------------ | ------------------------------------------------ |
| `create(text, granular=False, **options)`                                       | main               | Primary API. Returns `Metadata` (Object-Format). |
| `gen_text_code_semantic(text, **options)`                                       | code_semantic_text | Low-level. Returns `dict` (Index-Format).        |
| `code_text_semantic(fp, **options)`                                             | code_semantic_text | Same, reading a UTF-8 file path.                 |
| `soft_hash_text_semantic(text)`                                                 | code_semantic_text | Raw 384-bit digest (`bytes`), no header.         |
| `embed_chunks(chunks, batch_size=100)`                                          | code_semantic_text | Chunk list -> embedding array.                   |
| `Metadata`, `FeatureSet`, `Feature`                                             | models             | Result schema + converters.                      |
| `SctOptions`, `sct_opts`                                                        | options            | Settings model + global instance.                |
| `iscc_distance`, `hamming_distance`, `cosine_similarity`, `granular_similarity` | utils              | Similarity metrics.                              |
| `encode_base32`, `decode_base32`, `encode_base64`, `decode_base64`              | utils              | Codecs.                                          |
| `char_to_byte_offsets`, `get_model`, `MODEL_PATH`, `timer`                      | utils              | Helpers.                                         |
| `__version__`                                                                   | `__init__`         | `"0.2.0"`.                                       |

## Decision dispatch

### Which entry point?

| Goal                                                           | Use                                                     |
| -------------------------------------------------------------- | ------------------------------------------------------- |
| One ISCC code from a string, ready-to-use object               | `create(text)` -> `Metadata` (Object-Format)            |
| Granular per-chunk features (simprints/offsets/sizes/contents) | `create(text, granular=True)`                           |
| Compact parallel-array result for storage/indexing             | `gen_text_code_semantic(text, ...)` (Index-Format dict) |
| Process a text file from disk                                  | `code_text_semantic(path)`                              |
| Just the raw 384-bit vector digest                             | `soft_hash_text_semantic(text)` -> `bytes`              |
| Command line / batch over files                                | `iscc-sct <glob>`                                       |

### Which similarity metric?

| Inputs                                | Use                                       | Returns                                 |
| ------------------------------------- | ----------------------------------------- | --------------------------------------- |
| Two full ISCC code strings            | `iscc_distance(iscc1, iscc2)`             | Hamming distance in bits (`int`)        |
| Two raw digests of equal length       | `hamming_distance(a, b)`                  | bit distance                            |
| Two raw digests, normalized score     | `cosine_similarity(a, b)`                 | `int` in `[-100, 100]`                  |
| Two `Metadata` objects with simprints | `granular_similarity(a, b, threshold=80)` | `[(Feature, similarity, Feature), ...]` |

### Which feature format?

| Format        | Produced by                | Shape                                                    | Convert with                  |
| ------------- | -------------------------- | -------------------------------------------------------- | ----------------------------- |
| Index-Format  | `gen_text_code_semantic()` | parallel arrays `simprints`/`offsets`/`sizes`/`contents` | `Metadata.to_index_format()`  |
| Object-Format | `create()`                 | list of self-contained `Feature` objects                 | `Metadata.to_object_format()` |

## Constraints and invariants

- **No base ONNX runtime.** The base package declares no `onnxruntime`. Install exactly one of the
    mutually exclusive `cpu` / `gpu` extras. `onnxruntime` and `onnxruntime-gpu` unpack into the
    same directory and clobber each other (issue #23); `tool.uv.conflicts` enforces exclusivity for
    uv.
- **ONNX runtime is imported lazily.** `import iscc_sct`, the `iscc-sct` CLI, and `iscc-sct doctor`
    load without any runtime installed. `load_onnxruntime()` (called from `model()` on first code
    generation) raises a guarded `ImportError` with install instructions - keep that guard intact.
- **Codes are base32; simprints are base64url.** Never cross the codecs.
- **ISCC header is fixed:** MainType `SEMANTIC` (`0001`), SubType `TEXT` (`0000`), Version `0000`,
    plus a length nibble from `BIT_LEN_MAP`. 2 bytes, big-endian. `iscc_distance` strips the 5-char
    `ISCC:` prefix, base32-decodes, drops the 2-byte header, then compares bodies of equal length.
- **`bits` constraints:** `32 <= bits <= 256`, `multiple_of=32`. Same for `bits_granular`.
- **Default bit-length differs by entry point:** `SctOptions.bits` / `create()` default to **64**;
    the **CLI** `--bits` defaults to **256**.
- **`SctOptions` validates on assignment.** Mutate only via `.override(dict)`, which deep-copies and
    sets fields individually so validators run. Options flow as `**opts.model_dump()` through all
    layers. Env vars use the `ISCC_SCT_` prefix; `.env` is loaded.
- **Chunk boundaries are frozen** by `tests/chunking_vectors.json` and identical on both the normal
    (`splitter`) and guarded (`splitter_guarded`) paths. `needs_split_guard()` routes texts whose
    positions sit more than `SPLIT_GUARD_GAP` (8192) chars from the next paragraph separator to the
    guarded sizer (PDF-extracted text, issue #24).
- **Singletons:** `tokenizer()`, `chunking_tokenizer()`, `splitter(**opts)`,
    `splitter_guarded(**opts)`, `model()` are `@cache`d. The splitter cache keys on the option
    kwargs, so all option values must be hashable.
- **Two tokenizers:** `tokenizer()` keeps the vendored truncation (128) + padding and embeds chunks;
    `chunking_tokenizer()` disables truncation/padding and sizes chunks. Truncation on the sizer
    makes `tokenizers` >=0.23 emit overflow encodings, the root cause of the issue #24 super-linear
    chunking. Both yield identical boundaries, so don't merge them.
- **Coverage must stay at 100%** (`--cov-fail-under=100`). GPU branches and model download use
    `# pragma: no cover`. `dev.py`, `demo.py`, and `tests/` are omitted from coverage.
- **Style:** PEP 484 **type comments** (first line after `def`), PEP 585 generics, PEP 604 unions.
    Short pure functions, max 3 args, no nested functions, sphinx-style docstrings. Ruff line length
    100, LF endings, rule F401 disabled.

## Side effects catalog

| Function                                                                  | Effect                                                                                                                                                                               |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `model()` (first call)                                                    | If the model file is absent/corrupt, `get_model()` **downloads ~450 MB** to the platformdirs user-data dir and verifies a blake3 checksum. Creates a cached ONNX `InferenceSession`. |
| `get_model()`                                                             | Network download + disk write to `MODEL_PATH`; blake3 integrity check (`# pragma: no cover`).                                                                                        |
| `tokenizer()`, `chunking_tokenizer()`, `splitter()`, `splitter_guarded()` | Populate process-global `@cache` (loaded once).                                                                                                                                      |
| `model()`                                                                 | Calls `rt.preload_dlls()` when CUDA is available; logs a warning via `warn_gpu_shadowed()` if `onnxruntime-gpu` is installed but CUDA is missing.                                    |
| `import iscc_sct.options`                                                 | `load_dotenv()` reads `.env`; instantiates `sct_opts`.                                                                                                                               |
| `import iscc_sct.utils`                                                   | `os.makedirs(user_data_dir)` (idempotent).                                                                                                                                           |
| `create()`, `gen_text_code_semantic()`                                    | Pure given a loaded model: no disk writes, no mutation of inputs.                                                                                                                    |
| CLI `main()`                                                              | Reads files, prints to stdout, removes the loguru logger unless `--debug`.                                                                                                           |

## Task recipes

### Generate a code (library)

```python
import iscc_sct as sct

meta = sct.create("This is some sample text.", bits=256)
print(meta.iscc)  # "ISCC:CAD..."
print(meta.characters)  # input length in characters
```

### Granular per-chunk features

```python
import iscc_sct as sct

meta = sct.create(long_text, bits=256, granular=True)
for feature in meta.features[0].simprints:  # Object-Format
    print(feature.offset, feature.size, feature.simprint, feature.content[:40])
```

### Compare two texts (cross-lingual)

```python
import iscc_sct as sct

a = sct.create("An ISCC applies to a specific digital asset...")
b = sct.create("Ein ISCC bezieht sich auf ein bestimmtes digitales Gut...")
print(sct.iscc_distance(a.iscc, b.iscc))  # low bit distance => similar
```

### Match granular simprints

```python
import iscc_sct as sct

a = sct.create(doc_a, granular=True)
b = sct.create(doc_b, granular=True)
for feat_a, similarity, feat_b in sct.granular_similarity(a, b, threshold=80):
    print(similarity, feat_a.offset, feat_b.offset)
```

### Configure via options

```python
import iscc_sct as sct

# Per-call override (validated copy of the global settings):
meta = sct.create(text, bits=128, contents=True)

# Or globally via environment: ISCC_SCT_BITS=128, ISCC_SCT_MAX_TOKENS=127, ...
```

## Change playbook

| If you change...                                                                    | Also update...                                                                                                                                                         |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The embedding model file                                                            | `MODEL_FILENAME`, `MODEL_URL`, `MODEL_CHECKSUM` in `utils.py`; verify codes are bit-stable or bump version + mark BREAKING in CHANGELOG.                               |
| Chunking (`max_tokens`, `overlap`, `split_text`, `needs_split_guard`, token sizing) | Regenerate `tests/chunking_vectors.json` via `uv run python tests/test_chunking_vectors.py` (only for an intentional algorithm change); multi-chunk codes shift.       |
| A public function name/signature                                                    | The module's `__all__` (it is star-exported through `__init__.py`); README/docs examples.                                                                              |
| Add a processing option                                                             | Add a field to `SctOptions` (env var name, constraints); it flows via `model_dump()`. Wire into `create()` if it is a granular toggle; add to the README config table. |
| `Metadata` / `FeatureSet` / `Feature` schema                                        | Keep `to_index_format()`, `to_object_format()`, `get_content()`, `get_overlaps()` consistent.                                                                          |
| ISCC header constants (`MAINTYPE`/`SUBTYPE`/`SCT_VERSION`/`BIT_LEN_MAP`)            | Codes change - this is BREAKING; bump version and document.                                                                                                            |
| onnxruntime import / provider logic                                                 | Keep the `cpu`/`gpu` extras, `tool.uv.conflicts`, the `ImportError` guard, and `warn_gpu_shadowed()` in sync.                                                          |
| Any code path                                                                       | Add tests to keep coverage at 100%; use `# pragma: no cover` only for GPU/download branches. Run `uv run poe all`.                                                     |
| Dependencies                                                                        | Regenerate `requirements.txt` via `uv run poe export-requirements` (never hand-edit it).                                                                               |

## Common mistakes

**NEVER** add `onnxruntime` as a base dependency to "make install easier."

```toml
# WRONG - re-breaks issue #23: [gpu] then installs both wheels and CPU clobbers GPU
dependencies = ["onnxruntime"]
```

**ALWAYS** keep it behind the mutually exclusive `cpu` / `gpu` extras.

---

**NEVER** use inline type annotations.

```python
def binarize(vec: NDArray) -> bytes: ...  # WRONG for this codebase
```

**ALWAYS** use PEP 484 type comments as the first line after `def`.

```python
def binarize(vec):
    # type: (NDArray) -> bytes
    ...
```

---

**NEVER** expect Object-Format from the low-level function.

```python
meta = sct.gen_text_code_semantic(text)  # returns a dict in INDEX-Format
meta.features[0].simprints  # WRONG - it is a dict, not Metadata
```

**ALWAYS** use `create()` for an Object-Format `Metadata`, or wrap the dict: `Metadata(**data)`.

---

**NEVER** mutate `SctOptions` fields directly or assume CLI/library defaults match.

```python
sct.sct_opts.bits = 256  # avoid - mutates the global; use .override()
sct.create(text)  # library default bits=64, NOT the CLI's 256
```

**ALWAYS** pass overrides per call (`create(text, bits=256)`) or via `sct_opts.override({...})`.

---

**NEVER** hand-edit `tests/chunking_vectors.json`, `requirements.txt`, `iscc_sct/tokenizer.json`, or
the model file. Each is generated or byte-exact.

**ALWAYS** regenerate via the documented command and run `uv run poe all` before reporting done.

---

**NEVER** mix codecs: ISCC codes are RFC4648 base32 (no padding); granular simprints are base64url
(no padding). Decode with the matching `decode_base32` / `decode_base64`.
