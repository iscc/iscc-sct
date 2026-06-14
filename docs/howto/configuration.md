---
icon: lucide/settings
description: Configure iscc-sct with per-call options, environment variables, or a global settings override.
---

# Configuration

This guide shows how to control code length, chunking, and output detail through options — per call,
through environment variables, or globally.

Every option has the same name everywhere: as a keyword argument, as an `ISCC_SCT_*` environment
variable, and as a field on the `SctOptions` model.

## Options reference

| Option          | Env variable             | Default | Notes                                                   |
| --------------- | ------------------------ | ------- | ------------------------------------------------------- |
| `bits`          | `ISCC_SCT_BITS`          | `64`    | Document code length. 32–256, multiple of 32.           |
| `bits_granular` | `ISCC_SCT_BITS_GRANULAR` | `64`    | Granular simprint length. 32–256, multiple of 32.       |
| `characters`    | `ISCC_SCT_CHARACTERS`    | `True`  | Include the document character count.                   |
| `embedding`     | `ISCC_SCT_EMBEDDING`     | `False` | Include the global document embedding vector.           |
| `precision`     | `ISCC_SCT_PRECISION`     | `8`     | Max fractional digits when storing the embedding.       |
| `simprints`     | `ISCC_SCT_SIMPRINTS`     | `False` | Include granular per-chunk simprints.                   |
| `offsets`       | `ISCC_SCT_OFFSETS`       | `False` | Include per-chunk offsets.                              |
| `byte_offsets`  | `ISCC_SCT_BYTE_OFFSETS`  | `False` | Report UTF-8 byte offsets instead of character offsets. |
| `sizes`         | `ISCC_SCT_SIZES`         | `False` | Include per-chunk sizes.                                |
| `contents`      | `ISCC_SCT_CONTENTS`      | `False` | Include the per-chunk text.                             |
| `max_tokens`    | `ISCC_SCT_MAX_TOKENS`    | `127`   | Max tokens per chunk. Cannot exceed 127.                |
| `overlap`       | `ISCC_SCT_OVERLAP`       | `48`    | Max tokens shared between adjacent chunks.              |
| `trim`          | `ISCC_SCT_TRIM`          | `False` | Trim whitespace from chunks.                            |

The `granular=True` shortcut on `create()` is equivalent to setting `simprints`, `offsets`, `sizes`,
and `contents` to `True` at once.

## Override per call

Pass options as keyword arguments to `create()`. They apply to that call only:

```python
import iscc_sct as sct

text = "This is some sample text. It can be a longer document or even an entire book."
meta = sct.create(text, bits=128, simprints=True, contents=True)
print(meta.iscc)
# ISCC:CABV3GG6JH3XEVRNSVYGCLJ7AAV3A
```

## Set defaults with environment variables

Set any option through its `ISCC_SCT_*` variable. This changes the default for every call in the
process:

```bash
export ISCC_SCT_BITS=128
export ISCC_SCT_MAX_TOKENS=100
```

You can also place these in a `.env` file in the working directory — `iscc-sct` loads it
automatically on import:

```ini
# .env
ISCC_SCT_BITS=128
ISCC_SCT_OVERLAP=24
```

## Override the global settings

`sct_opts` is the global settings instance. Use `override()` to get a validated copy with some
fields changed, without mutating the global:

```python
import iscc_sct as sct

opts = sct.sct_opts.override({"bits": 128})
print(opts.bits)  # 128
print(sct.sct_opts.bits)  # 64 — the global is unchanged
```

Prefer `override()` or per-call keyword arguments over assigning to `sct_opts` fields directly. Both
keep the global default predictable for other code in the same process.

## Validation

Options are validated whenever they are set. An out-of-range or wrong-shaped value raises a pydantic
`ValidationError`:

```python
import iscc_sct as sct

sct.sct_opts.override({"bits": 100})  # not a multiple of 32 -> ValidationError
sct.sct_opts.override({"max_tokens": 200})  # exceeds 127 -> ValidationError
```

!!! warning "Library and CLI defaults differ"

    `create()` and `SctOptions` default `bits` to **64**. The `iscc-sct` command-line tool defaults
    `--bits` to **256**. The same text therefore yields a different code length depending on which entry
    point you use. Set `bits` explicitly when you need a specific length.

## Related pages

- **[Compare texts](compare-texts.md)** — How `bits` affects matching.
- **[Granular features](granular-features.md)** — `bits_granular`, offsets, and chunk contents.
- **[Command line](command-line.md)** — Setting options from the CLI.
- **[API reference](../reference/api.md)** — The `SctOptions` model in full.
