---
icon: lucide/rocket
description: Install iscc-sct, generate your first Semantic Text-Code, and match a translation across languages.
---

# Getting started

Install `iscc-sct`, generate your first Semantic Text-Code, and watch two languages produce
near-identical codes.

## Prerequisites

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Install

`iscc-sct` needs an ONNX runtime, selected through an install extra. The `cpu` extra works
everywhere:

=== "uv"

    ```bash
    uv add "iscc-sct[cpu]"
    ```

=== "pip"

    ```bash
    pip install "iscc-sct[cpu]"
    ```

!!! warning "Choose exactly one runtime"

    Install the `cpu` extra, or the `gpu` extra for NVIDIA CUDA acceleration — never both. The
    `onnxruntime` and `onnxruntime-gpu` packages unpack into the same directory and overwrite each
    other. A plain `pip install iscc-sct` installs no runtime, and the first code generation then fails
    with install instructions. If you are unsure which extra fits your machine, run `iscc-sct doctor`
    (see the [command-line guide](../howto/command-line.md)).

### Verify the installation

```python
import iscc_sct

print(iscc_sct.__version__)
```

## Generate your first code

Pass any text to `create()`. The first call downloads the embedding model (about 450 MB) to your
user data directory; later calls reuse it.

```python
import iscc_sct as sct

text = "This is some sample text. It can be a longer document or even an entire book."
print(sct.create(text, bits=256).iscc)
# ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI
```

The `bits` argument sets the code length. Longer codes carry more detail; 64 bits is the library
default, and 256 bits is the most precise. See [comparing texts](../howto/compare-texts.md) for how
length affects matching.

## Match a translation

The point of a Semantic Text-Code is that meaning survives translation. Generate codes for the same
passage in English and German, then measure their distance:

```python
import iscc_sct as sct

english = (
    "An ISCC applies to a specific digital asset and is a data-descriptor deterministically "
    "constructed from multiple hash digests using the algorithms and rules in this document. "
    "This document does not provide information on registration of ISCCs."
)
german = (
    "Ein ISCC bezieht sich auf ein bestimmtes digitales Gut und ist ein Daten-Deskriptor, der "
    "deterministisch aus mehreren Hash-Digests unter Verwendung der Algorithmen und Regeln in "
    "diesem Dokument erstellt wird. Dieses Dokument enthält keine Informationen über die "
    "Registrierung von ISCCs."
)

a = sct.create(english)
b = sct.create(german)

print(sct.iscc_distance(a.iscc, b.iscc))
# 3
```

A distance of `3` bits out of 64 means the translation is a near-match. For contrast, an unrelated
sentence lands far away:

```python
import iscc_sct as sct

unrelated = (
    "The recipe calls for two cups of flour, a pinch of salt, and three ripe bananas mashed "
    "until smooth before folding the mixture into the buttered baking tin."
)

print(sct.iscc_distance(a.iscc, sct.create(unrelated).iscc))
# 33
```

Low distance means similar meaning; high distance means unrelated content. Translation matching is
the behavior that lexical (word-based) codes cannot provide.

## Look inside the text

Set `granular=True` to get per-chunk features alongside the document code. Each chunk carries its
own offset, size, similarity-preserving fingerprint (simprint), and text:

```python
import iscc_sct as sct

text = "This is some sample text. It can be a longer document or even an entire book."
meta = sct.create(text, bits=256, granular=True)

feature = meta.features[0].simprints[0]
print(feature.offset, feature.size, feature.simprint)
# 0 77 XZjeSfdyVi0
```

Granular features let you match individual passages across documents, even when the surrounding text
differs. The [granular features guide](../howto/granular-features.md) covers chunk-level matching
and content reconstruction.

## Next steps

- **[Compare texts](../howto/compare-texts.md)** — Measure semantic and cross-lingual similarity.
- **[Granular features](../howto/granular-features.md)** — Work with per-chunk simprints and
    offsets.
- **[Configuration](../howto/configuration.md)** — Tune bit-length, chunking, and output options.
- **[How it works](../explanation/how-it-works.md)** — Understand why the codes match across
    languages.
