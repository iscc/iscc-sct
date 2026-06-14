---
icon: lucide/house
description: Cross-lingual, similarity-preserving semantic ISCC code for text content. Semantically similar texts, including translations, produce codes with low hamming distance.
---

# iscc-sct

[![Tests](https://github.com/iscc/iscc-sct/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-sct/actions/workflows/tests.yml)
[![Version](https://img.shields.io/pypi/v/iscc-sct.svg)](https://pypi.python.org/pypi/iscc-sct/)
[![Downloads](https://pepy.tech/badge/iscc-sct)](https://pepy.tech/project/iscc-sct)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/iscc/iscc-sct/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iscc/iscc-sct)

**A cross-lingual, similarity-preserving binary code for text. Semantically similar texts, including
translations, produce codes with low hamming distance.**

!!! warning "Proof of concept"

    All releases below v1.0.0 may break backward compatibility and produce incompatible Semantic
    Text-Codes. The algorithms in `iscc-sct` are experimental and **not** part of the official
    [ISO 24138:2024](https://www.iso.org/standard/77899.html) standard.

## Introduction

The [ISCC](https://iscc.codes) framework already includes a Text-Code based on lexical similarity
for near-duplicate matching. The Semantic Text-Code (SCT) is a planned additional ISCC-UNIT that
captures a broader, more abstract similarity. It is engineered to be robust against rephrasing and,
most notably, translations that lexical matching cannot detect.

`iscc-sct` turns any text into a compact binary code built from a binarized, multilingual document
embedding. The same content expressed in different languages maps to **(near-)identical codes**,
opening up cross-lingual content identification and similarity detection.

| Feature       | ISCC Content-Code Text   | ISCC Semantic-Code Text           |
| ------------- | ------------------------ | --------------------------------- |
| Focus         | Lexical similarity       | Semantic similarity               |
| Cross-lingual | No                       | Yes                               |
| Use case      | Near-duplicate detection | Semantic similarity, translations |

**Key features:**

- **Semantic similarity** - deep multilingual embeddings capture the meaning of the text
- **Translation matching** - near-identical codes for the same content across languages
- **Bit-length flexibility** - codes from 32 up to 256 bits for adjustable granularity
- **ISCC compatible** - codes integrate directly with existing ISCC-based systems
- **60+ languages** - one model covers a broad range of scripts and language pairs

## Quick start

=== "pip"

    ```bash
    pip install "iscc-sct[cpu]"
    ```

=== "uv"

    ```bash
    uv add "iscc-sct[cpu]"
    ```

=== "uvx (no install)"

    ```bash
    uvx "iscc-sct[cpu]" "path/to/textfile.txt"
    ```

Generate a Semantic Text-Code from Python:

```python
import iscc_sct as sct

text = "This is some sample text. It can be a longer document or even an entire book."
print(sct.create(text, bits=256).iscc)
# ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI
```

Or from the command line:

```bash
iscc-sct "path/to/textfile.txt"
```

!!! note "Choose exactly one ONNX runtime"

    A plain `pip install iscc-sct` installs no ONNX runtime; the first code generation then fails with
    install instructions. Install the `cpu` extra (works everywhere) or the `gpu` extra for NVIDIA CUDA
    acceleration (requires CUDA 12.x and cuDNN 9.x), or run `iscc-sct doctor` to detect and install the
    right one. Never install both: `onnxruntime` and `onnxruntime-gpu` unpack into the same directory
    and silently overwrite each other.

## How it works

```text
Text -> split -> embed -> mean-pool -> binarize -> ISCC header + base32
```

1. **Split** the text into overlapping chunks at syntactically sensible boundaries.
1. **Embed** each chunk with a multilingual sentence-transformer running on ONNX.
1. **Aggregate** the chunk embeddings into one mean-pooled, normalized document vector.
1. **Binarize** the vector (positive components become 1-bits), truncate to the requested
    bit-length, prefix the ISCC header, and base32-encode it.

This process is robust to variation and translation, enabling cross-lingual matching from a short
Simprint.

## Documentation

<div class="grid cards" markdown>

- **[Getting started](tutorials/getting-started.md)** - Tutorial

    Install, generate your first code, and match a translation across languages.

- **[How-to guides](howto/compare-texts.md)** - Task recipes

    Compare texts, work with granular features, configure options, and use the CLI.

- **[How it works](explanation/how-it-works.md)** - Understand the design

    The pipeline, cross-lingual matching, and why binarized vectors still match.

- **[API reference](reference/api.md)** - Library details

    Generated reference for `create()`, the data model, options, and similarity functions.

- **[For Coding Agents](reference/for-coding-agents.md)** - Build on iscc-sct

    A dense architecture map, constraints, and task recipes for AI coding agents.

- **[Live Demo :lucide-external-link:](https://huggingface.co/spaces/iscc/iscc-sct)** - Try it in
    the browser

    Generate and compare Semantic Text-Codes interactively on Hugging Face Spaces.

- **[Source Code :lucide-external-link:](https://github.com/iscc/iscc-sct)** - Read the
    implementation

    The full algorithm, CLI, and Gradio demo on GitHub.

- **[Full text for LLMs :lucide-external-link:](llms-full.txt)** - Machine-readable docs

    Every page concatenated into a single file for language models.

</div>

??? note "Supported languages (60+)"

    Arabic, Armenian, Bengali, Bosnian, Bulgarian, Burmese, Catalan, Chinese (China), Chinese (Taiwan),
    Croatian, Czech, Danish, Dutch, English, Estonian, Farsi, Finnish, French, French (Canada),
    Galician, German, Greek, Gujarati, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian,
    Japanese, Kannada, Korean, Kurdish, Latvian, Lithuanian, Macedonian, Malay, Malayalam, Marathi,
    Mongolian, Norwegian Bokmål, Persian, Polish, Portuguese, Portuguese (Brazil), Romanian, Russian,
    Serbian, Sinhala, Slovak, Slovenian, Spanish, Swedish, Tamil, Telugu, Thai, Turkish, Ukrainian,
    Urdu, Vietnamese.

[Source code on GitHub :lucide-external-link:](https://github.com/iscc/iscc-sct){ .md-button }
