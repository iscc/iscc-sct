# ISCC - Semantic Text-Code

[![Tests](https://github.com/iscc/iscc-sct/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-sct/actions/workflows/tests.yml)
[![Version](https://img.shields.io/pypi/v/iscc-sct.svg)](https://pypi.python.org/pypi/iscc-sct/)
[![Downloads](https://pepy.tech/badge/iscc-sct)](https://pepy.tech/project/iscc-sct)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iscc/iscc-sct)

> [!CAUTION]
> **This is a proof of concept.** All releases with version numbers below v1.0.0 may break backward
> compatibility and produce incompatible Semantic Text-Codes. The algorithms of this `iscc-sct`
> repository are experimental and not part of the official
> [ISO 24138:2024](https://www.iso.org/standard/77899.html) standard.

`iscc-sct` is a **Semantic-Code Text** implementation for the [ISCC](https://core.iscc.codes)
(*International Standard Content Code*). The Semantic-Code Text is a new ISCC-UNIT for semantic text
identification. The algorithm creates simmilar (low hamming distance) codes for semantically similar
text inputs across different languages. The SCT ISCC-UNIT is a compact binary code created from a
binarized document-vector text-embeddings.

## Quick Start

```bash
# Install the package with CPU inference
pip install "iscc-sct[cpu]"

# Generate a semantic code
python -c "import iscc_sct as sct; print(sct.create('Your text here').iscc)"

# Or use the CLI
iscc-sct "path/to/textfile.txt"

# Or run the CLI directly via uvx without installation
uvx "iscc-sct[cpu]" "path/to/textfile.txt"
```

## What is the ISCC

The ISCC is a combination of various similarity preserving fingerprints and an identifier for
digital media content.

ISCCs are generated algorithmically from digital content, just like cryptographic hashes. However,
instead of using a single cryptographic hash function to identify data only, the ISCC uses various
algorithms to create a composite identifier that exhibits similarity-preserving properties (soft
hash or Simprint).

The component-based structure of the ISCC identifies content at multiple levels of abstraction. Each
component is self-describing, modular, and can be used separately or with others to aid in various
content identification tasks. The algorithmic design supports content deduplication, database
synchronization, indexing, integrity verification, timestamping, versioning, data provenance,
similarity clustering, anomaly detection, usage tracking, allocation of royalties, fact-checking and
general digital asset management use-cases.

## Comparison with Standard ISCC Content-Code Text

| Feature       | ISCC Content-Code Text   | ISCC Semantic-Code Text           |
| ------------- | ------------------------ | --------------------------------- |
| Focus         | Lexical similarity       | Semantic similarity               |
| Cross-lingual | No                       | Yes                               |
| Use case      | Near-duplicate detection | Semantic similarity, translations |

## What is ISCC Semantic Text-Code?

The ISCC framework already includes a Text-Code based on lexical similarity for near-duplicate
matching. The ISCC Semantic Text-Code is a planned additional ISCC-UNIT focused on capturing a more
abstract and broader semantic similarity. It is engineered to be robust against a wide range of
variations and, most remarkably, translations of text that cannot be matched based on lexical
similarity alone.

### Translation Matching

One of the most interesting aspects of the Semantic Text-Code is its ability to generate
**(near)-identical codes for translations or paraphrased versions of the same text**. This means
that the same content, expressed in different languages, can be identified and linked, opening up
new possibilities for cross-lingual content identification and similarity detection.

## Key Features

- **Semantic Similarity**: Utilizes deep learning models to generate codes that reflect the semantic
    essence of text.
- **Translation Matching**: Creates nearly identical codes for text translations, enabling
    cross-lingual content identification.
- **Bit-Length Flexibility**: Supports generating codes of various bit lengths (up to 256 bits),
    allowing for adjustable granularity in similarity detection.
- **ISCC Compatible**: Generates codes fully compatible with the ISCC specification, facilitating
    seamless integration with existing ISCC-based systems.

## Installation

Ensure you have Python 3.10 or newer installed on your system. The package requires an ONNX runtime
that is selected via install extras. For CPU inference (works everywhere):

```bash
pip install "iscc-sct[cpu]"
```

For NVIDIA CUDA accelerated inference (requires CUDA 12.x and cuDNN 9.x):

```bash
pip install "iscc-sct[gpu]"
```

> [!NOTE]
> Install exactly **one** of the `cpu`/`gpu` extras. The underlying `onnxruntime` and
> `onnxruntime-gpu` packages unpack into the same directory and overwrite each other, so installing
> both silently disables GPU support. A plain `pip install iscc-sct` installs no ONNX runtime; the
> first code generation then fails with install instructions.

Not sure which extra to install, or seeing a GPU that won't engage? Run the built-in check — it
inspects your environment and, on confirmation, installs the right runtime:

```shell
iscc-sct doctor
```

## Usage

Generate a Semantic Text-Code using the create function:

```pycon
>>> import iscc_sct as sct
>>> text = "This is some sample text. It can be a longer document or even an entire book."
>>> sct.create(text, bits=256)
{
  "iscc": "ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI",
  "characters": 77
}

```

For granular (per chunk) feature outputs:

```pycon
>>> import iscc_sct as sct
>>> text = "This is some sample text. It can be a longer document or even an entire book."
>>> sct.create(text, bits=256, granular=True)
{
  "iscc": "ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI",
  "characters": 77,
  "features": [
    {
      "maintype": "semantic",
      "subtype": "text",
      "version": 0,
      "byte_offsets": false,
      "simprints": [
        {
          "simprint": "XZjeSfdyVi0",
          "offset": 0,
          "size": 77,
          "content": "This is some sample text. It can be a longer document or even an entire book."
        }
      ]
    }
  ]
}

```

> [!TIP]
> By default, granular features (simprints) report their offsets as character positions. If the
> `byte_offsets` option is enabled (via the ISCC_SCT_BYTE_OFFSETS environment variable or as an
> option in code), the offsets will be computed on the UTF-8 representation of the text. This can be
> useful when you need to retrieve individual text chunks via random access from remote storage.

### Comparing Two Texts

```python
import iscc_sct as sct

# Generate codes for two texts
text1 = """
An ISCC applies to a specific digital asset and is a data-descriptor deterministically constructed
from multiple hash digests using the algorithms and rules in this document. This document does not
provide information on registration of ISCCs.
"""

text2 = """
Ein ISCC bezieht sich auf ein bestimmtes digitales Gut und ist ein Daten-Deskriptor, der
deterministisch aus mehreren Hash-Digests unter Verwendung der Algorithmen und Regeln in diesem
Dokument erstellt wird. Dieses Dokument enthält keine Informationen über die Registrierung von ISCCs.
"""

code1 = sct.create(text1)
code2 = sct.create(text2)

distance = sct.iscc_distance(code1.iscc, code2.iscc)
print(f"Hamming distance in bits: {distance}")
```

The installation also provides an iscc-sct command-line tool:

```shell
usage: iscc-sct [-h] [-b BITS] [-g] [-d] [-y] [path]

Generate Semantic Text-Codes for text files.

positional arguments:
  path             Path to text files (glob patterns), 'doctor' to check the
                   ONNX runtime, or 'gui' for the demo.

options:
  -h, --help       show this help message and exit
  -b, --bits BITS  Bit-Length of Code (default 256)
  -g, --granular   Activate granular processing.
  -d, --debug      Show debugging messages.
  -y, --yes        Auto-confirm the 'doctor' runtime install.
```

## How It Works

```
Text Input → Text Chunking → Embedding Generation → Vector Aggregation → Binarization → ISCC Encoding
```

`iscc-sct` employs the following process:

1. Splits the text into overlaping chunks (using syntactically sensible breakpoints).
1. Uses a pre-trained deep learning model for text embedding.
1. Generates feature vectors capturing essential characteristics of the chunks.
1. Aggregates these vectors and binarizes them to produce a Semantic Text-Code.
1. Prefixes the binarized vector with the matching ISCC header, encodes it with base32, and adds the
    "ISCC:" prefix.

This process ensures robustness to variations and translations, enabling cross-lingual matching
based on a short Simprint.

## Configuration

ISCC-SCT can be configured using environment variables:

| Environment Variable | Description                          | Default |
| -------------------- | ------------------------------------ | ------- |
| ISCC_SCT_BITS        | Default bit-length of generated code | 64      |
| ISCC_SCT_MAX_TOKENS  | Maximum tokens per chunk             | 127     |
| ISCC_SCT_OVERLAP     | Maximum token overlap between chunks | 48      |

See iscc_sct/options.py for more configuration settings.

## Performance Considerations

- The embedding model will be downloaded on first execution
- **CPU vs GPU**: On systems with CUDA-compatible GPUs, install with `pip install "iscc-sct[gpu]"`
    for significantly faster processing (~17x on large documents).

## Development and Contributing

We welcome contributions to enhance the capabilities and efficiency of this proof of concept. For
development, install the project in development mode using [uv](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/iscc/iscc-sct.git
cd iscc-sct
uv sync --extra demo
```

Run code formatting and the test suite with:

```shell
uv run poe all
```

If you have suggestions for improvements or bug fixes, please open an issue or pull request. For
major changes, please open an issue first to discuss your ideas.

**We particularly welcome recommendations for other multilingual text embedding models trained with
Matryoshka Representation Learning (MRL) and optimized for binarization. Such contributions could
significantly improve the performance and efficiency of the ISCC Semantic Text-Code generation.**

## Gradio Demo

This repository also provides an interactive Gradio demo that allows you to explore the capabilities
of ISCC Semantic Text-Code. The demo showcases:

- Generation of ISCC Semantic Text-Codes for input texts
- Comparison of two texts and their similarity based on the generated codes
- Visualization of text chunking and granular matches
- Adjustable parameters like ISCC bit-length and maximum tokens per chunk

You can access the live version of the Gradio demo at:
[https://huggingface.co/spaces/iscc/iscc-sct](https://huggingface.co/spaces/iscc/iscc-sct)

### Running the Gradio Demo Locally

To run the Gradio demo locally, you first need to install the `iscc-sct` package with the optional
`demo` dependency:

```shell
pip install "iscc-sct[cpu,demo]"
```

This will ensure that Gradio and other necessary dependencies for the demo are installed.

After installation, you can use the `iscc-sct` command-line tool that comes with the package:

```shell
iscc-sct gui
```

This command will launch the Gradio interface in your default web browser, allowing you to interact
with the demo on your local machine.

## Current Limitations

- The semantic matching works best for texts with at least several sentences.
- Very short texts (a few words) may not generate reliable semantic codes.
- Performance may vary across different language pairs.
- The model size is approximately 450MB, which may impact initial loading time.

## Supported Languages

Arabic, Armenian, Bengali, Bosnian, Bulgarian, Burmese, Catalan, Chinese (China), Chinese (Taiwan),
Croatian, Czech, Danish, Dutch, English, Estonian, Farsi, Finnish, French, French (Canada),
Galician, German, Greek, Gujarati, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian,
Japanese, Kannada, Korean, Kurdish, Latvian, Lithuanian, Macedonian, Malay, Malayalam, Marathi,
Mongolian, Norwegian Bokmål, Persian, Polish, Portuguese, Portuguese (Brazil), Romanian, Russian,
Serbian, Sinhala, Slovak, Slovenian, Spanish, Swedish, Tamil, Telugu, Thai, Turkish, Ukrainian,
Urdu, Vietnamese.

## Citation

If you use ISCC-SCT in your research, please cite:

```bibtex
@software{iscc_sct,
  author = {Pan, Titusz},
  title = {ISCC-SCT: Semantic Text-Code for the International Standard Content Code},
  url = {https://github.com/iscc/iscc-sct},
  version = {0.2.0},
  year = {2026},
}
```

## Future Work

### Shift Resistant Semantic Chunking

The current chunking strategy uses tries to maximize chunk sizes (up to 127 tokens) while still
splitting at lexically sensible boundaries with an overlap of up to 48 tokens. See
[text-splitter](https://github.com/benbrandt/text-splitter).

Cross-document chunk matching via granular Simprints can likely be improved significantly with a
semantically aware and shift-resistant chunking strategy. Better shift resistance would improve the
chances that the bounderies detected for semantically similar text sequences in different documents
are aligned.

### MRL based Embeddings

A text embedding model trained with
[Matryoshka Representation Learning](https://arxiv.org/pdf/2205.13147) may yield better results with
short 64-bit Semantic Text-Codes.

### Larger Chunk Sizes

A text embedding model with support for a larger `max_token` size (currently 128) may yield
higher-order granular simprints based on larger chunks of text.

## Acknowledgements

- Text Chunking: [text-splitter](https://github.com/benbrandt/text-splitter)
- Text Embeddings:
    [Sentence-Transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
