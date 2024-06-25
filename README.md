# ISCC - Semantic Text-Code

[![Tests](https://github.com/iscc/iscc-sct/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-core/actions/workflows/tests.yml)
[![Version](https://img.shields.io/pypi/v/iscc-sct.svg)](https://pypi.python.org/pypi/iscc-sct/)
[![Downloads](https://pepy.tech/badge/iscc-sct)](https://pepy.tech/project/iscc-sct)

`iscc-sct` is a **proof of concept implementation** of a semantic Text-Code for the [ISCC](https://core.iscc.codes)
(*International Standard Content Code*). Semantic Text-Codes are designed to capture and represent the language
agnostic semantic content of text for improved similarity detection.

> [!CAUTION]
> **This is an early proof of concept.** All releases with version numbers below v1.0.0 may break backward
> compatibility and produce incompatible Semantic Text-Codes.

## What is ISCC Semantic Text-Code?

The ISCC framework already includes a Text-Code based on lexical similarity for near-duplicate matching.
The ISCC Semantic Text-Code is a planned additional ISCC-UNIT focused on capturing a more abstract and broader
semantic similarity. It is engineered to be robust against a wide range of variations and, most remarkably,
translations of text that cannot be matched based on lexical similarity alone.

### Translation Matching

One of the most interesting aspects of the Semantic Text-Code is its ability to generate **(near)-identical codes
for translations of the same text**. This means that the same content, expressed in different languages, can be
identified and linked, opening up new possibilities for cross-lingual content identification and similarity detection.

## Key Features

- **Semantic Similarity**: Utilizes deep learning models to generate codes that reflect the semantic essence of text.
- **Translation Matching**: Creates nearly identical codes for text translations, enabling cross-lingual content identification.
- **Bit-Length Flexibility**: Supports generating codes of various bit lengths (up to 256 bits), allowing for adjustable granularity in similarity detection.
- **ISCC Compatible**: Generates codes fully compatible with the ISCC specification, facilitating seamless integration with existing ISCC-based systems.

## Installation

Ensure you have Python 3.9 or newer installed on your system. Install the library using:

```bash
pip install iscc-sct
```

For systems with GPU CUDA support, enhance performance by installing with:

```bash
pip install iscc-sct[gpu]
```

## Usage

Generate a Semantic Text-Code using the create function:

```python-repl
>>> import iscc_sct as sct
>>> text = "This is some sample text. It can be a longer document or even an entire book."
>>> sct.create(text, bits=256)
{
  "iscc": "ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI",
  "characters": 77
}
```

For granular (per chunk) feature outputs:

```python-repl
>>> import iscc_sct as sct
>>> text = "This is some sample text. It can be a longer document or even an entire book."
>>> sct.create(text, bits=256, granular=True)
{
  "iscc": "ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI",
  "characters": 77,
  "features": [
    {
      "feature": "LWMN4SPXOJLC2",
      "offset": 0,
      "size": 77,
      "text": "This is some sample text. It can be a longer document or even an entire book."
    }
  ]
}
```

The installation also provides a sct command-line tool:

```shell
sct --help
usage: sct [-h] [-b BITS] [-g] [-d] [path]

Generate Semantic Text-Codes for text files.

positional arguments:
  path                  Path to text files (supports glob patterns).

options:
  -h, --help            show this help message and exit
  -b BITS, --bits BITS  Bit-Length of Code (default 256)
  -g, --granular        Activate granular processing.
  -d, --debug           Show debugging messages.
````

## How It Works

`iscc-sct` employs the following process:

1. Splits the text into semantically coherent chunks.
2. Uses a pre-trained deep learning model for text embedding.
3. Generates feature vectors capturing essential characteristics of the chunks.
4. Aggregates these vectors and binarizes them to produce a Semantic Text-Code.

This process ensures robustness to variations and translations, enabling cross-lingual matching.


## Development and Contributing

We welcome contributions to enhance the capabilities, efficiency, and compatibility of this proof of concept with the
broader ISCC ecosystem. For development, install the project in development mode using [Poetry](https://python-poetry.org):

```shell
git clone https://github.com/iscc/iscc-sct.git
cd iscc-sct
poetry install
```

If you have suggestions for improvements or bug fixes, please open an issue or pull request. For major changes, please
open an issue first to discuss your ideas.


## Acknowledgements

- Text Chunking: [semantic-text-splitter](https://github.com/benbrandt/text-splitter)
- Text Embedding: [Sentence-Transformer](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models)

## License

This project is licensed under the CC-BY-NC-SA-4.0 International License.
