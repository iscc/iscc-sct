# ISCC - Semantic Text-Code

[![Version](https://img.shields.io/pypi/v/iscc-sct.svg)](https://pypi.python.org/pypi/iscc-sct/)
[![Downloads](https://pepy.tech/badge/iscc-sct)](https://pepy.tech/project/iscc-sct)

`iscc-sct` is a **proof of concept implementation** of a semantic Text-Code for the
[ISCC](https://core.iscc.codes) (*International Standard Content Code*). Semantic Text-Codes are
designed to capture and represent the language agnostic semantic content of text for improved
similarity detection.

> \[!CAUTION\] This is an early proof of concept. All releases with release numbers below v1.0.0 may
> break backward compatibility and produce incompatible Semantic Text-Codes.

## What is ISCC Semantic Text-Code

The ISCC framework already comes with a Text-Code that is based on lexical similarity and can match
near duplicates. The ISCC Semantic Text-Code is planned as a new additional ISCC-UNIT focused on
capturing a more abstract and broad semantic similarity. As such the Semantic Text-Code is
engineered to be robust against a broader range of variations and translations of text that cannot
be matched based on lexical similarity.

## Features

- **Semantic Similarity**: Leverages deep learning models to generate codes that reflect the
  semantic content of text.
- **Bit-Length Flexibility**: Supports generating codes of various bit lengths (up to 256 bits),
  allowing for adjustable granularity in similarity detection.
- **ISCC Compatible**: Generates codes that are fully compatible with the ISCC specification,
  facilitating integration with existing ISCC-based systems.

## Installation

Before you can install `iscc-sct`, you need to have Python 3.8 or newer installed on your system.
Install the library as follows:

```bash
pip install iscc-sct[cpu]
```

If your system has GPU CUDA support you can improve perfomance by installing with GPU support:

```bash
pip install iscc-sct[gpu]
```

## Usage

To generate a Semantic Text-Code use the `code_text_semantic` function. You can specify the bit
length of the code to control the level of granularity in the semantic representation.

```python
import iscc_sct as sci

# Generate a 64-bit ISCC Semantic Text-Code for an image file
text = "This is some sample text. It can be a longer document or even an entire book."
semantic_code = sci.gen_text_code_semantic(text, bits=64)

print(semantic_code)
```

```shell
{'iscc': 'ISCC:CAAV3GG6JH3XEVRN', 'characters': 77}
```

## How It Works

`iscc-sct` splits the text into chunks and uses a pre-trained deep learning model for text
embedding. The model generates a feature vector that captures the essential characteristics of the
chunks. These vectors are aggregated and then binarized to produce a Semantic Text-Code that is
robust to variations/translations of the text.

## Development

This is a proof of concept and welcomes contributions to enhance its capabilities, efficiency, and
compatibility with the broader ISCC ecosystem. For development, you'll need to install the project
in development mode using [Poetry](https://python-poetry.org).

```shell
git clone https://github.com/iscc/iscc-sct.git
cd iscc-sct
poetry install -E cpu
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an
issue or pull request. For major changes, please open an issue first to discuss what you would like
to change.
