# ISCC - Semantic Text-Code

`iscc-sct` is a **proof of concept implementation** of a semantic Text-Code for the [ISCC](https://core.iscc.codes)
(*International Standard Content Code*). Semantic Text-Codes are designed to capture and represent the language
agnostic semantic content of text for improved similarity detection.

> [!CAUTION]
> **This is an early proof of concept.** All releases with release numbers below v1.0.0 may break backward
> compatibility and produce incompatible Semantic Text-Codes.

## What is ISCC Semantic Text-Code

The ISCC framework already comes with a Text-Code that is based on lexical similarity and can match near duplicates.
The ISCC Semantic Text-Code is planned as a new additional ISCC-UNIT focused on capturing a more abstract and broad
semantic similarity. As such the Semantic Text-Code is engineered to be robust against a broader range of variations
and translations of text that cannot be matched based on lexical similarity.

## Features

- **Semantic Similarity**: Leverages deep learning models to generate codes that reflect the semantic content of text.
- **Bit-Length Flexibility**: Supports generating codes of various bit lengths (up to 256 bits), allowing for
  adjustable granularity in similarity detection.
- **ISCC Compatible**: Generates codes that are fully compatible with the ISCC specification, facilitating integration
  with existing ISCC-based systems.

## Installation

Before you can install `iscc-sct`, you need to have Python 3.8 or newer installed on your system. Install the library
as follows:

```bash
pip install iscc-sct
```

If your system has GPU CUDA support you can improve perfomance by installing with GPU support:

```bash
pip install iscc-sct[gpu]
```

## Usage

To generate a Semantic Text-Code use the `create` function.

```python-repl
>>> import iscc_sct as sci
>>> text = "This is some sample text. It can be a longer document or even an entire book."
>>> sci.create(text)
{
  "iscc": "ISCC:CAAVZHGOJH3XUFRF",
  "characters": 89
}
```

You can also generate granular (per chunk) feature outputs:

```python-repl
>>> import iscc_sct as sci
>>> text = "This is some sample text. It can be a longer document or even an entire book."
>>> sci.create(text, granular=True)
{
  "iscc": "ISCC:CAAV3GG6JH3XEVRN",
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

Installation also creates a simple `sct` command line tool in you python bin/Scripts folder:

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

`iscc-sct` splits the text into chunks and uses a pre-trained deep learning model for text embedding. The model
generates a feature vector that captures the essential characteristics of the chunks. These vectors are aggregated and
then binarized to produce a Semantic Text-Code that is robust to variations/translations of the text.

## Development

This is a proof of concept and welcomes contributions to enhance its capabilities, efficiency, and compatibility with
the broader ISCC ecosystem. For development, you'll need to install the project in development mode using
[Poetry](https://python-poetry.org).

```shell
git clone https://github.com/iscc/iscc-sct.git
cd iscc-sct
poetry install
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or pull request.
For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the CC-BY-NC-SA-4.0 International License.
