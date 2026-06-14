---
icon: lucide/terminal
description: Generate Semantic Text-Codes from the command line, process files in bulk, run the runtime doctor, and launch the demo.
---

# Command line

This guide covers the `iscc-sct` command-line tool: generating codes from files, processing many
files at once, checking the ONNX runtime, and launching the demo.

The CLI is installed with the package. Run `iscc-sct --help` to see every option:

```text
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

## Generate a code from a file

Pass a path to print its Semantic Text-Code:

```bash
iscc-sct article.txt
# ISCC:CADV3GG6JH3XEVRNSVYGCLJ7AAV3BOT5J7EHEZKPFXEGRJ2CTWACGZI
```

The CLI defaults to **256-bit** codes. Set a different length with `--bits`:

```bash
iscc-sct --bits 64 article.txt
```

!!! note "The CLI default differs from the library"

    `iscc-sct` defaults to 256 bits, but `create()` in Python defaults to 64. Pass `--bits` (or the
    `bits` argument) explicitly when the length matters. See [configuration](configuration.md).

## Process multiple files

The `path` argument accepts a glob pattern. Quote it so `iscc-sct` expands the pattern itself rather
than the shell:

```bash
iscc-sct "texts/*.txt"
```

Each matching file prints its code. Files that cannot be decoded as UTF-8 are decoded with a
detected character set, and empty files are skipped.

## Granular output

Add `--granular` to print the full metadata — document code plus per-chunk simprints, offsets,
sizes, and contents — as JSON:

```bash
iscc-sct --granular article.txt
```

See [granular features](granular-features.md) for what the fields mean.

## Run without installing

With [`uv`](https://docs.astral.sh/uv/), run the CLI in one line without a permanent install:

```bash
uvx "iscc-sct[cpu]" article.txt
```

## Check the ONNX runtime

`iscc-sct` needs exactly one ONNX runtime (`cpu` or `gpu` extra). The `doctor` command inspects your
environment, names the problem, and recommends the right extra:

```bash
iscc-sct doctor
```

```text
iscc-sct ONNX runtime check

  ONNX runtime:   not installed
  CUDA provider:  no
  NVIDIA GPU:     no

Status: no ONNX runtime installed.
Recommended: pip install "iscc-sct[cpu]"
```

When a fix is available, `doctor` offers to run it. Add `--yes` to install without the prompt:

```bash
iscc-sct doctor --yes
```

`doctor` also detects the case where `onnxruntime-gpu` is installed but a plain `onnxruntime`
package has shadowed it (issue #23), and reinstalls only the GPU build.

## Launch the demo

The `gui` command launches the interactive Gradio demo in your browser. It needs the `demo` extra:

```bash
pip install "iscc-sct[cpu,demo]"
iscc-sct gui
```

## Related pages

- **[Getting started](../tutorials/getting-started.md)** — Install and first code.
- **[Configuration](configuration.md)** — Options and environment variables.
- **[How it works](../explanation/how-it-works.md)** — What happens behind a code.
