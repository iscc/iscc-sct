# -*- coding: utf-8 -*-
"""Chunking test vectors that freeze the exact `split_text` boundaries.

Chunk boundaries determine granular simprints and the final Semantic-Code, so any change to
the chunking pipeline (text-splitter version, tokenizer, chunk sizer implementation) must
reproduce these vectors bit-exactly to remain backward compatible.

The vectors in `chunking_vectors.json` pin chunk offsets and sizes for a variety of text
shapes (including the pathological PDF-extraction shape from issue #24) and option
combinations. Texts are built deterministically from the committed fixtures `en.txt` and
`de.txt` plus small synthetic literals; a sha256 of each input text guards against drift
between the vector file and the text construction below.

Regenerate the vector file (ONLY for an intentional, versioned chunking algorithm change):

    uv run python tests/test_chunking_vectors.py
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from iscc_sct.code_semantic_text import split_text


HERE = Path(__file__).parent.absolute()
VECTORS_PATH = HERE / "chunking_vectors.json"


def collapse_newlines(text):
    # type: (str) -> str
    """Collapse all newline runs to single newlines (PDF-extraction-like text shape)."""
    return re.sub(r"[\r\n]+", "\n", text)


def pathological(text):
    # type: (str) -> str
    """
    Apply the issue #24 trigger shape to text.

    No blank lines anywhere except a single trailing paragraph break. This makes the
    text-splitter probe from each chunk position all the way to the distant trailing
    separator, which exposed the super-linear chunking behavior.
    """
    return collapse_newlines(text) + "\n\nEnde."


def vector_cases():
    # type: () -> dict[str, tuple[str, dict]]
    """Build all vector cases as {name: (text, split_text options)}."""
    en = (HERE / "en.txt").read_text(encoding="utf-8")
    de = (HERE / "de.txt").read_text(encoding="utf-8")
    cjk = "数据是新的石油它推动着现代经济的发展与变革。" * 500
    long_word = "hypermodularization" * 600
    unk_runs = ("𓀀" * 100 + " ") * 120
    level3 = ("Ein kurzer Absatz über die Dinge des Lebens. " * 5 + "\n\n") * 100 + "\n\nEnde."
    # PDF-extraction shape whose words are separated by NBSP (U+00A0) instead of ASCII space;
    # routed to the guarded splitter and exercises its Unicode-whitespace cut search.
    nbsp = (chr(0xA0).join(["Inhalt"] * 4000)) + "\n\nEnde."
    return {
        "en-default": (en, {}),
        "de-default": (de, {}),
        "en-trim": (en, {"trim": True}),
        "de-overlap0": (de, {"overlap": 0}),
        "de-small-chunks": (de, {"max_tokens": 32, "overlap": 16}),
        "en-byte-offsets": (en, {"byte_offsets": True}),
        "en-pathological": (pathological(en), {}),
        "de-pathological": (pathological(de), {}),
        "de-pathological-trim": (pathological(de), {"trim": True}),
        "en-collapsed-no-breaks": (collapse_newlines(en), {}),
        "en-crlf": (en.replace("\n", "\r\n"), {}),
        "de-multi-blank": (de.replace("\n\n", "\n\n\n\n"), {}),
        "cjk-pathological": (cjk + "\n\n完", {}),
        "long-word-pathological": (long_word + " Ende\n\nEnde.", {}),
        "unk-runs-pathological": (unk_runs + "\n\nEnde.", {}),
        "nbsp-pathological": (nbsp, {}),
        "mixed-level-pathological": (level3, {}),
        "whitespace-only": ("  \t \n\n     ", {}),
        "tiny": ("Hello, World! Schöne Grüße aus München. 😀", {}),
        "unicode-mix": (
            "Café ‍naïve 😀🎉 سلام z̧álgo\n\nEnde.",
            {},
        ),
    }


def text_hash(text):
    # type: (str) -> str
    """Return the sha256 hex digest of text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunks_hash(chunks):
    # type: (list[str]) -> str
    """Return the sha256 hex digest over all chunk contents."""
    return hashlib.sha256("\x1f".join(chunks).encode("utf-8")).hexdigest()


def load_vectors():
    # type: () -> dict
    """Load the frozen chunking vectors from chunking_vectors.json."""
    with VECTORS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_vectors():
    # type: () -> dict
    """Compute chunking vectors for all cases with the current implementation."""
    vectors = {}
    for name, (text, options) in vector_cases().items():
        result = split_text(text, **options)
        offsets = [offset for offset, _ in result]
        sizes = [len(chunk) for _, chunk in result]
        vectors[name] = {
            "options": options,
            "text_sha256": text_hash(text),
            "offsets": " ".join(str(o) for o in offsets),
            "sizes": " ".join(str(s) for s in sizes),
            "chunks_sha256": chunks_hash([chunk for _, chunk in result]),
        }
    return vectors


def test_vector_file_covers_all_cases():
    assert set(load_vectors()) == set(vector_cases())


@pytest.mark.parametrize("name", vector_cases())
def test_chunking_vector(name):
    text, options = vector_cases()[name]
    expected = load_vectors()[name]
    assert text_hash(text) == expected["text_sha256"], "input text construction drifted"
    result = split_text(text, **options)
    offsets = " ".join(str(offset) for offset, _ in result)
    sizes = " ".join(str(len(chunk)) for _, chunk in result)
    assert offsets == expected["offsets"]
    assert sizes == expected["sizes"]
    assert chunks_hash([chunk for _, chunk in result]) == expected["chunks_sha256"]


if __name__ == "__main__":  # pragma: no cover
    VECTORS_PATH.write_text(
        json.dumps(build_vectors(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(build_vectors())} vectors to {VECTORS_PATH}")
