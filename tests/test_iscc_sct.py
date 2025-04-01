from pathlib import Path

import pytest
from blake3 import blake3

import iscc_sct as sct
from iscc_sct.code_semantic_text import (
    split_text,
    tokenize_chunks,
    embed_tokens,
    embed_chunks,
    compress,
)
import numpy as np


HERE = Path(__file__).parent.absolute()

TEXT = """
`iscc-sct` is a **proof of concept implementation** of a semantic Text-Code for the
[ISCC](https://core.iscc.codes) (*International Standard Content Code*). Semantic Text-Codes are
designed to capture and represent the language agnostic semantic content of text for improved
similarity detection.

The ISCC framework already comes with a Text-Code that is based on lexical similarity and can match
near duplicates. The ISCC Semantic Text-Code is planned as a new additional ISCC-UNIT focused on
capturing a more abstract and broad semantic similarity. As such the Semantic Text-Code is
engineered to be robust against a broader range of variations and translations of text that cannot
be matched based on lexical similarity.
"""


def test_version():
    assert sct.__version__ == "0.1.3"


def test_code_text_semantic_default():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp)
    assert result == {
        "iscc": "ISCC:CAA636IXQD736IGJ",
        "characters": 12076,
    }


def test_code_text_semantic_no_chars():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, characters=False)
    assert result == {"iscc": "ISCC:CAA636IXQD736IGJ"}


def test_code_text_semantic_embedding():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, embedding=True)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert len(result["features"][0]["embedding"]) == 384


def test_code_text_semantic_features():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, simprints=True)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert result["characters"] == 12076
    assert result["features"][0]["simprints"][:3] == ["5wkXkfEx4lE", "b2UVwfc3wgk", "qvlV0W63s90"]
    assert result["features"][0]["simprints"][-3:] == ["PNsX9eGZQEs", "fFk3M2u5Qkk", "TPuXs2sRtk8"]


def test_code_text_semantic_offsets():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, offsets=True)
    assert result["features"][0]["offsets"][:3] == [0, 277, 612]


def test_code_text_semantic_chunks():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, contents=True)
    assert len(result["features"][0]["contents"]) == 39
    assert result["features"][0]["contents"][0].startswith("\n Thank ")
    assert result["features"][0]["contents"][-1].endswith("(Applause)\n")


def test_code_text_semantic_sizes():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, sizes=True)
    # fmt: off
    assert result["features"][0]["sizes"] == [
        440, 396, 431, 385, 440, 380, 406, 477, 415, 536, 280, 449, 446, 442, 443, 444, 451, 485,
        477, 439, 517, 430, 468, 394, 531, 448, 421, 503, 376, 403, 513, 477, 393, 375, 555, 533,
        312, 455, 413
    ]
    # fmt: on


def test_gen_text_code_semantic_empty():
    with pytest.raises(ValueError) as excinfo:
        sct.gen_text_code_semantic("")
    assert str(excinfo.value) == "Input text cannot be empty."


def test_gen_text_code_semantic_granular():
    result = sct.gen_text_code_semantic(
        TEXT,
        simprints=True,
        offsets=True,
        contents=True,
    )
    assert result == {
        "characters": 726,
        "iscc": "ISCC:CAARISHPJHEXQAYL",
        "features": [
            {
                "maintype": "semantic",
                "subtype": "text",
                "version": 0,
                "simprints": ["FWjtTcl4Aws", "lAjHSc1wAws"],
                "offsets": [0, 297],
                "contents": [
                    "\n"
                    "`iscc-sct` is a **proof of concept implementation** of a semantic "
                    "Text-Code for the\n"
                    "[ISCC](https://core.iscc.codes) (*International Standard Content "
                    "Code*). Semantic Text-Codes are\n"
                    "designed to capture and represent the language agnostic semantic "
                    "content of text for improved\n"
                    "similarity detection.\n"
                    "\n",  # NOTE: end of first chunk (see comma :)
                    "\n"
                    "\n"
                    "The ISCC framework already comes with a Text-Code that is based "
                    "on lexical similarity and can match\n"
                    "near duplicates. The ISCC Semantic Text-Code is planned as a new "
                    "additional ISCC-UNIT focused on\n"
                    "capturing a more abstract and broad semantic similarity. As such "
                    "the Semantic Text-Code is\n"
                    "engineered to be robust against a broader range of variations and "
                    "translations of text that cannot\n"
                    "be matched based on lexical similarity.\n",
                ],
            }
        ],
    }


def test_gen_text_code_semantic_checks_bits():
    with pytest.raises(ValueError):
        sct.gen_text_code_semantic("Test", bits=99)


def test_split_text(text_en):
    chunks = split_text(text_en)
    assert chunks[0][1][:8] == "\n Thank "
    assert chunks[-1][1][:8] == "\n (Laugh"


def test_split_text_override():
    text = "Try some very small and granular text splitting. Use options override for it."
    chunks = split_text(text, max_tokens=8, overlap=4)
    assert chunks == [
        (0, "Try some very small and granular text "),
        (20, "and granular text splitting. "),
        (49, "Use options override for it."),
    ]


def test_tokenize_chunks():
    chunks = ["Hello World", "These are chunks"]
    result = tokenize_chunks(chunks)
    np.testing.assert_array_equal(
        result["input_ids"],
        np.array([[0, 35378, 6661, 2, 1, 1], [0, 32255, 621, 7839, 1224, 2]], dtype=np.int64),
    )


def test_embed_tokens():
    chunks = ["Hello World", "These are chunks"]
    tokens = tokenize_chunks(chunks)
    embeddings = embed_tokens(tokens)
    assert list(embeddings[0][0][:3]) == pytest.approx(
        [0.05907335, 0.11408358, 0.12727071], rel=1e-2
    )


def test_embed_chunks():
    chunks = ["Hello World"]
    expected = [0.008697219, 0.038051583, 0.043976285]
    embeddings = embed_chunks(chunks)
    assert list(embeddings[0][:3]) == pytest.approx(expected, rel=1e-3)


def test_gen_text_code_semantic(text_en):
    result = sct.gen_text_code_semantic(text_en, embedding=True)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert result["features"][0]["embedding"][:3] == pytest.approx(
        [0.03241169825196266, 0.022712377831339836, 0.050273094326257706],
        rel=1e-3,
    )


def test_cross_lingual_match(text_en, text_de):
    a = sct.gen_text_code_semantic(text_en)["iscc"]
    assert a == "ISCC:CAA636IXQD736IGJ"
    b = sct.gen_text_code_semantic(text_de)["iscc"]
    assert b == "ISCC:CAA636IXQD4TMIGL"  # hamming distance for the codes is 6 bits


def test_tokenizer_integrity(text_en):
    # test if updates break tokenizer compatibility
    hasher = blake3()
    for idx, chunk in split_text(text_en):
        hasher.update(chunk.encode("utf-8"))
    checksum = hasher.hexdigest()
    assert checksum == "7a7ad1ce83c36f853d31390150403e225bac7825a5573dd5c9e326b0917c7b52"


def test_soft_hash_text_semantic():
    result = sct.soft_hash_text_semantic("Hello World")
    assert (
        result.hex()
        == "f36789d8d1bbe351106bdf8e9b5006a3fc4cb1eb4042c75ea26b5058857c9177705429237858e9940e133c8b12ee1a3d"
    )


def test_shift_resistance(text_en):
    a = sct.soft_hash_text_semantic(text_en)
    shifted = "Just put another sentence in the begginging of the text!\n" + text_en
    b = sct.soft_hash_text_semantic(shifted)
    # TODO improve algorithm with more shift resistant semantic chunking
    # On 256-bit code
    assert sct.hamming_distance(a, b) == 6
    # On 64-bit code
    assert sct.hamming_distance(b[:16], a[:16]) == 1


def test_compress():
    arr1 = np.array([3.0, 15294.7789, 32977.7])
    arr2 = np.array([3.0, 15294.7789, 32977.7], dtype=np.float32)
    expected = [3.0, 15294.8, 32977.7]
    assert compress(arr1, 1) == expected
    assert compress(arr2, 1) == expected


def test_utf32be_chunk_retrieval():
    """Test that we can retrieve text chunks using UTF-32BE encoding with offset/size * 4."""
    # Generate text code with features
    text = (
        "Hello world! 你好世界! こんにちは! 안녕하세요! مرحبا! שלום! Ç 가 Ω ℍ ① ︷ i⁹ ¼ ǆ ⫝̸ ȴ ȷ ɂ ć "
        "Iñtërnâtiôn\nàlizætiøn☃💩 –  is a tric\t ky   thing!\r"
    )
    text += TEXT
    result = sct.gen_text_code_semantic(
        text, simprints=True, offsets=True, sizes=True, contents=True
    )

    # Convert text to UTF-32BE
    text_utf32be = text.encode("utf-32be")

    # For each feature, retrieve the chunk using offset and size
    features = result["features"][0]
    for i, simprint in enumerate(features["simprints"]):
        offset = features["offsets"][i]
        size = features["sizes"][i]
        original_chunk = features["contents"][i]

        # Calculate byte offset and size in UTF-32BE
        byte_offset = offset * 4
        byte_size = size * 4

        # Retrieve chunk from UTF-32BE encoded text
        chunk_bytes = text_utf32be[byte_offset : byte_offset + byte_size]
        retrieved_chunk = chunk_bytes.decode("utf-32be")

        # Verify retrieved chunk matches the original
        assert retrieved_chunk == original_chunk, f"Chunk mismatch at index {i}"


def test_embedding_precision():
    d16 = sct.gen_text_code_semantic("Hello World", embedding=True, precision=4)
    assert d16["features"][0]["embedding"][0] == 0.0087
