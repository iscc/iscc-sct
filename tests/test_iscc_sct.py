from pathlib import Path

import pytest
from blake3 import blake3

import iscc_sct as sct
from iscc_sct.code_semantic_text import (
    split_text,
    tokenize_chunks,
    embed_tokens,
    embed_chunks,
)
import numpy as np


HERE = Path(__file__).parent.absolute()


def test_version():
    assert sct.__version__ == "0.1.0"


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
    assert len(result["embedding"]) == 384


def test_code_text_semantic_features():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, features=True)
    assert result == {
        "iscc": "ISCC:CAA636IXQD736IGJ",
        "characters": 12076,
        "features": [
            "44ERPEPRGHRFC",
            "N5SRLQPXG7BAS",
            "VL4VLULOW6Z52",
            "UM6BG4DRT6ZFQ",
            "U34VPVDTQ6JNY",
            "424ZPBD7A6JP2",
            "7IOQ3Z6VV5BW2",
            "556U7RFZW6R7S",
            "5POV7RHLW7QPG",
            "5NOR7QFPV6YHO",
            "BX6D7BX3U6JX2",
            "4IYVPEMRWORVS",
            "7ZRRORV3WBJUS",
            "HR6RPEXXVDAG6",
            "FTOSNUNFHTQAK",
            "PRBCJUHNXU2CC",
            "HXEQ5QC5DIRIW",
            "3562MRA3DYQIW",
            "7XNQERLLWYQIE",
            "VW3YAAMLDYRMU",
            "PFV3ECTKGYQAW",
            "7R3QGS3OX3AAW",
            "7F7QOFUVXIAAG",
            "6HWQJF4VDYQYW",
            "LPWC7B5UFYQIS",
            "FT3ZFEJQFYAIC",
            "7IIQEUTUFBA6S",
            "7OI2DMAWCAG7A",
            "RX2IPIKWMEUPG",
            "VT2K7ELVXKQXS",
            "BRPP6AMVCASPS",
            "JVN5NI7NCE2OO",
            "JTNRPM3LA4YKG",
            "VHAQKQDZCQQYC",
            "QHIRKAD3CUUKS",
            "JXPZJA7LS5QOS",
            "HTNRP5PBTFAEW",
            "PRMTOM3LXFBES",
            "JT5ZPM3LCG3E6",
        ],
    }


def test_code_text_semantic_offsets():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, offsets=True)
    assert result["offsets"][:3] == [0, 277, 612]


def test_code_text_semantic_chunks():
    fp = HERE / "en.txt"
    result = sct.code_text_semantic(fp, chunks=True)
    assert len(result["chunks"]) == 39
    assert result["chunks"][0].startswith("\n Thank ")
    assert result["chunks"][-1].endswith("(Applause)\n")


def test_gen_text_code_semantic_empty():
    with pytest.raises(ValueError) as excinfo:
        sct.gen_text_code_semantic("")
    assert str(excinfo.value) == "Input text cannot be empty."


def test_gen_text_code_semantic_checks_bits():
    with pytest.raises(ValueError):
        sct.gen_text_code_semantic("Test", bits=99)


def test_split_text(text_en):
    chunks = split_text(text_en)
    assert chunks[0][1][:8] == "\n Thank "
    assert chunks[-1][1][:8] == "\n (Laugh"


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
    assert result["embedding"][:3] == pytest.approx(
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
