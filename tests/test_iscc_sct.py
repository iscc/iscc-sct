import pytest
import iscc_sct as sct
from iscc_sct.code_semantic_text import (
    split_text,
    tokenize_chunks,
    embed_tokens,
    embed_chunks,
    embed_text,
)
import numpy as np


def test_version():
    assert sct.__version__ == "0.1.0"


def test_split_text(text_en):
    chunks = split_text(text_en)
    assert chunks[0][:8] == "\n Thank "
    assert chunks[-1][:8] == "\n (Laugh"


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


def test_embed_text(text_en):
    sct.code_semantic_text._model = None
    sct.onnx_providers = ["CPUExecutionProvider"]
    result = embed_text(text_en)
    assert len(result) == 384
    assert list(result[:3]) == pytest.approx([0.0324117, 0.022712378, 0.050273094], rel=1e-3)


def test_gen_text_code_semantic(text_en):
    result = sct.gen_text_code_semantic(text_en)
    assert result["iscc"] == "ISCC:CAA636IXQD736IGJ"
    assert result["features"][:3] == pytest.approx(
        [0.03241169825196266, 0.022712377831339836, 0.050273094326257706],
        rel=1e-3,
    )


def test_cross_lingual_match(text_en, text_de):
    a = sct.gen_text_code_semantic(text_en)["iscc"]
    assert a == "ISCC:CAA636IXQD736IGJ"
    b = sct.gen_text_code_semantic(text_de)["iscc"]
    assert b == "ISCC:CAA636IXQD4TMIGL"  # hamming distance for the codes is 6 bits
