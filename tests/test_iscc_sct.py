import iscc_sct as sct
from iscc_sct.code_semantic_text import split_text, tokenize_chunks, embed_tokens
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
    np.testing.assert_array_equal(
        embeddings[0][0][:3], np.array([0.05907335, 0.11408358, 0.12727071], dtype=np.float32)
    )


# def test_code_text_semantic_default(text_en):
#     result = sct.code_text_semantic(text_en)
#     assert result["iscc"] == "ISCC:..."
#
#
# def test_code_image_semantic_256bit(text_en):
#     result = sct.code_text_semantic(text_en, bits=256)
#     assert result["iscc"] == "ISCC:..."
#
#
# def test_gen_image_code_semantic():
#     result = sct.gen_text_code_semantic([1.1, 2.2])
#     assert result["iscc"] == "ISCC:..."
#
#
# def test_models():
#     from iscc_sct.code_semantic_text import model
#     engine = model()
#     assert engine
