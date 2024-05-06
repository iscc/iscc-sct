import iscc_sct as sct


def test_version():
    assert sct.__version__ == "0.1.0"


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
