from iscc_sct.demo import (
    compute_iscc_code,
    compare_codes,
    hamming_to_cosine,
    generate_similarity_bar,
)


def test_compute_iscc_code():
    text1 = "Hello, world!"
    text2 = "Hallo, Welt!"
    bit_length = 64

    result = compute_iscc_code(text1, text2, bit_length)
    assert len(result) == 3
    assert all(isinstance(code, str) for code in result[:2])
    assert isinstance(result[2], str)


def test_compare_codes():
    code_a = "ISCC:EAAQCVG2TABD6"
    code_b = "ISCC:EAAQCVG2TABD6"
    bits = 64

    result = compare_codes(code_a, code_b, bits)
    assert isinstance(result, str)
    assert "100.00%" in result

    result = compare_codes(None, code_b, bits)
    assert result is None


def test_hamming_to_cosine():
    assert hamming_to_cosine(0, 64) == 1.0
    assert hamming_to_cosine(32, 64) == 0.0
    assert hamming_to_cosine(64, 64) == -1.0


def test_generate_similarity_bar():
    result = generate_similarity_bar(1.0)
    assert "100.00%" in result
    assert "green" in result

    result = generate_similarity_bar(-0.5)
    assert "-50.00%" in result
    assert "red" in result


from unittest.mock import patch, MagicMock
import gradio as gr
from iscc_sct.demo import process_text


@patch("iscc_sct.demo.sct.gen_text_code_semantic")
def test_process_text(mock_gen_text_code):
    mock_gen_text_code.return_value = {"iscc": "ISCC:EAAQCVG2TABD6"}

    # Test with valid input
    result = process_text("Hello, world!", 64, "a")
    assert isinstance(result, dict)
    assert len(result) == 1
    key, value = next(iter(result.items()))
    assert isinstance(key, gr.components.Textbox)
    assert isinstance(value, gr.components.Textbox)
    assert value.value == "ISCC:EAAQCVG2TABD6"

    # Test with empty input
    result = process_text("", 64, "b")
    assert result is None

    # Test with different bit length
    process_text("Test", 128, "a")
    mock_gen_text_code.assert_called_with("Test", bits=128)

    # Test with different suffix
    result = process_text("Test", 64, "b")
    assert len(result) == 1
    key, value = next(iter(result.items()))
    assert isinstance(key, gr.components.Textbox)
    assert isinstance(value, gr.components.Textbox)