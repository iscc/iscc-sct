from iscc_sct.demo import (
    compute_iscc_code,
    compare_codes,
    hamming_to_cosine,
    generate_similarity_bar,
    recalculate_iscc,
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


def test_process_text():
    # Test with valid input
    result = process_text("Hello, world!", 64, "a")
    assert isinstance(result, dict)
    assert len(result) == 2
    key, value = next(iter(result.items()))
    assert isinstance(key, gr.components.Textbox)
    assert isinstance(value, gr.components.Textbox)
    assert value.value == "ISCC:CAA7GY4JTDI3XZYV"

    # Test with empty input
    result = process_text("", 64, "b")
    assert result is None

    # Test with different suffix
    result = process_text("Test", 64, "b")
    assert len(result) == 2
    key, value = next(iter(result.items()))
    assert isinstance(key, gr.components.Textbox)
    assert isinstance(value, gr.components.Textbox)


@patch("iscc_sct.demo.sct.gen_text_code_semantic")
@patch("iscc_sct.demo.compare_codes")
def test_recalculate_iscc(mock_compare_codes, mock_gen_text_code):
    mock_gen_text_code.side_effect = lambda text, bits: {"iscc": f"ISCC:{text[:4].upper()}{bits}"}
    mock_compare_codes.return_value = "<similarity_html>"

    # Test with both texts non-empty
    result = recalculate_iscc("Hello", "World", 64)
    assert len(result) == 3
    assert isinstance(result[0], gr.components.Textbox)
    assert result[0].value == "ISCC:HELL64"
    assert isinstance(result[1], gr.components.Textbox)
    assert result[1].value == "ISCC:WORL64"
    assert result[2] == "<similarity_html>"

    # Test with first text empty
    result = recalculate_iscc("", "World", 128)
    assert len(result) == 3
    assert isinstance(result[0], gr.components.Textbox)
    assert result[0].value is None
    assert isinstance(result[1], gr.components.Textbox)
    assert result[1].value == "ISCC:WORL128"
    assert result[2] is None

    # Test with second text empty
    result = recalculate_iscc("Hello", "", 256)
    assert len(result) == 3
    assert isinstance(result[0], gr.components.Textbox)
    assert result[0].value == "ISCC:HELL256"
    assert isinstance(result[1], gr.components.Textbox)
    assert result[1].value is None
    assert result[2] is None

    # Test with both texts empty
    result = recalculate_iscc("", "", 64)
    assert len(result) == 3
    assert isinstance(result[0], gr.components.Textbox)
    assert result[0].value is None
    assert isinstance(result[1], gr.components.Textbox)
    assert result[1].value is None
    assert result[2] is None

    # Verify function calls
    assert mock_gen_text_code.call_count == 4
    assert mock_compare_codes.call_count == 1
